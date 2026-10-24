"""Transfer sampling-plan XY values to the agreed DB100 layout (100 units/mm)."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import ipaddress
import json
from pathlib import Path
import struct
import threading

CAPACITY = 64
POINT_START = 10
POINT_SIZE = 10
SCALE = 100
HERE = Path(__file__).resolve().parent


class TransferError(RuntimeError):
    def __init__(self, message, *, notification="not_sent"):
        super().__init__(message)
        self.notification = notification


@dataclass(frozen=True)
class PLCConfig:
    host: str = "192.168.6.10"
    rack: int = 0
    slot: int = 1
    port: int = 102
    db: int = 100

    def __post_init__(self):
        ipaddress.IPv4Address(self.host)
        for name, low, high in (("rack", 0, 7), ("slot", 0, 31),
                                ("port", 1, 65535), ("db", 1, 65535)):
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"PLC 参数 {name} 无效。")

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8-sig")))


def scaled_coordinate(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        raise ValueError(f"{label} 必须是数值。")
    try:
        number = Decimal(str(value))
        if not number.is_finite() or abs(number) > Decimal("21474836.48"):
            raise ValueError(f"{label} 超出 DInt 范围或不是有限数值。")
        scaled = int((number * SCALE).to_integral_value(rounding=ROUND_HALF_UP))
    except InvalidOperation:
        raise ValueError(f"{label} 无法转换。") from None
    if not -(2**31) <= scaled < 2**31:
        raise ValueError(f"{label} 换算后超出 DInt 范围。")
    return scaled


def prepare_plan(plan):
    if not isinstance(plan, dict) or not isinstance(plan.get("points"), list):
        raise ValueError("请选择取样程序导出的完整规划 JSON（必须包含 points）。")
    points = plan["points"]
    if not 1 <= len(points) <= CAPACITY:
        raise ValueError(f"当前有 {len(points)} 个点；DB100 一次只能接收 1–64 个点。未发送任何数据。")
    if "count" in plan and (type(plan["count"]) is not int or plan["count"] != len(points)):
        raise ValueError("规划 count 与实际点数不一致。")
    rows = []
    for index, point in enumerate(points, 1):
        if not isinstance(point, dict) or not {"x_mm", "y_mm"} <= point.keys():
            raise ValueError(f"第 {index} 个点缺少 x_mm 或 y_mm。")
        x = scaled_coordinate(point["x_mm"], f"点 {index} X")
        y = scaled_coordinate(point["y_mm"], f"点 {index} Y")
        offset = POINT_START + (index - 1) * POINT_SIZE
        rows.append({"index": index, "point_id": point.get("id", index),
                     "x_mm": str(point["x_mm"]), "y_mm": str(point["y_mm"]),
                     "x": x, "y": y, "x_offset": offset, "y_offset": offset + 4,
                     "compensation_offset": offset + 8})
    return {"plan_id": plan.get("plan_id"), "count": len(rows), "units_per_mm": SCALE,
            "rounding": "四舍五入到 0.01 mm，负数按绝对值四舍五入",
            "compensation": "保留各点 Int 原值；不将 Z 深度写入补偿字段",
            "points": rows}


class S7Connection:
    def __init__(self, config):
        try:
            from s7 import Client
            from s7.type import Area, WordLen
        except ImportError:
            raise TransferError("当前 Python 缺少通信依赖，请安装 python-snap7==3.3.0。") from None
        self.client = Client(auto_reconnect=False)
        self.config = config
        self.area = Area.DB
        self.bit = WordLen.Bit

    def connect(self):
        c = self.config
        self.client.connect(c.host, c.rack, c.slot, tcp_port=c.port)

    def read(self, offset, size):
        data = bytes(self.client.db_read(self.config.db, offset, size))
        if len(data) != size:
            raise TransferError("PLC 返回的数据长度不完整。")
        return data

    def write(self, offset, data):
        self.client.db_write(self.config.db, offset, bytearray(data))

    def set_ready(self, value):
        # BIT addresses are in bits: DBX4.0 = 4 * 8. Do not rewrite byte 4.
        self.client.write_area(self.area, self.config.db, 4 * 8,
                               bytearray([int(value)]), word_len=self.bit)

    def disconnect(self):
        self.client.disconnect()


class PLCTransfer:
    def __init__(self, config=None, connection_factory=S7Connection):
        self.config = config or PLCConfig()
        self.factory = connection_factory
        self.lock = threading.Lock()

    @contextmanager
    def connection(self):
        if not self.lock.acquire(blocking=False):
            raise TransferError("PLC 通信正在进行，请等待本次操作结束。")
        connection = None
        try:
            connection = self.factory(self.config)
            connection.connect()
            yield connection
        finally:
            if connection is not None:
                try:
                    connection.disconnect()
                except Exception:
                    pass
            self.lock.release()

    @staticmethod
    def read_status(connection):
        header = connection.read(0, 5)
        return {"count": struct.unpack(">i", header[:4])[0],
                "pc_ready": bool(header[4] & 1),
                "plc_taken": bool(header[4] & 4)}

    def status(self):
        try:
            with self.connection() as connection:
                result = self.read_status(connection)
                result.update(connected=True, host=self.config.host, db=self.config.db,
                              note="PLC点位取完仅表示取数反馈，不代表设备动作完成。")
                return result
        except TransferError:
            raise
        except Exception as exc:
            raise TransferError(f"读取 PLC 失败：{type(exc).__name__}: {exc}") from exc

    def send(self, plan):
        prepared = prepare_plan(plan)  # Validate the entire batch before connecting.
        notification = "not_sent"
        stage = "连接"
        try:
            with self.connection() as connection:
                stage = "检查握手状态"
                status = self.read_status(connection)
                if status["pc_ready"] or status["plc_taken"]:
                    raise TransferError("握手尚未回到初始状态：发送前 DBX4.0 和 DBX4.2 都应为 False。请先读取并核对 PLC 状态。")
                stage = "写入坐标"
                for point in prepared["points"]:
                    # Write XY only. The following two compensation bytes are untouched.
                    connection.write(point["x_offset"], struct.pack(">ii", point["x"], point["y"]))
                stage = "核对坐标"
                data = connection.read(POINT_START, prepared["count"] * POINT_SIZE)
                for point in prepared["points"]:
                    start = (point["index"] - 1) * POINT_SIZE
                    if data[start:start + 8] != struct.pack(">ii", point["x"], point["y"]):
                        raise TransferError(f"第 {point['index']} 个点读回不一致，未发送完成信号。")
                stage = "写入数量"
                encoded_count = struct.pack(">i", prepared["count"])
                connection.write(0, encoded_count)
                if connection.read(0, 4) != encoded_count:
                    raise TransferError("点位数量读回不一致，未发送完成信号。")
                status = self.read_status(connection)
                if status["pc_ready"] or status["plc_taken"]:
                    raise TransferError("传输期间握手状态改变，未发送完成信号；请检查是否有其他程序在写 DB100。")
                stage = "发送完成信号"
                notification = "unknown"
                connection.set_ready(True)
                notification = "sent"
                result = {"ok": True, "count": prepared["count"], "verified": True,
                          "notification": notification, "plan_id": prepared["plan_id"],
                          "message": f"已核对 {prepared['count']} 个点及数量，并发送 PC 完成信号。"}
                try:
                    result["status"] = self.read_status(connection)
                except Exception:
                    result["warning"] = "完成信号已成功写入，但后续状态读取失败。请读取状态，勿直接重复发送。"
                return result
        except TransferError:
            raise
        except Exception as exc:
            if notification == "unknown":
                detail = "完成信号可能已经被 PLC 接收；请先核对 PLC 状态，不要直接重发。"
            else:
                detail = "未发送完成信号；PLC 内可能留有部分数据，排除问题后可重新发送完整批次。"
            raise TransferError(f"{stage}失败：{type(exc).__name__}: {exc}。{detail}",
                                notification=notification) from exc

    def write_coordinates(self, plan):
        """Overwrite XY/count for data tests, keeping PC-ready low throughout."""
        prepared = prepare_plan(plan)
        stage = '撤回 PC 完成信号'
        try:
            with self.connection() as connection:
                connection.set_ready(False)
                if self.read_status(connection)['pc_ready']:
                    raise TransferError('PC 完成信号未能清除，未开始覆盖坐标。')
                stage = '覆盖坐标'
                for point in prepared['points']:
                    connection.write(point['x_offset'], struct.pack('>ii', point['x'], point['y']))
                stage = '核对坐标'
                data = connection.read(POINT_START, prepared['count'] * POINT_SIZE)
                for point in prepared['points']:
                    start = (point['index'] - 1) * POINT_SIZE
                    if data[start:start+8] != struct.pack('>ii', point['x'], point['y']):
                        raise TransferError(f"第 {point['index']} 个点读回不一致；未发送联动通知。")
                stage = '覆盖数量'
                count = struct.pack('>i', prepared['count'])
                connection.write(0, count)
                if connection.read(0, 4) != count:
                    raise TransferError('点位数量读回不一致；未发送联动通知。')
                status = self.read_status(connection)
                if status['pc_ready']:
                    raise TransferError('检测到其他程序将 PC 完成信号置为 True，请核对 PLC 状态。')
                return {'ok': True, 'mode': 'coordinates_only', 'notification': 'not_sent',
                        'count': prepared['count'], 'plan_id': prepared['plan_id'], 'verified': True,
                        'status': status,
                        'message': f"已覆盖并核对 {prepared['count']} 个点及数量；4.0 保持 False，未发送联动通知。"}
        except TransferError:
            raise
        except Exception as exc:
            raise TransferError(f'{stage}失败：{type(exc).__name__}: {exc}。'
                                '本次未发送联动通知；PLC 内可能已有部分新数据，请核对后重新测试。') from exc

    def release(self):
        """Withdraw the PC-owned flag only after the PLC has acknowledged receipt."""
        notification = "not_sent"
        try:
            with self.connection() as connection:
                status = self.read_status(connection)
                if not status["plc_taken"]:
                    raise TransferError("PLC 尚未反馈取完点位，不清除 PC 完成信号。")
                notification = "unknown"
                connection.set_ready(False)
                notification = "cleared"
                return {"ok": True, "notification": notification,
                        "message": "已清除 DBX4.0；DBX4.2 由 PLC 程序复位。两位都为 False 后才能再次发送。"}
        except TransferError:
            raise
        except Exception as exc:
            raise TransferError(f"清除 PC 完成信号失败：{exc}。请先读取状态。",
                                notification=notification) from exc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "plc_config.json")
    parser.add_argument("--plan", type=Path, help="分割程序导出的完整规划 JSON")
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--send", action="store_true", help="实际写入 PLC 并发送完成信号")
    actions.add_argument("--test-write", action="store_true", help="仅覆盖坐标和数量，清除并保持 PC 完成信号为 False")
    actions.add_argument("--status", action="store_true", help="只读取 PLC 状态")
    actions.add_argument("--release", action="store_true", help="取数确认后清除 PC 完成信号")
    args = parser.parse_args()
    try:
        if args.status or args.release:
            transfer = PLCTransfer(PLCConfig.load(args.config))
            result = transfer.status() if args.status else transfer.release()
        else:
            if not args.plan:
                parser.error("预览或发送需要 --plan 文件路径。")
            plan = json.loads(args.plan.read_text(encoding="utf-8-sig"), parse_float=Decimal)
            if args.test_write:
                result = PLCTransfer(PLCConfig.load(args.config)).write_coordinates(plan)
            elif args.send:
                result = PLCTransfer(PLCConfig.load(args.config)).send(plan)
            else:
                result = prepare_plan(plan)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (OSError, ValueError, TransferError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
