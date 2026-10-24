"""Local-only HTTP entry points for the DB100 transfer adapter."""
import ipaddress
import json
import re

from plc_transfer import TransferError, prepare_plan


def handle_plc(handler):
    local_hosts = {f"127.0.0.1:{handler.server.server_port}",
                   f"localhost:{handler.server.server_port}"}
    origin = handler.headers.get("Origin")
    if (not ipaddress.ip_address(handler.client_address[0]).is_loopback
            or handler.headers.get("Host") not in local_hosts
            or (origin and origin not in {"http://" + host for host in local_hosts})
            or handler.headers.get("X-PLC-Request") != "1"):
        return handler.reply(403, {"error": "PLC 接口仅允许本机页面调用。"})
    if handler.headers.get("Content-Type", "").split(";")[0] != "application/json":
        return handler.reply(415, {"error": "PLC 接口需要 JSON 请求。"})
    try:
        length = int(handler.headers.get("Content-Length", "0"))
        if not 0 < length <= 4096:
            raise ValueError("PLC 请求为空或过大。")
        handler.connection.settimeout(10)
        body = json.loads(handler.rfile.read(length))
        if not isinstance(body, dict):
            raise ValueError("PLC 请求必须是 JSON 对象。")
        transfer = handler.server.plc_transfer
        automatic = getattr(handler.server, 'automatic', None)
        if (handler.path in ('/api/plc/send', '/api/plc/test-write', '/api/plc/release') and automatic is not None
                and (automatic.enabled or automatic.active_id)):
            raise TransferError('自动处理正在管理数据交接，请先停止自动处理并等当前任务结束，再手动发送或清除信号。')
        if handler.path == "/api/plc/status":
            result = transfer.status()
        elif handler.path == "/api/plc/release":
            result = transfer.release()
        elif handler.path in ("/api/plc/preview", "/api/plc/send", "/api/plc/test-write"):
            prediction_id, plan_id = body.get("prediction_id"), body.get("plan_id")
            if any(not isinstance(v, str) or not re.fullmatch(r"[0-9a-f]{32}", v)
                   for v in (prediction_id, plan_id)):
                raise ValueError("请先生成并保存当前取样排布。")
            folder = handler.server.predictor.result_root / prediction_id
            path = folder / "plans" / plan_id / "plan.json"
            if not path.is_file():
                return handler.reply(404, {"error": "取样规划不存在，请重新生成。"})
            plan = json.loads(path.read_text(encoding="utf-8"))
            if handler.path.endswith('preview'):
                result = prepare_plan(plan)
            elif handler.path.endswith('test-write'):
                result = transfer.write_coordinates(plan)
            else:
                result = transfer.send(plan)
        else:
            return handler.reply(404, {"error": "未找到 PLC 接口。"})
        return handler.reply(200, result)
    except TransferError as exc:
        return handler.reply(409, {"error": str(exc), "notification": exc.notification})
    except (ValueError, OSError) as exc:
        return handler.reply(400, {"error": str(exc)})
    except Exception as exc:
        print(f"PLC request failed: {exc}", flush=True)
        return handler.reply(500, {"error": "PLC 操作失败，请查看启动窗口；重新发送前先读取 PLC 状态。"})
