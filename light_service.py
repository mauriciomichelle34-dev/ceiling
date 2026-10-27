"""DCS3 ASCII commands over the local RS232 Ethernet bridge."""
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import ipaddress
import json
from pathlib import Path
import socket
import threading
import time


class LightError(RuntimeError):
    pass


@dataclass(frozen=True)
class LightConfig:
    host: str = '192.168.6.15'
    port: int = 9232
    channel: int = 1
    enabled: bool = True
    brightness: int = 170
    warmup_ms: int = 100
    timeout_seconds: float = 2.0

    def __post_init__(self):
        if not isinstance(self.host, str) or not ipaddress.IPv4Address(self.host).is_private:
            raise ValueError('光源地址必须为本地 IPv4 地址。')
        for name, low, high in [('port', 1, 65535), ('channel', 1, 4),
                                ('brightness', 1, 255), ('warmup_ms', 0, 1000)]:
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f'光源 {name} 应在 {low}–{high} 范围内。')
        if type(self.enabled) is not bool:
            raise ValueError('拍照补光开关必须为布尔值。')
        if type(self.timeout_seconds) not in (int, float) or not 0.1 <= self.timeout_seconds <= 5:
            raise ValueError('光源等待时间应在 0.1–5 秒范围内。')

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding='utf-8-sig')))


def checksum(body):
    value = 0
    for byte in ('$' + body).encode('ascii'):
        value ^= byte
    return f'{value:02X}'


def command_frame(command, channel, value=0):
    if command not in '1234567890AB' or len(command) != 1:
        raise ValueError('光源命令无效。')
    if type(channel) is not int or not 1 <= channel <= 4:
        raise ValueError('光源通道应为 1–4。')
    if type(value) is not int or not 0 <= value <= 999:
        raise ValueError('光源命令数值无效。')
    body = f'{command}{channel}{value:03X}'
    return ('$' + body + checksum(body)).encode('ascii')


def read_value(frame, command, channel):
    try:
        text = frame.decode('ascii')
        if (len(text) != 8 or text[:3] != f'${command}{channel}'
                or text[6:].upper() != checksum(text[1:6])):
            raise ValueError()
        return int(text[3:6], 16)
    except (UnicodeError, ValueError) as exc:
        raise LightError('光源返回的数据格式或校验码不正确。') from exc


class LightService:
    def __init__(self, config, config_path=None, connector=socket.create_connection,
                 sleeper=time.sleep, clock=time.monotonic):
        self.config = config
        self.config_path = Path(config_path) if config_path else None
        self.connector, self.sleep, self.clock = connector, sleeper, clock
        self.operation = threading.Lock()
        self.state_lock = threading.RLock()
        self._last_command_at = float('-inf')
        self.connected = False
        self.last_error = None
        self.actual_brightness = None
        self.off_confirmed = None
        self.last_flash = None
        self.last_off = None

    def status(self):
        with self.state_lock:
            return {**asdict(self.config), 'connected': self.connected,
                    'busy': self.operation.locked(), 'last_error': self.last_error,
                    'actual_brightness': self.actual_brightness,
                    'off_confirmed': self.off_confirmed,
                    'last_off': dict(self.last_off) if self.last_off else None,
                    'last_flash': dict(self.last_flash) if self.last_flash else None}

    @contextmanager
    def _operation(self):
        if not self.operation.acquire(blocking=False):
            raise LightError('光源正在用于拍照，请稍后操作。')
        try:
            yield
        except Exception as exc:
            with self.state_lock:
                self.last_error = str(exc)
                self.connected = False
            raise
        finally:
            self.operation.release()

    def configure(self, body):
        if not isinstance(body, dict) or set(body) != {'enabled', 'brightness'}:
            raise ValueError('请提供拍照补光开关及亮度。')
        with self._operation():
            updated = replace(self.config, **body)
            if not updated.enabled:
                self._turn_off_locked('disabled')
            if self.config_path:
                temp = self.config_path.with_suffix('.tmp')
                temp.write_text(json.dumps(asdict(updated), ensure_ascii=False, indent=2), encoding='utf-8')
                temp.replace(self.config_path)
            with self.state_lock:
                self.config = updated
                self.last_error = None
        return self.status()

    def _turn_off_locked(self, reason):
        # A previous acknowledgement is historical, not a live hardware reading.
        with self.state_lock:
            self.off_confirmed = False
        connection = self._connect()
        try:
            self._ensure_off(connection)
            with self.state_lock:
                self.off_confirmed = True
                self.connected, self.last_error = True, None
                self.last_off = {'reason': reason, 'confirmed_at': datetime.now(timezone.utc).isoformat()}
        finally:
            connection.close()

    def turn_off(self, reason='idle'):
        """Clear a latched ON state even when flash illumination is disabled."""
        with self._operation():
            self._turn_off_locked(reason)
        return self.status()

    def _connect(self):
        try:
            return self.connector((self.config.host, self.config.port), timeout=self.config.timeout_seconds)
        except OSError as exc:
            raise LightError(f'无法连接光源 {self.config.host}:{self.config.port}：{exc}') from exc

    def _read_exact(self, connection, size):
        data = bytearray()
        deadline = self.clock() + self.config.timeout_seconds
        while len(data) < size:
            remaining = deadline - self.clock()
            if remaining <= 0:
                raise LightError('等待光源返回数据超时。')
            connection.settimeout(remaining)
            part = connection.recv(size - len(data))
            if not part:
                raise LightError('光源连接已断开。')
            data.extend(part)
        return bytes(data)

    def _command(self, connection, command, value=0, reading=False):
        remaining = .02 - (self.clock() - self._last_command_at)
        if remaining > 0:
            self.sleep(remaining)
        try:
            connection.settimeout(self.config.timeout_seconds)
            connection.sendall(command_frame(command, self.config.channel, value))
            first = self._read_exact(connection, 1)
            if first == b'&':
                raise LightError('光源拒绝了指令，请检查串口参数与控制器工作模式。')
            if first != b'$':
                raise LightError('光源返回了无法识别的响应。')
            return read_value(first + self._read_exact(connection, 7), command, self.config.channel) if reading else None
        except OSError as exc:
            raise LightError(f'光源通信失败：{exc}') from exc
        finally:
            self._last_command_at = self.clock()

    def probe(self):
        """Only reads current brightness; never turns on a light."""
        with self._operation():
            connection = self._connect()
            try:
                value = self._command(connection, '4', reading=True)
                if not 0 <= value <= 255:
                    raise LightError('光源返回的亮度超出范围。')
                with self.state_lock:
                    self.actual_brightness, self.connected, self.last_error = value, True, None
            finally:
                connection.close()
        return self.status()

    def _ensure_off(self, connection):
        try:
            self._command(connection, '2')
        except LightError:
            # Off is idempotent: retry once over a fresh connection after an uncertain reply.
            connection.close()
            fresh = self._connect()
            try:
                self._command(fresh, '2')
            finally:
                fresh.close()

    @contextmanager
    def exposure(self):
        """Hold CH1 on only for trigger/frame acquisition; off before encoding/inference."""
        with self._operation():
            if not self.config.enabled:
                yield None
                return
            connection = self._connect()
            attempted_on = False
            started = None
            record = {'channel': self.config.channel, 'brightness': self.config.brightness,
                      'off_confirmed': False, 'started_at': datetime.now(timezone.utc).isoformat()}
            try:
                value = self._command(connection, '4', reading=True)
                if value != self.config.brightness:
                    self._command(connection, '3', self.config.brightness)
                    value = self._command(connection, '4', reading=True)
                    if value != self.config.brightness:
                        raise LightError('光源亮度读回与设置值不一致，本次未拍照。')
                with self.state_lock:
                    self.actual_brightness, self.off_confirmed = value, False
                attempted_on = True
                started = self.clock()
                self._command(connection, '1')
                self.sleep(self.config.warmup_ms / 1000)
                yield record
            finally:
                try:
                    if attempted_on:
                        try:
                            self._ensure_off(connection)
                        except LightError as exc:
                            with self.state_lock:
                                self.off_confirmed = False
                            raise LightError(f'未能确认关灯，已停止后续识别和下发；请检查 CH{self.config.channel} 灯光：{exc}') from exc
                        record['off_confirmed'] = True
                        record['duration_ms'] = round((self.clock() - started) * 1000, 1)
                        with self.state_lock:
                            self.off_confirmed = True
                            self.last_flash = dict(record)
                            self.last_off = {'reason': 'capture', 'confirmed_at': datetime.now(timezone.utc).isoformat()}
                    with self.state_lock:
                        self.connected, self.last_error = True, None
                finally:
                    connection.close()
