"""Local image and control API used by the webpage and, later, the camera SDK."""
import ipaddress
import json
from pathlib import PureWindowsPath
import re
from urllib.parse import unquote

from auto_workflow import WorkflowError
from image_limits import MAX_IMAGE_BYTES, IMAGE_CONTENT_TYPES


def handle_automatic(handler):
    local_hosts = {f'127.0.0.1:{handler.server.server_port}', f'localhost:{handler.server.server_port}'}
    origin = handler.headers.get('Origin')
    if (not ipaddress.ip_address(handler.client_address[0]).is_loopback
            or handler.headers.get('Host') not in local_hosts
            or (origin and origin not in {'http://' + host for host in local_hosts})
            or handler.headers.get('X-PLC-Request') != '1'):
        return handler.reply(403, {'error': '自动下发接口仅允许本机程序和本机页面调用。'})
    try:
        service = handler.server.automatic
        length = int(handler.headers.get('Content-Length', '0'))
        content_type = handler.headers.get('Content-Type', '').split(';')[0]
        if handler.path == '/api/automatic/images':
            if content_type not in IMAGE_CONTENT_TYPES:
                return handler.reply(415, {'error': '请提交 JPG、PNG 或 BMP 图片字节。'})
            if not 0 < length <= MAX_IMAGE_BYTES:
                return handler.reply(413, {'error': '图片需在 100 MB 以内。'})
            handler.connection.settimeout(120)
            raw = handler.rfile.read(length)
            if len(raw) != length:
                raise ValueError('图片传输不完整。')
            filename = PureWindowsPath(unquote(handler.headers.get('X-File-Name', 'camera.jpg'))).name
            filename = re.sub(r'[\x00-\x1f<>:"/\\|?*]', '_', filename)[:160] or 'camera.jpg'
            job = service.submit(raw, filename, handler.headers.get('X-Capture-ID'))
            return handler.reply(200 if job.get('duplicate') else 202, job)
        if content_type != 'application/json':
            return handler.reply(415, {'error': '自动处理控制接口需要 JSON 请求。'})
        if not 0 < length <= 32768:
            raise ValueError('请求为空或过大。')
        handler.connection.settimeout(10)
        body = json.loads(handler.rfile.read(length))
        if not isinstance(body, dict):
            raise ValueError('请求必须是 JSON 对象。')
        if handler.path == '/api/automatic/status':
            result = service.snapshot()
        elif handler.path == '/api/automatic/start':
            result = service.start(body.get('settings'))
        elif handler.path == '/api/automatic/stop':
            result = service.stop()
        elif handler.path == '/api/automatic/job':
            result = service.get_job(body.get('job_id'))
        else:
            return handler.reply(404, {'error': '未找到自动处理接口。'})
        return handler.reply(200, result)
    except WorkflowError as exc:
        return handler.reply(409, {'error': str(exc)})
    except (ValueError, OSError) as exc:
        return handler.reply(400, {'error': str(exc)})
    except Exception as exc:
        print(f'Automatic request failed: {exc}', flush=True)
        return handler.reply(500, {'error': '自动处理请求失败，请查看启动窗口和处理记录。'})
