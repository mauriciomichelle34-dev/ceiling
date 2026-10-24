"""Local webpage API for MVS software-trigger captures."""
import copy
import hashlib
import ipaddress
import json

from camera_service import CameraError


def is_local(handler):
    hosts = {f'127.0.0.1:{handler.server.server_port}', f'localhost:{handler.server.server_port}'}
    origin = handler.headers.get('Origin')
    return (ipaddress.ip_address(handler.client_address[0]).is_loopback
            and handler.headers.get('Host') in hosts
            and (not origin or origin in {'http://' + host for host in hosts}))


def handle_camera(handler):
    if not is_local(handler) or handler.headers.get('X-PLC-Request')!='1':
        return handler.reply(403, {'error':'相机接口仅允许本机程序和本机页面调用。'})
    try:
        if handler.headers.get('Content-Type','').split(';')[0]!='application/json':
            return handler.reply(415, {'error':'相机控制需要 JSON 请求。'})
        length = int(handler.headers.get('Content-Length','0'))
        if not 0<length<=2048:
            raise ValueError('相机请求为空或过大。')
        handler.connection.settimeout(30)
        body = json.loads(handler.rfile.read(length))
        if not isinstance(body,dict):
            raise ValueError('请求必须是 JSON 对象。')
        camera = handler.server.camera
        path = handler.path
        if path=='/api/camera/status':
            result = camera.status()
        elif path=='/api/camera/search':
            result = camera.search()
        elif path=='/api/camera/connect':
            result = camera.connect(body.get('device_id'))
        elif path=='/api/camera/disconnect':
            result = camera.disconnect()
        elif path=='/api/camera/capture':
            workflow = handler.server.automatic
            with workflow.lock:
                automatic = workflow.enabled
                settings = copy.deepcopy(workflow.settings)
            route = ('automatic:' + hashlib.sha256(json.dumps(settings,sort_keys=True).encode()).hexdigest()
                     if automatic else 'analysis')

            def process(raw, filename, capture_id):
                if automatic:
                    with workflow.lock:
                        if not workflow.enabled or workflow.settings!=settings:
                            raise CameraError('拍照期间自动处理参数或模式已改变，本次照片未下发。')
                        job = workflow.submit(raw,filename,'camera-'+capture_id)
                    return {'kind':'automatic','job':job}
                predictor = handler.server.predictor
                if not predictor.slots.acquire(timeout=10):
                    raise CameraError('识别服务忙，请稍后再拍照。')
                locked = False
                try:
                    locked = predictor.lock.acquire(timeout=60)
                    if not locked:
                        raise CameraError('等待识别服务超时。')
                    return {'kind':'analysis','meta':predictor.predict(raw,filename)}
                finally:
                    if locked:
                        predictor.lock.release()
                    predictor.slots.release()

            result = camera.capture(body.get('capture_id'),route,process)
        else:
            return handler.reply(404, {'error':'未找到相机接口。'})
        return handler.reply(200,result)
    except (CameraError, OSError) as exc:
        return handler.reply(409, {'error':str(exc)})
    except ValueError as exc:
        return handler.reply(400, {'error':str(exc)})
    except Exception as exc:
        print(f'Camera request failed: {exc}',flush=True)
        return handler.reply(500, {'error':'相机操作失败，请查看启动窗口。'})
