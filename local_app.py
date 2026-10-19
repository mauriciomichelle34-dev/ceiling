"""Local upload and segmentation app. See README.md for setup."""
import argparse
import hashlib
import io
import json
import re
import threading
import socket
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PureWindowsPath
from urllib.parse import quote, unquote, urlsplit

import cv2
import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

from model import UNet
from inference import predict_probability
from dish import detect_dish, restrict_to_dish, DishNotFound
from sampling import make_plan, with_depths, csv_bytes, layout_svg, measure_area
from device_interface import PreviewDevice

HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = HERE/'mycelium_v3.pt'
APP_VERSION = 10
MAX_BYTES = 40*1024*1024
Image.MAX_IMAGE_PIXELS = 25_000_000


def save_region_previews(mask, preview_size, folder, boundary=None):
    """Save transparent animation layers without changing the original mask."""
    selection = cv2.resize(mask, preview_size, interpolation=cv2.INTER_NEAREST) > 0
    inside = None
    if boundary:
        height, width = mask.shape
        y, x = np.ogrid[:preview_size[1], :preview_size[0]]
        inside = ((x*width/preview_size[0]-boundary['center_x_px'])**2
                  + (y*height/preview_size[1]-boundary['center_y_px'])**2 <= boundary['radius_px']**2)
        selection &= inside
    region = np.zeros((*selection.shape, 4), dtype=np.uint8)
    region[..., :3] = (0, 255, 100)
    region[..., 3] = selection.astype(np.uint8)*255
    Image.fromarray(region).save(folder/'region.png')
    edge = cv2.morphologyEx(selection.astype(np.uint8), cv2.MORPH_GRADIENT,
                            np.ones((3, 3), dtype=np.uint8),
                            borderType=cv2.BORDER_CONSTANT, borderValue=0)
    if inside is not None:
        edge[~inside] = 0
    outline = np.full((*selection.shape, 4), 255, dtype=np.uint8)
    outline[..., 3] = edge*255
    Image.fromarray(outline).save(folder/'region_outline.png')
    bounds = list(cv2.boundingRect(mask))
    return selection, bounds if bounds[2] and bounds[3] else None


class Predictor:
    def __init__(self, checkpoint_path, result_root=None):
        torch.set_num_threads(4)
        cv2.setNumThreads(0)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        self.config = checkpoint['config']
        self.model_name = self.config.get('model_version', checkpoint_path.stem)
        self.model_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        self.result_root = result_root or HERE/'predictions'
        self.network = UNet().to(self.device)
        self.network.load_state_dict(checkpoint['model'])
        self.network.eval()
        self.lock = threading.Lock()
        self.slots = threading.BoundedSemaphore(6)

    def predict(self, raw, name):
        with Image.open(io.BytesIO(raw)) as im:
            if im.format not in {'JPEG', 'PNG'}:
                raise ValueError('请选择 JPG 或 PNG 图片。')
            if im.width*im.height > 25_000_000:
                raise ValueError('图片过大，请使用不超过 2500 万像素的图片。')
            source = im.convert('RGB')
        w, h = source.size
        boundary, dish_suggestion = detect_dish(source)
        probability = predict_probability(self.network, source, self.config, self.device)
        mask = (probability >= self.config['threshold']).astype(np.uint8)
        mask = restrict_to_dish(mask, boundary)
        identifier = uuid.uuid4().hex
        folder = self.result_root/identifier
        folder.mkdir(parents=True, exist_ok=False)
        Image.fromarray(mask).save(folder/'mask.png')
        preview = source.copy()
        preview.thumbnail((1400, 1050), Image.Resampling.LANCZOS)
        preview.save(folder/'original.jpg', quality=92)
        selection, region_bounds = save_region_previews(mask, preview.size, folder, boundary)
        overlay = np.asarray(preview).copy()
        overlay[selection] = (.55*overlay[selection]+.45*np.array([0, 255, 100])).astype(np.uint8)
        Image.fromarray(overlay).save(folder/'overlay.jpg', quality=92)
        meta = {'id': identifier, 'name': name, 'width': w, 'height': h,
                'foreground_fraction': float(mask.mean()), 'model': self.model_name,
                'original': f'/results/{identifier}/original.jpg',
                'overlay': f'/results/{identifier}/overlay.jpg',
                'mask': f'/results/{identifier}/mask.png',
                'region': f'/results/{identifier}/region.png',
                'region_outline': f'/results/{identifier}/region_outline.png',
                'region_bounds': region_bounds,
                'dish_boundary': boundary, 'dish_suggestion': dish_suggestion}
        (folder/'metadata.json').write_text(json.dumps(meta, ensure_ascii=False), encoding='utf-8')
        return meta


class Handler(BaseHTTPRequestHandler):
    def public_origin(self):
        path = getattr(self.server, 'public_origin_file', None)
        if path and path.is_file():
            value = path.read_text(encoding='utf-8').strip()
            if re.fullmatch(r'https://[a-z0-9-]+\.trycloudflare\.com', value):
                return value
        return None

    def reply(self, status, body, kind='application/json; charset=utf-8', download=None):
        if isinstance(body, dict):
            body = json.dumps(body, ensure_ascii=False, allow_nan=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', kind)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Referrer-Policy', 'no-referrer')
        self.send_header('X-Robots-Tag', 'noindex, nofollow')
        if download:
            self.send_header('Content-Disposition', "attachment; filename*=UTF-8''"+quote(download))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def valid_host(self):
        hosts = {f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}'}
        origin = self.public_origin()
        if origin:
            hosts.add(urlsplit(origin).netloc)
        return self.headers.get('Host') in hosts

    def do_GET(self):
        if not self.valid_host():
            return self.reply(403, {'error': '仅支持本机访问。'})
        path = urlsplit(self.path).path
        if path == '/':
            return self.reply(200, (HERE/'web.html').read_bytes(), 'text/html; charset=utf-8')
        if path in ('/planner.js', '/planner.css'):
            kind = 'text/javascript; charset=utf-8' if path.endswith('.js') else 'text/css; charset=utf-8'
            return self.reply(200, (HERE/path[1:]).read_bytes(), kind)
        if path == '/api/health':
            return self.reply(200, {'app': 'mycelium-local', 'ready': True, 'version': APP_VERSION,
                                    'original_image_inference': True,
                                    'dish_interior_only': True,
                                    'area_measurement': True, 'region_animation': True,
                                    'model': self.server.predictor.model_name,
                                    'model_sha256': self.server.predictor.model_sha256,
                                    'model_preprocessing': self.server.predictor.config.get('preprocessing', 'legacy_stretch_rgb_0_1'),
                                    'model_input_size': [self.server.predictor.config['width'], self.server.predictor.config['height']],
                                    'sampling_order': 'outer_boundary_layers',
                                    'sampling_planner': True, 'hardware_ready': False,
                                    'shared': bool(getattr(self.server, 'public_origin_file', None)),
                                    'device': 'GPU' if self.server.predictor.device.type == 'cuda' else 'CPU'})
        plan_match = re.fullmatch(r'/results/([0-9a-f]{32})/plans/([0-9a-f]{32})/(plan\.json|points\.csv|preview\.json|layout\.svg)', path)
        if plan_match:
            folder = self.server.predictor.result_root/plan_match[1]
            meta_path = folder/'metadata.json'
            if meta_path.is_file() and not json.loads(meta_path.read_text(encoding='utf-8')).get('dish_boundary'):
                return self.reply(409, {'error': '旧识别结果未限定皿内范围，请重新上传照片。'})
            plan_path = folder/'plans'/plan_match[2]/'plan.json'
            if not plan_path.is_file():
                return self.reply(404, {'error': '未找到取样方案。'})
            plan = json.loads(plan_path.read_text(encoding='utf-8'))
            kind = plan_match[3]
            stem = Path(plan['source_name']).stem
            if kind == 'points.csv':
                return self.reply(200, csv_bytes(plan), 'text/csv; charset=utf-8', stem+'_points.csv')
            if kind == 'preview.json':
                return self.reply(200, PreviewDevice().preview(plan), download=stem+'_device_preview.json')
            if kind == 'layout.svg':
                return self.reply(200, layout_svg(plan, (folder/'original.jpg').read_bytes()), 'image/svg+xml', stem+'_sampling.svg')
            return self.reply(200, plan, download=stem+'_plan.json')
        match = re.fullmatch(r'/results/([0-9a-f]{32})/(mask\.png|original\.jpg|overlay\.jpg|region\.png|region_outline\.png)', path)
        if match:
            folder = self.server.predictor.result_root/match[1]
            target = folder/match[2]
            if target.is_file():
                meta = json.loads((folder/'metadata.json').read_text(encoding='utf-8'))
                if match[2] != 'original.jpg' and not meta.get('dish_boundary'):
                    return self.reply(409, {'error': '旧识别结果未限定皿内范围，请重新上传照片。'})
                download = None
                if match[2] == 'mask.png':
                    download = Path(meta['name']).stem+'.png'
                return self.reply(200, target.read_bytes(), 'image/png' if match[2].endswith('.png') else 'image/jpeg', download)
        self.reply(404, {'error': '未找到文件。'})

    def handle_plan(self):
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 2*1024*1024:
                raise ValueError('取样参数过大或为空。')
            self.connection.settimeout(120)
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError('取样参数格式错误。')
            prediction_id = payload.get('prediction_id', '')
            if not isinstance(prediction_id, str) or not re.fullmatch('[0-9a-f]{32}', prediction_id):
                raise ValueError('图片编号无效。')
            folder = self.server.predictor.result_root/prediction_id
            if not (folder/'metadata.json').is_file():
                return self.reply(404, {'error': '图片不存在，请重新上传。'})
            if not json.loads((folder/'metadata.json').read_text(encoding='utf-8')).get('dish_boundary'):
                return self.reply(409, {'error': '旧识别结果未限定皿内范围，请重新上传照片。'})
            if not self.server.predictor.slots.acquire(blocking=False):
                return self.reply(429, {'error': '排队人数已满，请稍后重试。'})
            acquired = False
            try:
                acquired = self.server.predictor.lock.acquire(timeout=60)
                if not acquired:
                    return self.reply(429, {'error': '服务忙，请稍后重试。'})
                if self.path == '/api/area':
                    with Image.open(folder/'mask.png') as image:
                        mask = np.asarray(image)
                    return self.reply(200, measure_area(mask, payload.get('calibration')))
                if self.path == '/api/plans/create':
                    with Image.open(folder/'mask.png') as image:
                        mask = np.asarray(image)
                    plan = make_plan(mask, payload.get('calibration'), payload.get('diameter_mm'),
                                     payload.get('gap_mm'), payload.get('default_depth_mm'))
                    meta = json.loads((folder/'metadata.json').read_text(encoding='utf-8'))
                    plan.update(prediction_id=prediction_id, source_name=meta['name'])
                else:
                    plan_id = payload.get('plan_id', '')
                    if not isinstance(plan_id, str) or not re.fullmatch('[0-9a-f]{32}', plan_id):
                        raise ValueError('取样方案编号无效。')
                    path = folder/'plans'/plan_id/'plan.json'
                    if not path.is_file():
                        return self.reply(404, {'error': '取样方案不存在。'})
                    old = json.loads(path.read_text(encoding='utf-8'))
                    plan = with_depths(old, payload.get('default_depth_mm'), payload.get('overrides', {}))
                destination = folder/'plans'/plan['plan_id']
                destination.mkdir(parents=True, exist_ok=True)
                temporary = destination/'plan.tmp'
                temporary.write_text(json.dumps(plan, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
                temporary.replace(destination/'plan.json')
                self.reply(200, plan)
            finally:
                if acquired:
                    self.server.predictor.lock.release()
                self.server.predictor.slots.release()
        except (ValueError, TypeError) as exc:
            self.reply(400, {'error': str(exc)})
        except Exception as exc:
            print(f'Planning failed: {exc}', flush=True)
            self.reply(500, {'error': '计算失败，请查看启动窗口或调整参数后重试。'})

    def do_POST(self):
        origin = self.headers.get('Origin')
        allowed = {f'http://127.0.0.1:{self.server.server_port}', f'http://localhost:{self.server.server_port}'}
        public_origin = self.public_origin()
        if public_origin:
            allowed.add(public_origin)
        if not self.valid_host() or (origin and origin not in allowed):
            return self.reply(403, {'error': '仅支持从本地页面提交图片。'})
        if self.path in ('/api/area', '/api/plans/create', '/api/plans/update'):
            return self.handle_plan()
        if self.path != '/api/predict':
            return self.reply(404, {'error': '未找到接口。'})
        try:
            length = int(self.headers.get('Content-Length', '0'))
        except ValueError:
            return self.reply(400, {'error': '无效的文件大小。'})
        if not 0 < length <= MAX_BYTES:
            self.close_connection = True
            return self.reply(413, {'error': '请选择 40 MB 以内的图片。'})
        if not self.server.predictor.slots.acquire(blocking=False):
            self.close_connection = True
            return self.reply(429, {'error': '排队人数已满，请稍后再试。'})
        locked = False
        try:
            self.connection.settimeout(120)
            raw = self.rfile.read(length)
            if len(raw) != length:
                raise ValueError('Incomplete upload')
            name = PureWindowsPath(unquote(self.headers.get('X-File-Name', 'image.jpg'))).name
            name = re.sub(r'[\x00-\x1f<>:"/\\|?*]', '_', name)[:160] or 'image.jpg'
            locked = self.server.predictor.lock.acquire(timeout=60)
            if not locked:
                return self.reply(429, {'error': '等待时间过长，请稍后重试。'})
            result = self.server.predictor.predict(raw, name)
            self.reply(200, result)
        except socket.timeout:
            self.close_connection = True
            self.reply(408, {'error': '上传超时，请检查网络后重试。'})
        except DishNotFound as exc:
            self.reply(422, {'error': str(exc)})
        except (ValueError, UnidentifiedImageError, Image.DecompressionBombError):
            self.reply(400, {'error': '无法读取图片。请使用 40 MB、2500 万像素以内的有效 JPG 或 PNG。'})
        except Exception as exc:
            print(f'Prediction failed: {exc}', flush=True)
            self.reply(500, {'error': '处理失败，请查看启动窗口，或稍后重试。'})
        finally:
            if locked:
                self.server.predictor.lock.release()
            self.server.predictor.slots.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    parser.add_argument('--open', action='store_true')
    parser.add_argument('--public-origin-file', type=Path)
    parser.add_argument('--result-root', type=Path)
    args = parser.parse_args()
    if not args.model.is_file():
        parser.error(f'模型文件不存在：{args.model}。请按 README.md 准备权重，或用 --model 指定已有模型文件。')
    requested_hash = hashlib.sha256(args.model.read_bytes()).hexdigest()
    url = f'http://127.0.0.1:{args.port}'
    # Reopening the launcher reuses this app rather than starting a second model.
    from urllib.request import urlopen
    try:
        with urlopen(url+'/api/health', timeout=1) as response:
            running = json.load(response)
            if running.get('app') == 'mycelium-local':
                if running.get('version') != APP_VERSION or running.get('model_sha256') != requested_hash:
                    raise SystemExit('旧版服务或另一个模型还在运行。请先在旧启动窗口按 Ctrl+C 关闭，再运行本启动文件。')
                print('Already running:', url, flush=True)
                if args.open:
                    webbrowser.open(url)
                return
    except Exception:
        pass
    print('Loading model...', flush=True)
    predictor = Predictor(args.model, args.result_root)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    server.predictor = predictor
    server.public_origin_file = args.public_origin_file
    print('Ready:', url, flush=True)
    print('Keep this window open. Press Ctrl+C to stop.', flush=True)
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
