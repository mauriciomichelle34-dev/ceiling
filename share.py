"""Temporary public sharing. Run directly on the owner's computer."""
import argparse
import hashlib
import json
import queue
import re
import subprocess
import sys
import threading
import time
import uuid
import webbrowser
from pathlib import Path
from urllib.request import Request, urlopen

HERE = Path(__file__).resolve().parent
STATE = HERE/'sharing'
PORT = 8766
LOCAL = f'http://127.0.0.1:{PORT}'
HEADERS = {'User-Agent': 'Mycelium-Temporary-Sharing/1.0'}


def fetch_json(url, timeout=10):
    with urlopen(Request(url, headers=HEADERS), timeout=timeout) as response:
        return json.load(response)


def download_tool():
    target = HERE/'tools'/'cloudflared.exe'
    if target.exists() and target.stat().st_size > 1_000_000:
        return target
    print('首次启动：正在从 Cloudflare 官方 GitHub 仓库下载连接工具…', flush=True)
    release = fetch_json('https://api.github.com/repos/cloudflare/cloudflared/releases/latest', 30)
    asset = next(a for a in release['assets'] if a['name'] == 'cloudflared-windows-amd64.exe')
    url = asset['browser_download_url']
    if not url.startswith('https://github.com/cloudflare/cloudflared/releases/download/'):
        raise RuntimeError('下载地址不属于官方仓库。')
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix('.download')
    digest = hashlib.sha256()
    with urlopen(Request(url, headers=HEADERS), timeout=60) as response, temporary.open('wb') as f:
        while chunk := response.read(256*1024):
            f.write(chunk)
            digest.update(chunk)
    expected = asset.get('digest')
    if expected and expected != 'sha256:'+digest.hexdigest():
        raise RuntimeError('下载校验失败，请重试。')
    if temporary.stat().st_size != asset['size']:
        raise RuntimeError('下载不完整，请重试。')
    temporary.replace(target)
    (target.parent/'cloudflared_source.json').write_text(json.dumps({
        'url': url, 'release': release['tag_name'], 'sha256': digest.hexdigest(),
        'official_digest_checked': bool(expected)}, indent=2), encoding='utf-8')
    return target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stop', action='store_true')
    args = parser.parse_args()
    STATE.mkdir(exist_ok=True)
    stop_file = STATE/'stop.request'
    if args.stop:
        stop_file.write_text('stop', encoding='utf-8')
        print('已请求停止临时分享。本地 8765 服务不受影响。', flush=True)
        return
    try:
        info = fetch_json(LOCAL+'/api/health', 1)
        if info.get('app') == 'mycelium-local':
            link_file = STATE/'分享链接.txt'
            print('分享服务已经运行。请查看原来的分享窗口。')
            if link_file.exists():
                print(link_file.read_text(encoding='utf-8'))
            return
        raise RuntimeError('8766 端口已被其他服务占用。')
    except OSError:
        pass
    if stop_file.exists():
        stop_file.unlink()
    tool = download_tool()
    origin_file = STATE/'public_origin.txt'
    origin_file.write_text('', encoding='utf-8')
    session = uuid.uuid4().hex
    origin = subprocess.Popen([sys.executable, str(HERE/'local_app.py'), '--port', str(PORT),
        '--public-origin-file', str(origin_file), '--result-root', str(HERE/'predictions_shared'/session)],
        cwd=HERE, stdout=(STATE/'app.log').open('w', encoding='utf-8'), stderr=subprocess.STDOUT)
    tunnel = None
    try:
        print('正在加载模型…', flush=True)
        for _ in range(45):
            if origin.poll() is not None:
                raise RuntimeError('模型启动失败，请查看 sharing/app.log。')
            try:
                if fetch_json(LOCAL+'/api/health', 1).get('shared'):
                    break
            except OSError:
                pass
            time.sleep(1)
        else:
            raise RuntimeError('模型启动超时。')
        print('正在建立无密码临时链接…', flush=True)
        tunnel = subprocess.Popen([str(tool), 'tunnel', '--no-autoupdate', '--protocol', 'http2',
            '--url', LOCAL, '--http-host-header', f'127.0.0.1:{PORT}'],
            cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            encoding='utf-8', errors='replace')
        messages = queue.Queue()
        def read_logs():
            with (STATE/'tunnel.log').open('w', encoding='utf-8') as log:
                for line in tunnel.stdout:
                    log.write(line)
                    log.flush()
                    messages.put(line)
        threading.Thread(target=read_logs, daemon=True).start()
        deadline, public_url = time.monotonic()+90, None
        while time.monotonic() < deadline:
            if tunnel.poll() is not None:
                raise RuntimeError('连接工具已退出，请查看 sharing/tunnel.log。')
            if stop_file.exists():
                return
            try:
                line = messages.get(timeout=1)
            except queue.Empty:
                continue
            found = re.search(r'https://[a-z0-9-]+\.trycloudflare\.com', line)
            if found:
                public_url = found.group()
                break
        if not public_url:
            raise RuntimeError('未能生成链接，请查看 sharing/tunnel.log。')
        origin_file.write_text(public_url, encoding='utf-8')
        (STATE/'分享链接.txt').write_text(public_url+'\n保持分享窗口和电脑运行；重启后链接可能改变。\n', encoding='utf-8')
        print('\n临时分享链接（不需要密码）：\n'+public_url, flush=True)
        print('\n正在检查公网连接…', flush=True)
        verified = False
        for _ in range(6):
            try:
                verified = fetch_json(public_url+'/api/health', 8).get('app') == 'mycelium-local'
                if verified:
                    break
            except OSError:
                pass
            time.sleep(2)
        print('公网连接检查通过。' if verified else '已生成链接，但尚未通过公网检查，请查看 tunnel.log 或稍后重试。', flush=True)
        print('请让大陆用户实际测试访问和上传。不要把本机 127.0.0.1 地址发给客户。', flush=True)
        print('保持此窗口打开。按 Ctrl+C 或双击 Stop_Sharing.cmd 停止分享。', flush=True)
        webbrowser.open(public_url)
        while not stop_file.exists():
            if tunnel.poll() is not None or origin.poll() is not None:
                raise RuntimeError('分享服务已退出，请重新启动。')
            time.sleep(1)
    finally:
        for child in [tunnel, origin]:
            if child is not None and child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
        origin_file.write_text('', encoding='utf-8')
        (STATE/'分享链接.txt').write_text('分享已停止。重新运行 Start_Sharing.cmd 可生成新链接。\n', encoding='utf-8')
        print('临时分享已停止。', flush=True)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f'启动失败：{exc}', flush=True)
        print('若下载失败，请检查此电脑是否能访问 github.com，再重新运行。', flush=True)
        sys.exit(1)
