"""Submit one encoded JPEG/PNG capture to the automatic pipeline on this PC."""
import argparse
import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote, urlsplit
from urllib.request import Request, urlopen


def submit_image(image_bytes, filename, capture_id, base_url='http://127.0.0.1:8766'):
    """Reuse capture_id when retrying the same capture; use a new ID for a new frame."""
    url = urlsplit(base_url)
    if url.scheme != 'http' or url.hostname not in ('127.0.0.1', 'localhost') or url.username or url.password:
        raise ValueError('图片处理入口必须是本机 HTTP 地址。')
    request = Request(base_url.rstrip('/')+'/api/automatic/images', data=image_bytes,
                      headers={'Content-Type':'application/octet-stream', 'X-PLC-Request':'1',
                               'X-Capture-ID':capture_id, 'X-File-Name':quote(filename)})
    try:
        with urlopen(request, timeout=30) as response:
            return json.load(response)
    except HTTPError as exc:
        error = json.load(exc)
        raise RuntimeError(error.get('error', str(exc))) from exc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', required=True, type=Path)
    parser.add_argument('--capture-id', required=True, help='每次拍摄的唯一编号；网络重试沿用原编号')
    parser.add_argument('--url', default='http://127.0.0.1:8766')
    args = parser.parse_args()
    try:
        print(json.dumps(submit_image(args.image.read_bytes(), args.image.name, args.capture_id, args.url),
                         ensure_ascii=False, indent=2))
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f'{exc}\n')


if __name__ == '__main__':
    main()
