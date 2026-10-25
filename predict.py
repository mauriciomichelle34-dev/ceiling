"""Export raw model masks and overlays without dish or component filtering."""
from pathlib import Path
import argparse
import numpy as np
from PIL import Image
import torch

from model import UNet
from inference import predict_probability


def main():
    parser = argparse.ArgumentParser(description='菌丝体图片分割')
    parser.add_argument('image', type=Path, help='输入照片')
    parser.add_argument('--model', type=Path, default=Path(__file__).with_name('mycelium_v4_full16.pt'))
    parser.add_argument('--output', type=Path, default=Path('results'))
    args = parser.parse_args()
    torch.set_num_threads(4)
    checkpoint = torch.load(args.model, map_location='cpu', weights_only=True)
    config = checkpoint['config']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    network = UNet().to(device).eval()
    network.load_state_dict(checkpoint['model'])
    with Image.open(args.image) as image:
        source = image.convert('RGB')
    probability = predict_probability(network, source, config, device)
    mask = probability >= float(config.get('threshold', .5))
    args.output.mkdir(parents=True, exist_ok=True)
    stem = args.image.stem
    Image.fromarray(mask.astype(np.uint8)*255).save(args.output/f'{stem}_mask.png')
    overlay = np.asarray(source).copy()
    overlay[mask] = (.58*overlay[mask]+.42*np.array([0, 245, 100])).astype(np.uint8)
    Image.fromarray(overlay).save(args.output/f'{stem}_overlay.jpg', quality=95)
    print('已保存分割掩码和预览：', args.output.resolve())


if __name__ == '__main__':
    main()
