"""Check image/mask pairs, write reproducible splits, and preview annotations.

Usage: python prepare_data.py --data-root data/MyceliumSeg
Requires Pillow and numpy; does not modify the source dataset.
"""
import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def collect(root, relative):
    folder = root / relative
    images = {p.stem: p for p in (folder / 'image').glob('*.jpg')
              if not p.name.startswith('._')}
    masks = {p.stem: p for p in (folder / 'mask').glob('*.png')
             if not p.name.startswith('._')}
    if not images or images.keys() != masks.keys():
        raise ValueError(f'Missing data or unmatched names: {folder}')
    rows = []
    for name in sorted(images):
        with Image.open(images[name]) as im, Image.open(masks[name]) as mask:
            im.load()
            mask.load()
            if im.size != mask.size:
                raise ValueError(f'Size mismatch: {images[name]}')
            if mask.mode != 'L' or mask.getextrema() != (0, 1):
                raise ValueError(f'Expected grayscale 0/1 labels: {masks[name]}')
            rows.append({'id': name, 'group': relative,
                         'image': images[name].relative_to(root).as_posix(),
                         'mask': masks[name].relative_to(root).as_posix(),
                         'width': im.width, 'height': im.height})
    print(f'Checked {relative}: {len(rows)} pairs', flush=True)
    return rows


def preview(root, rows, dest):
    w, h, label_h = 400, 300, 30
    canvas = Image.new('RGB', (3*w, 40 + len(rows)*(h+label_h)), 'white')
    draw = ImageDraw.Draw(canvas)
    for col, title in enumerate(['Original image', 'Mask (white = mycelium)', 'Overlay (green = mycelium)']):
        draw.text((col*w+10, 12), title, fill='black')
    for i, row in enumerate(rows):
        with Image.open(root / row['image']) as source:
            rgb = np.asarray(source.convert('RGB').resize((w, h), Image.Resampling.LANCZOS)).copy()
        with Image.open(root / row['mask']) as source:
            mask = np.asarray(source.resize((w, h), Image.Resampling.NEAREST)) == 1
        overlay = rgb.copy()
        overlay[mask] = (rgb[mask]*0.55 + np.array([0, 255, 100])*0.45).astype(np.uint8)
        panels = [Image.fromarray(rgb), Image.fromarray(mask.astype(np.uint8)*255).convert('RGB'), Image.fromarray(overlay)]
        y = 40 + i*(h+label_h)
        draw.text((10, y+8), f"{row['group']} / {row['id']}", fill='black')
        for col, panel in enumerate(panels):
            canvas.paste(panel, (col*w, y+label_h))
    canvas.save(dest)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, default=Path(__file__).resolve().parent / 'data' / 'MyceliumSeg')
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent / 'prepared')
    args = parser.parse_args()
    root = args.data_root.resolve()
    groups = ['labeled-GL/trainset', 'labeled-GL/testset',
              'labeled-MYG_PDA_TEMP/MYG', 'labeled-MYG_PDA_TEMP/PDA',
              'labeled-MYG_PDA_TEMP/TEMP15']
    data = {g: collect(root, g) for g in groups}
    training = data[groups[0]].copy()
    random.Random(42).shuffle(training)
    validation_count = round(len(training)*0.1)
    splits = {'train': sorted(training[validation_count:], key=lambda r: r['id']),
              'val': sorted(training[:validation_count], key=lambda r: r['id']),
              'test': data[groups[1]],
              'external': [r for g in groups[2:] for r in data[g]]}
    all_ids = [r['id'] for rows in splits.values() for r in rows]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError('Duplicate image IDs across splits')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        with (args.output_dir / f'{name}.csv').open('w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['id', 'group', 'image', 'mask', 'width', 'height'])
            writer.writeheader()
            writer.writerows(rows)
    report = {'data_root': str(root), 'seed': 42,
              'counts': {name: len(rows) for name, rows in splits.items()},
              'split_method': 'Provisional image-level split. If images share culture dishes or time series, use group-level splitting before reliable evaluation.',
              'test_policy': 'Do not use test or external labels to select models or tune thresholds.',
              'source_files_modified': False}
    (args.output_dir / 'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    preview(root, [data[g][len(data[g])//2] for g in groups], args.output_dir / 'annotation_preview.png')
    print(json.dumps(report['counts']), flush=True)
    print(f'Results: {args.output_dir.resolve()}', flush=True)


if __name__ == '__main__':
    main()
