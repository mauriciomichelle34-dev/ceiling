"""Evaluate a frozen model on held-out sets and export full-size 0/1 masks.

Does not train, select a checkpoint, or tune a threshold on held-out data.
"""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

from model import UNet


def read_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def scores(prediction, reference):
    intersection = np.count_nonzero(prediction & reference)
    predicted = np.count_nonzero(prediction)
    actual = np.count_nonzero(reference)
    union = predicted + actual - intersection
    return (2*intersection/(predicted+actual) if predicted+actual else 1.,
            intersection/union if union else 1.)


def make_preview(rows, root, output, destination):
    ordered = sorted(rows, key=lambda r: r['dice_original'])
    selected = [ordered[0], ordered[len(ordered)//2], ordered[-1]]
    width, height, label = 384, 288, 34
    canvas = Image.new('RGB', (3*width, 30+3*(height+label)), 'white')
    draw = ImageDraw.Draw(canvas)
    for i, text in enumerate(['Original', 'Reference annotation', 'Model prediction']):
        draw.text((i*width+8, 9), text, fill='black')
    for i, (rank, row) in enumerate(zip(['Lowest Dice', 'Middle Dice', 'Highest Dice'], selected)):
        source = root/row['image']
        reference = root/row['mask']
        prediction = output/row['prediction']
        with Image.open(source) as im:
            rgb = np.asarray(im.convert('RGB').resize((width, height), Image.Resampling.LANCZOS)).copy()
        panels = [rgb]
        for path in [reference, prediction]:
            with Image.open(path) as im:
                mask = np.asarray(im.resize((width, height), Image.Resampling.NEAREST)) > 0
            overlay = rgb.copy()
            overlay[mask] = (.55*rgb[mask]+.45*np.array([0, 255, 100])).astype(np.uint8)
            panels.append(overlay)
        y = 30+i*(height+label)
        draw.text((8, y+10), f"{rank} | {row['group']} | {row['id']} | Dice={row['dice_original']:.4f}", fill='black')
        for j, panel in enumerate(panels):
            canvas.paste(Image.fromarray(panel), (j*width, y+label))
    canvas.save(destination)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--split', choices=['test', 'external', 'both'], default='both')
    args = parser.parse_args()
    torch.set_num_threads(4)
    cv2.setNumThreads(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(args.model, map_location='cpu', weights_only=True)
    config = checkpoint['config']
    root, prepared = Path(config['data_root']), Path(config['prepared'])
    width, height = config['width'], config['height']
    threshold = float(config['threshold'])
    network = UNet().to(device)
    network.load_state_dict(checkpoint['model'])
    network.eval()
    splits = ['test', 'external'] if args.split == 'both' else [args.split]
    rows = [(split, row) for split in splits for row in read_rows(prepared/f'{split}.csv')]
    used = {r['image'] for name in ['train', 'val'] for r in read_rows(args.model.parent/f'{name}.csv')}
    if not rows or any(row['image'] in used for _, row in rows):
        raise ValueError('Empty evaluation set or overlap with training/validation images')
    output = args.model.parent/('evaluation_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    output.mkdir(parents=True, exist_ok=False)
    print(f'Device: {device}; model epoch: {checkpoint["epoch"]}; threshold: {threshold}', flush=True)
    print(f'Output: {output}', flush=True)
    results = []
    with torch.inference_mode():
        for index, (split, row) in enumerate(rows, 1):
            with Image.open(root/row['image']) as im:
                original_size = im.size
                image = np.asarray(im.convert('RGB').resize((width, height), Image.Resampling.BILINEAR), dtype=np.float32)/255
            with Image.open(root/row['mask']) as im:
                if im.size != original_size:
                    raise ValueError(f'Mismatched image/mask size: {row["id"]}')
                reference = np.asarray(im)
                if not set(np.unique(reference)).issubset({0, 1}):
                    raise ValueError(f'Unexpected mask values: {row["id"]}')
                reference_small = np.asarray(im.resize((width, height), Image.Resampling.NEAREST)) > 0
                reference = reference > 0
            tensor = torch.from_numpy(image.transpose(2, 0, 1).copy())[None].to(device)
            probability = network(tensor).sigmoid()[0, 0].cpu().numpy()
            if not np.isfinite(probability).all():
                raise RuntimeError(f'Non-finite prediction: {row["id"]}')
            prediction_small = probability >= threshold
            # Resize probabilities before thresholding to produce the deliverable mask.
            prediction = cv2.resize(probability, original_size, interpolation=cv2.INTER_LINEAR) >= threshold
            dice, iou = scores(prediction, reference)
            dice_small, iou_small = scores(prediction_small, reference_small)
            group = 'GL-test' if split == 'test' else Path(row['group']).name
            relative = Path('masks')/group/(row['id']+'.png')
            (output/relative).parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(prediction.astype(np.uint8)).save(output/relative)
            results.append({'id': row['id'], 'group': group, 'image': row['image'], 'mask': row['mask'],
                            'prediction': relative.as_posix(), 'dice_original': dice, 'iou_original': iou,
                            'dice_resized': dice_small, 'iou_resized': iou_small,
                            'reference_pixels': int(reference.sum()), 'predicted_pixels': int(prediction.sum())})
            if index % 10 == 0 or index == len(rows):
                print(f'Evaluated {index}/{len(rows)}', flush=True)
    with (output/'per_image.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    groups = defaultdict(list)
    for row in results:
        groups[row['group']].append(row)
    summaries = {}
    print('\nGroup | Count | Mean Dice (original) | Mean IoU (original) | Lowest Dice', flush=True)
    for group, values in groups.items():
        summary = {'count': len(values),
                   **{f'mean_{metric}': float(np.mean([r[metric] for r in values]))
                      for metric in ['dice_original', 'iou_original', 'dice_resized', 'iou_resized']},
                   'min_dice_original': min(r['dice_original'] for r in values),
                   'empty_predictions': sum(r['predicted_pixels'] == 0 for r in values)}
        summaries[group] = summary
        print(f"{group} | {summary['count']} | {summary['mean_dice_original']:.4f} | {summary['mean_iou_original']:.4f} | {summary['min_dice_original']:.4f}", flush=True)
        make_preview(values, root, output, output/f'preview_{group}.png')
    report = {'model': str(args.model.resolve()), 'model_sha256': hashlib.sha256(args.model.read_bytes()).hexdigest(),
              'model_epoch': checkpoint['epoch'], 'threshold': threshold, 'inference_precision': 'fp32',
              'network_size': [width, height], 'aggregation': 'unweighted mean of per-image scores',
              'original_metrics': 'Predicted probabilities resized to original image dimensions before thresholding; compared with original masks.',
              'resized_metrics': 'Scores at model input resolution; same mask resizing as training validation.',
              'predictions': 'Grayscale PNG, original dimensions, background=0, mycelium=1.',
              'training_performed': False, 'groups': summaries}
    (output/'summary.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    lines = ['# 模型独立评估结果', '', f'模型：{args.model.resolve()}',
             f'训练轮次：{checkpoint["epoch"]}；固定分割阈值：{threshold}', '',
             '| 数据组 | 图片数 | 平均 Dice | 平均 IoU | 最低单图 Dice |',
             '|---|---:|---:|---:|---:|']
    for group, s in summaries.items():
        lines.append(f"| {group} | {s['count']} | {s['mean_dice_original']:.4f} | {s['mean_iou_original']:.4f} | {s['min_dice_original']:.4f} |")
    lines += ['', '以上是在原始分辨率下逐张计算后取平均的结果。模型输入为 512×384（默认），恢复尺寸不能恢复缩小时丢失的边缘细节。',
              '本次没有训练模型或使用测试结果调整阈值。训练分组暂按图片划分，尚未核实培养皿/拍摄序列是否存在组间重叠。',
              '', '每组预览依次展示该组 Dice 最低、中间、最高的图片；绿色分别代表参考标注或预测区域。',
              '导出的 masks 文件夹保存原图大小、同名的 0/1 PNG；普通看图软件中可能几乎全黑，这是标签值为 1 而非 255 的正常表现。']
    for group in summaries:
        lines += ['', f'## {group}', '', f'![{group}](preview_{group}.png)']
    (output/'评估报告.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f'Finished. Report: {output / "summary.json"}', flush=True)


if __name__ == '__main__':
    main()
