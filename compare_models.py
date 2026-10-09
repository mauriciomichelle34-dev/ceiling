"""Compare frozen checkpoints at original resolution; never tune on test labels."""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

from model import UNet


def read_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def scores(pred, truth):
    tp = int(np.count_nonzero(pred & truth))
    fp = int(np.count_nonzero(pred & ~truth))
    fn = int(np.count_nonzero(~pred & truth))
    return {'dice': 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.,
            'precision': tp/(tp+fp) if tp+fp else (0. if tp+fn else 1.),
            'recall': tp/(tp+fn) if tp+fn else 1., 'fp': fp, 'fn': fn}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    cv2.setNumThreads(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    root = Path(json.loads((args.prepared/'report.json').read_text(encoding='utf-8'))['data_root'])
    models, configs, hashes = {}, {}, {}
    for name, path in [('baseline', args.baseline), ('candidate', args.candidate)]:
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        models[name] = UNet().to(device).eval()
        models[name].load_state_dict(checkpoint['model'])
        configs[name] = checkpoint['config']
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    rows = [(split, r) for split in ['val', 'test', 'external'] for r in read_rows(args.prepared/f'{split}.csv')]
    used = {hashlib.sha256((root/r['image']).read_bytes()).hexdigest()
            for r in read_rows(args.prepared/'train.csv')}
    seen = set()
    for split, row in rows:
        key = hashlib.sha256((root/row['image']).read_bytes()).hexdigest()
        if key in used or key in seen:
            raise ValueError('Duplicate evaluation content or training leakage: ' + row['image'])
        seen.add(key)
    args.output.mkdir(parents=True, exist_ok=False)
    records, visuals = [], {}
    with torch.inference_mode():
        for i, (split, row) in enumerate(rows, 1):
            with Image.open(root/row['image']) as im:
                source = im.convert('RGB')
            with Image.open(root/row['mask']) as im:
                if im.size != source.size or im.mode != 'L' or im.getextrema()[1] > 1:
                    raise ValueError('Invalid annotation: ' + row['mask'])
                truth = np.asarray(im) > 0
            group = Path(row['group']).name
            small_truth = cv2.resize(truth.astype(np.uint8), (400, 300), interpolation=cv2.INTER_NEAREST) > 0
            rgb = np.asarray(source.resize((400, 300), Image.Resampling.LANCZOS)).copy()
            panels = [rgb]
            ref = rgb.copy()
            ref[small_truth] = (.5*ref[small_truth]+.5*np.array([0, 255, 80])).astype(np.uint8)
            panels.append(ref)
            for name, network in models.items():
                c = configs[name]
                image = np.asarray(source.resize((c['width'], c['height']), Image.Resampling.BILINEAR), dtype=np.float32)/255
                tensor = torch.from_numpy(image.transpose(2, 0, 1).copy())[None].to(device)
                probability = network(tensor).sigmoid()[0, 0].cpu().numpy()
                if not np.isfinite(probability).all():
                    raise RuntimeError('Non-finite prediction')
                pred = cv2.resize(probability, source.size, interpolation=cv2.INTER_LINEAR) >= c['threshold']
                record = {'model': name, 'split': split, 'group': group, 'id': row['id'], **scores(pred, truth)}
                records.append(record)
                small_pred = cv2.resize(pred.astype(np.uint8), (400, 300), interpolation=cv2.INTER_NEAREST) > 0
                diff = rgb.copy()
                for mask, color in [(small_pred & ~small_truth, [255, 40, 40]), (~small_pred & small_truth, [40, 110, 255])]:
                    diff[mask] = (.35*diff[mask]+.65*np.array(color)).astype(np.uint8)
                panels.append(diff)
            if row['group'].startswith('extra/'):
                visuals[(group, row['id'])] = panels
            if i % 10 == 0 or i == len(rows):
                print(f'Compared {i}/{len(rows)}', flush=True)
    with (args.output/'per_image.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    grouped = defaultdict(list)
    for r in records:
        grouped[(r['split'], r['group'], r['model'])].append(r)
    summary = {}
    for (split, group, name), values in grouped.items():
        summary.setdefault(split+'/'+group, {})[name] = {
            'count': len(values), **{metric: float(np.mean([r[metric] for r in values])) for metric in ['dice', 'precision', 'recall']},
            'fp': sum(r['fp'] for r in values), 'fn': sum(r['fn'] for r in values)}
    report = {'groups': summary, 'checkpoint_sha256': hashes,
              'thresholds': {n: c['threshold'] for n, c in configs.items()},
              'aggregation': 'Mean per-image metrics at original resolution; FP/FN are pixel totals.',
              'validation_policy': 'Validation used for checkpoint selection. These are development results, not an untouched final test.',
              'regression_policy': 'Test/external labels excluded from fine-tuning and checkpoint selection; exact image hashes checked.',
              'threshold_tuned': False}
    (args.output/'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    if visuals:
        canvas = Image.new('RGB', (1600, 35+len(visuals)*335), 'white')
        draw = ImageDraw.Draw(canvas)
        for j, title in enumerate(['Original', 'Reference: green', 'Baseline: red FP / blue FN', 'Candidate: red FP / blue FN']):
            draw.text((j*400+8, 10), title, fill='black')
        for i, ((group, ident), panels) in enumerate(sorted(visuals.items())):
            a, b = [next(r for r in records if r['group'] == group and r['id'] == ident and r['model'] == n) for n in ['baseline', 'candidate']]
            y = 35+i*335
            draw.text((8, y+8), f"{group}/{ident} Dice {a['dice']:.3f} -> {b['dice']:.3f}; precision {a['precision']:.3f} -> {b['precision']:.3f}; recall {a['recall']:.3f} -> {b['recall']:.3f}", fill='black')
            for j, panel in enumerate(panels):
                canvas.paste(Image.fromarray(panel), (j*400, y+35))
        canvas.save(args.output/'validation_comparison.jpg', quality=92)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
