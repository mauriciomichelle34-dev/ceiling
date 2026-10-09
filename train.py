"""Train a baseline using train.csv and val.csv only. No extra downloads.

All model choices use validation data. Test and external sets remain untouched.
"""
import argparse
import csv
import hashlib
import json
import random
import time
from datetime import datetime
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from model import UNet

HERE = Path(__file__).resolve().parent


def read_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


class Pairs(Dataset):
    def __init__(self, root, rows, cache, augment, width, height):
        self.augment = augment
        self.items = []
        cache.mkdir(parents=True, exist_ok=True)
        for i, row in enumerate(rows):
            image_path, mask_path = root/row['image'], root/row['mask']
            # Invalidate cached arrays when a source file changes.
            identity = '|'.join(str(p.resolve()) + ':' + str(p.stat().st_mtime_ns)
                                + ':' + str(p.stat().st_size) for p in [image_path, mask_path])
            key = hashlib.sha256(f'v1|{width}|{height}|{identity}'.encode()).hexdigest()[:24]
            target = cache / (key + '.npy')
            if not target.exists():
                with Image.open(image_path) as im:
                    image = np.asarray(im.convert('RGB').resize((width, height), Image.Resampling.BILINEAR))
                with Image.open(mask_path) as im:
                    mask = np.asarray(im.resize((width, height), Image.Resampling.NEAREST))
                if not set(np.unique(mask)).issubset({0, 1}):
                    raise ValueError(f'Invalid mask: {mask_path}')
                array = np.concatenate([image, mask[..., None]], axis=2)
                temporary = target.with_suffix('.tmp')
                with temporary.open('wb') as f:
                    np.save(f, array)
                temporary.replace(target)
            self.items.append(target)
            if (i+1) % 50 == 0 or i+1 == len(rows):
                print(f'Preparing images: {i+1}/{len(rows)}', flush=True)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        array = np.load(self.items[index], allow_pickle=False)
        # Spatial transformations must always be identical for image and mask.
        if self.augment:
            if random.random() < .5:
                array = array[:, ::-1]
            if random.random() < .5:
                array = array[::-1]
        image = array[..., :3].astype(np.float32) / 255
        if self.augment:
            image = np.clip((image-.5)*random.uniform(.9, 1.1)+.5, 0, 1)
            image = np.clip(image * random.uniform(.85, 1.15), 0, 1)
        mask = array[..., 3].astype(np.float32)
        return torch.from_numpy(image.transpose(2, 0, 1).copy()), torch.from_numpy(mask[None].copy())


def loss_fn(logits, target):
    logits = logits.float()
    bce = F.binary_cross_entropy_with_logits(logits, target)
    probability = logits.sigmoid()
    intersection = (probability*target).sum((1, 2, 3))
    dice = (2*intersection+1) / (probability.sum((1, 2, 3))+target.sum((1, 2, 3))+1)
    return .5*bce + .5*(1-dice.mean())


def run_epoch(model, loader, device, optimizer=None, precision='fp32'):
    training = optimizer is not None
    model.train(training)
    loss_total, dice_total, iou_total, count = 0., 0., 0., 0
    per_image = []
    with torch.set_grad_enabled(training):
        for image, target in loader:
            image, target = image.to(device), target.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            # BF16 has a wider exponent range than FP16 and does not need loss
            # scaling. Keep FP32 as the fallback on unsupported hardware.
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16,
                                    enabled=precision == 'bf16'):
                logits = model(image)
            # Reductions and the segmentation loss are always computed in FP32.
            loss = loss_fn(logits, target)
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite loss; training stopped.')
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
            pred = logits.detach().float().sigmoid() >= .5
            truth = target > .5
            intersection = (pred & truth).sum((1, 2, 3)).float()
            total = pred.sum((1, 2, 3)) + truth.sum((1, 2, 3))
            union = (pred | truth).sum((1, 2, 3)).float()
            dice_values = ((2*intersection+1e-7)/(total+1e-7))
            per_image.extend(dice_values.cpu().tolist())
            dice_total += dice_values.sum().item()
            iou_total += ((intersection+1e-7)/(union+1e-7)).sum().item()
            count += image.shape[0]
            loss_total += loss.item()*image.shape[0]
    return loss_total/count, dice_total/count, iou_total/count, per_image


def save_preview(model, dataset, device, dest):
    model.eval()
    rows = []
    with torch.no_grad():
        for i in range(min(3, len(dataset))):
            image, target = dataset[i]
            pred = model(image[None].to(device)).float().sigmoid()[0, 0].cpu().numpy() >= .5
            rgb = (image.numpy().transpose(1, 2, 0)*255).astype(np.uint8)
            truth = target[0].numpy() > .5
            panels = [rgb]
            for mask in [truth, pred]:
                overlay = rgb.copy()
                overlay[mask] = (.55*rgb[mask]+.45*np.array([0, 255, 100])).astype(np.uint8)
                panels.append(overlay)
            rows.append(np.concatenate(panels, axis=1))
    body = Image.fromarray(np.concatenate(rows, axis=0))
    canvas = Image.new('RGB', (body.width, body.height+30), 'white')
    canvas.paste(body, (0, 30))
    draw = ImageDraw.Draw(canvas)
    for i, title in enumerate(['Original', 'Reference annotation', 'Model prediction']):
        draw.text((i*(body.width//3)+8, 8), title, fill='black')
    canvas.save(dest)


def save_checkpoint(path, payload):
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepared', type=Path, default=HERE/'prepared')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--height', type=int, default=384)
    parser.add_argument('--smoke-test', action='store_true')
    parser.add_argument('--precision', choices=['auto', 'bf16', 'fp32'], default='auto')
    parser.add_argument('--run-root', type=Path, default=HERE/'runs')
    parser.add_argument('--init-model', type=Path, help='Fine-tune a compatible U-Net checkpoint.')
    parser.add_argument('--learning-rate', type=float, default=.001)
    parser.add_argument('--extra-fraction', type=float, default=0., help='Sampling fraction for extra/ groups; 0 uses uniform image sampling.')
    parser.add_argument('--group-balanced-selection', action='store_true')
    parser.add_argument('--cache-dir', type=Path, default=HERE/'work'/'mycelium_cache')
    args = parser.parse_args()
    if args.learning_rate <= 0 or not 0 <= args.extra_fraction < 1:
        parser.error('Positive learning rate and 0 <= extra-fraction < 1 required.')
    if args.epochs < 1 or args.batch_size < 1 or min(args.width, args.height) < 32 or args.width % 16 or args.height % 16:
        parser.error('Positive epochs/batch size required; width and height must be multiples of 16, at least 32.')
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(4)
    cv2.setNumThreads(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    supports_bf16 = device.type == 'cuda' and torch.cuda.is_bf16_supported()
    precision = ('bf16' if supports_bf16 else 'fp32') if args.precision == 'auto' else args.precision
    if precision == 'bf16' and not supports_bf16:
        parser.error('BF16 is not supported on this device; use --precision fp32.')
    print('Device:', torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU (slow)', flush=True)
    print('Precision:', precision, '(FP32 loss; finite-gradient checks enabled)', flush=True)
    report = json.loads((args.prepared/'report.json').read_text(encoding='utf-8'))
    root = Path(report['data_root'])
    train_rows, val_rows = read_rows(args.prepared/'train.csv'), read_rows(args.prepared/'val.csv')
    if not train_rows or not val_rows:
        raise ValueError('Empty train or validation split')
    if {r['image'] for r in train_rows} & {r['image'] for r in val_rows}:
        raise ValueError('Train and validation images overlap')
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out = args.run_root / ('smoke_' + stamp if args.smoke_test else stamp)
    out.mkdir(parents=True, exist_ok=False)
    if args.smoke_test:
        def smoke_subset(rows):
            groups = sorted({r['group'] for r in rows})
            return [row for group in groups for row in [r for r in rows if r['group'] == group][:2]]
        train_rows, val_rows = smoke_subset(train_rows), smoke_subset(val_rows)
        args.epochs = 1
    config = {**{k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()}, 'prepared': str(args.prepared.resolve()), 'data_root': str(root),
              'run_root': str(args.run_root.resolve()), 'precision_used': precision,
              'device': str(device), 'train_count': len(train_rows), 'val_count': len(val_rows),
              'architecture': 'UNet GroupNorm base16', 'seed': 42,
              'threshold': .5, 'resize': f'whole image, {args.width}x{args.height}; nearest-neighbor masks',
              'selection': 'highest equal-group mean validation Dice' if args.group_balanced_selection else 'highest mean per-image validation Dice',
              'test_used': False, 'external_used': False,
              'split_note': report['split_method'],
              'torch_version': str(torch.__version__)}
    (out/'config.json').write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding='utf-8')
    for name, rows in [('train', train_rows), ('val', val_rows)]:
        with (out/f'{name}.csv').open('w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print('Output:', out, flush=True)
    print('First run prepares a smaller local cache; source images stay unchanged.', flush=True)
    cache = args.cache_dir
    train_set = Pairs(root, train_rows, cache, True, args.width, args.height)
    val_set = Pairs(root, val_rows, cache, False, args.width, args.height)
    sampler = None
    if args.extra_fraction:
        extra_groups = sorted({r['group'] for r in train_rows if r['group'].startswith('extra/')})
        base_count = sum(not r['group'].startswith('extra/') for r in train_rows)
        if not extra_groups or not base_count:
            raise ValueError('Balanced sampling requires base images and extra/ groups')
        counts = {g: sum(r['group'] == g for r in train_rows) for g in extra_groups}
        weights = [args.extra_fraction/len(extra_groups)/counts[r['group']] if r['group'] in counts
                   else (1-args.extra_fraction)/base_count for r in train_rows]
        sampler = WeightedRandomSampler(weights, len(train_rows), replacement=True)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=sampler is None, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = UNet().to(device)
    if args.init_model:
        initial = torch.load(args.init_model, map_location='cpu', weights_only=True)
        model.load_state_dict(initial['model'])
        print('Fine-tuning:', args.init_model, flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best = -1.
    fields = ['epoch', 'train_loss', 'val_loss', 'val_dice', 'val_iou', 'selection_score', 'group_dice', 'seconds']
    with (out/'history.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, args.epochs+1):
            start = time.perf_counter()
            train_loss, _, _, _ = run_epoch(model, train_loader, device, optimizer, precision)
            val_loss, dice, iou, values = run_epoch(model, val_loader, device, precision=precision)
            grouped = defaultdict(list)
            for row, value in zip(val_rows, values):
                grouped[row['group']].append(value)
            group_dice = {g: float(np.mean(v)) for g, v in grouped.items()}
            score = float(np.mean(list(group_dice.values()))) if args.group_balanced_selection else dice
            scheduler.step()
            record = dict(epoch=epoch, train_loss=train_loss, val_loss=val_loss,
                          val_dice=dice, val_iou=iou, selection_score=score,
                          group_dice=json.dumps(group_dice), seconds=round(time.perf_counter()-start, 1))
            writer.writerow(record)
            f.flush()
            payload = {'model': model.state_dict(), 'config': config, 'epoch': epoch,
                       'val_dice': dice, 'val_iou': iou, 'selection_score': score, 'group_dice': group_dice}
            save_checkpoint(out/'last_model.pt', payload)
            improved = score > best
            if improved:
                best = score
                save_checkpoint(out/'best_model.pt', payload)
                save_preview(model, val_set, device, out/'best_preview.png')
            print(f"Epoch {epoch:02d}/{args.epochs} | loss={train_loss:.4f} | val Dice={dice:.4f} | score={score:.4f} | groups={group_dice} | {record['seconds']}s" + (' | saved best' if improved else ''), flush=True)
    # Confirm the saved model can be reloaded for later inference.
    checkpoint = torch.load(out/'best_model.pt', map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model'])
    print(f'Finished. Best validation selection score: {best:.4f}', flush=True)
    print(f'Model: {out / "best_model.pt"}', flush=True)
    print(f'Preview: {out / "best_preview.png"}', flush=True)
    if args.smoke_test:
        print('SMOKE TEST ONLY: these weights are not a trained annotation model.', flush=True)


if __name__ == '__main__':
    main()
