"""Whole-image inference shared by the website and the command-line tool."""
import cv2
import numpy as np
import torch
from PIL import Image


def prepare_input(source, config):
    width, height = int(config['width']), int(config['height'])
    preprocessing = config.get('preprocessing', 'legacy_stretch_rgb_0_1')
    if preprocessing == 'letterbox_rgb_0_1':
        scale = min(width/source.width, height/source.height)
        resized_w = max(1, round(source.width*scale))
        resized_h = max(1, round(source.height*scale))
        left, top = (width-resized_w)//2, (height-resized_h)//2
        image = np.zeros((height, width, 3), dtype=np.uint8)
        image[top:top+resized_h, left:left+resized_w] = cv2.resize(
            np.asarray(source), (resized_w, resized_h), interpolation=cv2.INTER_AREA)
        crop = (left, top, resized_w, resized_h)
    elif preprocessing == 'legacy_stretch_rgb_0_1':
        image = np.asarray(source.resize((width, height), Image.Resampling.BILINEAR))
        crop = (0, 0, width, height)
    else:
        raise ValueError('Unsupported model preprocessing: '+str(preprocessing))
    return image, crop


def predict_probability(network, source, config, device):
    """Restore raw probabilities to source size; no dish or component filtering."""
    image, (left, top, width, height) = prepare_input(source, config)
    if config.get('preprocessing', 'legacy_stretch_rgb_0_1') == 'legacy_stretch_rgb_0_1':
        normalized = np.asarray(image, dtype=np.float32)/255
        tensor = torch.from_numpy(normalized.transpose(2, 0, 1).copy())[None].to(device)
    else:
        tensor = torch.from_numpy(image.transpose(2, 0, 1).copy()).float()[None].to(device)/255
    with torch.inference_mode():
        probability = network(tensor).sigmoid()[0, 0].cpu().numpy()
    if not np.isfinite(probability).all():
        raise RuntimeError('Non-finite model prediction')
    return cv2.resize(probability[top:top+height, left:left+width], source.size,
                      interpolation=cv2.INTER_LINEAR)
