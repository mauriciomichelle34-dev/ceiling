"""Authoritative image origin configured on the server, independent of browsers."""
import json
import math
from pathlib import Path

DISH_DIAMETER_MM = 90


def validate_origin(value):
    if not isinstance(value, dict) or value.get('kind') != 'fixed_screw':
        raise ValueError('后台原点配置必须为固定螺丝中心。')
    width, height = value.get('image_width'), value.get('image_height')
    if any(type(v) is not int or v < 1 for v in (width, height)) or width*height > 25_000_000:
        raise ValueError('后台原点配置的照片尺寸无效。')
    x, y = value.get('x_px'), value.get('y_px')
    if any(isinstance(v, bool) or not isinstance(v, (int,float)) or not math.isfinite(v) for v in (x,y)):
        raise ValueError('后台螺丝中心需要有限像素坐标。')
    if not 0 <= x <= width-1 or not 0 <= y <= height-1:
        raise ValueError('后台螺丝中心超出照片范围。')
    return dict(kind='fixed_screw', x_px=x, y_px=y, image_width=width, image_height=height)


def load_origin(path):
    return validate_origin(json.loads(Path(path).read_text(encoding='utf-8-sig')))


def origin_for_image(config, width, height):
    origin = validate_origin(config)
    if (width, height) != (origin['image_width'], origin['image_height']):
        raise ValueError(f"照片尺寸与后台螺丝原点不匹配，需要 {origin['image_width']} × {origin['image_height']} px；请使用相同相机画幅。")
    return origin


def apply_origin(calibration, config, width, height):
    if not isinstance(calibration, dict):
        raise ValueError('请先填写培养皿尺寸标定。')
    return dict(calibration, coordinate_origin=origin_for_image(config, width, height))


def automatic_calibration(meta, config, width=None, height=None):
    """Use the detected circle and fixed 90 mm diameter, never browser geometry."""
    from sampling import validate_calibration
    suggestion = meta.get('dish_suggestion')
    if not isinstance(suggestion, dict):
        raise ValueError('没有本张照片的培养皿定位结果，无法自动标定。')
    width = meta.get('width') if width is None else width
    height = meta.get('height') if height is None else height
    origin = origin_for_image(config, width, height)
    c = dict(suggestion, dish_diameter_mm=DISH_DIAMETER_MM, coordinate_origin=origin)
    return validate_calibration(c, width, height, require_origin=True)
