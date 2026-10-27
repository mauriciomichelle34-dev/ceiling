"""Locate the lower-right silver screw; cache only successful detections."""
from dataclasses import dataclass
from datetime import datetime, timezone
import copy
import json
import math
from pathlib import Path
import threading
import uuid

import cv2
import numpy as np

from fixed_origin import validate_origin


class ScrewDetectionError(ValueError):
    pass


@dataclass(frozen=True)
class ScrewConfig:
    image_width: int = 5472
    image_height: int = 3648
    search_roi: tuple = (.50, .55, .94, .98)

    def __post_init__(self):
        if any(type(v) is not int or v < 100 for v in (self.image_width, self.image_height)):
            raise ValueError('螺丝定位画幅无效。')
        if self.image_width * self.image_height > 25_000_000:
            raise ValueError('螺丝定位画幅超过 2500 万像素。')
        x0, y0, x1, y1 = self.search_roi
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in self.search_roi) or not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
            raise ValueError('螺丝搜索区域无效。')

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding='utf-8-sig')))

    def frame(self):
        return {'image_width':self.image_width, 'image_height':self.image_height}


def _score(gray, saturation, x, y, radius):
    reach = int(math.ceil(2.1 * radius))
    cx, cy = int(round(x)), int(round(y))
    if cx-reach < 0 or cy-reach < 0 or cx+reach >= gray.shape[1] or cy+reach >= gray.shape[0]:
        return None
    patch = gray[cy-reach:cy+reach+1, cx-reach:cx+reach+1]
    sat = saturation[cy-reach:cy+reach+1, cx-reach:cx+reach+1]
    yy, xx = np.indices(patch.shape)
    dx, dy = xx + cx-reach-x, yy + cy-reach-y
    distance = np.hypot(dx, dy)
    ring = (distance >= .75*radius) & (distance <= 1.17*radius)
    outside = (distance >= 1.5*radius) & (distance <= 2*radius)
    inside = distance <= .45*radius
    ring_gray = float(np.median(patch[ring]))
    outer_gray = float(np.median(patch[outside]))
    center_gray = float(np.median(patch[inside]))
    silver = (patch >= 100) & (sat < 90)
    fraction = float(np.mean(silver[ring]))
    angles = ((np.arctan2(dy, dx) + np.pi) * 12/(2*np.pi)).astype(int) % 12
    sectors = sum(np.count_nonzero(ring & (angles == sector) & silver) >= 2 for sector in range(12))
    if fraction < .25 or sectors < 9 or ring_gray-outer_gray < 28 or ring_gray-center_gray < 12 or outer_gray > 105:
        return None
    return {'score':round(fraction + sectors/12 + (ring_gray-outer_gray)/255, 4),
            'silver_fraction':round(fraction, 4), 'silver_sectors':int(sectors),
            'ring_gray':ring_gray, 'outer_gray':outer_gray, 'center_gray':center_gray}


def detect_screw(image, boundary=None, search_roi=(.50,.55,.94,.98)):
    rgb = np.asarray(image.convert('RGB'))
    height, width = rgb.shape[:2]
    scale = min(1.0, 1600/width)
    small = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    saturation = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)[:, :, 1]
    h, w = gray.shape
    x0,y0,x1,y1 = [int(v) for v in (search_roi[0]*w,search_roi[1]*h,search_roi[2]*w,search_roi[3]*h)]
    minimum, maximum = max(3, int(w*.004)), max(6, int(w*.023))
    circles = cv2.HoughCircles(cv2.GaussianBlur(gray[y0:y1,x0:x1],(5,5),1),
        cv2.HOUGH_GRADIENT, 1, max(12,minimum*2), param1=80,param2=18,
        minRadius=minimum,maxRadius=maximum)
    candidates=[]
    if circles is not None:
        for xx,yy,radius in circles[0]:
            x,y=float(xx+x0),float(yy+y0)
            if boundary:
                dish_radius=boundary.get('outer_radius_px',boundary.get('radius_px',0))
                if math.hypot(x/scale-boundary['center_x_px'],y/scale-boundary['center_y_px']) <= dish_radius+radius/scale:
                    continue
            # Hough may follow the inner edge of the bright ring; evaluate its head radius too.
            variants=[]
            for factor in (.85,1.0,1.2,1.4):
                head_radius=float(radius)*factor
                if not minimum <= head_radius <= maximum:
                    continue
                score=_score(gray,saturation,x,y,head_radius)
                if score is not None:
                    variants.append(dict(x=x/scale,y=y/scale,radius=head_radius/scale,**score))
            if variants:
                candidates.append(max(variants,key=lambda c:c['score']))
    candidates.sort(key=lambda c:c['score'],reverse=True)
    unique=[]
    for candidate in candidates:
        if not any(math.hypot(candidate['x']-other['x'],candidate['y']-other['y']) < max(candidate['radius'],other['radius']) for other in unique):
            unique.append(candidate)
    if not unique:
        raise ScrewDetectionError('未识别到右下角银色螺丝圆环与内孔。')
    if len(unique)>1:
        raise ScrewDetectionError('右下角出现多个银色螺丝候选，无法确定原点。')
    found=unique[0]
    # Refine the silver head on the original pixels, rather than using a thumbnail center.
    reach=int(math.ceil(found['radius']*2))
    left,top=max(0,int(found['x'])-reach),max(0,int(found['y'])-reach)
    right,bottom=min(width,int(found['x'])+reach+1),min(height,int(found['y'])+reach+1)
    patch=rgb[top:bottom,left:right]
    full_gray=cv2.cvtColor(patch,cv2.COLOR_RGB2GRAY)
    refined=cv2.HoughCircles(cv2.GaussianBlur(full_gray,(5,5),1),cv2.HOUGH_GRADIENT,1,
        max(10,int(found['radius'])),param1=90,param2=26,
        minRadius=max(3,int(found['radius']*.73)),maxRadius=max(5,int(found['radius']*1.4)))
    if refined is not None:
        choices=[(float(x+left),float(y+top),float(r)) for x,y,r in refined[0]
                 if math.hypot(x+left-found['x'],y+top-found['y']) < found['radius']*.4]
        if choices:
            x,y,r=min(choices,key=lambda c:math.hypot(c[0]-found['x'],c[1]-found['y']))
            found.update(x=x,y=y,radius=r)
    return {'origin':validate_origin({'kind':'fixed_screw','x_px':round(found['x'],3),
        'y_px':round(found['y'],3),'image_width':width,'image_height':height}),
        'radius_px':round(found['radius'],3),'evidence':{key:found[key] for key in
        ('score','silver_fraction','silver_sectors','ring_gray','outer_gray','center_gray')},
        'method':'silver_ring_dark_socket_v1'}


def origin_from_metadata(meta):
    record=meta.get('screw_origin')
    if not isinstance(record,dict) or record.get('status') not in ('detected','reused_last'):
        raise ScrewDetectionError('本张照片没有自动螺丝原点，请重新识别照片并生成坐标。')
    origin=validate_origin(record.get('origin'))
    if (origin['image_width'],origin['image_height']) != (meta.get('width'),meta.get('height')):
        raise ScrewDetectionError('本张照片的螺丝原点画幅不匹配。')
    return origin


class AutomaticScrewOrigin:
    def __init__(self, config, state_path, detector=detect_screw):
        self.config,self.path,self.detector=config,Path(state_path),detector
        self.lock=threading.RLock()
        self.last=None
        self.last_error=None
        if self.path.is_file():
            try:
                value=json.loads(self.path.read_text(encoding='utf-8'))
                origin=origin_from_metadata(dict(value,width=config.image_width,height=config.image_height))
                record=value['screw_origin']
                if record['status']!='detected' or not record.get('detection_id') or not record.get('detected_at'):
                    raise ValueError('Saved origin is not a successful detection')
                self.last=copy.deepcopy(dict(record,origin=origin))
            except (ValueError,KeyError,TypeError,OSError) as exc:
                self.last_error=f'上次螺丝定位记录不可用：{exc}'

    def status(self):
        with self.lock:
            return {'mode':'auto_screw','frame':self.config.frame(),
                    'last_success':copy.deepcopy(self.last),'last_error':self.last_error}

    def resolve(self, image, boundary=None):
        if image.size != (self.config.image_width,self.config.image_height):
            raise ScrewDetectionError(f'照片画幅不匹配，需要 {self.config.image_width} × {self.config.image_height} px；不能沿用其他画幅的螺丝原点。')
        with self.lock:
            try:
                result=self.detector(image,boundary,self.config.search_roi)
            except ScrewDetectionError as exc:
                self.last_error=str(exc)
                if self.last is None:
                    raise ScrewDetectionError(f'{exc} 尚无成功定位记录，本张未生成坐标。') from exc
                return {**copy.deepcopy(self.last),'status':'reused_last','reason':str(exc)}
            record={**result,'status':'detected','detection_id':uuid.uuid4().hex,
                    'detected_at':datetime.now(timezone.utc).isoformat()}
            self.path.parent.mkdir(parents=True,exist_ok=True)
            temporary=self.path.with_suffix('.tmp')
            temporary.write_text(json.dumps({'screw_origin':record},ensure_ascii=False,indent=2),encoding='utf-8')
            temporary.replace(self.path)
            self.last=copy.deepcopy(record)
            self.last_error=None
            return copy.deepcopy(record)
