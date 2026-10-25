"""Locate one compact red-halo/bright-core sensor spot without model labels.

This detector is for the fixed colour-camera setup. It does not infer hidden
mycelium or erase a rectangular neighbourhood around the sensor spot.
"""
import cv2
import numpy as np
from PIL import Image


class InfraredDetectionError(ValueError):
    pass


def detect_infrared(source, boundary=None):
    source = source.convert('RGB')
    preview = source.copy()
    preview.thumbnail((1024,1024), Image.Resampling.LANCZOS)
    rgb = np.asarray(preview).astype(np.int16)
    height,width = rgb.shape[:2]
    unit = width/1024
    sx,sy = source.width/width,source.height/height
    yy,xx = np.ogrid[:height,:width]
    inside = np.ones((height,width),bool)
    if boundary is not None:
        inside = ((xx*sx-boundary['center_x_px'])**2
                  +(yy*sy-boundary['center_y_px'])**2 <= boundary['radius_px']**2)
    r,g,b = rgb[:,:,0],rgb[:,:,1],rgb[:,:,2]
    red = (r>160)&(r-g>45)&(r-b>45)&inside
    white = (r>225)&(g>165)&(b>100)&inside
    empty = np.zeros((source.height,source.width),bool)
    info = {'method':'red_halo_bright_core_v1','status':'not_detected','spots':[],
            'excluded_pixels':0,'removed_foreground_pixels':0}
    if int(red.sum()) < max(8,round(30*unit*unit)):
        return empty,info
    # White background must be close to actual red evidence. The inscribed
    # radius favours the compact spot over long rim reflections or red text.
    nearest_red = cv2.distanceTransform((~red).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    support = (red | (white & (nearest_red<=22*unit))).astype(np.uint8)
    kernel = np.ones((3,3),np.uint8)
    support = cv2.morphologyEx(support,cv2.MORPH_CLOSE,kernel)
    support[~inside]=0
    interior = cv2.distanceTransform(support,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    maximum = float(interior.max())
    if maximum < max(2.,8*unit):
        return empty,info
    if maximum > 40*unit:
        raise InfraredDetectionError('红色亮区过大，无法可靠区分红外亮点与其他物体；请检查原图后重拍。')
    peaks = (interior>=maximum*.86).astype(np.uint8)
    # Small disconnected maxima can belong to one saturated spot. Merge only
    # nearby peak islands, retaining original peak pixels for the centroid.
    merge_size=max(3,int(round(10*unit))|1)
    merged=cv2.morphologyEx(peaks,cv2.MORPH_CLOSE,np.ones((merge_size,merge_size),np.uint8))
    count,labels,stats,centers = cv2.connectedComponentsWithStats(merged)
    candidates=[]
    for k in range(1,count):
        if stats[k,cv2.CC_STAT_AREA]<max(2,round(3*unit*unit)):
            continue
        py,px=np.where((labels==k)&(peaks>0))
        if not len(px):
            continue
        cx,cy=float(px.mean()),float(py.mean())
        local = ((xx-cx)/(38*unit))**2+((yy-cy)/(40*unit))**2<=1
        core_window = ((xx-cx)/(24*unit))**2+((yy-cy)/(28*unit))**2<=1
        ry,rx = np.where(red&local)
        if len(rx)<max(12,round(80*unit*unit)) or np.count_nonzero(white&core_window)<max(8,round(50*unit*unit)):
            continue
        sectors = np.floor((np.arctan2(ry-cy,rx-cx)+np.pi)*12/(2*np.pi)).astype(int)%12
        covered = int(np.count_nonzero(np.bincount(sectors,minlength=12)>=max(1,round(2*unit*unit))))
        if covered<8:
            continue
        hot = ((red | (white&core_window))&local).astype(np.uint8)
        hot = cv2.morphologyEx(hot,cv2.MORPH_CLOSE,kernel)
        contours,_ = cv2.findContours(hot,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        contours=[c for c in contours if cv2.contourArea(c)>=max(12,80*unit*unit)]
        if not contours:
            continue
        contour=max(contours,key=lambda c:cv2.pointPolygonTest(c,(cx,cy),True))
        # Fill the saturated centre, then add a two-preview-pixel guard. Avoid
        # inpainting: excluded pixels cannot contain verified visual evidence.
        spot=np.zeros((height,width),np.uint8)
        cv2.drawContours(spot,[contour],-1,1,cv2.FILLED)
        spot=cv2.dilate(spot,kernel,iterations=2)
        spot[~inside]=0
        if np.count_nonzero(spot)>max(100,.04*np.count_nonzero(inside)):
            raise InfraredDetectionError('疑似红外区域范围异常，本次停止识别，请检查照片。')
        candidates.append((cx,cy,covered,spot))
    if not candidates:
        return empty,info
    if len(candidates)>1:
        raise InfraredDetectionError('检测到多个相近强度的红色亮点，无法可靠确定传感器亮点；本次停止识别，请检查照片。')
    cx,cy,sectors,spot=candidates[0]
    native=cv2.resize(spot,source.size,interpolation=cv2.INTER_NEAREST)>0
    # One source pixel covers coordinate rounding at the restored boundary.
    native=cv2.dilate(native.astype(np.uint8),np.ones((3,3),np.uint8))>0
    if boundary is not None:
        y,x=np.ogrid[:source.height,:source.width]
        native &= ((x-boundary['center_x_px'])**2+(y-boundary['center_y_px'])**2<=boundary['radius_px']**2)
    x,y,w,h=cv2.boundingRect(native.astype(np.uint8))
    info.update(status='excluded',excluded_pixels=int(native.sum()),spots=[{
        'center_x_px':cx*sx,'center_y_px':cy*sy,'bounds_px':[x,y,w,h],
        'red_angular_sectors':sectors,'inscribed_radius_px':maximum*sx}])
    return native,info


def exclude_infrared(mask, ignored):
    if mask.ndim!=2 or mask.shape!=ignored.shape:
        raise ValueError('红外掩码与分割图尺寸不一致。')
    result=mask.copy()
    result[np.asarray(ignored,dtype=bool)]=0
    return result
