"""Locate concentric dish rims independently of the mycelium prediction."""
import cv2
import numpy as np
from PIL import Image


class DishNotFound(ValueError):
    pass


def _profile(gx, gy, cx, cy, radii):
    angles = np.linspace(0, 2*np.pi, 360, endpoint=False, dtype=np.float32)
    cosine, sine = np.cos(angles), np.sin(angles)
    x = (cx + radii[:, None]*cosine).astype(np.float32)
    y = (cy + radii[:, None]*sine).astype(np.float32)
    dx = cv2.remap(gx, x, y, cv2.INTER_LINEAR)
    dy = cv2.remap(gy, x, y, cv2.INTER_LINEAR)
    magnitude = np.hypot(dx, dy)
    radial = np.abs(dx*cosine + dy*sine)
    # A straight conveyor edge only supports a small part of a circle.
    supported = (radial >= 12) & (radial >= magnitude*.75)
    coverage = supported.mean(axis=1)
    sectors = (supported.reshape(len(radii), 12, 30).mean(axis=2) >= .35).sum(axis=1)
    strength = np.minimum(radial, 100).mean(axis=1)
    return coverage, sectors, strength


def _refine_circle(gx, gy, cx, cy, radius):
    """Fit distributed rim pixels rather than trusting a Hough center alone."""
    angles = np.linspace(0, 2*np.pi, 360, endpoint=False, dtype=np.float32)
    cosine, sine = np.cos(angles), np.sin(angles)
    for _ in range(4):
        band = max(4, radius*.045)
        offsets = np.arange(-band, band+.25, .5, dtype=np.float32)
        radii = radius+offsets
        x = (cx+radii[:, None]*cosine).astype(np.float32)
        y = (cy+radii[:, None]*sine).astype(np.float32)
        dx = cv2.remap(gx, x, y, cv2.INTER_LINEAR)
        dy = cv2.remap(gy, x, y, cv2.INTER_LINEAR)
        radial = np.abs(dx*cosine+dy*sine)
        aligned = radial >= np.hypot(dx, dy)*.75
        weighted = np.minimum(radial, 100)*aligned*np.exp(-.5*(offsets[:, None]/(band*.6))**2)
        rows = weighted.argmax(axis=0)
        good = radial[rows, np.arange(360)] >= 12
        if good.sum() < 180:
            break
        points = np.column_stack((x[rows, np.arange(360)][good], y[rows, np.arange(360)][good])).astype(float)
        fit = np.ones(len(points), dtype=bool)
        for _ in range(3):
            p = points[fit]
            if len(p) < 150:
                break
            a, b, c = np.linalg.lstsq(np.column_stack((2*p, np.ones(len(p)))), (p*p).sum(axis=1), rcond=None)[0]
            r = np.sqrt(max(0, c+a*a+b*b))
            residual = np.abs(np.hypot(points[:, 0]-a, points[:, 1]-b)-r)
            fit = residual <= max(1.5, float(np.median(residual))*2)
        if len(p) < 150 or abs(r-radius) > band or np.hypot(a-cx, b-cy) > band:
            break
        cx, cy, radius = float(a), float(b), float(r)
    return cx, cy, radius


def detect_dish(source):
    """Return an outer calibration circle and a guarded inner recognition circle.

    Require distributed circular evidence and two distinct concentric edges.
    Missing/partial dishes fail closed rather than receiving a guessed ROI.
    Pixel geometry is returned at the source resolution.
    """
    preview = source.copy()
    preview.thumbnail((900, 900), Image.Resampling.BILINEAR)
    image = np.asarray(preview)
    height, width = image.shape[:2]
    shortest = min(width, height)
    gray = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), (5, 5), 1)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edges = cv2.Canny(gray, 30, 80)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contour_circles = []
    for contour in contours:
        if len(contour) < 100:
            continue
        (x, y), (a, b), _ = cv2.fitEllipse(contour)
        r = (a+b)/4
        if not shortest*.16 <= r <= shortest*.495 or min(a, b)/max(a, b) < .94:
            continue
        if cv2.contourArea(contour) < np.pi*r*r*.8:
            continue
        residual = np.abs(np.hypot(contour[:, 0, 0]-x, contour[:, 0, 1]-y)-r)
        if np.percentile(residual, 90) <= max(2, r*.015):
            contour_circles.append((x, y, r))
    if contour_circles:
        circles = np.array([contour_circles], dtype=np.float32)
    else:
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1,
            minDist=max(20, shortest*.08), param1=80, param2=32,
            minRadius=max(12, int(shortest*.16)), maxRadius=int(shortest*.495))
    candidates = []
    if circles is not None:
        for cx, cy, radius in circles[0]:
            cx, cy, radius = _refine_circle(gx, gy, float(cx), float(cy), float(radius))
            if min(cx-radius, cy-radius, width-1-cx-radius, height-1-cy-radius) < 1:
                continue
            radii = np.arange(radius-3, radius+3.25, .5, dtype=np.float32)
            coverage, sectors, strength = _profile(gx, gy, cx, cy, radii)
            best = int(np.argmax(coverage + np.minimum(strength/100, .3)))
            if coverage[best] >= .58 and sectors[best] >= 10:
                # Select the outer rim, not a stronger smaller colony/inner ring.
                score = radii[best]
                candidates.append((score, float(cx), float(cy), float(radii[best])))
    if not candidates:
        raise DishNotFound('未能可靠定位完整培养皿内边缘。请使用清晰、完整的俯视照片后重试；本次未输出菌丝结果。')
    _, cx, cy, radius = max(candidates)
    # Locate the rim family, including weak inner edges, around the voted circle.
    for _ in range(2):
        radii = np.arange(np.floor(radius*.70), min(cx-1, cy-1, width-2-cx, height-2-cy), .5, dtype=np.float32)
        coverage, sectors, strength = _profile(gx, gy, cx, cy, radii)
        # Merge the two flanks of a blurred edge before looking for rim peaks.
        # A fixed radial grid avoids changing the chosen wall with Hough rounding.
        strength = np.convolve(strength, np.ones(5)/5, mode='same')
        supported = (coverage >= .55) & (sectors >= 10)
        valid = np.convolve(supported.astype(int), np.ones(5), mode='same') > 0
        peaks = np.flatnonzero(valid & (strength >= np.r_[0, strength[:-1]]) & (strength >= np.r_[strength[1:], 0]))
        if len(peaks) < 2:
            break
        if _ == 0:
            cx, cy, radius = _refine_circle(gx, gy, cx, cy, float(radii[peaks[-1]]))
    if len(peaks) < 2:
        raise DishNotFound('培养皿内外边缘无法区分。请换一张边缘清晰的俯视照片后重试；本次未输出菌丝结果。')
    outer = float(radii[peaks[-1]])
    inner_peaks = [i for i in peaks if outer*.75 <= radii[i] <= outer-3]
    if not inner_peaks:
        raise DishNotFound('培养皿内外边缘无法区分。请换一张边缘清晰的俯视照片后重试；本次未输出菌丝结果。')
    inner_index = inner_peaks[0]
    inner = float(radii[inner_index])
    scale = source.width/preview.width
    # Keep antialiasing, highlights and preview-outline glow inside the wall.
    guard = max(1.5*scale, 1.)
    boundary = {'center_x_px': cx*scale, 'center_y_px': cy*scale,
                'radius_px': inner*scale-guard, 'inner_edge_radius_px': inner*scale,
                'outer_radius_px': outer*scale, 'margin_px': guard,
                'edge_support': float(coverage[inner_index]),
                'method': 'concentric_radial_edges', 'status': 'detected'}
    suggestion = {'center_x_px': cx*scale, 'center_y_px': cy*scale,
                  'dish_diameter_px': outer*2*scale, 'source': 'detected_outer_rim'}
    return boundary, suggestion


def restrict_to_dish(mask, boundary):
    """Set every source pixel outside the detected interior to zero."""
    y, x = np.ogrid[:mask.shape[0], :mask.shape[1]]
    inside = ((x-boundary['center_x_px'])**2 + (y-boundary['center_y_px'])**2
              <= boundary['radius_px']**2)
    return ((mask > 0) & inside).astype(np.uint8)


def prepare_model_image(source, boundary):
    """Replace bright exterior interference with the photo's dark background.

    Keep the ordinary background texture and every dish pixel unchanged; a
    solid black exterior introduces a sharp new edge and changes model input.
    """
    support = restrict_to_dish(np.ones((source.height, source.width), np.uint8),
                              dict(boundary, radius_px=boundary['outer_radius_px']))
    image = np.asarray(source).copy()
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    step = max(1, max(source.size)//900)
    samples = image[::step, ::step]
    dark = (support[::step, ::step] == 0) & (gray[::step, ::step] <= 120)
    if dark.sum() < 64:
        dark = gray[::step, ::step] <= 120
    background = np.median(samples[dark], axis=0) if dark.any() else np.array([24, 24, 24])
    image[(support == 0) & (gray > 120)] = background.astype(np.uint8)
    return Image.fromarray(image)
