"""Equal-circle planning in the predicted mask. Coordinates are NOT machine coordinates.

Search rotated/translated hexagonal grids, then fill remaining feasible gaps.
This is a deterministic heuristic, not a proof of a globally maximum packing.
Containment uses full-resolution Euclidean distance with a conservative pixel
boundary allowance; background holes and separate components are retained.
"""
import base64
import copy
import csv
import io
import json
import math
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

MAX_POINTS = 5000


def number(value, label, minimum=0, maximum=10000, positive=False):
    if isinstance(value, bool):
        raise ValueError(f'{label}需要填写数字。')
    try:
        result = float(value)
    except (ValueError, TypeError):
        raise ValueError(f'{label}需要填写数字。') from None
    if not math.isfinite(result) or result < minimum or result > maximum or (positive and result == 0):
        raise ValueError(f'{label}超出允许范围。')
    return result


def validate_calibration(data, width, height):
    if not isinstance(data, dict):
        raise ValueError('请先完成固定拍摄标定。')
    cx = number(data.get('center_x_px'), '皿中心 X', maximum=width-1)
    cy = number(data.get('center_y_px'), '皿中心 Y', maximum=height-1)
    diameter = number(data.get('dish_diameter_px'), '图中皿直径', maximum=2*max(width, height), positive=True)
    mm = number(data.get('dish_diameter_mm'), '培养皿实际直径', positive=True)
    radius = diameter/2
    if cx-radius < -.5 or cy-radius < -.5 or cx+radius > width-.5 or cy+radius > height-.5:
        raise ValueError('标定圆超出图片，请核对皿中心和直径，使用完整俯视照片。')
    return dict(center_x_px=cx, center_y_px=cy, dish_diameter_px=diameter,
                dish_diameter_mm=mm, mm_per_pixel=mm/diameter,
                image_width=width, image_height=height)


class Geometry:
    def __init__(self, mask, calibration, radius):
        self.h, self.w = mask.shape
        self.calibration = calibration
        self.radius = radius
        # Pad the image: even an all-foreground image has a finite outer edge.
        self.distance = cv2.distanceTransform(np.pad(mask.astype(np.uint8), 1), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]

    def valid(self, points):
        points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        if not len(points):
            return np.zeros(0, dtype=bool)
        x, y = points[:, 0], points[:, 1]
        ix, iy = np.rint(x).astype(int), np.rint(y).astype(int)
        bounds = (ix >= 0) & (ix < self.w) & (iy >= 0) & (iy < self.h)
        clearance = np.zeros(len(points))
        # Distance to nearest zero-center minus half pixel diagonal is a lower
        # bound on distance to the union of background pixel squares.
        clearance[bounds] = self.distance[iy[bounds], ix[bounds]] - np.hypot(x[bounds]-ix[bounds], y[bounds]-iy[bounds]) - math.sqrt(.5)
        c = self.calibration
        dish_ok = np.hypot(x-c['center_x_px'], y-c['center_y_px']) + self.radius <= c['dish_diameter_px']/2 + 1e-8
        return bounds & dish_ok & (clearance >= self.radius+1e-6)


def add_candidates(seed, candidates, pitch):
    """Spatial bins avoid quadratic all-pairs checks while filling gaps."""
    points = [tuple(map(float, p)) for p in seed]
    bins = {}
    def register(point):
        key = (math.floor(point[0]/pitch), math.floor(point[1]/pitch))
        bins.setdefault(key, []).append(point)
    for point in points:
        register(point)
    for x, y in candidates:
        key = (math.floor(x/pitch), math.floor(y/pitch))
        neighbors = (q for dx in (-1, 0, 1) for dy in (-1, 0, 1) for q in bins.get((key[0]+dx, key[1]+dy), ()))
        if any((x-a)**2+(y-b)**2 < (pitch-1e-7)**2 for a, b in neighbors):
            continue
        point = (float(x), float(y))
        points.append(point)
        register(point)
        if len(points) > MAX_POINTS:
            raise ValueError('取样点超过 5000 个，请增大口径或间隙后重试。')
    return np.asarray(points, dtype=float).reshape(-1, 2)


def pack(mask, calibration, diameter_mm, gap_mm):
    scale = calibration['mm_per_pixel']
    radius, pitch = diameter_mm/(2*scale), (diameter_mm+gap_mm)/scale
    if radius < 2:
        raise ValueError('取样圆半径小于 2 个原图像素，无法可靠判断边界，请增大口径或提高照片分辨率。')
    if not np.any(mask) or diameter_mm > calibration['dish_diameter_mm']:
        return np.empty((0, 2)), radius
    geometry = Geometry(mask, calibration, radius)
    h, w = mask.shape
    if (math.hypot(w, h)/pitch+4)**2 > 100000:
        raise ValueError('预计候选点过多，请增大口径或间隙。')
    diagonal = math.hypot(w, h)/2
    row_step = pitch*math.sqrt(3)/2
    rows = np.arange(-math.ceil(diagonal/row_step)-2, math.ceil(diagonal/row_step)+3)
    cols = np.arange(-math.ceil(diagonal/pitch)-2, math.ceil(diagonal/pitch)+3)
    col_grid, row_grid = np.meshgrid(cols, rows)
    lattice = np.column_stack(((col_grid+(.5*(row_grid % 2))).ravel()*pitch, row_grid.ravel()*row_step))
    best_grids = []
    # 12 rotations x 16 offsets, with deterministic tie handling.
    for angle in range(0, 60, 5):
        radians = math.radians(angle)
        rotation = np.array([[math.cos(radians), -math.sin(radians)], [math.sin(radians), math.cos(radians)]])
        for phase_y in range(4):
            for phase_x in range(4):
                points = (lattice+np.array([phase_x*pitch/4, phase_y*row_step/4])) @ rotation.T + [w/2, h/2]
                points = points[geometry.valid(points)]
                best_grids.append(points)
                best_grids.sort(key=len, reverse=True)
                best_grids = best_grids[:3]
    # A finer search adds circles in leftovers without moving valid seed circles.
    step = max(1., pitch/5)
    if math.ceil(w/step)*math.ceil(h/step) > 2_000_000:
        raise ValueError('细化搜索过密，请增大口径或间隙。')
    gx, gy = np.meshgrid(np.arange(0, w, step), np.arange(0, h, step))
    candidates = np.column_stack((gx.ravel(), gy.ravel()))
    candidates = candidates[geometry.valid(candidates)]
    # Local distance maxima rescue small disconnected regions missed by the grids.
    peaks = (geometry.distance >= cv2.dilate(geometry.distance, np.ones((3, 3), np.uint8))) & (geometry.distance >= radius+math.sqrt(.5)+1e-6)
    py, px = np.nonzero(peaks)
    if len(px):
        peaks_xy = np.column_stack((px, py))[::max(1, len(px)//50000)]
        candidates = np.concatenate((candidates, peaks_xy[geometry.valid(peaks_xy)]))
    if len(candidates):
        clearance = geometry.distance[np.rint(candidates[:, 1]).astype(int), np.rint(candidates[:, 0]).astype(int)]
        candidates = candidates[np.argsort(clearance, kind='stable')]
    best = best_grids[0]
    if len(best) > MAX_POINTS:
        raise ValueError('取样点超过 5000 个，请增大口径或间隙。')
    for seed in best_grids:
        completed = add_candidates(seed, candidates, pitch)
        if len(completed) > len(best):
            best = completed
    if len(best) > MAX_POINTS:
        raise ValueError('取样点超过 5000 个，请增大口径或间隙。')
    if len(best) and not np.all(geometry.valid(best)):
        raise RuntimeError('取样圆边界校验失败。')
    # Stable top-to-bottom serpentine ordering; no claim of a shortest route.
    band_height = max(1., row_step)
    best = sorted(best.tolist(), key=lambda p: (int(p[1]/band_height), p[0] if int(p[1]/band_height) % 2 == 0 else -p[0]))
    return np.asarray(best).reshape(-1, 2), radius


def contour_positions(points, contour):
    """Project centers onto a closed polyline; return arc-length positions."""
    vertices = contour.reshape(-1, 2).astype(float)
    if len(vertices) < 2:
        return np.zeros(len(points))
    vectors = np.roll(vertices, -1, axis=0)-vertices
    lengths = np.linalg.norm(vectors, axis=1)
    starts = np.concatenate(([0.], np.cumsum(lengths[:-1])))
    best_distance = np.full(len(points), np.inf)
    positions = np.zeros(len(points))
    # Bound temporary memory for large, jagged contours.
    for p0 in range(0, len(points), 64):
        p1 = min(p0+64, len(points))
        for s0 in range(0, len(vertices), 1024):
            s1 = min(s0+1024, len(vertices))
            delta = points[p0:p1, None, :]-vertices[None, s0:s1, :]
            t = np.clip(np.sum(delta*vectors[None, s0:s1, :], axis=2)
                        / np.maximum(lengths[None, s0:s1]**2, 1e-12), 0, 1)
            distance = np.sum((delta-t[:, :, None]*vectors[None, s0:s1, :])**2, axis=2)
            nearest = distance.argmin(axis=1)
            value = distance[np.arange(p1-p0), nearest]
            improve = value < best_distance[p0:p1]
            arc = starts[s0+nearest]+t[np.arange(p1-p0), nearest]*lengths[s0+nearest]
            positions[p0:p1] = np.where(improve, arc, positions[p0:p1])
            best_distance[p0:p1] = np.minimum(value, best_distance[p0:p1])
    return positions


def contour_walk(indices, arcs, points, previous):
    """Keep contour order; pick an open-cycle direction and a shorter link."""
    indices = np.asarray(indices, dtype=int)
    cycle = indices[np.lexsort((points[indices, 0], points[indices, 1], arcs))]
    if len(cycle) < 2:
        return cycle.tolist()
    xy = points[cycle]
    edge_lengths = np.linalg.norm(xy-np.roll(xy, -1, axis=0), axis=1)
    perimeter = edge_lengths.sum()
    if previous is None:
        # A repeatable first point, independent of the dish-center position.
        start = int(np.lexsort((xy[:, 0], xy[:, 1]))[0])
        forward = perimeter-edge_lengths[(start-1) % len(cycle)]
        backward = perimeter-edge_lengths[start]
        direction = 1 if forward <= backward else -1
    else:
        link = np.linalg.norm(xy-previous, axis=1)
        costs = np.stack((link+perimeter-np.roll(edge_lengths, 1),
                          link+perimeter-edge_lengths))
        option, start = np.unravel_index(int(costs.argmin()), costs.shape)
        direction = 1 if option == 0 else -1
    return [int(cycle[(start+direction*k) % len(cycle)]) for k in range(len(cycle))]


def outer_layer_order(mask, points, radius, pitch):
    """Number the same centers by external-boundary depth, then contour walks.

    Fill enclosed holes ONLY for ordering. Packing continues to use the original
    mask, so holes never become sampling sites. Concave exterior edges remain.
    """
    if not len(points):
        return [], np.zeros(0, dtype=int), np.zeros(0)
    binary = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    external = np.zeros_like(binary)
    cv2.drawContours(external, contours, -1, 1, cv2.FILLED)
    distance = cv2.distanceTransform(np.pad(external, 1), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]
    ix, iy = np.rint(points[:, 0]).astype(int), np.rint(points[:, 1]).astype(int)
    edge_depth = np.maximum(0, distance[iy, ix]-np.hypot(points[:, 0]-ix, points[:, 1]-iy)-math.sqrt(.5)-radius)
    layers = np.floor(edge_depth/pitch+1e-10).astype(int)+1
    order, previous = [], None
    for layer in np.unique(layers):
        indices = np.flatnonzero(layers == layer)
        # Eroded external outlines follow irregular growth boundaries, rather
        # than sorting polar angles around the dish center.
        core = (distance >= radius+math.sqrt(.5)+(int(layer)-1)*pitch-1e-7).astype(np.uint8)
        _, labels = cv2.connectedComponents(core, connectivity=8)
        paths, _ = cv2.findContours(core, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        by_label = {int(labels[path[0, 0, 1], path[0, 0, 0]]): path for path in paths}
        point_labels = labels[iy[indices], ix[indices]]
        groups = []
        for label in np.unique(point_labels):
            members = indices[point_labels == label]
            path = by_label.get(int(label))
            if label == 0 or path is None:
                raise RuntimeError('外到内轮廓分组校验失败。')
            groups.append((members, contour_positions(points[members], path)))
        while groups:
            if previous is None:
                group = min(range(len(groups)), key=lambda g: min((points[k, 1], points[k, 0]) for k in groups[g][0]))
            else:
                group = min(range(len(groups)), key=lambda g: np.linalg.norm(points[groups[g][0]]-previous, axis=1).min())
            members, arcs = groups.pop(group)
            walk = contour_walk(members, arcs, points, previous)
            order.extend(walk)
            previous = points[walk[-1]]
    if len(order) != len(points) or len(set(order)) != len(points):
        raise RuntimeError('取样顺序存在遗漏或重复。')
    return order, layers, edge_depth


def make_plan(mask, calibration, diameter_mm, gap_mm, default_depth_mm):
    if mask.ndim != 2:
        raise ValueError('标注图必须为单通道。')
    calibration = validate_calibration(calibration, mask.shape[1], mask.shape[0])
    diameter = number(diameter_mm, '取样口径', positive=True)
    gap = number(gap_mm, '圆边缘间隙')
    depth = number(default_depth_mm, '默认取样深度', maximum=1000)
    points, radius = pack(mask > 0, calibration, diameter, gap)
    # Reorder a permutation of the existing packing, without moving any circle.
    indices, layers, edge_depths = outer_layer_order(mask, points, radius,
                                                   (diameter+gap)/calibration['mm_per_pixel'])
    c = calibration
    items = [{'id': i+1, 'image_x_px': float(points[k, 0]), 'image_y_px': float(points[k, 1]),
              'layer': int(layers[k]), 'outer_edge_depth_mm': float(edge_depths[k]*c['mm_per_pixel']),
              'x_mm': (points[k, 0]-c['center_x_px'])*c['mm_per_pixel'],
              'y_mm': (c['center_y_px']-points[k, 1])*c['mm_per_pixel'], 'z_mm': depth,
              'depth_overridden': False} for i, k in enumerate(indices)]
    return {'schema_version': 1, 'plan_id': uuid.uuid4().hex,
            'created_at': datetime.now(timezone.utc).isoformat(),
            'units': 'mm', 'calibration': c,
            'coordinate_frame': {'xy_origin': 'dish_center', 'x_positive': 'image_right',
                                 'y_positive': 'image_up', 'z_zero': 'medium_surface', 'z_positive': 'downward_depth'},
            'diameter_mm': diameter, 'gap_mm': gap, 'default_depth_mm': depth, 'radius_px': radius,
            'count': len(items), 'points': items, 'hardware_ready': False,
            'sampling_order': {'strategy': 'outer_boundary_layers', 'layer_width_mm': diameter+gap,
                               'layer_count': int(layers.max()) if len(layers) else 0,
                               'boundary': 'external_contours_ignoring_internal_holes',
                               'within_layer': 'contour_walk_with_shorter_links',
                               'globally_shortest_route_guaranteed': False},
            'method': 'rotated_hex_search_and_greedy_fill', 'global_optimum_guaranteed': False,
            'constraint_basis': 'predicted_mask_at_original_resolution_and_calibrated_dish_circle',
            'coverage_fraction': len(items)*math.pi*radius**2/max(1, int(np.count_nonzero(mask))),
            'notes': ['圆边界保守避开背景像素；满足约束不代表模型分割一定正确。',
                      '编号沿菌丝实际外轮廓逐层向内；内部孔洞不作为最外层。',
                      '启发式搜索尽量增加圆数量，不保证全局最优。',
                      'XYZ 是样本坐标和深度，尚未转换为设备坐标。']}


def with_depths(plan, default_depth_mm, overrides):
    if not isinstance(overrides, dict):
        raise ValueError('单点深度需要按编号填写。')
    known = {str(p['id']) for p in plan['points']}
    if not set(overrides).issubset(known):
        raise ValueError('存在无效的取样点编号。')
    result = copy.deepcopy(plan)
    result['default_depth_mm'] = number(default_depth_mm, '默认取样深度', maximum=1000)
    for point in result['points']:
        key = str(point['id'])
        point['depth_overridden'] = key in overrides
        point['z_mm'] = number(overrides[key], f'点 {key} 深度', maximum=1000) if key in overrides else result['default_depth_mm']
    result['updated_at'] = datetime.now(timezone.utc).isoformat()
    return result


def csv_bytes(plan):
    output = io.StringIO(newline='')
    writer = csv.writer(output)
    writer.writerow(['point_id', 'x_mm', 'y_mm', 'z_depth_mm', 'diameter_mm', 'edge_gap_mm', 'depth_overridden', 'layer'])
    for p in plan['points']:
        writer.writerow([p['id'], f"{p['x_mm']:.6f}", f"{p['y_mm']:.6f}", f"{p['z_mm']:.6f}", plan['diameter_mm'], plan['gap_mm'], p['depth_overridden'], p.get('layer', '')])
    return output.getvalue().encode('utf-8-sig')


def layout_svg(plan, background):
    c = plan['calibration']
    width, height = c['image_width'], c['image_height']
    thickness = max(1., width/900)
    font_size = max(5., min(plan['radius_px']*.7, width/65))
    image = base64.b64encode(background).decode('ascii')
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             f'<image width="{width}" height="{height}" href="data:image/jpeg;base64,{image}"/>']
    for p in plan['points']:
        x, y = p['image_x_px'], p['image_y_px']
        parts.append(f'<circle cx="{x}" cy="{y}" r="{plan["radius_px"]}" fill="#22c55e" fill-opacity="0.15" stroke="#fde047" stroke-width="{thickness}"/>')
        parts.append(f'<text x="{x}" y="{y}" fill="white" stroke="#16392b" stroke-width="{font_size/12}" paint-order="stroke" text-anchor="middle" dominant-baseline="central" font-family="sans-serif" font-size="{font_size}">{p["id"]}</text>')
    parts.append('</svg>')
    return ''.join(parts).encode('utf-8')
