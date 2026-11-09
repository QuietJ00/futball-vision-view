"""Register a surveyed pitch to a rotating camera on a fixed pivot."""
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp

from . import camera, camfit, field_solver, hybrid_calib, pitch

CENTER_CIRCLE = 17
GROUND_LINK_FRAMES = 26


def _map(H, points):
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    p = points @ H[:, :2].T + H[:, 2]
    xy = np.full((len(points), 2), np.nan)
    visible = p[:, 2] > 1e-9
    xy[visible] = p[visible, :2] / p[visible, 2, None]
    return xy


@dataclass
class Calibration:
    meta: dict
    reference: int
    pivot: np.ndarray
    lens_offset_m: float
    focal_px: float
    matrices: np.ndarray
    same_segment: np.ndarray

    def supports(self, a, b=None):
        end = a if b is None else b
        return bool(self.same_segment[a:end+1].all())

    def to_img(self, frame, xy):
        return _map(self.matrices[frame], xy)

    def to_pitch(self, frame, pixels):
        return _map(np.linalg.inv(self.matrices[frame]), pixels)


def _motions(cumulative, reference, focal):
    K = np.array([[focal, 0, camfit.CX], [0, focal, camfit.CY], [0, 0, 1.]])
    Ki = np.linalg.inv(K)
    relative = Ki @ cumulative @ np.linalg.inv(cumulative[reference]) @ K
    # A homography has an arbitrary sign and scale. Recover the closest proper rotation.
    relative *= np.sign(np.linalg.det(relative))[:, None, None]
    u, _, vt = np.linalg.svd(relative)
    u[:, :, 2] *= np.linalg.det(u @ vt)[:, None]
    return K, u @ vt


def _matrices(params, cumulative, reference, indices, corrections, lens_offset_m):
    focal = np.exp(params[3])
    K, relative = _motions(cumulative, reference, focal)
    rotation = corrections @ relative[indices] @ field_solver._rotation(*params[:3])
    pivot = params[4:7] * [1, 1, -1]
    return _homography(K, rotation, pivot, lens_offset_m)


def _homography(K, rotation, pivot, lens_offset_m):
    # In world space C_i = pivot + R_i.T @ offset_camera.
    translation = -(rotation @ pivot) - np.array([0., 0., lens_offset_m])
    return K @ np.concatenate((rotation[..., :2], translation[..., None]), axis=-1)


def _observations(marks, meta, frame_count):
    features = pitch.model_segments(meta)
    observations = []
    for frame, strokes in sorted(marks.items()):
        if not 0 <= frame < frame_count:
            raise ValueError(f'Frame {frame} lies outside the video.')
        for index, pixels in sorted(strokes.items()):
            if not 0 <= index < len(features):
                raise ValueError(f'Unknown pitch line {index}.')
            points = np.asarray(pixels, float)
            if (points.ndim != 2 or points.shape[1] != 2 or len(points) < 2
                    or not np.isfinite(points).all()
                    or (points < 0).any() or (points >= [camfit.W, camfit.H]).any()):
                raise ValueError(f'Frame {frame}, line {index} needs at least two image points in bounds.')
            kind, *geometry = features[index]
            if kind == 'line':
                world = np.cross(np.r_[geometry[0], 1], np.r_[geometry[1], 1])
            else:
                world = geometry[0]
            observations.append((frame, index, points, kind, world))
    return observations


def _residual(H, points, kind, world):
    if kind == 'line':
        line = np.linalg.solve(H.T, world)
        return (points @ line[:2] + line[2]) / np.linalg.norm(line[:2])
    projected = _map(H, world)
    a, delta = projected[:-1], np.diff(projected, axis=0)
    visible = np.isfinite(delta).all(axis=1)
    if not visible.any():
        return np.full(len(points), 2000.)
    a, delta = a[visible], delta[visible]
    vectors = points[:, None] - a
    t = np.clip(np.sum(vectors * delta, axis=2) / np.maximum(np.sum(delta**2, axis=1), 1e-9), 0, 1)
    return np.min(np.linalg.norm(vectors - t[..., None]*delta, axis=2), axis=1)


def _center_ground_links(images, boxes, cumulative, frames, output, same_segment, meta, marks):
    """Refine camera links locally on stationary turf around observed centre paint."""
    half = meta.get('half_x', meta['length']/2)
    radius = 9.15
    patch = np.array([[half-1.8*radius, -1.6*radius], [half+1.8*radius, -1.6*radius],
                      [half+1.8*radius, 1.6*radius], [half-1.8*radius, 1.6*radius]])
    height, width = images.shape[1:3]
    links = {}
    for i, reference in enumerate(frames):
        if CENTER_CIRCLE not in marks[reference]:
            continue
        pts_ref = np.asarray(marks[reference][CENTER_CIRCLE], float)
        world_focus = _map(np.linalg.inv(output[reference]), [pts_ref.mean(axis=0)])[0]
        if not np.isfinite(world_focus).all():
            world_focus = np.array([half, 0.])
        focus = np.array([_map(H, [world_focus])[0] for H in output])
        outline = _map(output[reference], patch)
        if not np.isfinite(outline).all():
            continue
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.rint(outline).astype(np.int32), 255)
        start = max(0, reference-GROUND_LINK_FRAMES, frames[i-1]+1 if i else 0)
        end = min(len(output), reference+GROUND_LINK_FRAMES+1,
                  frames[i+1] if i+1 < len(frames) else len(output))
        nearby = [k for k in range(start, end) if k != reference and same_segment[k]
                  and ((focus[k] >= 0) & (focus[k] < [width, height])).all()]
        result = camera.refine_ground_motion(images, cumulative, boxes, reference, nearby,
                                             focus, ground_mask=mask)
        links.update({(i, k): item['reference_to_image'] for k, item in result.items()
                      if item['accepted']})
    return links


def _propagate(params, cumulative, frames, key_rotations, seed_rotations, K, meta, offset,
               same_segment, marks, images=None, boxes=None, edges=None):
    """Follow local image motion while keeping every pose on the same pivot."""
    x, y = np.meshgrid(np.linspace(0, meta['length'], 25),
                       np.linspace(-meta['width']/2, meta['width']/2, 13))
    grid = np.c_[x.ravel(), y.ravel()]
    pivot = params[4:7] * [1, 1, -1]
    key_matrices = _homography(K, key_rotations, pivot, offset)
    planes = [_map(H, grid) for H in key_matrices]
    inverse = np.linalg.inv(cumulative[frames])
    output = _homography(K, seed_rotations, pivot, offset)
    output[frames] = key_matrices
    ground_links = (_center_ground_links(images, boxes, cumulative, frames, output, same_segment, meta, marks)
                    if images is not None else {})
    edges = edges or {}
    middle = np.linalg.norm(grid-[meta.get('half_x', meta['length']/2), 0.], axis=1) <= 16.
    v_prev = np.zeros(3)
    has_prev = False
    focal_val = float(K[0, 0])
    for k in np.flatnonzero(same_segment):
        j = np.searchsorted(frames, k)
        if j < len(frames) and frames[j] == k:
            neighbours = [(j, 1.)]
        elif 0 < j < len(frames):
            t = (k-frames[j-1])/(frames[j]-frames[j-1])
            neighbours = [(j-1, 1-t), (j, t)]
        else:
            neighbours = [(min(j, len(frames)-1), 1.)]
        observations, edge_obs = [], []
        for i, time_weight in neighbours:
            source = planes[i]
            visible = ((source >= 0) & (source < [camfit.W, camfit.H])).all(axis=1)
            M_ik = cumulative[k] @ inverse[i]
            pixels = camera.warp(source[visible], M_ik)
            pt_w = np.ones(visible.sum())
            link = ground_links.get((i, k))
            if link is not None:
                use = middle[visible]
                pixels[use] = camera.warp(source[visible][use], link)
                pt_w[use] = 3.5
            inside = ((pixels >= 0) & (pixels < [camfit.W, camfit.H])).all(axis=1)
            if inside.sum() >= 8:
                observations.append((grid[visible][inside], pixels[inside],
                                     time_weight * pt_w[inside] / pt_w[inside].sum()))
            ref_frame = int(frames[i])
            if ref_frame in edges:
                ew = camera.warp(edges[ref_frame], M_ik)
                ein = ((ew >= 0) & (ew < [camfit.W, camfit.H])).all(axis=1)
                if ein.sum() >= 3:
                    edge_obs.append((ew[ein], time_weight))
        if not observations:
            has_prev = False
            continue
        world = np.concatenate([item[0] for item in observations])
        target = np.concatenate([item[1] for item in observations])
        weight = np.concatenate([np.sqrt(w) for _, _, w in observations])

        def matrix(v):
            rotation = Rotation.from_rotvec(v).as_matrix() @ seed_rotations[k]
            return _homography(K, rotation, pivot, offset)

        def loss(v):
            H_k = matrix(v)
            projected = _map(H_k, world)
            res = (np.nan_to_num(projected-target, nan=2000.) * weight[:, None]).ravel()
            priors = [0.03 * focal_val * v]
            if has_prev:
                priors.append(0.04 * focal_val * (v - v_prev))
            if not edge_obs:
                return np.concatenate([res, *priors])
            er = [2.2 * np.sqrt(w / len(ep)) * np.maximum(_edge_errors(H_k, meta['width'], ep) - 1.5, 0.)
                  for ep, w in edge_obs]
            return np.concatenate([res, *er, *priors])

        x0 = v_prev if has_prev else np.zeros(3)
        fitted = least_squares(loss, x0, bounds=(-.25, .25),
                               loss='soft_l1', f_scale=2., max_nfev=35)
        if fitted.success or np.any(fitted.x != 0):
            output[k] = matrix(fitted.x)
            v_prev = fitted.x
            has_prev = True
        else:
            has_prev = False
    return output


def _edge_errors(H, width, points):
    """Positive where the projected far touchline lies above the playing surface."""
    line = np.linalg.solve(H.T, [0., 1., width/2])
    if abs(line[1]) < 1e-9*np.linalg.norm(line[:2]):
        return np.full(len(points), 2000.)
    return points[:, 1] + (line[0]*points[:, 0] + line[2])/line[1]


def _edge_points(samples, frame_count):
    edges = {}
    for k, points in samples.items():
        points = np.asarray(points, float)
        if (not 0 <= k < frame_count or points.ndim != 2 or points.shape[1] != 2
                or len(points) < 3 or not np.isfinite(points).all()
                or (points < 0).any() or (points >= [camfit.W, camfit.H]).any()
                or np.ptp(points[:, 0]) < 50):
            raise ValueError(f'Frame {k} needs three surface-edge points spread across the image.')
        edges[k] = points
    return edges


def fit(cumulative, inliers, meta, marks, lens_offset_m=0., far_edges=None, *, images=None, boxes=None,
        seed_homographies=None):
    """Fit the fixed pivot and focal length, with a measured forward optical-centre offset."""
    if not np.isfinite(lens_offset_m) or abs(lens_offset_m) > 1:
        raise ValueError('Lens offset must be between -1 and 1 metre.')
    if (images is None) != (boxes is None):
        raise ValueError('Ground motion refinement needs both images and player detections.')
    cumulative = camera.repair_cumulative(cumulative)
    frames = sorted(marks)
    if len(frames) < 2:
        raise ValueError('Mark named pitch lines on at least two frames.')
    observations = _observations(marks, meta, len(cumulative))
    edges = _edge_points(far_edges or {}, len(cumulative))
    if edges.keys() - marks.keys():
        raise ValueError('A training surface edge must share a frame with training paint.')
    segments = camera.motion_segments(inliers, len(cumulative))
    if len({segments[k] for k in frames}) != 1:
        raise ValueError('Training marks cross a camera tracking break. Process each continuous shot separately.')

    # Each frame with four identifiable straight lines gives a possible starting camera.
    features = hybrid_calib._features(meta)
    candidates = []
    for frame in frames:
        H = (seed_homographies or {}).get(frame)
        if H is None:
            lines = [{'segment': index, 'points': points} for index, points in marks[frame].items()]
            H = hybrid_calib.homography_from_lines(lines, meta, features)
        if H is None:
            continue
        focal, rotation, centre = camfit.decompose(H)
        R = camfit.rodrigues(rotation)
        pivot = centre - lens_offset_m*R[2]
        params = field_solver._parameters(R, pivot, focal)
        if np.isfinite(params).all() and 1 < params[6] < 40 and 250 < focal < 6000:
            candidates.append((frame, params))
    if not candidates:
        raise ValueError('Mark at least four distinct straight lines on one clear reference frame.')

    # Keep each annotated stroke comparable even when one was clicked more densely.
    rows = {frame: i for i, frame in enumerate(frames)}
    frame_indices = np.asarray(frames)
    lower, upper = (
        [-4*np.pi, -1.55, -np.pi, np.log(250), -2*meta['length'], -3*meta['width'], .5],
        [4*np.pi, 1.55, np.pi, np.log(6000), 3*meta['length'], 3*meta['width'], 80],
    )

    # Calculate world X-span of observed lines to anchor pivot/focal when geometry is degenerate
    obs_indices = {item[1] for item in observations}
    all_x = []
    for idx in obs_indices:
        feat = features[idx]
        if feat[0] == 'line':
            all_x.extend([feat[1][0], feat[2][0]])
        else:
            all_x.extend(feat[1][:, 0])
    x_span = float(max(all_x) - min(all_x)) if all_x else 0.0

    w_pivot = 3.5 if x_span < meta['length'] * 0.55 else 1.5
    w_focal = 18.0 if x_span < meta['length'] * 0.55 else 8.0

    def corrections(params, reference):
        # The reference rotation is already represented by the first three parameters.
        vectors = np.insert(params[7:].reshape(-1, 3), rows[reference], 0., axis=0)
        return Rotation.from_rotvec(vectors)

    def loss(params, reference, p_seed):
        H = _matrices(params, cumulative, reference, frame_indices,
                      corrections(params, reference).as_matrix(), lens_offset_m)
        paint = [_residual(H[rows[k]], points, kind, geometry) * np.sqrt(6/len(points))
                 for k, _, points, kind, geometry in observations]
        # Crossing the surface boundary is stronger evidence than a small paint-centre error.
        edge_loss = [2.0 * np.maximum(_edge_errors(H[rows[k]], meta['width'], points) - 1., 0.)
                     * np.sqrt(6/len(points)) for k, points in edges.items()]
        prior_pivot = w_pivot * (params[4:7] - p_seed[4:7])
        prior_focal = w_focal * (params[3] - p_seed[3])
        # The feature chain guides the correction while labeled paint can move the pose.
        return np.concatenate([*paint, *edge_loss, prior_pivot, [prior_focal], .05*np.exp(params[3])*params[7:]])

    fitted = []
    extra = 3*(len(frames)-1)
    bounds = np.r_[lower, np.full(extra, -.2)], np.r_[upper, np.full(extra, .2)]
    for frame, params in candidates:
        p_seed = params.copy()
        params = np.clip(np.r_[params, np.zeros(extra)], bounds[0]+1e-6, bounds[1]-1e-6)
        result = least_squares(loss, params, args=(frame, p_seed), bounds=bounds, loss='soft_l1',
                               f_scale=3., x_scale='jac', max_nfev=150)
        if result.success and np.linalg.matrix_rank(result.jac) == len(params):
            fitted.append((result.cost, frame, result.x))
    if not fitted:
        raise ValueError('The marked lines did not constrain a fixed camera. Check their labels and spread.')
    _, reference, params = min(fitted, key=lambda item: item[0])
    indices = np.arange(len(cumulative))
    interp = Slerp(frame_indices, corrections(params, reference))
    smooth = interp(np.clip(indices, frame_indices[0], frame_indices[-1])).as_matrix()
    focal = float(np.exp(params[3]))
    K, relative = _motions(cumulative, reference, focal)
    seed_rotations = smooth @ relative @ field_solver._rotation(*params[:3])
    same_segment = segments == segments[reference]
    matrices = _propagate(params, cumulative, frame_indices, seed_rotations[frame_indices],
                          seed_rotations, K, meta, lens_offset_m, same_segment, marks, images, boxes, edges)
    return Calibration(meta, reference, params[4:7], float(lens_offset_m), focal,
                       matrices, same_segment)


def errors(calibration, marks):
    """Residuals against labeled paint, in source-image pixels, reported by frame and line."""
    out = {}
    for frame, index, points, kind, world in _observations(marks, calibration.meta, len(calibration.matrices)):
        distances = np.abs(_residual(calibration.matrices[frame], points, kind, world))
        out.setdefault(frame, {})[index] = {
            'count': len(points), 'median_px': float(np.median(distances)),
            'p90_px': float(np.percentile(distances, 90)),
        }
    return out


def far_edge_errors(calibration, samples):
    """Compare the predicted far line to a manually identified grass/track boundary."""
    out = {}
    for k, points in _edge_points(samples, len(calibration.matrices)).items():
        if calibration.same_segment[k]:
            excess = np.maximum(_edge_errors(calibration.matrices[k], calibration.meta['width'], points), 0.)
            out[k] = float(np.percentile(excess, 90))
    return out


def player_on_pitch(matrices, boxes, meta, margin=4.0):
    """Fraction of detected player feet that project inside pitch bounds."""
    if boxes is None or len(boxes) == 0:
        return 1.0
    total, on_pitch = 0, 0
    L, W = meta['length'], meta['width']
    for k in range(min(len(matrices), len(boxes))):
        dets = np.asarray(boxes[k])
        if len(dets) == 0:
            continue
        h = dets[:, 3] - dets[:, 1]
        valid = (h > 25) & (dets[:, 2] > dets[:, 0])
        if not valid.any():
            continue
        dets = dets[valid]
        feet = np.c_[(dets[:, 0] + dets[:, 2]) / 2, dets[:, 3]]
        invH = np.linalg.inv(matrices[k])
        world = camera.warp(feet, invH)
        inside = (world[:, 0] >= -margin) & (world[:, 0] <= L + margin) & (np.abs(world[:, 1]) <= W/2 + margin)
        total += len(world)
        on_pitch += inside.sum()
    return float(on_pitch / max(total, 1))


def supported_frames(calibration, holdout_errors, threshold_px, max_gap_frames, edge_errors=None, boxes=None):
    """Accept only intervals near an independent multi-line paint check."""
    meta = getattr(calibration, 'meta', None)
    if meta is not None:
        features = hybrid_calib._features(meta)
        observed_lines = {line for lines in holdout_errors.values() for line in lines}
        all_x = []
        has_far_edge = bool(edge_errors and len(edge_errors) > 0)
        for idx in observed_lines:
            if 0 <= idx < len(features):
                feat = features[idx]
                if feat[0] == 'line':
                    all_x.extend([feat[1][0], feat[2][0]])
                else:
                    all_x.extend(feat[1][:, 0])
        x_span = float(max(all_x) - min(all_x)) if all_x else 0.0
        if x_span < meta['length'] * 0.55 and not has_far_edge:
            return np.zeros(len(calibration.matrices), dtype=bool)

    frames = sorted(k for k, lines in holdout_errors.items()
                    if len(lines) >= 2 and calibration.same_segment[k])
    valid = np.zeros(len(calibration.matrices), dtype=bool)
    if frames:
        valid_frames = [f for f in frames if all(line['p90_px'] <= threshold_px
                                                 for line in holdout_errors[f].values())]
        total_checking = len([k for k in range(6, len(calibration.matrices)-1, 12) if calibration.same_segment[k]]) + 2
        if len(frames) >= 4 and len(valid_frames) / max(total_checking, 1) < 0.80:
            return valid
        for k in np.flatnonzero(calibration.same_segment):
            nearest = min(frames, key=lambda j: (abs(j-k), j))
            valid[k] = (abs(nearest-k) <= max_gap_frames
                        and all(line['p90_px'] <= threshold_px
                                for line in holdout_errors[nearest].values()))
            if valid[k] and edge_errors:
                edge = min(edge_errors, key=lambda j: (abs(j-k), j))
                if abs(edge-k) <= max_gap_frames and edge_errors[edge] > 5.:
                    valid[k] = False
    if valid.any() and boxes is not None:
        if player_on_pitch(calibration.matrices, boxes, calibration.meta) < 0.85:
            valid[:] = False
    return valid
