"""Synthetic clips with exact truth: camera, pitch coordinates and player paths.

Observations are generated directly from the true camera, so these tests measure geometry. An
oracle clip is not evidence about any line detector; noisy clips imitate an imperfect one.
"""

from dataclasses import asdict, dataclass, field

import cv2
import numpy as np

from autohl import field_solver as fs, hybrid_calib as hc, pitch
from autohl.camera import warp

META = {'name': 'synthetic', 'length': 100., 'width': 60., 'half_x': 50.,
        'box_half': 18.85, 'box_depth': 16.4, 'ga_depth': 5.4, 'ga_half': 8.75}
FRAMES, KEY_STEP, FPS = 121, 15, 30
FOCAL = 1600.
RUN_SPEED = 7.  # m/s
GROUPS = {'centre_circle': [17], 'box_part': [6, 7, 18], 'touch_halfway': [1, 2, 4],
          'all': list(range(20))}


@dataclass
class Scene:
    name: str
    mode: str = 'oracle'         # oracle | noisy
    height: float = 5.
    side: float = 15.            # metres behind the near touchline
    x: float = 35.               # camera position along the touchline
    pan: float = 45.             # peak-to-peak half range in degrees
    offset: float = 0.           # lens distance from the rotation axis, metres
    noise: float = 1.            # px
    occlusion: float = 0.        # share of paint hidden
    lines: str = 'all'
    drift: float = 0.            # metres the camera moves during the clip
    k1: float = 0.
    missing: float = 0.          # share of lines the detector loses, per key
    wrong: float = 0.            # share of lines given a wrong class, per key
    tripod: bool = True          # whether the camera really turns about a fixed vertical axis
    tags: tuple = field(default_factory=tuple)


def _true_camera(scene, t):
    """Camera parameters at time t in 0..1, as a field_solver parameter vector."""
    target = np.array([50., 0.])
    base = np.array([scene.x, META['width']/2+scene.side])
    yaw0 = np.arctan2(target[0]-base[0], base[1]-target[1])
    yaw = yaw0+np.deg2rad(scene.pan)*np.sin(2*np.pi*t)
    forward = np.array([np.sin(yaw), -np.cos(yaw)])
    centre = base+scene.offset*forward+np.array([scene.drift*(t-.5), 0.])
    tilt = np.arctan2(scene.height, np.linalg.norm(target-base))
    return np.r_[yaw, tilt, np.deg2rad(.3), np.log(FOCAL), centre, scene.height]


def _matrix(p):
    # Unnormalised, so the sign of the third row still says what is in front of the camera.
    focal = np.exp(p[3])
    K = np.array([[focal, 0, fs.CENTRE[0]], [0, focal, fs.CENTRE[1]], [0, 0, 1.]])
    R = fs._rotation(*p[:3])
    return K @ np.c_[R[:, :2], -R @ np.array([p[4], p[5], -p[6]])]


def _perturb(H, sigma, rng):
    if not sigma:
        return H
    corners = np.float32([[0, 0], [1920, 0], [1920, 1080], [0, 1080]])
    moved = corners+rng.normal(0, sigma, corners.shape).astype(np.float32)
    return cv2.getPerspectiveTransform(corners, moved) @ H


def _visible(uv, depth_ok):
    return depth_ok & (uv[:, 0] >= 0) & (uv[:, 0] < 1920) & (uv[:, 1] >= 0) & (uv[:, 1] < 1080)


def _project(H, world, k1):
    q = (H @ np.c_[world, np.ones(len(world))].T).T
    front = q[:, 2] > 1e-6
    uv = q[:, :2]/np.where(front, q[:, 2], 1)[:, None]
    return hc.distort(uv, k1, FOCAL), front


def _observe(scene, H, rng, classes):
    out = []
    for segment, feature in enumerate(pitch.model_segments(META)):
        if segment not in classes:
            continue
        if feature[0] == 'line':
            n = max(2, int(np.linalg.norm(feature[2]-feature[1])/.4))
            world = np.linspace(feature[1], feature[2], n)
        else:
            world = feature[1]
        uv, front = _project(H, world, scene.k1)
        uv = uv[_visible(uv, front)]
        if len(uv) < 5:
            continue
        step = np.r_[0, np.cumsum(np.linalg.norm(np.diff(uv, axis=0), axis=1))]
        uv = uv[np.unique(np.searchsorted(step, np.arange(0, step[-1], 8.)).clip(0, len(uv)-1))]
        if scene.occlusion:
            # Players hide runs of paint, not scattered points.
            keep = np.ones(len(uv), bool)
            while 1-keep.mean() < scene.occlusion:
                a = rng.integers(0, len(uv))
                keep[a:a+rng.integers(3, 12)] = False
            uv = uv[keep]
        if len(uv) < 4:
            continue
        out.append({'segment': segment, 'points': uv+rng.normal(0, scene.noise, uv.shape) if scene.noise else uv,
                    'confidence': .9})
    count = len(pitch.model_segments(META))
    damaged = []
    for line in out:
        if rng.random() < scene.missing:
            continue
        if rng.random() < scene.wrong:
            line = {**line, 'segment': int(rng.integers(0, count)), 'confidence': .5}
        damaged.append(line)
    return damaged


def make_clip(scene, seed):
    rng = np.random.default_rng(seed)
    cameras = [_true_camera(scene, i/(FRAMES-1)) for i in range(FRAMES)]
    H = [_matrix(p) for p in cameras]
    # Camera motion: exact plane-to-plane motion plus the drift a feature tracker accumulates.
    cumulative = [np.eye(3)]
    for i in range(1, FRAMES):
        step = _perturb(H[i] @ np.linalg.inv(H[i-1]), .03, rng)
        cumulative.append(step @ cumulative[-1])
    cumulative = np.stack(cumulative)
    keys = list(range(0, FRAMES, KEY_STEP))
    classes = set(GROUPS[scene.lines])
    observations = {k: _observe(scene, H[k], rng, classes) for k in keys}
    return {'scene': scene, 'seed': seed, 'cameras': cameras, 'H': H, 'cumulative': cumulative,
            'observations': observations, 'keys': keys, 'players': _players(scene, H, rng), 'meta': META}


def _players(scene, H, rng):
    # One runner and one standing player, placed where the camera sees them mid-clip.
    middle = FRAMES//2
    y, x = np.mgrid[250:1000:30, 120:1800:40]
    ground = warp(np.c_[x.ravel(), y.ravel()].astype(float), np.linalg.inv(H[middle]))
    inside = ground[(np.abs(ground[:, 0]-50) < 45) & (np.abs(ground[:, 1]) < 27)]
    if not len(inside):
        return None
    start = inside[rng.integers(0, len(inside))]
    runner = np.array([start+[RUN_SPEED*(i-middle)/FPS, 0.] for i in range(FRAMES)])
    runner = np.c_[runner[:, 0].clip(-5, 105), runner[:, 1]]
    still = np.tile(inside[rng.integers(0, len(inside))], (FRAMES, 1))
    seen = {}
    for name, path in (('runner', runner), ('still', still)):
        pixels = np.full((FRAMES, 2), np.nan)
        for i in range(FRAMES):
            uv, front = _project(H[i], path[i:i+1], scene.k1)
            if _visible(uv, front)[0]:
                pixels[i] = uv[0]+rng.normal(0, 1.5, 2)
        seen[name] = {'world': path, 'pixels': pixels}
    return seen


def holdout(clip, frame):
    """Pitch points the camera sees at this frame: never used by any method."""
    y, x = np.mgrid[-30:30.1:4, 0:100.1:4]
    world = np.c_[x.ravel(), y.ravel()].astype(float)
    uv, front = _project(clip['H'][frame], world, clip['scene'].k1)
    keep = _visible(uv, front)
    return world[keep], uv[keep]


def render_key(clip, k):
    """Observed paint drawn as an image, for the methods that read pixels."""
    image = np.full((1080, 1920, 3), (40, 110, 50), np.uint8)
    for line in clip['observations'].get(k, ()):
        p = np.int32(line['points'])
        gaps = np.linalg.norm(np.diff(p, axis=0), axis=1) > 40
        for part in np.split(p, np.flatnonzero(gaps)+1):
            if len(part) > 1:
                cv2.polylines(image, [part], False, (235, 240, 235), 3, cv2.LINE_AA)
    return image


def describe(scene):
    return asdict(scene)
