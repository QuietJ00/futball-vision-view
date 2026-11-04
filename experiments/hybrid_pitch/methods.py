"""Algorithms A to E behind one interface: run(clip) returns a calibration or None."""

import numpy as np

from autohl import camfit, hybrid_calib as hc, pitch
from . import synthetic_data as sd


class Frames:
    """camfit reads images by index; only its key frames need to exist."""

    def __init__(self, clip):
        self.clip, self.cache = clip, {}

    def __len__(self):
        return sd.FRAMES

    def __getitem__(self, k):
        if k not in self.cache:
            self.cache[k] = sd.render_key(self.clip, k)
        return self.cache[k]


def _on_visible_paint(world, scene):
    # A corner can only be clicked if its paint is showing in this scenario.
    near = np.zeros(len(world), bool)
    for segment in sd.GROUPS[scene.lines]:
        feature = pitch.model_segments(sd.META)[segment]
        line = np.linspace(feature[1], feature[2], 200) if feature[0] == 'line' else feature[1]
        near |= np.linalg.norm(world[:, None]-line[None], axis=2).min(1) < 1.
    return near


def _landmarks(clip, k, rng, click_px=2.):
    # What a person clicks: named painted corners in view. Their accuracy is set by the screen,
    # not by the line detector, so this path takes no semantic input.
    scene = clip['scene']
    named = pitch.landmarks(sd.META['length'], sd.META['width'], sd.META)
    names = list(named)
    world = np.array([named[n] for n in names], float)
    uv, front = sd._project(clip['H'][k], world, scene.k1)
    keep = sd._visible(uv, front) & _on_visible_paint(world, scene)
    keep &= rng.random(len(names)) >= scene.occlusion
    return [(names[i], tuple(uv[i]+rng.normal(0, click_px, 2))) for i in np.flatnonzero(keep)]


def run_a(clip):
    """Production path for an unknown venue: clicked landmarks at one frame, then the chain."""
    rng = np.random.default_rng(clip['seed']+11)
    views = {k: _landmarks(clip, k, rng) for k in clip['keys']}
    k = max(views, key=lambda v: len(views[v]))
    clicks = views[k]
    if len(clicks) < 4:
        return None
    try:
        H = pitch.from_points(k, clicks, sd.META)
    except Exception:  # OpenCV raises on degenerate point sets; that is a failed calibration
        return None
    return hc.Calibration([k], {k: H}, clip['cumulative'], sd.META, cameras=None)


class _Start:
    # camfit starts from any object with img_from_pitch and meta.
    def __init__(self, calibration):
        self.c, self.meta = calibration, sd.META

    def img_from_pitch(self, k):
        return self.c.img_from_pitch(k)


class _CamFit:
    def __init__(self, fitted):
        self.fitted, self.meta = fitted, sd.META

    def supports(self, a, b=None):
        return True

    def to_pitch(self, k, p):
        return self.fitted.to_pitch(k, p)

    def to_img(self, k, p):
        return self.fitted.to_img(k, p)

    def img_from_pitch(self, k):
        return self.fitted.img_from_pitch(k)


def run_b(clip):
    """Existing shared camera (camfit.fit) started from A, reading rendered paint."""
    start = run_a(clip)
    if start is None:
        return None
    try:
        fitted = camfit.fit(Frames(clip), clip['cumulative'], _Start(start), 0, step=sd.KEY_STEP,
                            log=lambda *_: None, free_dims=False)[0]
    except Exception:
        return None
    return _CamFit(fitted)


def run_c(clip):
    return hc.fit_homographies(clip['observations'], clip['cumulative'], sd.META)


def run_d0(clip):
    """Shared camera, no ties through the camera chain: isolates what temporal ties add."""
    return hc.fit_shared(clip['observations'], clip['cumulative'], sd.META, temporal=False)


def run_d(clip):
    return hc.fit_shared(clip['observations'], clip['cumulative'], sd.META)


def run_e(clip):
    return hc.fit_tripod(clip['observations'], clip['cumulative'], sd.META)


def run_auto(clip):
    return hc.calibrate(clip['observations'], clip['cumulative'], sd.META)


def run_truth(clip):
    """The true camera: what is left is tracker noise, for separating it from calibration."""
    H = {k: clip['H'][k]/clip['H'][k][2, 2] for k in clip['keys']}
    return hc.Calibration(clip['keys'], H, clip['cumulative'], sd.META, k1=clip['scene'].k1, focal=sd.FOCAL)


METHODS = {'TRUTH': run_truth, 'A': run_a, 'B': run_b, 'C': run_c, 'D0': run_d0, 'D': run_d, 'E': run_e, 'AUTO': run_auto}
