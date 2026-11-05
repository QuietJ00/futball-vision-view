"""Which speed estimator is closest to a known run, given the noise a real clip has?

Each trial replays one known speed profile through the real clip's measurement geometry:
its per-frame ground noise, its occluded frames, plus the foot bob of a running stride.

    python -m experiments.instant_speed.benchmark /path/to/geometry.npz
"""
import sys

import numpy as np

from autohl import speed

FPS = 30


def profile(rng, count):
    """Speed in m/s: a jog, a burst at a human acceleration, a hold, a slowdown."""
    start = rng.integers(40, 150)
    jog, top = rng.uniform(3, 5), rng.uniform(7, 9.2)
    accel, brake = rng.uniform(2.5, 5), rng.uniform(2.5, 5)
    rise, hold = int((top-jog)/accel*FPS), rng.integers(15, 60)
    fall = int((top-jog)/brake*FPS)
    knots = [0, start, start+rise, start+rise+hold, start+rise+hold+fall, count-1]
    return np.interp(np.arange(count), knots, [jog, jog, top, top, jog, jog])


def run(rng, frames, covariance, seen):
    count = len(frames)
    v = profile(rng, count)
    heading = rng.uniform(-.5, .5)+np.pi*rng.integers(0, 2)
    unit = np.array([np.cos(heading), np.sin(heading)])
    along = np.cumsum(v)/FPS
    # Strides: the visible foot point moves ahead of and behind the body at the step rate.
    step = np.cumsum(1.7+.12*v)/FPS*2*np.pi
    bob = rng.uniform(.1, .3)*np.sin(step+rng.uniform(0, 6.28))
    truth = np.outer(along+bob, unit)
    xy = np.full((count, 2), np.nan)
    for i in np.flatnonzero(seen):
        xy[i] = truth[i]+np.linalg.cholesky(covariance[i]+1e-9*np.eye(2)) @ rng.normal(0, 1, 2)
    return xy, v*3.6


def window(xy, seen, half):
    out = np.full(len(xy), np.nan)
    for k in range(half, len(xy)-half):
        try:
            out[k] = speed.measure_window(xy, seen, k-half, k+half)['speed_kmh']
        except ValueError:
            pass
    return out


def difference(xy):
    return np.linalg.norm(np.gradient(xy, axis=0), axis=1)*FPS*3.6


def evaluate(geometry, trials, seed):
    frames = geometry['frames']
    covariance, seen = geometry['covariance'], geometry['seen']
    R = covariance*(3/3)**2
    estimators = {'frame difference': lambda x, s, c: difference(x),
                  'window 1 s': lambda x, s, c: window(x, s, 15)}
    for q in (.5, 1., 2.):
        estimators[f'velocity model q={q:g}'] = lambda x, s, c, q=q: speed.instant_speed(x, s, c, order=2, q=q)['kmh']
    estimators['acceleration model q=5'] = lambda x, s, c: speed.instant_speed(x, s, c, order=3, q=5.)['kmh']
    results = {}
    for name, estimate in estimators.items():
        rng = np.random.default_rng(seed)
        rmse, peak, late, held = [], [], [], []
        for _ in range(trials):
            xy, truth = run(rng, frames, R, seen)
            found = estimate(xy, seen, R)
            inner = np.arange(20, len(truth)-20)
            rmse.append(np.sqrt(np.nanmean((found[inner]-truth[inner])**2)))
            peak.append(np.nanmax(found[inner])-truth[inner].max())
            top, _ = speed.sustained_top(np.where(np.isin(np.arange(len(found)), inner), found, np.nan))
            held.append(top-truth[inner].max())
            ramp = inner[np.abs(np.gradient(truth)[inner]) > .05]
            late.append(np.sqrt(np.nanmean((found[ramp]-truth[ramp])**2)) if len(ramp) else np.nan)
        results[name] = rmse, peak, late, held
    return results


def main():
    z = np.load(sys.argv[1])
    geometry = {'frames': z['frames'], 'covariance': z['covariance'], 'seen': z['seen']}
    print(f"{'method':26s} {'rmse':>6s} {'while changing':>15s} {'peak bias':>10s} {'held top bias':>14s} {'spread':>7s}")
    for name, (rmse, peak, late, held) in evaluate(geometry, int(sys.argv[2]) if len(sys.argv) > 2 else 60, 7).items():
        print(f'{name:26s} {np.mean(rmse):6.2f} {np.nanmean(late):15.2f} {np.mean(peak):+10.2f} {np.nanmean(held):+14.2f} {np.nanstd(held):7.2f}', flush=True)


if __name__ == '__main__':
    main()
