"""Time alignment for Alfheim camera segments and ZXY reference speeds."""
import csv
from datetime import datetime
from pathlib import Path

import numpy as np

EPOCH = datetime(1970, 1, 1)
MAX_GAP_S = .15


def seconds(timestamp):
    # Video filenames and ZXY CSV both use the venue's local clock.
    return (datetime.fromisoformat(timestamp) - EPOCH).total_seconds()


def frame_times(clip, fps, count):
    size = round(clip['segment_seconds'] * fps)
    if size * len(clip['sources']) != count:
        raise ValueError('Frame count disagrees with source timestamps')
    starts = [seconds(Path(source).stem.split('_', 1)[1]) for source in clip['sources']]
    if len(starts) > 1 and np.max(np.abs(np.diff(starts) - size/fps)) > 1/fps + 1e-5:
        raise ValueError('Sensor fixture contains a timestamp break')
    return np.concatenate([start + np.arange(size)/fps for start in starts])


def read_tags(path, times):
    start, end = min(times)-.2, max(times)+.2
    records = {}
    with path.open(newline='') as data:
        for row in csv.reader(data):
            try:
                when = seconds(row[0])
            except ValueError:
                if row[0].lower() == 'timestamp':
                    continue
                raise
            if start <= when <= end:
                records.setdefault(int(row[1]), []).append([when, float(row[2]), float(row[3]), float(row[7])*3.6])
    return {tag: np.asarray(sorted(values)) for tag, values in records.items()}


def aligned(samples, times):
    times = np.asarray(times, dtype=float)
    truth = np.full((len(times), 3), np.nan)
    if len(samples) < 2:
        return truth
    times_seen = samples[:, 0]
    right = np.clip(np.searchsorted(times_seen, times, side='right'), 1, len(times_seen)-1)
    left = right - 1
    gap = times_seen[right] - times_seen[left]
    keep = ((times >= times_seen[left]) & (times <= times_seen[right])
            & (gap > 0) & (gap <= MAX_GAP_S))
    fraction = ((times[keep] - times_seen[left[keep]]) / gap[keep])[:, None]
    truth[keep] = samples[left[keep], 1:] + fraction * (samples[right[keep], 1:] - samples[left[keep], 1:])
    return truth
