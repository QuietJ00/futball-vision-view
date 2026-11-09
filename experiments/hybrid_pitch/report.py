"""Turn comparison.csv and real.json into summary.json, figures and tables for report.md."""

import csv
import json
import os
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

from autohl import field_solver as fs, pitch
from . import methods, synthetic_data as sd

RESULTS = Path(__file__).parent/'results'
ORDER = ['A', 'B', 'C', 'D0', 'D', 'E', 'AUTO']
COLOURS = dict(zip(ORDER, ['#7f7f7f', '#b07aa1', '#4e79a7', '#76b7b2', '#59a14f', '#edc948', '#e15759']))
GROUPS = {'grid': '오라클 카메라 격자', 'noisy': '노이즈 라인', 'sparse': '선이 적음', 'stress': '가림+큰 회전',
          'moving': '카메라 이동', 'tripod': '삼각대 오프셋', 'distortion': '렌즈 왜곡', 'classes': '클래스 오류',
          'occlusion': '라인 가림', 'noise': '관측 노이즈'}
METRICS = ['hold_median_m', 'hold_p95_m', 'hold_rmse_m', 'near_median_m', 'far_median_m', 'reproject_px', 'camera_position_m',
           'camera_rotation_deg', 'distance_error_m', 'mean_speed_error_kmh', 'top_speed_error_kmh',
           'still_fake_speed_kmh', 'seconds']


def load():
    rows = list(csv.DictReader(open(RESULTS/'comparison.csv')))
    for r in rows:
        for m in METRICS:
            r[m] = float(r[m]) if r.get(m) not in (None, '') else np.nan
    return rows


def med(values):
    v = np.array(values, float)
    v = v[np.isfinite(v)]
    return float(np.median(v)) if len(v) else None


def table(rows):
    out = {}
    for m in ['TRUTH']+ORDER:
        group = [r for r in rows if r['method'] == m]
        if not group:
            continue
        good = [r for r in group if r['status'] == 'ok']
        out[m] = {'runs': len(group), 'success': len(good)/len(group), **{k: med([r[k] for r in good]) for k in METRICS}}
    return out


def paired(rows, base, other, key='hold_median_m', seed=7):
    """Median ratio other/base over clips both solved, with a bootstrap interval."""
    b = {(r['scene'], r['seed']): r[key] for r in rows if r['method'] == base and r['status'] == 'ok'}
    o = {(r['scene'], r['seed']): r[key] for r in rows if r['method'] == other and r['status'] == 'ok'}
    both = [k for k in b if k in o and np.isfinite(b[k]) and np.isfinite(o[k]) and b[k] > 1e-4]
    if len(both) < 5:
        return None
    ratio = np.array([o[k]/b[k] for k in both])
    rng = np.random.default_rng(seed)
    sample = [np.median(rng.choice(ratio, len(ratio))) for _ in range(2000)]
    return {'clips': len(both), 'median_ratio': float(np.median(ratio)), 'ci95': [float(np.percentile(sample, 2.5)), float(np.percentile(sample, 97.5))]}


def bars(rows, key, title, path, log=True):
    tags = [t for t in GROUPS if any(r['tags'] == t for r in rows)]
    fig, ax = plt.subplots(figsize=(13, 4.8))
    width = .12
    for i, m in enumerate(ORDER):
        values = []
        for t in tags:
            v = med([r[key] for r in rows if r['tags'] == t and r['method'] == m and r['status'] == 'ok'])
            values.append(np.nan if v is None else min(v, 1e3))
        ax.bar(np.arange(len(tags))+i*width, values, width, label=m, color=COLOURS[m])
    ax.set_xticks(np.arange(len(tags))+3*width, [t for t in tags], rotation=20)
    if log:
        ax.set_yscale('log')
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:g}'))
    ax.set_title(title)
    ax.legend(ncol=7, fontsize=9)
    ax.grid(axis='y', alpha=.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def heatmap(path):
    scene = sd.Scene('heatmap', mode='noisy', height=5., side=15., noise=3., missing=.2, wrong=.1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
    for ax, name in zip(axes, ('A', 'AUTO')):
        cells = np.zeros((16, 26))
        counts = np.zeros((16, 26))
        for seed in range(42, 47):
            clip = sd.make_clip(scene, seed)
            c = methods.METHODS[name](clip)
            if c is None:
                continue
            for k in clip['keys']:
                world, uv = sd.holdout(clip, k)
                if len(world):
                    e = np.linalg.norm(c.to_pitch(k, uv)-world, axis=1)
                    i, j = ((world[:, 1]+32)//4).astype(int), ((world[:, 0]+2)//4).astype(int)
                    np.add.at(cells, (i, j), e)
                    np.add.at(counts, (i, j), 1)
        shown = ax.imshow(np.where(counts > 0, cells/np.maximum(counts, 1), np.nan), origin='lower', extent=[-2, 102, -32, 32],
                          cmap='magma', norm=matplotlib.colors.LogNorm(.02, 10), aspect='auto')
        ax.set_title(f'{name}: 평균 위치 오차(m), 노이즈 라인 합성 장면')
        fig.colorbar(shown, ax=ax, format='%g')
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def overlay(image, H, colour=(40, 40, 255)):
    for feature in pitch.model_segments(pitch.venues()['noksi']):
        line = np.linspace(feature[1], feature[2], 300) if feature[0] == 'line' else feature[1]
        p = (H @ np.c_[line, np.ones(len(line))].T).T
        p = p[:, :2]/p[:, 2:]
        p = p[np.isfinite(p).all(1) & (np.abs(p).max(1) < 6000)]
        if len(p) > 1:
            cv2.polylines(image, [np.int32(p)], False, colour, 2, cv2.LINE_AA)
    return image


def figures():
    # Needs the local job folder; skipped on a machine without it.
    from autohl import pipeline, semantic_pitch
    j = pipeline.Job('ye-0920-goal')
    if not os.path.exists(j.f('frames.npy')):
        return
    frame, boxes = np.array(j.frames()[245]), j.dets()[245]
    strokes, _, _ = fs._strokes(frame, boxes, False)
    named = 'not run'
    if os.environ.get('AUTOHL_LINE_WEIGHTS'):
        named = len(semantic_pitch.SemanticLines().detect(frame)['lines'])
    shown = frame.copy()
    for s in strokes.astype(int):
        cv2.line(shown, tuple(s[:2]), tuple(s[2:]), (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(shown, f'red: OpenCV strokes {len(strokes)} | AI line classes: {named}', (30, 60), 0, 1.1, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(RESULTS/'line_detection.png'), cv2.resize(shown, (1280, 720)))
    solved = RESULTS.parent/'solved_ye.npy'
    if solved.exists():
        shown = overlay(frame.copy(), np.load(solved))
        cv2.putText(shown, 'camera solver, frame 245, no human input', (30, 60), 0, 1.1, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(RESULTS/'pitch_overlay.png'), cv2.resize(shown, (1280, 720)))


def main():
    plt.rcParams['font.family'] = ['AppleGothic', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    rows = load()
    info = json.loads((RESULTS/'run_info.json').read_text())
    summary = {'environment': info, 'by_group': {}, 'gains': {}, 'against_A': {}}
    for tag in GROUPS:
        group = [r for r in rows if r['tags'] == tag]
        summary['by_group'][tag] = table(group)
    oracle = [r for r in rows if r['mode'] == 'oracle']
    noisy = [r for r in rows if r['mode'] == 'noisy']
    summary['all_oracle'], summary['all_noisy'] = table(oracle), table(noisy)
    for name, subset in (('oracle', oracle), ('noisy', noisy)):
        summary['gains'][name] = {
            'lines_vs_clicks (C/A)': paired(subset, 'A', 'C'),
            '3d_model (D0/C)': paired(subset, 'C', 'D0'),
            'temporal (D/D0)': paired(subset, 'D0', 'D'),
            'tripod (E/D)': paired(subset, 'D', 'E'),
            'auto_vs_A (AUTO/A)': paired(subset, 'A', 'AUTO'),
            'cost AUTO/A seconds': paired(subset, 'A', 'AUTO', 'seconds')}
    (RESULTS/'real.json').exists() and summary.update(real=json.loads((RESULTS/'real.json').read_text()))
    (RESULTS/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    bars(rows, 'hold_median_m', '보이는 경기장 점의 위치 오차 중앙값 (m, 로그)', RESULTS/'position_error.png')
    bars(rows, 'mean_speed_error_kmh', '동일 선수 경로의 평균 속도 오차 (km/h, 로그)', RESULTS/'speed_error.png')
    bars(rows, 'camera_position_m', '카메라 위치 오차 (m, 로그)', RESULTS/'camera_error.png')
    bars(rows, 'seconds', '클립당 보정 시간 (초, 로그)', RESULTS/'processing_time.png')
    heatmap(RESULTS/'position_heatmap.png')
    figures()
    print('written')


if __name__ == '__main__':
    main()
