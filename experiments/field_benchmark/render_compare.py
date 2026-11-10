"""Side-by-side benchmark video: PivotAnchor vs TurfLens on the 7 external clips.

usage: python experiments/field_benchmark/render_compare.py RUN_DIR OUT.mp4 [--clip ID ...] [--cache DIR]
Run after `bench.py score`. Reads RUN_DIR/manifest/fixtures.json (or the repo fixtures), RUN_DIR/report.json,
RUN_DIR/speed_trace.csv and the saved predictions; draws each engine's projected pitch
lines on the shared frames, the hand-marked reference points on labelled frames, the
speed targets, and the per-clip scores.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from autohl import camera, detect, pitch, track  # noqa: E402
import imageio_ffmpeg  # noqa: E402

CACHE = ROOT / 'work/external_benchmark/cache'
# Any font with Hangul. The first is macOS; the others are common Linux and Windows installs.
FONTS = ('/System/Library/Fonts/AppleSDGothicNeo.ttc', '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
         '/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc', 'C:/Windows/Fonts/malgun.ttf')
W, H = 1920, 1080
PW, PH = 960, 540
TOP = 72
ENG = ('pivotanchor', 'turflens')
NAME = {'pivotanchor': 'PivotAnchor', 'turflens': 'TurfLens', 'manual_control': '수동 보정'}
# BGR for cv2, RGB for PIL
BGR = {'pivotanchor': (255, 180, 40), 'turflens': (40, 150, 255), 'sensor': (235, 235, 235),
       'manual_control': (150, 150, 150)}
RGB = {k: v[::-1] for k, v in BGR.items()}
BG = (22, 24, 28)
GROUP = {'elevated_fixed': '상단 고정 카메라', 'elevated_pan_zoom': '상단 팬·줌',
         'near_sideline_pan_stress': '관중석 하단 스트레스 (진단)',
         'near_sideline_pan_zoom_stress': '관중석 하단 스트레스 (진단)',
         'negative_unmarked': '거부 대조군 (구장선 없음)'}
GATE = {'no_crash': '오류 없음', 'field_coverage': '커버리지 ≥90%', 'field_accuracy_yield': '선 p95≤20px 프레임 ≥80%',
        'speed': '속도 MAE≤3 km/h', 'no_false_accept': '오탐 0'}


def gate_label(k, row):
    if k != 'speed':
        return GATE.get(k, k)
    why = []
    for t in row['speed']:
        if not t['identity_audit_pass']:
            why.append(f"{t['id']} 신원 감사 실패")
        elif not t['error_kmh'] or t['scored'] < .8 * t['eligible']:
            why.append(f"{t['id']} 표본 부족")
        elif t['error_kmh']['mae'] > 3:
            why.append(f"{t['id']} MAE>3")
    return '속도 (' + ', '.join(why) + ')'


def font(size, bold=False):
    for path in FONTS:
        if Path(path).exists():
            # Apple SD Gothic Neo keeps its bold face at index 6; other files have one face.
            return ImageFont.truetype(path, size, index=6 if bold and path.endswith('Neo.ttc') else 0)
    raise SystemExit('No Hangul font found; add one to FONTS in render_compare.py')


_FONTS = {}


def text(img, xy, s, size=22, color=(230, 230, 230), bold=False, anchor='la'):
    if (size, bold) not in _FONTS:
        _FONTS[size, bold] = font(size, bold)
    ImageDraw.Draw(img).text(xy, s, font=_FONTS[size, bold], fill=color, anchor=anchor)


class Proj:
    def __init__(self, npz):
        d = np.load(npz)
        self.M, self.valid = d['matrices'], d['valid']
        self.k1, self.focal = float(d['k1']), float(d['focal'])

    def img(self, k, pts):
        p = np.c_[pts, np.ones(len(pts))] @ self.M[k].T
        ok = p[:, 2] > 1e-9
        out = np.full((len(pts), 2), np.nan)
        out[ok] = p[ok, :2] / p[ok, 2:]
        out[ok] = camera.distort(out[ok], self.k1, self.focal)
        return out


def polylines(meta):
    out = []
    for kind, *g in pitch.model_segments(meta):
        if kind == 'line':
            a, b = g
            out.append(a + (b - a) * np.linspace(0, 1, 80)[:, None])
        else:
            out.append(g[0])
    return out


def draw_pitch(panel, P, k, lines, color):
    for pts in lines:
        q = P.img(k, pts) * (PW / W)
        good = np.isfinite(q).all(1) & (np.abs(q) < 4000).all(1)
        run = []
        for g, p in zip(good, q):
            if g:
                run.append(p)
            elif len(run) > 1:
                cv2.polylines(panel, [np.int32(np.round(np.array(run) * 4))], False, color, 2, cv2.LINE_AA, 2)
                run = []
            else:
                run = []
        if len(run) > 1:
            cv2.polylines(panel, [np.int32(np.round(np.array(run) * 4))], False, color, 2, cv2.LINE_AA, 2)


def fmt(v, d=1, unit=''):
    return '—' if v is None else f'{v:.{d}f}{unit}'


def load_speed(run, clip_id):
    trace = {}
    with open(run / 'speed_trace.csv') as f:
        for r in csv.DictReader(f):
            if r['clip'] != clip_id:
                continue
            t = trace.setdefault(r['target'], {})
            k = int(r['frame'])
            t.setdefault('sensor', {})[k] = float(r['sensor_kmh'])
            t.setdefault(r['engine'], {})[k] = float(r['estimate_kmh']) if r['estimate_kmh'] else np.nan
    return trace


def target_feet(clip_id, frames, boxes, labels):
    out = {}
    targets = labels.get('targets', [])
    if not targets:
        return out
    with np.load(CACHE / clip_id / 'camera.npz') as c:
        cum = c['cumulative']
    fe = track.features(frames, boxes)
    for t in targets:
        anchors = {int(k): v for k, v in t['anchors'].items()}
        path = track.track(frames, boxes, cum, anchors, FE=fe, log=lambda *_: None)
        out[t['id']] = (path['foot'], t['frames'])
    return out


def metrics_panel(clip, rep, size):
    img = Image.new('RGB', size, BG)
    x0 = [20, 300, 620]
    text(img, (x0[0], 14), '지표', 22, (160, 160, 160), True)
    for i, e in enumerate(ENG):
        text(img, (x0[i + 1], 14), NAME[e], 24, RGB[e], True)
    rows = []
    eng = rep['engines']
    def g(e):
        return eng[e]
    rows.append(('지원 프레임', [f"{g(e)['supported']}/{g(e)['frames']}" for e in ENG]))
    if any(g(e)['paint']['labelled_frames'] for e in ENG):
        rows.append(('선 오차 중앙값 / p95', [
            (f"{fmt(g(e)['paint']['line_error_px']['median'])} / {fmt(g(e)['paint']['line_error_px']['p95'])} px"
             if g(e)['paint']['line_error_px'] else '—') for e in ENG]))
        rows.append(('within 10/20/40 px', [
            ' / '.join(str(g(e)['paint']['within_px'][x]) for x in ('10', '20', '40')) +
            f"  (of {g(e)['paint']['labelled_frames']})" for e in ENG]))
        rows.append(('잔디 1초 재투영 p95', [
            (f"{fmt(g(e)['motion']['error_px']['p95'], 2)} px (n={g(e)['motion']['error_px']['n']})"
             if g(e)['motion']['error_px'] else f"— (0/{g(e)['motion']['eligible_pairs']})") for e in ENG]))
    if eng['pivotanchor'].get('paint', {}).get('landmark_error_m') or eng['turflens'].get('paint', {}).get('landmark_error_m'):
        rows.append(('고정점 오차 p95', [fmt((g(e)['paint']['landmark_error_m'] or {}).get('p95'), 2, ' m') for e in ENG]))
    for t in range(len(eng['pivotanchor']['speed'])):
        rows.append((f"속도 {eng['pivotanchor']['speed'][t]['id']} MAE (표본)", [
            (f"{g(e)['speed'][t]['error_kmh']['mae']:.2f} km/h ({g(e)['speed'][t]['scored']}/{g(e)['speed'][t]['eligible']})"
             if g(e)['speed'][t]['error_kmh'] else f"— (0/{g(e)['speed'][t]['eligible']})") for e in ENG]))
    if clip['class'].startswith('negative_'):
        rows.append(('오탐 프레임', [str(g(e)['false_accept_frames']) for e in ENG]))
    y = 54
    for label, vals in rows:
        text(img, (x0[0], y), label, 20, (190, 190, 190))
        for i, v in enumerate(vals):
            text(img, (x0[i + 1], y), v, 20)
        y += 34
    y += 6
    ImageDraw.Draw(img).line((20, y, size[0] - 20, y), fill=(60, 60, 66), width=1)
    y += 12
    diagnostic = clip['class'].startswith('near_sideline')
    for i, e in enumerate(ENG):
        gates = g(e)['gates']
        failed = [k for k, ok in gates.items() if not ok]
        if diagnostic:
            status, col = '진단 (자동 인식 성공으로 세지 않음)', (170, 170, 170)
        elif failed:
            status, col = 'FAIL · ' + ', '.join(gate_label(k, g(e)) for k in failed), (255, 95, 95)
        else:
            status, col = 'PASS', (90, 220, 120)
        text(img, (x0[0], y), NAME[e], 22, RGB[e], True)
        text(img, (x0[1] - 120, y), status, 22, col, True)
        y += 34
        if g(e)['reason'] and g(e)['supported'] == 0:
            text(img, (x0[1] - 120, y - 4), f"사유: {g(e)['reason'][:48]}", 18, (150, 150, 150))
            y += 26
    return np.array(img)[:, :, ::-1].copy()


def chart_panel(clip, rep, trace, n, size):
    """Returns (static BGR image, x-mapper from frame to pixel)."""
    w, h = size
    img = Image.new('RGB', size, BG)
    left, right = (70 if trace else 150), w - 20
    fx = lambda k: left + (right - left) * k / max(1, n - 1)
    d = ImageDraw.Draw(img)
    if trace:
        ids = sorted(trace)
        ch = (h - 40) // len(ids)
        allv = [v for t in trace.values() for s in t.values() for v in s.values() if np.isfinite(v)]
        vmax = max(5, np.ceil(max(allv) / 5) * 5) if allv else 30
        for i, tid in enumerate(ids):
            y0, y1 = 30 + i * ch + 26, 30 + (i + 1) * ch - 14
            fy = lambda v: y1 - (y1 - y0) * v / vmax
            text(img, (left, y0 - 26), f'선수 {tid} 속도 (km/h) — 센서 ZXY vs 엔진', 20, (200, 200, 200), True)
            d.rectangle((left, y0, right, y1), outline=(60, 60, 66))
            for v in range(0, int(vmax) + 1, 10):
                d.line((left, fy(v), right, fy(v)), fill=(45, 45, 50))
                text(img, (left - 8, fy(v)), str(v), 16, (130, 130, 130), anchor='rm')
            for key in ('manual_control', 'sensor', 'pivotanchor', 'turflens'):
                s = trace[tid].get(key)
                if not s:
                    continue
                ks = sorted(s)
                pts, seg = [], []
                for k in ks:
                    if np.isfinite(s[k]):
                        seg.append((fx(k), fy(s[k])))
                    elif seg:
                        pts.append(seg); seg = []
                if seg:
                    pts.append(seg)
                col = RGB['sensor'] if key == 'sensor' else RGB[key]
                for seg in pts:
                    if len(seg) > 1:
                        d.line(seg, fill=col, width=4 if key == 'sensor' else 3, joint='curve')
        lx = left
        for key, lab in (('sensor', '센서'), ('pivotanchor', 'PivotAnchor'), ('turflens', 'TurfLens'), ('manual_control', '수동 보정')):
            d.line((lx, 14, lx + 26, 14), fill=RGB['sensor'] if key == 'sensor' else RGB[key], width=4)
            text(img, (lx + 32, 14), lab, 18, (200, 200, 200), anchor='lm')
            lx += 175
    else:
        text(img, (left, 14), '프레임별 좌표 지원 (채움 = 지원)', 20, (200, 200, 200), True)
        for i, e in enumerate(ENG):
            y0 = 56 + i * 56
            text(img, (left - 10, y0 + 16), NAME[e], 20, RGB[e], anchor='rm')
            d.rectangle((left, y0, right, y0 + 32), outline=(60, 60, 66))
        text(img, (left, 186), '수동 표식 프레임별 구장선 오차 p95 (px), 점선 = 합격선 20 px', 20, (200, 200, 200), True)
    return np.array(img)[:, :, ::-1].copy(), fx


def support_bars(chart, valid, fx, k):
    for i, e in enumerate(ENG):
        y0 = 56 + i * 56
        v = valid[e][:k + 1]
        for j in np.flatnonzero(v):
            cv2.rectangle(chart, (int(fx(j)), y0 + 2), (int(fx(j + 1)) + 1, y0 + 30), BGR[e], -1)


def p95_bars(chart, rep, labels, fx_size):
    w, h = fx_size
    frames = sorted(int(f) for f in labels.get('frames', {}))
    if not frames:
        return
    left, base, top = 90, h - 34, 215
    vmax = 40
    fy = lambda v: base - (base - top) * min(v, vmax) / vmax
    cv2.line(chart, (left, int(fy(20))), (w - 20, int(fy(20))), (90, 90, 230), 1, cv2.LINE_AA)
    slot = (w - 20 - left) / len(frames)
    for i, fr in enumerate(frames):
        for j, e in enumerate(ENG):
            pf = rep['engines'][e]['paint']['per_frame'].get(str(fr))
            x = int(left + slot * i + 30 + j * 70)
            if pf:
                cv2.rectangle(chart, (x, int(fy(pf['p95']))), (x + 56, base), BGR[e], -1)
                cv2.putText(chart, f"{pf['p95']:.1f}", (x, int(fy(pf['p95'])) - 6), cv2.FONT_HERSHEY_SIMPLEX, .5, (230, 230, 230), 1, cv2.LINE_AA)
            else:
                cv2.putText(chart, 'n/a', (x + 10, base - 6), cv2.FONT_HERSHEY_SIMPLEX, .5, (140, 140, 140), 1, cv2.LINE_AA)
        cv2.putText(chart, f'frame {fr}', (int(left + slot * i + 50), base + 22), cv2.FONT_HERSHEY_SIMPLEX, .5, (170, 170, 170), 1, cv2.LINE_AA)


def card(lines_spec):
    img = Image.new('RGB', (W, H), BG)
    for xy, s, size, color, bold in lines_spec:
        text(img, xy, s, size, color, bold)
    return np.array(img)[:, :, ::-1].copy()


def title_card(manifest, run):
    e = manifest['engines']
    scope = '고정 위치 카메라' if manifest.get('suite') == 'fixed_mount_v2' else '외부 실제 경기'
    spec = [((120, 260), f"{scope} {len(manifest['clips'])}개 클립 벤치마크", 60, (240, 240, 240), True),
            ((120, 350), '구장 인식 · 선수 속도 비교', 44, (200, 200, 200), False),
            ((120, 480), f"PivotAnchor  {e['pivotanchor']}", 28, RGB['pivotanchor'], True),
            ((120, 530), f"TurfLens        {e['turflens']}", 28, RGB['turflens'], True),
            ((120, 640), '두 엔진 모두 같은 공용 입력(프레임·YOLO 검출·ORB 추적)에서 자동 보정만 사용', 24, (170, 170, 170), False),
            ((120, 680), '수동 표식(reference.json)은 채점에만 사용 — 엔진에는 주입하지 않음', 24, (170, 170, 170), False),
            ((120, 760), '화면 범례: 파랑 = PivotAnchor 투영 구장선, 주황 = TurfLens 투영 구장선, 흰 점 = 수동 표식', 24, (200, 200, 200), False),
            ((120, 960), f'결과 폴더: {run}', 20, (120, 120, 120), False)]
    return card(spec)


def summary_card(manifest, report):
    img = Image.new('RGB', (W, H), BG)
    text(img, (80, 50), '종합 결과', 44, (240, 240, 240), True)
    cols = [80, 420, 700, 940, 1180, 1440]
    heads = ['클립', '엔진', '지원', '선 p95 (px)', '속도 MAE A/B', '판정']
    for x, s in zip(cols, heads):
        text(img, (x, 130), s, 22, (150, 150, 150), True)
    y = 172
    for clip in manifest['clips']:
        rep = report[clip['id']]
        for i, e in enumerate(ENG):
            r = rep['engines'][e]
            failed = [k for k, ok in r['gates'].items() if not ok]
            if clip['class'].startswith('near_sideline'):
                st, col = '진단', (170, 170, 170)
            elif failed:
                st, col = 'FAIL ' + ', '.join(gate_label(k, r) for k in failed), (255, 95, 95)
            else:
                st, col = 'PASS', (90, 220, 120)
            lp = r['paint']['line_error_px']
            sp = ' / '.join(fmt(t['error_kmh']['mae'], 2) if t['error_kmh'] else '—' for t in r['speed']) or ''
            if i == 0:
                text(img, (cols[0], y), clip['id'], 22, (230, 230, 230), True)
            text(img, (cols[1], y), NAME[e], 22, RGB[e], True)
            text(img, (cols[2], y), f"{r['supported']}/{r['frames']}", 22)
            text(img, (cols[3], y), fmt(lp['p95'] if lp else None, 2), 22)
            text(img, (cols[4], y), sp, 22)
            text(img, (cols[5], y), st, 20, col, True)
            y += 50 if i else 36
        ImageDraw.Draw(img).line((80, y - 12, W - 80, y - 12), fill=(50, 50, 56))
    policy = manifest['policy']
    caption = (f"커버리지 ≥{policy['elevated_frame_coverage']:.0%}, 표식 프레임의 "
               f"≥{policy['elevated_good_labelled_share']:.0%}가 선 p95 ≤{policy['line_p95_px']:g} px, "
               f"속도 표본 ≥{policy['speed_sample_coverage']:.0%} 및 MAE ≤{policy['speed_mae_kmh']:g} km/h. ")
    if any(clip['class'].startswith('negative_') for clip in manifest['clips']):
        caption += '대조군 오탐 0. '
    if manifest.get('suite') == 'fixed_mount_v2':
        caption += '구장별 공개 치수 입력. '
    text(img, (80, H - 60), caption + '센서는 Alfheim의 두 경기 3초 구간.', 18, (140, 140, 140))
    return np.array(img)[:, :, ::-1].copy()


def render_clip(write, clip, rep, run, labels, hold=12):
    cid = clip['id']
    cache = CACHE / cid
    frames = np.load(cache / 'frames.npy', mmap_mode='r')
    boxes = detect.load(cache / 'dets.npz')
    n = len(frames)
    P = {e: Proj(run / f'{cid}__{e}.npz') for e in ENG}
    lines = polylines(clip['pitch'])
    trace = load_speed(run, cid)
    feet = target_feet(cid, frames, boxes, labels) if trace else {}
    marks = {int(k): v for k, v in labels.get('frames', {}).items()}
    metrics = metrics_panel(clip, rep, (PW, H - TOP - PH - 8))
    chart0, fx = chart_panel(clip, rep, trace, n, (PW, H - TOP - PH - 8))
    if not trace:
        p95_bars(chart0, rep, labels, (PW, H - TOP - PH - 8))
    header = Image.new('RGB', (W, TOP), (12, 13, 16))
    text(header, (24, 36), f"{cid}", 34, (240, 240, 240), True, anchor='lm')
    text(header, (360, 36), f"{clip.get('venue_name', clip['match'])}  ·  {GROUP.get(clip['class'], clip['class'])}  ·  "
         f"{clip['pitch']['length']}×{clip['pitch']['width']} m ({clip['dimensions']})", 22, (180, 180, 180), anchor='lm')
    header = np.array(header)[:, :, ::-1].copy()
    labels_img = {}
    for e in ENG:
        im = Image.new('RGBA', (PW, 40), (0, 0, 0, 0))
        text(im, (14, 6), NAME[e], 26, RGB[e], True)
        labels_img[e] = im
    nocal = {}
    for e in ENG:
        im = Image.new('RGB', (PW, 120), (0, 0, 0))
        text(im, (PW // 2, 40), '좌표 없음 (자동 보정 거부)', 34, (255, 110, 110), True, anchor='mm')
        reason = rep['engines'][e]['reason'] or ''
        text(im, (PW // 2, 88), reason[:60], 20, (200, 200, 200), anchor='mm')
        nocal[e] = np.array(im)[:, :, ::-1].copy()

    def compose(k, marked):
        canvas = np.full((H, W, 3), BG, np.uint8)
        canvas[:TOP] = header
        base = cv2.resize(np.ascontiguousarray(frames[k]), (PW, PH), interpolation=cv2.INTER_AREA)
        for i, e in enumerate(ENG):
            panel = base.copy()
            if P[e].valid[k]:
                draw_pitch(panel, P[e], k, lines, BGR[e])
            else:
                panel = (panel * .35).astype(np.uint8)
                y = PH // 2 - 60
                m = nocal[e] > 0
                panel[y:y + 120][m] = nocal[e][m]
            if marked:
                for pts in marks[k].get('lines', {}).values():
                    for x, y in np.asarray(pts) * (PW / W):
                        cv2.circle(panel, (int(x), int(y)), 3, (255, 255, 255), -1, cv2.LINE_AA)
                        cv2.circle(panel, (int(x), int(y)), 3, (0, 0, 0), 1, cv2.LINE_AA)
            for tid, (foot, (a, b)) in feet.items():
                if a <= k <= b and np.isfinite(foot[k]).all():
                    x, y = (foot[k] * (PW / W)).astype(int)
                    cv2.circle(panel, (x, y), 12, (0, 255, 255), 2, cv2.LINE_AA)
                    sp = trace.get(tid, {}).get(e, {}).get(k, np.nan)
                    ss = trace.get(tid, {}).get('sensor', {}).get(k, np.nan)
                    s = f"{tid} {'--' if not np.isfinite(sp) else f'{sp:.1f}'} / sensor {ss:.1f} km/h"
                    cv2.putText(panel, s, (x + 14, y - 10), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.putText(panel, s, (x + 14, y - 10), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 1, cv2.LINE_AA)
            lab = np.array(labels_img[e])
            a = lab[:, :, 3:4] / 255.
            cv2.rectangle(panel, (0, 0), (PW, 40), (0, 0, 0), -1)
            panel[:40] = (panel[:40] * (1 - a) + lab[:, :, 2::-1] * a).astype(np.uint8)
            tag = f'frame {k:3d}/{n - 1}  t={k / 30:4.2f}s'
            cv2.putText(panel, tag, (PW - 270, 27), cv2.FONT_HERSHEY_SIMPLEX, .6, (220, 220, 220), 1, cv2.LINE_AA)
            if marked:
                pf = rep['engines'][e]['paint']['per_frame'].get(str(k))
                s = f"REF frame {k}: line p95 {pf['p95']:.1f}px" if pf else f'REF frame {k}: not scored'
                cv2.rectangle(panel, (0, PH - 40), (PW, PH), (0, 0, 0), -1)
                cv2.putText(panel, s, (14, PH - 13), cv2.FONT_HERSHEY_SIMPLEX, .75,
                            (90, 220, 120) if pf and pf['p95'] <= 20 else (90, 90, 255), 2, cv2.LINE_AA)
            canvas[TOP:TOP + PH, i * PW:(i + 1) * PW] = panel
        cv2.line(canvas, (PW, TOP), (PW, TOP + PH), (0, 0, 0), 2)
        y0 = TOP + PH + 8
        canvas[y0:, :PW] = metrics
        chart = chart0.copy()
        if trace:
            x = int(fx(k))
            cv2.line(chart, (x, 30), (x, chart.shape[0] - 10), (0, 255, 255), 1, cv2.LINE_AA)
        else:
            support_bars(chart, {e: P[e].valid for e in ENG}, fx, k)
        canvas[y0:, PW:] = chart
        return canvas

    for k in range(n):
        marked = k in marks
        img = compose(k, marked)
        for _ in range(hold if marked else 1):
            write(img)
    for _ in range(20):
        write(img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run', type=Path)
    ap.add_argument('out', type=Path)
    ap.add_argument('--clip', action='append')
    ap.add_argument('--cache', type=Path, default=CACHE, help='shared frame cache from bench.py prepare')
    args = ap.parse_args()
    globals()['CACHE'] = args.cache
    mpath = args.run / 'manifest/fixtures.json'
    manifest = json.loads((mpath if mpath.exists() else ROOT / 'experiments/field_benchmark/fixtures.json').read_text())
    lpath = mpath.parent / 'reference.json'
    labels = json.loads((lpath if lpath.exists() else ROOT / 'experiments/field_benchmark/reference.json').read_text())
    report = json.loads((args.run / 'report.json').read_text())
    clips = [c for c in manifest['clips'] if not args.clip or c['id'] in args.clip]
    writer = imageio_ffmpeg.write_frames(str(args.out), (W, H), fps=30, codec='libx264', quality=None,
                                         output_params=['-crf', '20', '-preset', 'medium', '-pix_fmt', 'yuv420p', '-movflags', '+faststart'],
                                         macro_block_size=8)
    writer.send(None)
    write = lambda bgr: writer.send(np.ascontiguousarray(bgr[:, :, ::-1]))
    t = title_card(manifest, args.run)
    for _ in range(120):
        write(t)
    for clip in clips:
        print('render', clip['id'], flush=True)
        render_clip(write, clip, report[clip['id']], args.run, labels.get(clip['id'], {}))
    s = summary_card({**manifest, 'clips': clips}, report)
    for _ in range(210):
        write(s)
    writer.close()
    print('wrote', args.out)


if __name__ == '__main__':
    main()
