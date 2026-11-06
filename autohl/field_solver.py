"""Find the camera behind a clip from paint, people and camera motion, with no venue photos.

One homography per frame can fit whatever paint is in view and still be wrong a few metres away,
because it is free to describe a camera that cannot exist. Here every short window of the clip is
explained by one real camera (position, height, focal length, three angles). The position must
recur across windows, which rejects most wrong answers, and two checks that do not use paint
reject the rest: people must come out about 1.74 m tall and stand on the pitch.

World axes: X along the near touchline, Y toward the camera side, Z down, so the camera height
is a negative Z. A pitch has 180-degree symmetry; results are reported with the camera at +Y.
"""

from functools import lru_cache
from itertools import combinations

import cv2
import numpy as np
from scipy import ndimage
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from . import pitch
from .camera import warp

CENTRE = np.array([960., 540.])
PERSON_HEIGHT = 1.74
ALONG_LINES = [1, 2, 5, 7, 8, 10, 11, 13, 14, 16]
ACROSS_LINES = [0, 3, 4, 6, 9, 12, 15]
ARCS = (17, 18, 19)
STROKE_LINES = 18      # longest distinct strokes tried in the line-to-rectangle search
STARTS = 14            # camera hypotheses refined in each window
SAME_POSITION_M = 3.   # two windows see one camera when their solutions are this close
SAME_HEIGHT_M = 1.
MIN_FOCAL, MAX_FOCAL = 850., 4200.
MIN_HEIGHT, MAX_HEIGHT = 1.2, 30.
# Acceptance, set from clips with known answers and from low-camera clips the solver gets wrong.
MAX_PEOPLE_ERROR = .15  # median |log(predicted height / box height)|
MIN_ON_PITCH = .85
MIN_COVER = .4          # share of a window's strokes the model must explain to count as a key
MAX_KEY_GAP = 40.       # pixels two keys may differ by once carried to the same frame
# The ground error of a foot grows as distance squared over height and focal length. Noise from
# real clips (3 px on paint, 2 px on feet) gives p95 = 2.3 D^2/(h f) over the pitch, within 10% for
# lenses of 1000-5400 px, heights of 2-20 m and cameras 5-30 m behind the touchline.
GROUND_ERROR_GAIN = 2.3
MAX_GROUND_ERROR_M = 2.5
LOWEST_CAMERA_M = 3.  # Below this, a long lens does not help: wrong answers pass every check.


@lru_cache(maxsize=1)
def _kernels():
    return pitch._line_kernels(L=25, off=3, n=32)


def _lawn(small):
    # Pitch surface in a half-size frame: green or over-lit turf, painted yellow included, in the
    # longest vertical run of each column. A leak up a green fence is a narrow upward spike in
    # that boundary, which a wide closing removes.
    hue, sat, val = cv2.split(cv2.cvtColor(small, cv2.COLOR_BGR2HSV))
    grass = ((hue > 13) & (hue < 95) & (sat > 35) & (val > 62)) | ((val > 135) & (sat < 95))
    grass = cv2.morphologyEx(np.uint8(grass)*255, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0
    height, width = grass.shape
    top, bottom = np.full(width, float(height)), np.full(width, float(height))
    for x in range(width):
        rows = np.flatnonzero(grass[:, x])
        if len(rows):
            run = max(np.split(rows, np.flatnonzero(np.diff(rows) > 5)+1), key=len)
            top[x], bottom[x] = run[0], run[-1]
    top = ndimage.grey_opening(ndimage.grey_closing(top, size=201, mode='nearest'),
                               size=31, mode='nearest')
    rows = np.arange(height)[:, None]
    return (rows >= top+2) & (rows <= bottom)


def _strokes(image, boxes, yellow):
    """Straight paint in a frame, as segments in full-size pixels, plus surface and loose paint."""
    small = cv2.resize(image, (960, 540), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    grey = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32) - .35*hsv[..., 1]
    lawn = _lawn(small)
    top = cv2.morphologyEx(grey, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    if not yellow:
        top[(hsv[..., 0] > 13) & (hsv[..., 0] < 40) & (hsv[..., 1] > 80)] = 0
    top[~lawn] = 0
    for x1, y1, x2, y2, *_ in boxes:
        a, b, c, d = np.rint(np.array([x1, y1, x2, y2])/2).astype(int)
        top[max(0, b-2):d+3, max(0, a-2):c+3] = 0
    ridge = np.maximum.reduce([cv2.filter2D(top, -1, k) for k in _kernels()])
    ridge[ridge < 0] = 0
    # Turf texture is bright but no brighter than its surroundings: require local contrast.
    relative = ridge / (cv2.blur(ridge, (41, 41))+1)
    mask = np.uint8((relative >= 2.2) & (ridge >= 7))*255
    segments = cv2.HoughLinesP(mask, 1, np.pi/720, 22, minLineLength=28, maxLineGap=7)
    segments = np.empty((0, 4)) if segments is None else segments.reshape(-1, 4).astype(float)*2
    return segments, lawn, relative >= 1.5


def line_paint(loose, lawn, shortest=28., widest=5.):
    """Paint that is long and thin. Floodlit turf sparkles in short bright flecks that sit near any
    line, so taking them as paint pulls a fit toward nothing and makes every line look close."""
    mask = np.uint8(loose & lawn)
    count, label, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    span = np.hypot(stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT])
    keep = (span >= shortest) & (stats[:, cv2.CC_STAT_AREA] <= widest*span)
    keep[0] = False
    return keep[label]


def _far_edge(image, boxes):
    """The far touchline, found as the turf edge below a red running track.

    Paint at the very edge of the turf is thin at this distance and sits on the lawn mask's
    border, so the paint search misses it, and the far side of the pitch then floats freely.
    Short pieces, because the edge bows slightly across a wide lens.
    """
    hue, sat, val = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
    red = ((hue < 12) | (hue > 168)) & (sat > 70) & (val > 45)
    red = cv2.morphologyEx(np.uint8(red)*255, cv2.MORPH_OPEN, np.ones((3, 9), np.uint8)) > 0
    points = []
    for x in range(20, 1900, 12):
        rows = np.flatnonzero(red[450:950, x])+450
        if len(rows):
            strips = [g for g in np.split(rows, np.flatnonzero(np.diff(rows) > 4)+1) if 4 <= len(g) <= 60]
            if strips:
                points.append((x, strips[0][-1]+1))
    points = np.array(points, float).reshape(-1, 2)
    for x1, y1, x2, y2, *_ in boxes:
        points = points[~((points[:, 0] > x1-4) & (points[:, 0] < x2+4) & (points[:, 1] > y1-4) & (points[:, 1] < y2+4))]
    if len(points) < 20:
        return np.empty((0, 4))
    for _ in range(3):
        curve = np.polyfit(points[:, 0], points[:, 1], 2)
        error = np.abs(np.polyval(curve, points[:, 0])-points[:, 1])
        points = points[error < max(4., 2.5*np.median(error))]
        if len(points) < 20:
            return np.empty((0, 4))
    pieces = []
    for part in np.array_split(points, 4):
        if len(part) >= 5 and np.ptp(part[:, 0]) > 120:
            pieces.append(np.r_[part[0], part[-1]])
    return np.array(pieces).reshape(-1, 4)


def _people(boxes, to_base):
    # Foot points and box heights of whole people, in the window's reference frame.
    boxes = np.asarray(boxes, float).reshape(-1, 5)
    height = boxes[:, 3]-boxes[:, 1]
    whole = ((height > 26) & (height < 420) & (boxes[:, 1] > 6) & (boxes[:, 3] < 1072)
             & (boxes[:, 0] > 6) & (boxes[:, 2] < 1914) & (boxes[:, 2]-boxes[:, 0] < height*1.1))
    boxes = boxes[whole]
    if not len(boxes):
        return np.empty((0, 2)), np.empty(0)
    x = (boxes[:, 0]+boxes[:, 2])/2
    feet = warp(np.c_[x, boxes[:, 3]], to_base)
    heads = warp(np.c_[x, boxes[:, 1]], to_base)
    return feet, feet[:, 1]-heads[:, 1]


def _rotation(yaw, tilt, roll):
    z = np.array([np.sin(yaw)*np.cos(tilt), -np.cos(yaw)*np.cos(tilt), np.sin(tilt)])
    across = np.array([np.cos(yaw), np.sin(yaw), 0.])
    down = np.cross(z, across)
    return np.stack((np.cos(roll)*across+np.sin(roll)*down, -np.sin(roll)*across+np.cos(roll)*down, z))


def _homography(p):
    yaw, tilt, roll, log_focal, x, y, height = p
    focal = np.exp(log_focal)
    K = np.array([[focal, 0, CENTRE[0]], [0, focal, CENTRE[1]], [0, 0, 1.]])
    R = _rotation(yaw, tilt, roll)
    H = K @ np.c_[R[:, :2], -R @ np.array([x, y, -height])]
    return H / H[2, 2]


def _parameters(R, centre, focal):
    z = R[2]
    yaw = np.arctan2(z[0], -z[1])
    across = np.array([np.cos(yaw), np.sin(yaw), 0.])
    roll = np.arctan2(R[0] @ np.cross(z, across), R[0] @ across)
    return np.array([yaw, np.arcsin(np.clip(z[2], -1, 1)), roll, np.log(focal),
                     centre[0], centre[1], -centre[2]])


def _camera_side(p, meta):
    # The same camera after turning the pitch 180 degrees.
    if p[5] >= 0:
        return p
    p = p.copy()
    p[0] = np.arctan2(np.sin(p[0]+np.pi), np.cos(p[0]+np.pi))
    p[4], p[5] = meta['length']-p[4], -p[5]
    return p


def _people_fit(R, centre, focal, feet, seen, meta):
    """For many cameras: median log(predicted/seen height) and the share of feet on the pitch."""
    rays = np.stack(((feet[:, 0]-CENTRE[0])[None]/focal[:, None],
                     (feet[:, 1]-CENTRE[1])[None]/focal[:, None],
                     np.ones((len(focal), len(feet)))), 2) @ R
    front = rays[:, :, 2] > 1e-4
    ground = centre[:, None] + (-centre[:, 2:3]/np.maximum(rays[:, :, 2], 1e-6))[..., None]*rays
    head = ground.copy()
    head[:, :, 2] = -PERSON_HEIGHT
    view = (head-centre[:, None]) @ R.transpose(0, 2, 1)
    y = focal[:, None]*view[:, :, 1]/np.maximum(view[:, :, 2], 1e-6) + CENTRE[1]
    ratio = np.log(np.maximum(feet[:, 1][None]-y, 1e-3)/seen[None])
    ratio[~front] = np.nan
    on_pitch = ((ground[:, :, 0] > -6) & (ground[:, :, 0] < meta['length']+6)
                & (np.abs(ground[:, :, 1]) < meta['width']/2+6) & front)
    with np.errstate(all='ignore'):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            error = np.nanmedian(ratio, axis=1)
    return np.nan_to_num(error, nan=9.), on_pitch.mean(1)


def _people_weight(error, on_pitch):
    return np.exp(-(error/.22)**2)*np.clip(on_pitch/.7, 0, 1)


def _people_of(p, window, meta):
    if len(window['feet']) < 4:
        return 0., 1.
    R = _rotation(*p[:3])[None]
    error, on_pitch = _people_fit(R, np.array([[p[4], p[5], -p[6]]]), np.array([np.exp(p[3])]),
                                  window['feet'], window['seen'], meta)
    return float(error[0]), float(on_pitch[0])


def _distinct(segments):
    # A line seen in several frames is one piece of evidence, weighted by its total length.
    length = np.linalg.norm(segments[:, :2]-segments[:, 2:], axis=1)
    rays = np.c_[segments[:, :2]-CENTRE, np.full(len(segments), 1600.)]
    ends = np.c_[segments[:, 2:]-CENTRE, np.full(len(segments), 1600.)]
    normal = np.cross(rays, ends)
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    kept, weight, normals = [], [], []
    for i in np.argsort(-length):
        if normals:
            overlap = np.abs(np.array(normals) @ normal[i])
            if overlap.max() > .99996:
                weight[int(np.argmax(overlap))] += length[i]
                continue
        normals.append(normal[i])
        kept.append(segments[i])
        weight.append(length[i])
    return np.array(kept), np.sqrt(np.array(weight))


def _sample(segments, step):
    if not len(segments):
        return np.empty((0, 2))
    pieces = []
    for a, b in segments.reshape(-1, 2, 2):
        n = max(2, int(np.linalg.norm(b-a)/step))
        pieces.append(a+(b-a)*np.linspace(0, 1, n)[:, None])
    return np.vstack(pieces)


def _unit_square(q):
    # Homographies taking the unit square to each quadrilateral (a, b, c, d in grid order).
    (x0, y0), (x1, y1), (x2, y2), (x3, y3) = q.transpose(1, 2, 0)
    dx1, dx2, dy1, dy2 = x1-x3, x2-x3, y1-y3, y2-y3
    sx, sy = x0-x1-x2+x3, y0-y1-y2+y3
    den = dx1*dy2-dx2*dy1
    den = np.where(np.abs(den) < 1e-9, 1e-9, den)
    g, h = (sx*dy2-dx2*sy)/den, (dx1*sy-sx*dy1)/den
    return np.stack((np.stack((x1-x0+g*x1, x2-x0+h*x2, x0), 1),
                     np.stack((y1-y0+g*y1, y2-y0+h*y2, y0), 1),
                     np.stack((g, h, np.ones_like(g)), 1)), 1)


def _rectangles(meta):
    d = pitch.dims(meta)
    length, half = meta['length'], meta['width']/2
    xs = sorted({0, d['ga_depth'], d['box_depth'], meta.get('half_x', length/2),
                 length-d['box_depth'], length-d['ga_depth'], length})
    ys = sorted({-half, -d['box_half'], -d['ga_half'], d['ga_half'], d['box_half'], half})
    return np.array([(a, b, c, e) for a, b in combinations(xs, 2) for c, e in combinations(ys, 2)
                     if b-a >= 4 and e-c >= 4], float)


def _cameras_from_lines(lines, meta):
    """Every assignment of four image lines to a model rectangle a pinhole camera could produce."""
    a = np.c_[lines[:, :2], np.ones(len(lines))]
    b = np.c_[lines[:, 2:], np.ones(len(lines))]
    lines = np.cross(a, b)
    lines /= np.linalg.norm(lines[:, :2], axis=1, keepdims=True)
    quads = np.array(list(combinations(range(len(lines)), 4)))
    q = quads[:, [[0, 1, 2, 3], [0, 2, 1, 3], [0, 3, 1, 2]]].reshape(-1, 4)
    q = np.vstack((q, q[:, [2, 3, 0, 1]]))
    q = np.vstack((q, q[:, [1, 0, 2, 3]], q[:, [0, 1, 3, 2]], q[:, [1, 0, 3, 2]]))
    l = lines[q]
    corners = np.stack([np.cross(l[:, i], l[:, j]) for i, j in ((0, 2), (1, 2), (0, 3), (1, 3))], 1)
    keep = (np.abs(corners[:, :, 2]) > 1e-9).all(1)
    corners = corners[keep]
    xy = corners[:, :, :2]/corners[:, :, 2:]
    xy = xy[(np.abs(xy) < 6000).all(axis=(1, 2))]
    edge = np.roll(xy[:, [0, 1, 3, 2]], -1, 1)-xy[:, [0, 1, 3, 2]]
    turn = np.roll(edge, -1, 1)
    turn = edge[:, :, 0]*turn[:, :, 1]-edge[:, :, 1]*turn[:, :, 0]
    xy = xy[(turn > 0).all(1) | (turn < 0).all(1)]
    if not len(xy):
        return np.empty((0, 3, 3)), np.empty(0)
    unit = _unit_square(xy)
    v = unit.copy()
    v[:, :2] -= CENTRE[None, :, None]*v[:, 2:3]
    with np.errstate(divide='ignore', invalid='ignore'):
        focal2 = -(v[:, 0, 0]*v[:, 0, 1]+v[:, 1, 0]*v[:, 1, 1])/(v[:, 2, 0]*v[:, 2, 1])
    good = (focal2 > MIN_FOCAL**2) & (focal2 < MAX_FOCAL**2)
    unit, v, focal2 = unit[good], v[good], focal2[good]
    v[:, :2] /= np.sqrt(focal2)[:, None, None]
    ratio = np.linalg.norm(v[:, :, 0], axis=1)/np.linalg.norm(v[:, :, 1], axis=1)
    rect = _rectangles(meta)
    width, depth = rect[:, 1]-rect[:, 0], rect[:, 3]-rect[:, 2]
    i, j = np.nonzero(np.abs(np.log(ratio[:, None]/(width/depth))) < .10)
    H = unit[i].copy()
    H[:, :, 0] /= width[j, None]
    H[:, :, 1] /= depth[j, None]
    H[:, :, 2] -= H[:, :, 0]*rect[j, 0, None]+H[:, :, 1]*rect[j, 2, None]
    return H, np.sqrt(focal2[i])


def _poses(H, focal):
    m = H.copy()
    m[:, :2] -= CENTRE[None, :, None]*m[:, 2:3]
    m[:, :2] /= focal[:, None, None]
    m /= np.linalg.norm(m[:, :, 0], axis=1)[:, None, None]
    # The pitch centre is in front of the camera. The pitch origin need not be: a camera near
    # one end turned toward the other has that corner behind it.
    m *= np.sign(m[:, 2, 0]*50+m[:, 2, 2])[:, None, None]
    r1, r2, t = m[:, :, 0], m[:, :, 1], m[:, :, 2]
    R = np.stack((r1, r2, np.cross(r1, r2)), 2)
    return R, -(R.transpose(0, 2, 1) @ t[..., None])[..., 0]


def _soft_template(meta, cell=.25, sigma=.5):
    origin = np.array([-6., -meta['width']/2-6])
    lines = np.zeros((int((meta['width']+12)/cell), int((meta['length']+12)/cell)), np.float32)
    for line in pitch.model_lines(meta):
        cv2.polylines(lines, [np.int32(np.rint((line-origin)/cell))], False, 1., 1)
    distance = cv2.distanceTransform(np.uint8(lines == 0)*255, cv2.DIST_L2, 5)*cell
    return np.exp(-(distance/sigma)**2/2), origin, 1/cell


def _paint_rank(H, points, template):
    surface, origin, scale = template
    h, w = surface.shape
    uniform = np.c_[points, np.ones(len(points))].astype(np.float32)
    out = []
    for chunk in np.array_split(H, len(H)//4000+1):
        world = np.linalg.inv(chunk).astype(np.float32) @ uniform.T
        z = world[:, 2]
        z = np.where(np.abs(z) < 1e-7, 1e-7, z)
        ahead = z*np.sign(np.median(z, axis=1, keepdims=True)) > 1e-9
        x = np.nan_to_num(((world[:, 0]/z-origin[0])*scale).clip(-1, w)).astype(np.int32)
        y = np.nan_to_num(((world[:, 1]/z-origin[1])*scale).clip(-1, h)).astype(np.int32)
        ahead &= (x >= 0) & (x < w) & (y >= 0) & (y < h)
        out.append(np.where(ahead, surface[y.clip(0, h-1), x.clip(0, w-1)], 0).mean(1))
    return np.concatenate(out)


def _model_tree(meta):
    points, normals, ids = [], [], []
    for i, feature in enumerate(pitch.model_segments(meta)):
        if feature[0] == 'line':
            n = max(2, int(np.linalg.norm(feature[2]-feature[1])/.5))
            sample = np.linspace(feature[1], feature[2], n)
            tangent = np.tile(feature[2]-feature[1], (n, 1))
        else:
            sample, tangent = feature[1], np.gradient(feature[1], axis=0)
        tangent = tangent/np.linalg.norm(tangent, axis=1, keepdims=True)
        points.append(sample)
        normals.append(np.c_[-tangent[:, 1], tangent[:, 0]])
        ids.append(np.full(len(sample), i))
    points = np.vstack(points)
    return {'points': points, 'normals': np.vstack(normals), 'ids': np.concatenate(ids),
            'tree': cKDTree(points), 'lines': pitch.model_lines(meta)}


def _associate(p, points, model, limit):
    H = _homography(p)
    distance, nearest = model['tree'].query(warp(points, np.linalg.inv(H)))
    near = distance < limit
    normal = model['normals'][nearest[near]]
    return points[near], np.c_[normal, -(normal*model['points'][nearest[near]]).sum(1)]


def _line_error(p, points, lines):
    # Pixel distance from each image point to its pitch line, seen through the camera p.
    seen = lines @ np.linalg.inv(_homography(p))
    return ((seen[:, 0]*points[:, 0]+seen[:, 1]*points[:, 1]+seen[:, 2])
            / np.maximum(np.hypot(seen[:, 0], seen[:, 1]), 1e-12))


def _refine_free(p, window, model, meta):
    # All seven camera parameters, with people as a weak prior: they settle the height and depth
    # that paint seen at a shallow angle leaves loose.
    points = _sample(window['strokes'], 7)
    for limit in (3., 2., 1.3, .8, .5):
        near, lines = _associate(p, points, model, limit)
        if len(near) < 12:
            break

        def residual(v):
            error = _line_error(v, near, lines)
            if len(window['feet']) < 4:
                return error
            people = _people_of(v, window, meta)[0]
            return np.r_[error, np.full(max(4, len(near)//12), 8*people)]

        p = least_squares(residual, p, loss='soft_l1', f_scale=2., max_nfev=25,
                          x_scale=[.02, .02, .02, .05, .5, .5, .2]).x
    return p


def _with_far_edge(near, lines, far, meta):
    # The edge is known to be the far touchline, so it counts however far off the model sits.
    # Paint is only tied within a metre or so, which could never pull that line back.
    if not len(far):
        return near, lines
    return np.vstack((near, far)), np.vstack((lines, np.tile([0., 1., meta['width']/2], (len(far), 1))))


def _refine_angles(p, points, model, free_focal=True, far=()):
    # Camera position held: rotation, and zoom if allowed, may still change.
    n = 4 if free_focal else 3
    for limit in (2., 1.3, .8, .5):
        near, lines = _with_far_edge(*_associate(p, points, model, limit), far, model['meta'])
        if len(near) < 12:
            break
        p = np.r_[least_squares(lambda v: _line_error(np.r_[v, p[n:]], near, lines), p[:n],
                                loss='soft_l1', f_scale=2., x_scale=.01, max_nfev=20).x, p[n:]]
    return p


def _stroke_cover(p, strokes, model, tolerance=5., shortest=90.):
    """Share of each distinct stroke that lies on a projected model line. A long touchline counts
    as one line, so it cannot outvote a shorter line left unexplained."""
    if not len(strokes):
        return 0.
    left, top, scale = -1600, -700, .5
    to_canvas = np.diag([scale, scale, 1.])
    to_canvas[:2, 2] = -left*scale, -top*scale
    canvas = np.full((1300, 2600), 255, np.uint8)
    H = to_canvas @ _homography(p)
    for line in model['lines']:
        projected = warp(line, H)
        projected = projected[np.isfinite(projected).all(1) & (np.abs(projected).max(1) < 20000)]
        if len(projected) > 1:
            cv2.polylines(canvas, [np.int32(np.rint(projected*8))], False, 0, 1, cv2.LINE_8, 3)
    distance = cv2.distanceTransform(canvas, cv2.DIST_L2, 5)/scale
    distinct, _ = _distinct(strokes)
    length = np.linalg.norm(distinct[:, :2]-distinct[:, 2:], axis=1)
    total = covered = 0.
    for stroke, l in zip(distinct[length >= shortest], length[length >= shortest]):
        n = max(4, int(l/12))
        along = stroke[:2]+(stroke[2:]-stroke[:2])*np.linspace(0, 1, n)[:, None]
        pixel = np.rint((along-[left, top])*scale).astype(int)
        inside = (pixel[:, 0] >= 0) & (pixel[:, 0] < 2600) & (pixel[:, 1] >= 0) & (pixel[:, 1] < 1300)
        covered += np.sqrt(l)*float((distance[pixel[inside, 1], pixel[inside, 0]] < tolerance).sum())/n
        total += np.sqrt(l)
    return covered/total if total else 0.


def _model_seen(p, window, model):
    # Painted fraction of the model lines in view, and how many lines of each direction show.
    H = _homography(p)
    points, ids = model['points'], model['ids']
    uniform = np.c_[points, np.ones(len(points))]
    count = int(ids.max())+1
    visible, painted = np.zeros(count), np.zeros(count)
    for to_base, lawn, loose in window['frames'].values():
        projected = (np.linalg.inv(to_base) @ H @ uniform.T).T
        front = projected[:, 2] > 1e-9
        pixel = np.rint(projected[:, :2]/np.where(front, projected[:, 2], 1)[:, None]/2).astype(int)
        front &= (pixel[:, 0] >= 0) & (pixel[:, 0] < 960) & (pixel[:, 1] >= 0) & (pixel[:, 1] < 540)
        pixel, which = pixel[front], ids[front]
        on_lawn = lawn[pixel[:, 1], pixel[:, 0]]
        pixel, which = pixel[on_lawn], which[on_lawn]
        wide = cv2.dilate(np.uint8(loose), np.ones((5, 5), np.uint8)) > 0
        np.add.at(visible, which, 1)
        np.add.at(painted, which[wide[pixel[:, 1], pixel[:, 0]]], 1)
    shown = visible >= 40
    fraction = painted/np.maximum(visible, 1)
    lines = [sum(shown[i] and fraction[i] > .3 for i in group) for group in (ALONG_LINES, ACROSS_LINES)]
    arcs = sum(shown[i] and fraction[i] > .3 for i in ARCS)
    return float(painted[shown].sum()/max(visible[shown].sum(), 1)), lines[0], lines[1], arcs


def _rate(p, window, model, meta):
    cover = _stroke_cover(p, window['strokes'], model)
    seen, along, across, arcs = _model_seen(p, window, model)
    error, on_pitch = _people_of(p, window, meta)
    score = (cover**2*seen**.5*min(along, 3)*min(across, 3)*(1+.1*min(arcs, 2))
             * float(_people_weight(error, on_pitch)))
    return {'p': p, 'cover': cover, 'seen': seen, 'people': error, 'on_pitch': on_pitch, 'score': score}


def _read_window(F, cumulative, boxes, frames, base):
    window = {'frames': {}, 'base': base}
    white, every, feet, seen, far = [], [], [], [], []
    for k in frames:
        to_base = cumulative[base] @ np.linalg.inv(cumulative[k])
        found = boxes[k] if boxes is not None else np.empty((0, 5))
        strokes, lawn, loose = _strokes(F[k], found, False)
        coloured, _, _ = _strokes(F[k], found, True)
        edge = _far_edge(F[k], found)
        strokes = np.vstack((strokes, edge))
        if len(edge):
            far.append(warp(_sample(edge, 16), to_base))
        if len(strokes):
            white.append(warp(strokes.reshape(-1, 2), to_base).reshape(-1, 4))
        if len(coloured):
            every.append(warp(coloured.reshape(-1, 2), to_base).reshape(-1, 4))
        window['frames'][k] = to_base, lawn, loose
        a, b = _people(found, to_base)
        feet.append(a)
        seen.append(b)
    window['strokes'] = np.vstack(white) if white else np.empty((0, 4))
    window['every'] = np.vstack(every) if every else np.empty((0, 4))
    window['feet'], window['seen'] = np.vstack(feet), np.concatenate(seen)
    window['far'] = np.vstack(far) if far else np.empty((0, 2))
    return window


def _search_window(window, model, meta):
    """Candidate cameras for one window, best first."""
    strokes, feet, seen = window['strokes'], window['feet'], window['seen']
    if len(strokes) < 3:
        return []
    lines, weight = _distinct(np.vstack((strokes, window['every'])))
    lines = lines[np.argsort(-weight)[:STROKE_LINES]]
    if len(lines) < 4:
        return []
    H, focal = _cameras_from_lines(lines, meta)
    if not len(H):
        return []
    R, centre = _poses(H, focal)
    height = -centre[:, 2]
    inside = ((centre[:, 0] > -1) & (centre[:, 0] < meta['length']+1)
              & (np.abs(centre[:, 1]) < meta['width']/2+1))
    good = ((height > MIN_HEIGHT) & (height < MAX_HEIGHT) & (np.abs(centre[:, 1]) < 170)
            & (centre[:, 0] > -120) & (centre[:, 0] < 220) & ~inside & (np.linalg.det(R) > 0))
    H, focal, R, centre = H[good], focal[good], R[good], centre[good]
    if not len(H):
        return []
    rank = _paint_rank(H, _sample(strokes, 11), _soft_template(meta))
    if len(feet) >= 4:
        parts = [_people_fit(R[i], centre[i], focal[i], feet, seen, meta)
                 for i in np.array_split(np.arange(len(R)), len(R)//3000+1)]
        rank = rank*(.15+_people_weight(np.concatenate([a for a, _ in parts]),
                                        np.concatenate([b for _, b in parts])))
    found, used = [], []
    for i in np.argsort(-rank):
        if len(used) >= STARTS:
            break
        if any(np.linalg.norm(centre[i]-c) < 1.5 for c in used):
            continue
        used.append(centre[i])
        p = _refine_free(_parameters(R[i], centre[i], focal[i]), window, model, meta)
        found.append(_rate(p, window, model, meta))
    return sorted(found, key=lambda r: -r['score'])


def _credible(r):
    p = r['p']
    return (MIN_FOCAL <= np.exp(p[3]) <= MAX_FOCAL and MIN_HEIGHT <= p[6] <= MAX_HEIGHT
            and abs(p[2]) < .2 and r['on_pitch'] >= .75 and abs(r['people']) < .25)


def _vote(results, meta):
    """Group solutions by camera position. A true position recurs across windows."""
    entries = sorted(((w, _camera_side(r['p'], meta), r['score']) for w, found in results.items()
                      for r in found if _credible(r)), key=lambda e: -e[2])
    groups = []
    for w, p, score in entries:
        for group in groups:
            if (np.linalg.norm(group['position'][:2]-p[4:6]) < SAME_POSITION_M
                    and abs(group['position'][2]-p[6]) < SAME_HEIGHT_M):
                group['members'].append((w, p, score))
                break
        else:
            groups.append({'position': p[4:7].copy(), 'members': [(w, p, score)]})
    for group in groups:
        best = {}
        for w, p, score in group['members']:
            if w not in best or score > best[w][1]:
                best[w] = p, score
        weight = np.array([s for _, s in best.values()])
        group['best'] = best
        group['votes'] = float(weight.sum())
        group['position'] = (np.array([p[4:7] for p, _ in best.values()])*weight[:, None]).sum(0)/weight.sum()
    return sorted(groups, key=lambda g: -g['votes'])


def focal_from_motion(cumulative, frames, gap=30, step=15):
    """Focal length from pure rotation alone, using no paint. None for a clip that barely pans.

    Paint fixes focal length poorly when the camera is nearly level; the pan pins it to about 8%.
    """
    to_centre = np.array([[1, 0, -CENTRE[0]], [0, 1, -CENTRE[1]], [0, 0, 1.]])
    motions = []
    for a in range(int(frames[0]), int(frames[-1])-gap, step):
        M = cumulative[a+gap] @ np.linalg.inv(cumulative[a])
        if np.linalg.norm(warp([CENTRE], M)[0]-CENTRE) > 25:
            motions.append(to_centre @ M @ np.linalg.inv(to_centre))
    if len(motions) < 4:
        return None
    motions = np.array(motions)
    focals = np.geomspace(700, 4000, 100)
    cost = []
    for f in focals:
        R = np.diag([1/f, 1/f, 1.]) @ motions @ np.diag([f, f, 1.])
        R = R/np.cbrt(np.linalg.det(R))[:, None, None]
        cost.append(((R.transpose(0, 2, 1) @ R-np.eye(3))**2).sum(axis=(1, 2)))
    cost = np.array(cost)
    # Hand shake and tracking faults break the rotation model on some links: use the better half.
    fit = cost.min(0)
    clean = fit <= np.percentile(fit, 50)
    if fit[clean].mean() > .004:
        return None
    return float(focals[np.argmin(cost[:, clean].sum(1))])


def _search_angles(p, template, points):
    # Position and focal length known: only three angles remain, so try them all near p.
    f = np.exp(p[3])
    K = np.array([[f, 0, CENTRE[0]], [0, f, CENTRE[1]], [0, 0, 1.]])
    centre = np.array([p[4], p[5], -p[6]])
    grid = [(a, b, c) for a in np.deg2rad(np.linspace(-9, 9, 19))
            for b in np.deg2rad(np.linspace(-3.5, 3.5, 8)) for c in np.deg2rad(np.linspace(-2.5, 2.5, 6))]
    H = []
    for a, b, c in grid:
        R = _rotation(p[0]+a, p[1]+b, p[2]+c)
        H.append(K @ np.c_[R[:, :2], -R @ centre])
    H = np.array(H)
    H = H/H[:, 2:3, 2:3]
    a, b, c = grid[int(np.argmax(_paint_rank(H, points, template)))]
    return np.r_[p[0]+a, p[1]+b, p[2]+c, p[3:]]


def _paint_points(window):
    pieces = [_sample(window['strokes'], 7)]
    for to_base, lawn, loose in window['frames'].values():
        y, x = np.nonzero(loose & lawn)
        if len(y):
            keep = np.arange(0, len(y), max(1, len(y)//1500))
            pieces.append(warp(np.c_[x[keep], y[keep]]*2., to_base))
    return np.vstack(pieces)


def _angles_from(H, focal, position):
    M = np.linalg.inv(np.array([[focal, 0, CENTRE[0]], [0, focal, CENTRE[1]], [0, 0, 1.]])) @ H
    M = M/np.linalg.norm(M[:, 0])*np.sign(M[2, 2])
    u, _, vt = np.linalg.svd(np.c_[M[:, 0], M[:, 1], np.cross(M[:, 0], M[:, 1])])
    return _parameters(u @ vt, np.array([position[0], position[1], -position[2]]), focal)


def _solve_keys(windows, group, cumulative, focal, fixed_focal, model):
    """Windows sharing one position: search the angles of each, then re-fit the shared values."""
    position = group['position']
    template = _soft_template(model['meta'], sigma=.8)
    points = {w: _paint_points(window) for w, window in windows.items()}
    keys = {}
    for _ in range(2):
        keys = {}
        order = sorted(windows, key=lambda w: -group['best'].get(w, (0, 0))[1])
        for w in order:
            if w in group['best']:
                start = group['best'][w][0]
            elif keys:
                source = min(keys, key=lambda k: abs(k-w))
                start = _angles_from(pitch.chain(cumulative, source, w, _homography(keys[source])),
                                     focal, position)
            else:
                continue
            p = np.r_[start[:3], np.log(focal), position]
            if len(windows[w]['strokes']) >= 2:
                p = _search_angles(p, template, _sample(windows[w]['strokes'], 11))
                p = _refine_angles(p, points[w], model, far=windows[w]['far'])
            keys[w] = p
        tied = []
        for w, p in keys.items():
            if len(points[w]):
                near, lines = _associate(p, points[w], model, 1.)
                near, lines = _with_far_edge(near[::3], lines[::3], windows[w]['far'], model['meta'])
                if len(near) >= 12:
                    tied.append((p, near, lines))
        if not tied:
            break
        if fixed_focal:
            position = least_squares(
                lambda c: np.concatenate([_line_error(np.r_[p[:4], c], a, b) for p, a, b in tied]),
                position, loss='soft_l1', f_scale=2., x_scale=[.5, .5, .2], max_nfev=20).x
        else:
            shared = least_squares(
                lambda c: np.concatenate([_line_error(np.r_[p[:3], c[3], c[:3]], a, b) for p, a, b in tied]),
                np.r_[position, np.log(focal)], loss='soft_l1', f_scale=2.,
                x_scale=[.5, .5, .2, .03], max_nfev=20).x
            position, focal = shared[:3], float(np.exp(shared[3]))
    return {w: _refine_angles(np.r_[p[:3], np.log(focal), position], points[w], model, far=windows[w]['far']) if len(points[w])
            else np.r_[p[:3], np.log(focal), position] for w, p in keys.items()}, position, focal


def minimum_height(position, focal, meta):
    """Lowest camera, in metres, that keeps feet on the far touchline within the allowed error."""
    far = abs(position[1])+meta['width']/2
    if not 0 <= position[0] <= meta['length']:  # Behind a goal it looks down the whole pitch.
        far = max(far, abs(position[0]-meta['length']/2)+meta['length']/2)
    return max(LOWEST_CAMERA_M, GROUND_ERROR_GAIN*far**2/(MAX_GROUND_ERROR_M*focal))


def _rejection(keys, windows, gap, people, on_pitch, height, lowest=0.):
    """Why a solved camera should not be used, or None. A refusal sends the clip to manual marks."""
    if height < lowest:
        return f'카메라가 낮아요 ({height:.1f} m, 이 위치와 화각의 최소 높이 {lowest:.1f} m)'
    if keys < min(2, windows):
        return f'페인트선이 설명되는 구간이 부족해요 ({keys}개)'
    if gap > MAX_KEY_GAP:
        return f'구간끼리 어긋나요 ({gap:.0f} px)'
    if abs(people) > MAX_PEOPLE_ERROR:
        return f'사람 키가 맞지 않아요 ({people:+.2f})'
    if on_pitch < MIN_ON_PITCH:
        return f'선수가 경기장 밖에 놓여요 ({on_pitch:.0%})'
    return None


def solve(F, cumulative, meta, boxes, frames, log=print):
    """Pitch-to-image homographies at key frames of one connected shot, or None if not credible."""
    frames = np.asarray(frames)
    count = len(frames)
    reach = int(min(30, (count-1)//2))
    starts = np.unique(np.linspace(frames[0]+reach, frames[-1]-reach, int(np.clip(count/45, 1, 9))).astype(int))
    model = _model_tree(meta)
    model['meta'] = meta
    windows, results = {}, {}
    for w in map(int, starts):
        sample_frames = np.unique(np.linspace(w-reach, w+reach+(count-1) % 2*(reach == 0), 5).astype(int))
        windows[w] = _read_window(F, cumulative, boxes, sample_frames, w)
        results[w] = _search_window(windows[w], model, meta)
        log(f'카메라 위치 찾는 중 {len(windows)}/{len(starts)}')
    groups = _vote(results, meta)
    if not groups:
        return None
    group = groups[0]
    motion_focal = focal_from_motion(cumulative, frames)
    paint_focal = float(np.median([np.exp(p[3]) for p, _ in group['best'].values()]))
    trials = [motion_focal*x for x in (.92, .96, 1., 1.04, 1.08)] if motion_focal else [paint_focal]
    best = None
    for focal in trials:
        keys, position, focal = _solve_keys(windows, group, cumulative, focal, motion_focal is not None, model)
        rated = {w: _rate(p, windows[w], model, meta) for w, p in keys.items() if len(windows[w]['strokes'])}
        explained = sum(r['cover']**2 for r in rated.values())
        if best is None or explained > best[0]:
            best = explained, rated, position, focal
    _, rated, position, focal = best
    strong = {w: _homography(r['p']) for w, r in rated.items() if r['cover'] >= MIN_COVER}
    people = float(np.median([r['people'] for r in rated.values()])) if rated else 9.
    on_pitch = float(np.median([r['on_pitch'] for r in rated.values()])) if rated else 0.
    # Keys of one camera must agree through the camera motion. A wrong model fits each window on
    # its own terms, so a static camera can disagree with itself by hundreds of pixels.
    grid = pitch.GRID*[meta['length']/100, meta['width']/60]
    ordered = sorted(strong)
    gaps = [pitch._gap(pitch.chain(cumulative, a, b, strong[a]), strong[b], 0, grid)
            for a, b in zip(ordered, ordered[1:])]
    worst = max((g for g in gaps if np.isfinite(g)), default=0.)
    problem = _rejection(len(strong), len(windows), worst, people, on_pitch, position[2],
                         minimum_height(position, focal, meta))
    if problem:
        log(f'카메라 후보를 믿을 수 없어요: {problem}')
        return None
    log(f'카메라 위치 ({position[0]:.1f}, {position[1]:.1f}) 높이 {position[2]:.1f} m, 초점거리 {focal:.0f} px')
    return strong
