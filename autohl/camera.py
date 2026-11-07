import numpy as np,cv2
from scipy.linalg import fractional_matrix_power
from scipy.spatial.transform import Rotation
MIN_LINK_INLIERS=40

def motion_segments(inliers,count):
 # inliers[i] measures the link into frame i. Frame 0 has no preceding link.
 segments=np.zeros(count,dtype=int)
 if inliers is not None:segments[1:]=np.cumsum(inliers[1:]<MIN_LINK_INLIERS)
 return segments
# Frame-to-frame homographies for a panning/zooming camera. From a fixed spot (tripod or a
# hand-held phone that mostly rotates) one homography maps the whole scene between frames.
# CUM[i] maps frame 0 into frame i.
def _feats(orb,img,sky):
 g=cv2.cvtColor(np.ascontiguousarray(img),cv2.COLOR_BGR2GRAY);m=np.full(g.shape,255,np.uint8);m[:sky]=0
 return orb.detectAndCompute(g,m)
def sky_cut(F):
 # Floodlight halos and clouds move with exposure; skip the image rows above the pitch.
 # The turf is the large green band, so start a little above its first rows.
 hsv=cv2.cvtColor(np.ascontiguousarray(F[len(F)//2]),cv2.COLOR_BGR2HSV)
 green=((hsv[...,0]>25)&(hsv[...,0]<95)&(hsv[...,1]>40)).mean(1)
 rows=np.nonzero(green>.35)[0]
 return int(max(0,(rows[0] if len(rows) else 300)-80))
def plane_mask(img,sky):
 # Everything below the sky cut. From a tripod the camera only rotates, so fence, trees and
 # buildings move exactly like the pitch; the sparkle of floodlit artificial turf changes with
 # the viewing angle and would bias a turf-only match over a long pan.
 # The fence, trees and buildings above the turf carry most of the stable texture, so the cut sits
 # well above the turf edge; floodlights and their halos (blown-out pixels) are masked instead.
 m=np.full(img.shape[:2],255,np.uint8);m[:max(0,sky-260)]=0
 hot=(cv2.cvtColor(np.ascontiguousarray(img),cv2.COLOR_BGR2GRAY)>245).astype(np.uint8)
 m[cv2.dilate(hot,np.ones((41,41),np.uint8))>0]=0;return m
def solve(F,log=print):
 # Scene homographies between frames. Each frame is matched straight to the current keyframe, so
 # error only accumulates once per keyframe, not once per frame.
 N=len(F);sky=sky_cut(F)
 orb=cv2.ORB_create(5000,fastThreshold=8);bf=cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)
 def feats(i):
  g=cv2.cvtColor(np.ascontiguousarray(F[i]),cv2.COLOR_BGR2GRAY);return orb.detectAndCompute(g,plane_mask(F[i],sky))
 def match(A,B):
  if A[1] is None or B[1] is None:return None,0
  m=bf.match(A[1],B[1])
  if len(m)<15:return None,0
  a=np.float32([A[0][x.queryIdx].pt for x in m]);b=np.float32([B[0][x.trainIdx].pt for x in m])
  Hm,ok=cv2.findHomography(a,b,cv2.RANSAC,2.0,maxIters=4000,confidence=.999)
  if Hm is None or not np.isfinite(Hm).all() or abs(Hm[2,2])<1e-9:return None,0
  keep=ok.ravel().astype(bool);n=int(keep.sum())
  if n<MIN_LINK_INLIERS:return None,0
  for pts in (a[keep],b[keep]):
   if cv2.contourArea(cv2.convexHull(pts))<15000:return None,0
  return Hm,n
 cum=[np.eye(3)];inl=[0];K=0;KF=feats(0);prev=None
 for i in range(1,N):
  cur=feats(i);Hm,n=match(KF,cur)
  if Hm is None or n<250 or i-K>45:
   # Promote the previous frame to keyframe and match against it instead.
   if i-1!=K:K=i-1;KF=prev;Hm,n=match(KF,cur)
  if Hm is None:
   Hm=cum[-1].copy();n=0
   K=i;KF=cur  # Subsequent links start in the new view, never through the failed match.
  else:
   Hm=Hm@cum[K]
   if not np.isfinite(Hm).all() or abs(Hm[2,2])<1e-9:
    Hm=cum[-1].copy();n=0;K=i;KF=cur
  cum.append(Hm/Hm[2,2]);inl.append(n);prev=cur
  if i%60==0:log(f'카메라 움직임 분석 {i}/{N}')
 return np.array(cum),np.array(inl),sky
def warp(p,M):return cv2.perspectiveTransform(np.float32(p).reshape(-1,1,2),M).reshape(-1,2)


def refine_ground_motion(frames, cumulative, boxes, reference, indices, focus, ground_mask=None):
    """Correct camera motion locally using static turf texture near a ground trajectory."""
    gray = cv2.cvtColor(frames[reference], cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    to_reference = {k: cumulative[reference] @ np.linalg.inv(cumulative[k]) for k in indices}
    targets = {k: warp([focus[k]], H)[0] for k, H in to_reference.items()}
    if not targets or not np.isfinite(list(targets.values())).all():
        return {}
    x, y = np.mean(list(targets.values()), axis=0)
    left, top = max(0, int(x-390)), max(0, int(y-235))
    right, bottom = min(width, int(x+390)), min(height, int(y+210))
    mask = np.zeros_like(gray)
    mask[top:bottom, left:right] = 255
    if ground_mask is not None:
        mask[~np.asarray(ground_mask, dtype=bool)] = 0
    for x1, y1, x2, y2, *_ in boxes[reference].astype(int):
        mask[max(0, y1-8):min(height, y2+8), max(0, x1-8):min(width, x2+8)] = 0
    points = cv2.goodFeaturesToTrack(gray, 300, .02, 12, mask=mask, blockSize=7)
    if points is None or len(points) < 45:
        return {}
    points = points.reshape(-1, 2)
    lk = dict(winSize=(21, 21), maxLevel=2,
              criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 30, .03))
    result = {}
    for k, H in to_reference.items():
        if k == reference:
            continue
        current = cv2.cvtColor(frames[k], cv2.COLOR_BGR2GRAY)
        stabilized = cv2.warpPerspective(current, H, (width, height))
        moved, forward, _ = cv2.calcOpticalFlowPyrLK(gray, stabilized, points, None, **lk)
        back, backward, _ = cv2.calcOpticalFlowPyrLK(stabilized, gray, moved, None, **lk)
        observed = warp(moved, np.linalg.inv(H))
        keep = (forward.ravel().astype(bool) & backward.ravel().astype(bool)
                & (np.linalg.norm(points-back, axis=1) < .75)
                & (np.linalg.norm(points-moved, axis=1) < 12))
        for x1, y1, x2, y2, *_ in boxes[k]:
            keep &= ~((observed[:, 0] > x1-8) & (observed[:, 0] < x2+8)
                      & (observed[:, 1] > y1-8) & (observed[:, 1] < y2+8))
        train = keep & (np.arange(len(points)) % 3 != 0)
        test = keep & ~train
        if train.sum() < 30 or test.sum() < 15:
            continue
        affine, inliers = cv2.estimateAffine2D(points[train], moved[train],
                                               ransacReprojThreshold=1.)
        if affine is None or inliers.mean() < .6:
            continue
        hull = cv2.convexHull(points[train][inliers.ravel() != 0])
        if cv2.pointPolygonTest(hull, tuple(map(float, targets[k])), True) < -25:
            continue
        predicted = cv2.transform(points[test, None, :], affine)[:, 0]
        old = float(np.median(np.linalg.norm(points[test]-moved[test], axis=1)))
        new = float(np.median(np.linalg.norm(predicted-moved[test], axis=1)))
        shift = np.linalg.norm(affine @ np.r_[targets[k], 1]-targets[k])
        accepted = bool(old-new > .35 and new < 1.1 and shift < 12)
        result[k] = {'reference_to_image': np.linalg.inv(H) @ np.vstack((affine, [0, 0, 1]))
                     if accepted else np.linalg.inv(H), 'accepted': accepted,
                     'before_px': old, 'after_px': new if accepted else old,
                     'held_out': int(test.sum())}
    return result


def repair_cumulative(cumulative, tol_px=8.0):
    """Repair single-frame tracking spikes in cumulative camera homographies while preserving fast pans."""
    N = len(cumulative)
    if N < 3:
        return np.array(cumulative, dtype=float).copy()
    c = np.array(cumulative, dtype=float).copy()
    grid = np.array([[960., 540.], [300., 300.], [1620., 300.], [300., 780.], [1620., 780.]])
    for _ in range(3):
        spikes = []
        for k in range(1, N - 1):
            s_prev = c[k] @ np.linalg.inv(c[k-1])
            s_next = c[k+1] @ np.linalg.inv(c[k])
            s_direct = c[k+1] @ np.linalg.inv(c[k-1])
            d_prev = warp(grid, s_prev) - grid
            d_next = warp(grid, s_next) - grid
            d_direct = warp(grid, s_direct) - grid
            det = np.linalg.det(s_direct)
            if det <= 0:
                continue
            s_norm = s_direct / (det ** (1.0 / 3.0))
            s_half = np.real(fractional_matrix_power(s_norm, 0.5))
            H_mid = s_half @ c[k-1]
            H_mid /= H_mid[2, 2]
            dev_from_mid = np.linalg.norm(warp(grid, c[k]) - warp(grid, H_mid), axis=1).max()
            norm_direct = np.linalg.norm(d_direct, axis=1).max()
            norm_prev = np.linalg.norm(d_prev, axis=1).max()
            norm_next = np.linalg.norm(d_next, axis=1).max()
            if dev_from_mid > tol_px and (norm_prev + norm_next > norm_direct + tol_px):
                spikes.append((k, H_mid))
        if not spikes:
            break
        for k, H_mid in spikes:
            c[k] = H_mid
    return c


def refine_turf_motion(frames, boxes, cumulative, sky):
    """Refine cumulative camera motion using consecutive turf optical flow with player masking."""
    N = len(frames)
    if N < 2 or cumulative is None or len(cumulative) != N:
        return cumulative
    new_cum = [np.array(cumulative[0], dtype=float)]
    grid = np.array([[960., 540.], [300., 300.], [1620., 780.]])
    for k in range(N - 1):
        a = cv2.cvtColor(np.asarray(frames[k]), cv2.COLOR_BGR2GRAY)
        b = cv2.cvtColor(np.asarray(frames[k+1]), cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(np.asarray(frames[k]), cv2.COLOR_BGR2HSV)
        turf = ((hsv[..., 0] >= 25) & (hsv[..., 0] <= 95) & (hsv[..., 1] >= 30) & (hsv[..., 2] >= 30))
        mask = np.zeros(a.shape, np.uint8)
        mask[max(sky, 0):] = 255
        mask &= (turf.astype(np.uint8) * 255)
        if boxes is not None and k < len(boxes):
            for x0, y0, x1, y1, *_ in boxes[k]:
                mask[max(int(y0)-8, 0):min(int(y1)+8, a.shape[0]),
                     max(int(x0)-8, 0):min(int(x1)+8, a.shape[1])] = 0
        step_cam = cumulative[k+1] @ np.linalg.inv(cumulative[k])
        p = cv2.goodFeaturesToTrack(a, 1500, .005, 10, mask=mask)
        step = None
        if p is not None and len(p) >= 30:
            q, found, _ = cv2.calcOpticalFlowPyrLK(a, b, p, None, winSize=(31, 31), maxLevel=4)
            back, returned, _ = cv2.calcOpticalFlowPyrLK(b, a, q, None, winSize=(31, 31), maxLevel=4)
            keep = (found[:, 0] == 1) & (returned[:, 0] == 1) & (np.linalg.norm(back-p, axis=2)[:, 0] < .7)
            p, q = p[keep, 0], q[keep, 0]
            if len(p) >= 20:
                H, inlier = cv2.findHomography(p, q, cv2.RANSAC, 1.5)
                if H is not None and inlier is not None and inlier.sum() >= 20:
                    det = np.linalg.det(H)
                    if 0.5 < det < 2.0:
                        diff = np.linalg.norm(warp(grid, H) - warp(grid, step_cam), axis=1).max()
                        if diff <= 4.0:
                            step = H
        if step is None:
            step = step_cam
        nxt = step @ new_cum[-1]
        new_cum.append(nxt / nxt[2, 2])
    return np.array(new_cum)


