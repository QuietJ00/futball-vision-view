import numpy as np,cv2
MIN_LINK_INLIERS=40

def distort(points,k1,focal):
 # Pinhole pixels -> lens pixels, one radial term about the image centre.
 p=np.asarray(points,float)
 if not k1:return p
 x=(p-[960.,540.])/focal
 return np.array([960.,540.])+x*(1+k1*(x*x).sum(1,keepdims=True))*focal

def undistort(points,k1,focal):
 # Lens pixels -> pinhole pixels. Newton on x(1+k1 x^2)=r: a plain fixed-point loop diverges
 # toward the frame corners.
 p=np.asarray(points,float)
 if not k1:return p
 d=(p-[960.,540.])/focal;r=np.linalg.norm(d,axis=1);x=r.copy()
 for _ in range(8):x=x-(x*(1+k1*x*x)-r)/np.maximum(1+3*k1*x*x,.2)
 return np.array([960.,540.])+d*(np.where(r>1e-9,x/np.maximum(r,1e-9),1.))[:,None]*focal

def pinhole_motion(cum,k1,focal):
 # Camera motion measured on lens pixels, restated for pinhole pixels link by link.
 if not k1:return cum
 y,x=np.mgrid[60:1040:140,60:1880:180];grid=np.c_[x.ravel(),y.ravel()].astype(float);straight=undistort(grid,k1,focal)
 out=[np.eye(3)]
 for a,b in zip(cum,cum[1:]):
  moved=warp(grid,b@np.linalg.inv(a));keep=np.isfinite(moved).all(1)
  H,_=cv2.findHomography(straight[keep],undistort(moved[keep],k1,focal),0)
  out.append((H if H is not None else np.eye(3))@out[-1])
 return np.stack(out)

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
