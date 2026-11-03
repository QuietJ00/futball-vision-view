import numpy as np,cv2
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
  return (Hm,int(ok.sum())) if Hm is not None else (None,0)
 cum=[np.eye(3)];inl=[0];K=0;KF=feats(0);prev=None
 for i in range(1,N):
  cur=feats(i);Hm,n=match(KF,cur)
  if Hm is None or n<250 or i-K>45:
   # Promote the previous frame to keyframe and match against it instead.
   if i-1!=K:K=i-1;KF=prev;Hm,n=match(KF,cur)
  if Hm is None:Hm=(cum[-1]@np.linalg.inv(cum[-2]) if i>1 else np.eye(3))@cum[-1];n=0  # lost: keep the last motion
  else:Hm=Hm@cum[K]
  cum.append(Hm/Hm[2,2]);inl.append(n);prev=cur
  if i%60==0:log(f'카메라 움직임 분석 {i}/{N}')
 return np.array(cum),np.array(inl),sky
def warp(p,M):return cv2.perspectiveTransform(np.float32(p).reshape(-1,1,2),M).reshape(-1,2)
