import glob,json,os
import numpy as np,cv2
from .camera import distort, motion_segments, undistort, warp
# Pitch model: X along the pitch from the left goal line (metres), Y across it, +Y toward the
# camera's touchline. A venue preset keeps a few reference frames from a calibrated clip with
# their pitch->image homographies, so a new clip shot from the same spot calibrates itself.
VENUES=os.path.join(os.path.dirname(os.path.abspath(__file__)),'venues')

def landmarks(L,Wd,meta=None):
 # Points a user can click for manual calibration, at the venue's painted sizes.
 w=Wd/2;l=L;d=dims(meta or {});bd,bw,gd,gw,sp=d['box_depth'],d['box_half'],d['ga_depth'],d['ga_half'],d['spot']
 hx=(meta or {}).get('half_x',l/2)
 return {'왼쪽 골라인·가까운 터치라인 모서리':(0,w),'왼쪽 골라인·먼 터치라인 모서리':(0,-w),
  '오른쪽 골라인·가까운 터치라인 모서리':(l,w),'오른쪽 골라인·먼 터치라인 모서리':(l,-w),
  '하프라인·가까운 터치라인':(hx,w),'하프라인·먼 터치라인':(hx,-w),'센터 스폿':(hx,0),
  '센터서클 가까운 쪽 끝':(hx,9.15),'센터서클 먼 쪽 끝':(hx,-9.15),
  '왼쪽 페널티박스 가까운 앞모서리':(bd,bw),'왼쪽 페널티박스 먼 앞모서리':(bd,-bw),
  '왼쪽 페널티박스 가까운 골라인 점':(0,bw),'왼쪽 페널티박스 먼 골라인 점':(0,-bw),
  '왼쪽 골에어리어 가까운 앞모서리':(gd,gw),'왼쪽 골에어리어 먼 앞모서리':(gd,-gw),
  '왼쪽 페널티 스폿':(sp,0),
  '오른쪽 페널티박스 가까운 앞모서리':(l-bd,bw),'오른쪽 페널티박스 먼 앞모서리':(l-bd,-bw),
  '오른쪽 페널티박스 가까운 골라인 점':(l,bw),'오른쪽 페널티박스 먼 골라인 점':(l,-bw),
  '오른쪽 골에어리어 가까운 앞모서리':(l-gd,gw),'오른쪽 골에어리어 먼 앞모서리':(l-gd,-gw),
  '오른쪽 페널티 스폿':(l-sp,0)}

STD_DIMS={'box_depth':16.5,'box_half':20.16,'ga_depth':5.5,'ga_half':9.16,'spot':11.}
def dims(meta):return {k:float(meta.get(k,v)) for k,v in STD_DIMS.items()}
def model_lines(meta):
 # Painted lines of the pitch in metres. Box and goal-area sizes come from the venue when it
 # has measured them (amateur pitches often paint smaller boxes), else the laws' values.
 L,Wd=meta['length'],meta['width'];hx=meta.get('half_x',L/2);w=Wd/2;out=[];d=dims(meta)
 bd,bw,gd,gw,sp=d['box_depth'],d['box_half'],d['ga_depth'],d['ga_half'],d['spot']
 seg=lambda a,b:np.stack([np.linspace(a[0],b[0],80),np.linspace(a[1],b[1],80)],1)
 for a,b in [((0,-w),(0,w)),((0,w),(L,w)),((0,-w),(L,-w)),((L,-w),(L,w)),((hx,-w),(hx,w))]:out.append(seg(a,b))
 for x0,s in [(0,1),(L,-1)]:
  for a,b in [((0,bw),(bd,bw)),((bd,bw),(bd,-bw)),((bd,-bw),(0,-bw)),((0,gw),(gd,gw)),((gd,gw),(gd,-gw)),((gd,-gw),(0,-gw))]:
   out.append(seg((x0+s*a[0],a[1]),(x0+s*b[0],b[1])))
 t=np.linspace(0,2*np.pi,200);out.append(np.stack([hx+9.15*np.cos(t),9.15*np.sin(t)],1))
 # Penalty arcs: the part of the 9.15 m circle round each spot that lies outside the box. They fix
 # the box depth against its width, which straight lines alone leave loose.
 a=np.arccos(np.clip((bd-sp)/9.15,-1,1));t=np.linspace(-a,a,60)
 out.append(np.stack([sp+9.15*np.cos(t),9.15*np.sin(t)],1));out.append(np.stack([L-sp-9.15*np.cos(t),9.15*np.sin(t)],1))
 return out

def venues():
 out={}
 for m in sorted(glob.glob(os.path.join(VENUES,'*','meta.json'))):
  d=json.load(open(m));d['id']=os.path.basename(os.path.dirname(m));out[d['id']]=d
 return out
def venue_refs(vid):
 d=os.path.join(VENUES,vid);refs=[]
 for p in sorted(glob.glob(os.path.join(d,'ref_*.jpg'))):
  refs.append((cv2.imread(p),np.load(p[:-4]+'.npy')))
 return refs

_sift=None
def turf_mask(img,sky=0):
 # The playing surface: the big green region (with the paint on it) below the sky cut. Only
 # features on the pitch plane give the pitch homography; the fence, trees and buildings sit
 # at other depths and shift by parallax whenever the camera is not exactly where it was.
 hsv=cv2.cvtColor(np.ascontiguousarray(img),cv2.COLOR_BGR2HSV)
 g=((hsv[...,0]>28)&(hsv[...,0]<95)&(hsv[...,1]>35)&(hsv[...,2]>60)).astype(np.uint8)*255;g[:sky]=0
 g=cv2.morphologyEx(g,cv2.MORPH_CLOSE,np.ones((25,25),np.uint8))
 n,lb,st,_=cv2.connectedComponentsWithStats(g)
 if n<2:return g
 k=1+int(np.argmax(st[1:,cv2.CC_STAT_AREA]));m=(lb==k).astype(np.uint8)*255
 # Fill holes (players, paint) inside the pitch region, then shave the far edge (fence foot).
 cs,_=cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);m=np.zeros_like(m);cv2.drawContours(m,cs,-1,255,-1)
 return cv2.erode(m,np.ones((9,9),np.uint8))
def _feats(img,sky):
 global _sift
 if _sift is None:_sift=cv2.SIFT_create(4000,contrastThreshold=.02)
 g=cv2.cvtColor(np.ascontiguousarray(img),cv2.COLOR_BGR2GRAY)
 return _sift.detectAndCompute(g,turf_mask(img,sky))
def _match_features(a,b):
 # Restrict matches to the playing surface and require support over an area, rather than
 # accepting a homography estimated along a single painted stripe.
 if a[1] is None or b[1] is None or len(a[0])<20 or len(b[0])<20:return None,0
 mm=cv2.BFMatcher(cv2.NORM_L2).knnMatch(a[1],b[1],k=2)
 good=[m for m,n in (x for x in mm if len(x)==2) if m.distance<.8*n.distance]
 if len(good)<20:return None,0
 pa=np.float32([a[0][m.queryIdx].pt for m in good]);pb=np.float32([b[0][m.trainIdx].pt for m in good])
 Hm,ok=cv2.findHomography(pa,pb,cv2.RANSAC,3.,maxIters=8000,confidence=.999)
 if Hm is None or not np.isfinite(Hm).all():return None,0
 use=ok.ravel().astype(bool);n=int(use.sum())
 if n<MIN_INL or n/len(good)<.35:return None,n
 for pts in (pa[use],pb[use]):
  if cv2.contourArea(cv2.convexHull(pts))<15000:return None,n
 return Hm,n

def match_ref(img,ref,sky):
 return _match_features(_feats(ref,sky),_feats(img,sky))

def _line_kernels(L=31,off=4,n=24):
 # Oriented centre-surround bars: a thin bright line along the kernel scores high, scattered
 # glare on the turf averages out, broad bright patches cancel against the side bands.
 ks=[]
 for i in range(n):
  a=np.pi*i/n;k=np.zeros((L+2*off+2,)*2,np.float32);c=(k.shape[0]-1)/2;d=np.array([np.cos(a),np.sin(a)]);nn=np.array([-d[1],d[0]])
  for t in np.linspace(-L/2,L/2,L*2):
   for s_,w in [(0,1.),(off,-.5),(-off,-.5)]:
    x,y=c+d[0]*t+nn[0]*s_,c+d[1]*t+nn[1]*s_;k[int(round(y)),int(round(x))]+=w
  ks.append(k/np.abs(k[k>0]).sum())
 return ks
_KS=None
def line_mask(img,sky=0):
 # Painted football lines. Artificial turf under floodlights sparkles everywhere, so plain
 # brightness picks up mostly glare: the top-hat is averaged along 24 orientations over ~60 px,
 # which keeps long straight-ish strokes (lines, the circle locally) and drops the sparkle.
 # Yellow multi-sport markings and anything off the turf (fence, stands) are removed.
 global _KS
 if _KS is None:_KS=_line_kernels()
 img=np.ascontiguousarray(img);sm=cv2.resize(img,None,fx=.5,fy=.5,interpolation=cv2.INTER_AREA)
 g=cv2.cvtColor(sm,cv2.COLOR_BGR2GRAY).astype(np.float32);hsv=cv2.cvtColor(sm,cv2.COLOR_BGR2HSV)
 th=cv2.morphologyEx(g,cv2.MORPH_TOPHAT,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)))
 th[(hsv[...,0]>12)&(hsv[...,0]<40)&(hsv[...,1]>70)]=0
 r=np.max([cv2.filter2D(th,-1,k) for k in _KS],0);r[:sky//2]=0
 turf=cv2.dilate(turf_mask(sm,sky//2),np.ones((9,9),np.uint8))>0;r[~turf]=0
 v=r[sky//2:];ref=np.percentile(v[v>0],99.5) if (v>0).sum()>100 else 1.
 m=((r>max(.45*ref,12.))).astype(np.uint8)*255
 return cv2.resize(m,(img.shape[1],img.shape[0]),interpolation=cv2.INTER_NEAREST)
def white_mask(img,sky=0):
 # Bright, non-yellow thin structures (top-hat). Noisy on sparkling turf, but its density is what
 # the score thresholds were tuned on; the association fit uses line_mask instead.
 img=np.ascontiguousarray(img);g=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
 th=cv2.morphologyEx(g,cv2.MORPH_TOPHAT,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15)))
 hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV);yellow=(hsv[...,0]>12)&(hsv[...,0]<40)&(hsv[...,1]>70)
 yellow=cv2.dilate(yellow.astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool)
 m=((th>28)&~yellow).astype(np.uint8)*255;m[:sky]=0;return m
def line_score(img,Hp,meta,sky=0):
 # Share of visible model-line samples that land within 4 px of painted white.
 dt=cv2.distanceTransform(255-white_mask(img,sky),cv2.DIST_L2,3);hit=tot=0
 for l in model_lines(meta):
  p=warp(l,Hp);ok=np.isfinite(p).all(1)&(p[:,0]>=0)&(p[:,0]<img.shape[1]-1)&(p[:,1]>=max(sky,0))&(p[:,1]<img.shape[0]-1)
  if ok.sum()<5:continue
  q=p[ok].astype(int);hit+=int((dt[q[:,1],q[:,0]]<4).sum());tot+=len(q)
 return hit/tot if tot else 0.

def refine(img,Hp,meta,sky=0,iters=25,caps=(12,)):
 # Pull the projected model lines onto the painted lines: Levenberg-Marquardt on the 8
 # homography parameters, residual = distance to the nearest white pixel, capped so lines hidden
 # behind players or stray paint do not drag the fit. Wider caps let an 8-parameter fit latch
 # onto unrelated paint when only a few lines are visible, so the default stays tight.
 raw=cv2.distanceTransform(255-white_mask(img,sky),cv2.DIST_L2,3).astype(np.float32)
 L=np.concatenate(model_lines(meta));hh,ww=img.shape[:2]
 h=(Hp/Hp[2,2]).ravel()[:8].astype(float)
 for cap in caps:
  dt=np.minimum(raw,cap)
  def res(h):
   p=warp(L,np.append(h,1).reshape(3,3));ok=np.isfinite(p).all(1)&(p[:,0]>=0)&(p[:,0]<ww-1)&(p[:,1]>=sky)&(p[:,1]<hh-1)
   p=np.where(ok[:,None],p,0).astype(np.float32)
   r=cv2.remap(dt,p[:,0].reshape(1,-1),p[:,1].reshape(1,-1),cv2.INTER_LINEAR).ravel();return np.where(ok,r,0.)
  lam=1e-2;r=res(h)
  for _ in range(iters):
   J=np.zeros((len(r),8))
   for k in range(8):dh=np.zeros(8);dh[k]=max(abs(h[k])*1e-4,1e-7);J[:,k]=(res(h+dh)-r)/dh[k]
   A=J.T@J;g=J.T@r
   try:hn=h+np.linalg.solve(A+lam*np.diag(np.diag(A)+1e-9),-g)
   except np.linalg.LinAlgError:break
   rn=res(hn)
   if (rn**2).sum()<(r**2).sum():h,r=hn,rn;lam*=.4
   else:lam*=5
 return np.append(h,1).reshape(3,3)

MIN_SCORE=.5   # share of model lines on painted white below which a calibration is not trusted
AGREE_PX=20    # two calibrations agree when the pitch grid lands within this many pixels
GRID=np.array([[x,y] for x in range(0,101,5) for y in range(-30,31,5)],float)
def _gap(H1,H2,sky,grid=GRID):
 p1,p2=warp(grid,H1),warp(grid,H2);ok=(p1[:,1]>sky)&(p1[:,1]<1080)&(p1[:,0]>0)&(p1[:,0]<1920)&np.isfinite(p2).all(1)
 return float(np.median(np.linalg.norm(p1-p2,axis=1)[ok])) if ok.sum()>=6 else np.inf
def chain(CUM,a,b,Ha):
 # Calibration of frame a carried into frame b through the camera chain.
 M=CUM[b]@np.linalg.inv(CUM[a])@Ha;return M/M[2,2]
def blend(H1,H2,t):
 # Homographies are defined up to sign and scale, and dividing by H[2,2] flips the sign once the
 # pitch origin is behind the camera. Blending two such matrices directly can cancel them out.
 a,b=H1/np.linalg.norm(H1),H2/np.linalg.norm(H2)
 if (a*b).sum()<0:b=-b
 return (1-t)*a+t*b
def track_lines(F,CUM,seeds,meta,sky,step=10,log=print,boxes=None):
 # Follow the painted lines through the clip. From each trusted seed, walk outward keyframe by
 # keyframe: predict through the camera chain, then snap onto the white lines visible in that
 # frame. Each keyframe gets its own local fit, which also absorbs the lens distortion of wide
 # phone cameras near the image edges; a snap is kept only if it fits better and stays close.
 N=len(F);kfs=sorted(set(list(range(0,N,step))+[N-1]+list(seeds)));found={k:seeds[k] for k in seeds}
 order=sorted(seeds,key=lambda k:-line_score(F[k],seeds[k],meta,sky))
 from .camfit import decompose
 Cref=np.median(np.array([decompose(seeds[k])[2] for k in seeds]),0)
 done=0
 for s0 in order:
  for direction in (1,-1):
   i=kfs.index(s0);prev=s0
   while 0<=i+direction<len(kfs):
    i+=direction;k=kfs[i]
    if k in found and k not in seeds:break  # met the walk from a better seed
    if k in seeds:prev=k;continue
    pred=chain(CUM,prev,k,found[prev])
    Hr,inf=icp_fit(F[k],pred,meta,sky,boxes=boxes[k] if boxes is not None else ()) if USE_ICP else (None,{})
    # The association fit is trusted where box or goal-area lines are in view: their corners pin
    # the pitch in both directions. With only far touchline, halfway and circle visible it can
    # flatten the pitch toward the horizon, so those frames keep the gentler snap instead.
    ok=USE_ICP and Hr is not None and min(inf.get('lines',[0]))>=4 and min(inf.get('box',[0]))>=2 and inf['rms']<12 and _gap(pred,Hr,sky)<60
    if ok:
     _,_,Cn=decompose(Hr);ok=np.linalg.norm(Cn-Cref)<CAM_TOL  # the camera does not move in a clip
    if not ok:
     sp=line_score(F[k],pred,meta,sky);Hr=refine(F[k],pred,meta,sky,iters=25);Hr=Hr/Hr[2,2];sr=line_score(F[k],Hr,meta,sky)
     Hr=Hr if sr>sp+.01 and _gap(pred,Hr,sky)<AGREE_PX*1.5 else pred
    found[k]=Hr;prev=k;done+=1
    if done%10==0:log(f'경기장 라인 추적 {done}/{len(kfs)}')
 return [(k,found[k],0,line_score(F[k],found[k],meta,sky)) for k in sorted(found)]
def spread(F,CUM,seed,meta,sky,every=30,log=print):return track_lines(F,CUM,seed,meta,sky,log=log)
MIN_INL=80
# Per-frame association fit inside line tracking. Exact on box frames, but errors it leaves carry
# along the camera chain into far-away centre frames; off until it runs inside the shared-camera fit.
USE_ICP=False
CAM_TOL=2.5   # metres the implied camera position may differ from the clip's
def calib_score(img,C,fi,sky=0):
 # line_score for any calibration object (homography blend or camera model).
 dt=cv2.distanceTransform(255-white_mask(img,sky),cv2.DIST_L2,3);hit=tot=0
 for l in model_lines(C.meta):
  p=C.to_img(fi,l);ok=np.isfinite(p).all(1)&(p[:,0]>=0)&(p[:,0]<img.shape[1]-1)&(p[:,1]>=max(sky,0))&(p[:,1]<img.shape[0]-1)
  if ok.sum()<5:continue
  q=p[ok].astype(int);hit+=int((dt[q[:,1],q[:,0]]<4).sum());tot+=len(q)
 return hit/tot if tot else 0.
MAX_ANCHOR_AGE=45  # frames at 30 fps
ANCHOR_AGREEMENT=12  # pixels at 1080p

def anchor_coverage(bases,CUM,segments,sky,meta):
 # Reacquisition starts a new calibration on each side of a broken camera link.
 # Between competing anchors, suppress coordinates when the camera motion cannot reconcile them.
 valid=np.zeros(len(CUM),dtype=bool)
 grid=GRID*np.array([meta['length']/100,meta['width']/60])
 for k,_,_,_ in bases:
  a=max(0,k-MAX_ANCHOR_AGE);b=min(len(CUM),k+MAX_ANCHOR_AGE+1)
  valid[a:b]|=segments[a:b]==segments[k]
 for (a,H1,_,_),(b,H2,_,_) in zip(bases,bases[1:]):
  if segments[a]==segments[b] and 1<b-a<=2*MAX_ANCHOR_AGE:
   mid=(a+b)//2
   p=chain(CUM,a,mid,H1);q=chain(CUM,b,mid,H2)
   if max(_gap(p,q,sky,grid),_gap(q,p,sky,grid))>ANCHOR_AGREEMENT:
    valid[a+1:b]=False
 return valid

def auto(F,CUM,vid,sky,log=print,every=30,boxes=None,inliers=None):
 meta=venues()[vid]
 refs=[(_feats(img,0),H) for img,H in venue_refs(vid) if img is not None]
 if not refs:return None
 segments=motion_segments(inliers,len(F))
 breaks=np.flatnonzero(np.diff(segments))+1
 samples=sorted(set(range(0,len(F),every))|{len(F)-1}|set(breaks)|set(breaks-1))
 bases=[]
 for fi in samples:
  feat=_feats(F[fi],sky)
  best=None
  for ref,rH in refs:
   motion,n=_match_features(ref,feat)
   if motion is None:continue
   H=motion@rH
   if abs(H[2,2])<1e-9:continue
   H=H/H[2,2]
   score=line_score(F[fi],H,meta,sky)
   if score>=MIN_SCORE*.8 and (best is None or n>best[2]):best=(fi,H,n,score)
  if best is not None:bases.append(best)
  log(f'경기장 맞추는 중 {fi+1}/{len(F)}')
 if not bases:return None
 valid=anchor_coverage(bases,CUM,segments,sky,meta)
 return {'bases':bases,'meta':meta,'venue':vid,'score':float(np.mean([b[3] for b in bases])),
         'mode':'auto','valid':valid,'segments':segments}

def from_points(frame,clicks,meta):
 # Manual calibration: clicks = [(landmark name, (x, y)), ...], at least 4.
 lm=landmarks(meta['length'],meta['width'],meta)
 src=np.float32([lm[n] for n,_ in clicks]);dst=np.float32([p for _,p in clicks])
 if len(src)<4:raise ValueError('기준점을 4개 이상 찍어 주세요.')
 Hp,_=cv2.findHomography(src,dst,0 if len(src)==4 else cv2.RANSAC,8.)
 if Hp is None:raise ValueError('기준점으로 경기장을 계산하지 못했어요. 점 위치를 다시 확인해 주세요.')
 return Hp/Hp[2,2]
def manual(F,CUM,frame,clicks,meta,sky,log=print,inliers=None):
 if not 0<=frame<len(F):raise ValueError('영상 안의 프레임을 선택해 주세요.')
 Hp=from_points(frame,clicks,meta);Hr=refine(F[frame],Hp,meta,sky);Hr=Hr/Hr[2,2]
 # Keep the snapped fit only when it stays near the clicked one (it can latch onto other paint).
 if _gap(Hp,Hr,sky)<AGREE_PX*1.5 and line_score(F[frame],Hr,meta,sky)>line_score(F[frame],Hp,meta,sky):Hp=Hr
 segments=motion_segments(inliers,len(F));valid=segments==segments[frame]
 a,b=np.flatnonzero(valid)[[0,-1]]
 local=spread(F[a:b+1],CUM[a:b+1],{int(frame-a):Hp},meta,sky,log=log)
 bases=[(k+a,H,n,s) for k,H,n,s in local]
 return {'bases':bases,'meta':meta,'venue':meta.get('id'),'score':float(np.mean([x[3] for x in bases])),
         'mode':'manual','valid':valid,'segments':segments}

class Calib:
 def __init__(self,cal,CUM):
  self.CUM,self.meta=CUM,cal['meta'];self.ks=np.array([b[0] for b in cal['bases']])
  self.M=[np.linalg.inv(CUM[k])@b[1] for k,b in zip(self.ks,cal['bases'])]  # pitch -> frame-0 image, per base
  self.valid=cal.get('valid',np.ones(len(CUM),dtype=bool))
  self.matrix=cal.get('matrix')
  self.paint_score=cal.get('paint_score')
  self.mode=cal.get('mode')
  self.k1,self.focal=float(cal.get('k1',0.)),float(cal.get('focal',1.))  # matrices are pinhole; pixels are not
  self.anchor_age=MAX_ANCHOR_AGE if 'valid' in cal and cal.get('mode')=='auto' else None
  segments=cal.get('segments',np.zeros(len(CUM),dtype=int))
  self.segments=segments
  self.groups={s:np.flatnonzero(segments[self.ks]==s) for s in np.unique(segments[self.ks])}
 def supports(self,start,end=None):
  end=start if end is None else end
  return bool(0<=start<=end<len(self.valid) and self.valid[start:end+1].all()
              and self.segments[start]==self.segments[end])
 def _pair(self,fi):
  group=self.groups[self.segments[fi]]
  if self.anchor_age is not None:group=group[np.abs(self.ks[group]-fi)<=self.anchor_age]
  ks=self.ks[group];j=np.searchsorted(ks,fi)
  if j<len(ks) and ks[j]==fi:return [(group[j],1.)]
  if j<=0:return [(group[0],1.)]
  if j>=len(ks):return [(group[-1],1.)]
  a,b=ks[j-1],ks[j];t=(fi-a)/(b-a);return [(group[j-1],1-t),(group[j],t)]
 def img_from_pitch(self,fi,i=None):
  if not self.supports(fi):raise ValueError('이 프레임의 경기장 좌표를 확인할 수 없어요.')
  if self.matrix is not None:
   return self.matrix[fi]
  if i is None:i=max(self._pair(fi),key=lambda pair:pair[1])[0]
  M=self.CUM[fi]@self.M[i];return M/M[2,2]
 def to_pitch(self,fi,pts):
  if not self.supports(fi):return np.full((len(pts),2),np.nan)
  if self.matrix is not None:return warp(undistort(pts,self.k1,self.focal),np.linalg.inv(self.matrix[fi]))
  return sum(w*warp(pts,np.linalg.inv(self.img_from_pitch(fi,i))) for i,w in self._pair(fi))
 def to_img(self,fi,pts):
  if not self.supports(fi):return np.full((len(pts),2),np.nan)
  if self.matrix is not None:return distort(warp(pts,self.matrix[fi]),self.k1,self.focal)
  return sum(w*warp(pts,self.img_from_pitch(fi,i)) for i,w in self._pair(fi))

def save_venue(vid,name,meta,F,CUM,cal,frames):
 # Store reference frames from a calibrated clip, so later clips from the same spot match them.
 d=os.path.join(VENUES,vid);os.makedirs(d,exist_ok=True);C=Calib(cal,CUM)
 json.dump({**{k:v for k,v in meta.items() if k!='id'},'name':name},open(os.path.join(d,'meta.json'),'w'),ensure_ascii=False,indent=1)
 n0=len(glob.glob(os.path.join(d,'ref_*.jpg')))
 for i,fi in enumerate(frames):
  cv2.imwrite(os.path.join(d,f'ref_{n0+i:02d}.jpg'),np.ascontiguousarray(F[fi]),[cv2.IMWRITE_JPEG_QUALITY,93])
  np.save(os.path.join(d,f'ref_{n0+i:02d}.npy'),C.img_from_pitch(fi))

# ---------- line-association fit ----------
def model_segments(meta):
 # Model lines as separate features: ('line', a, b) straight segments and ('curve', pts) arcs.
 L,Wd=meta['length'],meta['width'];hx=meta.get('half_x',L/2);w=Wd/2;d=dims(meta)
 bd,bw,gd,gw,sp=d['box_depth'],d['box_half'],d['ga_depth'],d['ga_half'],d['spot'];out=[]
 for a,b in [((0,-w),(0,w)),((0,w),(L,w)),((0,-w),(L,-w)),((L,-w),(L,w)),((hx,-w),(hx,w))]:out.append(('line',np.array(a,float),np.array(b,float)))
 for x0,s in [(0,1),(L,-1)]:
  for a,b in [((0,bw),(bd,bw)),((bd,bw),(bd,-bw)),((bd,-bw),(0,-bw)),((0,gw),(gd,gw)),((gd,gw),(gd,-gw)),((gd,-gw),(0,-gw))]:
   out.append(('line',np.array((x0+s*a[0],a[1]),float),np.array((x0+s*b[0],b[1]),float)))
 t=np.linspace(0,2*np.pi,240);out.append(('curve',np.stack([hx+9.15*np.cos(t),9.15*np.sin(t)],1)))
 a=np.arccos(np.clip((bd-sp)/9.15,-1,1));t=np.linspace(-a,a,60)
 out.append(('curve',np.stack([sp+9.15*np.cos(t),9.15*np.sin(t)],1)));out.append(('curve',np.stack([L-sp-9.15*np.cos(t),9.15*np.sin(t)],1)))
 return out

def _visible(p,sky,W=1920,H=1080):return np.isfinite(p).all(1)&(p[:,0]>=0)&(p[:,0]<W)&(p[:,1]>=sky)&(p[:,1]<H)
def _associate(m_pts,meta,H,R,sky,min_px=40):
 # Claim white pixels for every visible model feature under homography H (see icp_fit).
 WP,=m_pts;obs=[]
 for fi_,(kind,*g) in enumerate(model_segments(meta)):
  if kind=='line':a,b=g;mp=a+(b-a)*np.linspace(0,1,60)[:,None]
  else:mp=g[0]
  p=warp(mp,H);vis=_visible(p,sky)
  if vis.sum()<6:continue
  pv=p[vis];lo,hi=pv.min(0)-R,pv.max(0)+R
  cand=WP[(WP[:,0]>=lo[0])&(WP[:,0]<=hi[0])&(WP[:,1]>=lo[1])&(WP[:,1]<=hi[1])]
  if len(cand)<min_px:continue
  if kind=='line':
   d=pv[-1]-pv[0];ld=np.linalg.norm(d)+1e-9;u=d/ld;n=np.array([-u[1],u[0]])
   dist=np.abs((cand-pv[0])@n);along=(cand-pv[0])@u
   near=cand[(dist<R)&(along>-R)&(along<ld+R)]
   if len(near)<min_px:continue
   best=None;rng=np.random.default_rng(fi_)
   for _ in range(80):
    i,j=rng.choice(len(near),2,replace=False);q=near[j]-near[i];nq=np.linalg.norm(q)
    if nq<10 or abs(q@u)/nq<np.cos(np.radians(12)):continue
    nn=np.array([-q[1],q[0]])/nq;inl=np.abs((near-near[i])@nn)<2.5
    if best is None or inl.sum()>best[0]:best=(inl.sum(),inl)
   if best is None or best[0]<min_px:continue
   pts=near[best[1]];vx,vy,x0,y0=cv2.fitLine(pts,cv2.DIST_HUBER,0,.01,.01).ravel()
   obs.append(('line',fi_,vis,(np.array([-vy,vx]),-(np.array([-vy,vx])@[x0,y0])),len(pts)))
  else:
   dd=np.min(np.linalg.norm(cand[:,None]-pv[None,::3],axis=2),1);near=cand[dd<R*.6]
   if len(near)<min_px:continue
   obs.append(('curve',fi_,None,near[::max(1,len(near)//150)],len(near)))
 return obs
def _white_pts(img,sky,boxes):
 m=line_mask(img,sky)
 for b in boxes:
  x1,y1,x2,y2=[int(v) for v in b[:4]];m[max(y1-4,0):y2+4,max(x1-4,0):x2+4]=0
 ys,xs=np.nonzero(m);return (np.stack([xs,ys],1).astype(np.float32),)
def _feature_pts(meta,fi_):
 kind,*g=model_segments(meta)[fi_]
 if kind=='line':a,b=g;return a+(b-a)*np.linspace(0,1,60)[:,None]
 return g[0]
DIMS_FREE=('box_depth','box_half','ga_depth','ga_half','spot')
def icp_multi(imgs,Hs,meta,sky,boxes=None,radii=(50,25,12),free_dims=False):
 # Several frames, each with its own homography, optionally sharing fitted box/goal-area sizes.
 from scipy.optimize import least_squares
 n=len(imgs);WPs=[_white_pts(im,sky,boxes[i] if boxes is not None else ()) for i,im in enumerate(imgs)]
 Hs=[H/H[2,2] for H in Hs];d0=np.array([dims(meta)[k] for k in DIMS_FREE]);dv=d0.copy();info={}
 def meta_of(dv):m=dict(meta);m.update(dict(zip(DIMS_FREE,map(float,dv))));return m
 for R in radii:
  mm=meta_of(dv);obs=[_associate(WPs[i],mm,Hs[i],R,sky) for i in range(n)]
  if sum(len([o for o in ob if o[0]=='line']) for ob in obs)<3*n:return None,{'lines':0}
  # Directions the visible lines do not pin down (few lines, all near-parallel) stay where the
  # starting fit had them: a weak pull of the pitch grid toward its starting projection.
  G0=[warp(GRID,H) for H in Hs]
  def unpack(p):
   H=[np.append(p[8*i:8*i+8],1).reshape(3,3) for i in range(n)];return H,(p[8*n:] if free_dims else dv)
  def res(p):
   H,dd=unpack(p);m2=meta_of(dd);out=[]
   for i in range(n):
    for kind,fi_,vis,ob,cnt in obs[i]:
     mp=_feature_pts(m2,fi_);q=warp(mp,H[i])
     if kind=='line':ln,c=ob;q=q[vis] if vis is not None and len(vis)==len(q) else q;out.append((q@ln+c)*np.sqrt(min(cnt,400)/len(q)))
     else:dd2=np.min(np.linalg.norm(ob[:,None]-q[None],axis=2),1);out.append(np.minimum(dd2,R)*.7)
    g=warp(GRID,H[i]);ok=np.isfinite(g).all(1)&np.isfinite(G0[i]).all(1)&(np.abs(G0[i]-[960,540]).max(1)<1500)
    out.append(((g-G0[i])[ok]*.03).ravel())
   if free_dims:out.append((dd-d0)/4.)
   return np.concatenate(out)
  p0=np.concatenate([np.concatenate([H.ravel()[:8] for H in Hs])]+([dv] if free_dims else []))
  sol=least_squares(res,p0,loss='soft_l1',f_scale=2.,x_scale='jac',max_nfev=300)
  Hs,dv=unpack(sol.x);Hs=[H/H[2,2] for H in Hs];dv=np.array(dv,float)
  info={'lines':[len([o for o in ob if o[0]=='line']) for ob in obs],'rms':float(np.sqrt(np.mean(res(sol.x)**2))),
        'box':[len([o for o in ob if o[0]=='line' and 5<=o[1]<=16]) for ob in obs]}
 info['dims']=dict(zip(DIMS_FREE,np.round(dv,2).tolist()))
 return Hs,info
def icp_fit(img,Hp,meta,sky=0,radii=(50,25,12),boxes=()):
 Hs,info=icp_multi([img],[Hp],meta,sky,[boxes],radii)
 return (None,info) if Hs is None else (Hs[0],info)
