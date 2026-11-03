import numpy as np,cv2
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from .camera import warp
from . import pitch
# Camera-model pitch registration. Within one clip the camera stands in one place: only its
# pan/tilt/roll and zoom change. So instead of a free 8-parameter homography per frame, every
# keyframe shares one camera position and one lens distortion, and has just a rotation and a
# focal length of its own. All keyframes are fitted together to the painted white lines, tied to
# each other by the frame-to-frame camera motion. Far fewer unknowns per frame means a frame
# that shows only a couple of lines cannot bend the pitch to fit stray paint, and the radial
# term straightens the edge-of-frame curvature of wide phone lenses.
STEP=10
MAXNFEV=200
CAP=15.        # px; residuals beyond this are lines hidden by players or not painted
W,H=1920,1080;CX,CY=W/2,H/2

def rodrigues(w):return cv2.Rodrigues(np.asarray(w,float).reshape(3,1))[0]
def project(P,C,w,f,k1):
 # Pitch points (X, Y) on Z=0 -> distorted pixels. Returns (uv, in_front).
 P=np.asarray(P,float);X=np.concatenate([P,np.zeros((len(P),1))],1)-C
 Xc=X@rodrigues(w).T;z=Xc[:,2];front=z>2.;z=np.where(front,z,2.)
 x,y=Xc[:,0]/z,Xc[:,1]/z;r2=x*x+y*y;d=1+k1*r2
 return np.stack([f*x*d+CX,f*y*d+CY],1),front
def backproject(uv,C,w,f,k1):
 # Distorted pixels -> pitch points on Z=0.
 uv=np.asarray(uv,float);xd,yd=(uv[:,0]-CX)/f,(uv[:,1]-CY)/f;x,y=xd.copy(),yd.copy()
 for _ in range(8):d=1+k1*(x*x+y*y);x,y=xd/d,yd/d
 ray=np.stack([x,y,np.ones_like(x)],1)@rodrigues(w)  # R^T applied row-wise
 s=-C[2]/np.where(np.abs(ray[:,2])<1e-9,1e-9,ray[:,2]);return C[:2]+s[:,None]*ray[:,:2]

def decompose(Hp):
 # Pitch->image homography -> focal, rotation vector, camera centre (principal point at centre).
 T=np.array([[1,0,-CX],[0,1,-CY],[0,0,1.]]);h=T@Hp;h=h/np.linalg.norm(h[:,0])
 a=-(h[0,0]*h[0,1]+h[1,0]*h[1,1])/(h[2,0]*h[2,1]+1e-12)
 b=(h[0,1]**2+h[1,1]**2-h[0,0]**2-h[1,0]**2)/(h[2,0]**2-h[2,1]**2+1e-12)
 cands=[v for v in (a,b) if v>100**2]
 f=float(np.sqrt(np.median(cands))) if cands else 1500.
 Ki=np.diag([1/f,1/f,1.]);M=Ki@h;lam=1/np.linalg.norm(M[:,0])
 r1,r2,t=M[:,0]*lam,M[:,1]*lam,M[:,2]*lam
 if t[2]<0:r1,r2,t=-r1,-r2,-t   # pitch in front of the camera
 R=np.stack([r1,r2,np.cross(r1,r2)],1);U,_,Vt=np.linalg.svd(R);R=U@Vt
 if np.linalg.det(R)<0:R=U@np.diag([1,1,-1])@Vt
 C=-R.T@t;return f,cv2.Rodrigues(R)[0].ravel(),C

def line_dt(img,sky,boxes=()):
 m=pitch.white_mask(img,sky)
 for b in boxes:
  x1,y1,x2,y2=[int(v) for v in b[:4]];m[max(y1-4,0):y2+4,max(x1-4,0):x2+4]=0  # white kit is not paint
 return np.minimum(cv2.distanceTransform(255-m,cv2.DIST_L2,3),CAP).astype(np.float32)

class CamCalib:
 # Same interface as pitch.Calib: to_img / to_pitch / img_from_pitch, ks, meta.
 def __init__(self,ks,glob,local,CUM,meta,score=None):
  self.ks=np.asarray(ks);self.glob=np.asarray(glob,float);self.local=np.asarray(local,float);self.CUM=CUM;self.meta=meta;self.score=score
  self.C,self.k1=self.glob[:3],self.glob[3]
 def cam(self,i):w=self.local[i,:3];f=np.exp(self.local[i,3]);return self.C,w,f,self.k1
 def _pair(self,fi):
  j=np.searchsorted(self.ks,fi)
  if j<=0:return [(0,1.)]
  if j>=len(self.ks):return [(len(self.ks)-1,1.)]
  a,b=self.ks[j-1],self.ks[j];t=(fi-a)/(b-a);return [(j-1,1-t),(j,t)]
 def to_img(self,fi,pts):
  # Project at the neighbouring keyframes, carry along the camera chain into this frame, blend.
  out=0;pts=np.asarray(pts,float).reshape(-1,2)
  for i,wt in self._pair(fi):
   k=int(self.ks[i]);uv,fr=project(pts,*self.cam(i))
   uv[~fr]=np.nan  # behind the camera or past the horizon
   if k!=fi:uv=warp(uv,self.CUM[fi]@np.linalg.inv(self.CUM[k]))
   out=out+wt*uv
  return out
 def to_pitch(self,fi,pts):
  out=0
  for i,wt in self._pair(fi):
   k=int(self.ks[i]);uv=np.asarray(pts,float).reshape(-1,2)
   if k!=fi:uv=warp(uv,self.CUM[k]@np.linalg.inv(self.CUM[fi]))
   out=out+wt*backproject(uv,*self.cam(i))
  return out
 def img_from_pitch(self,fi,i=None):
  # Best plain homography for this frame (for scoring and saved venue references).
  L,Wd=self.meta['length'],self.meta['width']
  g=np.array([[x,y] for x in np.linspace(0,L,21) for y in np.linspace(-Wd/2,Wd/2,9)])
  uv=self.to_img(fi,g);ok=np.isfinite(uv).all(1)&(np.abs(uv-[CX,CY]).max(1)<3000)
  if ok.sum()<4:return np.eye(3)
  Hm,_=cv2.findHomography(np.float32(g[ok]),np.float32(uv[ok]),0);return Hm/Hm[2,2]
 def save(self,path):
  import json;np.savez(path,ks=self.ks,glob=self.glob,local=self.local,score=self.score if self.score is not None else np.nan,meta=json.dumps(self.meta,ensure_ascii=False))

DIMK=['half_x','box_depth','box_half','ga_depth','ga_half','spot']
def fit(F,CUM,init,sky,dets=None,step=STEP,log=print,free_dims=True):
 # init: any calibration with img_from_pitch (auto or manual homographies) to start from.
 N=len(F);meta=init.meta;ks=sorted(set(list(range(0,N,step))+[N-1]));K=len(ks)
 lines=np.concatenate(pitch.model_lines(meta));L,Wd=meta['length'],meta['width']
 grid=np.array([[x,y] for x in np.arange(0,L+.1,5) for y in np.arange(-Wd/2,Wd/2+.1,5)])
 log('카메라 모델 초기화')
 dec=[decompose(init.img_from_pitch(k)) for k in ks]
 C0=np.median(np.array([d[2] for d in dec]),0)
 d0=pitch.dims(meta);d0['half_x']=float(meta.get('half_x',L/2));dim0=np.array([d0[k] for k in DIMK])
 glob0=np.array([*C0,0.,*dim0]);NG=len(glob0) if free_dims else 4;loc0=np.array([[*d[1],np.log(d[0])] for d in dec])
 # Re-solve each keyframe's rotation and zoom with the shared centre, from its own homography.
 for i,k in enumerate(ks):
  Hk=init.img_from_pitch(k);tgt=warp(grid,Hk);ok=(np.abs(tgt-[CX,CY])<[CX*1.2,CY*1.2]).all(1)&(tgt[:,1]>sky)
  if ok.sum()<6:continue
  def r(p):uv,_=project(grid[ok],C0,p[:3],np.exp(p[3]),0.);return (uv-tgt[ok]).ravel()
  loc0[i]=least_squares(r,loc0[i],method='lm',max_nfev=200).x
 DT=[line_dt(F[k],sky,dets[k] if dets is not None else ()) for k in ks]
 # Chain ties: pitch grid points seen at keyframe a must land where the camera motion takes them.
 vis=[]
 for i in range(K-1):
  uv,fr=project(grid,C0,loc0[i,:3],np.exp(loc0[i,3]),0.);ok=fr&(uv[:,0]>0)&(uv[:,0]<W)&(uv[:,1]>sky)&(uv[:,1]<H)
  vis.append(np.nonzero(ok)[0])
 Hrel=[CUM[ks[i+1]]@np.linalg.inv(CUM[ks[i]]) for i in range(K-1)]
 nl=len(lines);WC=.4
 rows_l=[(i*nl,(i+1)*nl) for i in range(K)];off=K*nl;rows_c=[]
 for i in range(K-1):n=2*len(vis[i]);rows_c.append((off,off+n));off+=n
 nres=off+2+(len(DIMK) if free_dims else 0)  # + distortion prior (+ dimension priors)
 def unpack(p):
  g=p[:NG] if free_dims else np.concatenate([p[:4],dim0]);return g,p[NG:].reshape(K,4)
 def mlines(g):
  m=dict(meta);m.update({k:float(v) for k,v in zip(DIMK,g[4:])});return np.concatenate(pitch.model_lines(m))
 def res_line(i,g,l):
  uv,fr=project(mlines(g) if free_dims else lines,g[:3],l[i,:3],np.exp(l[i,3]),g[3])
  ok=fr&(uv[:,0]>=0)&(uv[:,0]<W-1)&(uv[:,1]>=sky+8)&(uv[:,1]<H-1)
  q=np.where(ok[:,None],uv,0).astype(np.float32);r=cv2.remap(DT[i],q[:,0].reshape(1,-1),q[:,1].reshape(1,-1),cv2.INTER_LINEAR).ravel()
  return np.where(ok,r,0.)
 def res_chain(i,g,l):
  P=grid[vis[i]]
  if len(P)==0:return np.zeros(0)
  a,_=project(P,g[:3],l[i,:3],np.exp(l[i,3]),g[3]);b,_=project(P,g[:3],l[i+1,:3],np.exp(l[i+1,3]),g[3])
  return ((b-warp(a,Hrel[i]))*WC).ravel()
 def res(p):
  g,l=unpack(p);out=[res_line(i,g,l) for i in range(K)]+[res_chain(i,g,l) for i in range(K-1)]
  # Weak pull toward the laws' sizes (3 m), so a dimension no frame shows stays sensible.
  pri=[np.array([g[3]*20,0.])]+([(g[4:]-dim0)/3.] if free_dims else [])
  return np.concatenate(out+pri)
 S=lil_matrix((nres,NG+4*K),dtype=np.int8)
 for i,(a,b) in enumerate(rows_l):S[a:b,:NG]=1;S[a:b,NG+4*i:NG+4+4*i]=1
 for i,(a,b) in enumerate(rows_c):S[a:b,:4]=1;S[a:b,NG+4*i:NG+8+4*i]=1
 S[off,3]=1
 for q in range(len(DIMK) if free_dims else 0):S[off+2+q,4+q]=1
 p0=np.concatenate([glob0[:NG],loc0.ravel()])
 log('카메라 모델 맞추는 중')
 sol=least_squares(res,p0,jac_sparsity=S,method='trf',loss='soft_l1',f_scale=3.,x_scale='jac',max_nfev=MAXNFEV,verbose=0)
 g,l=unpack(sol.x);m=dict(meta);m.update({k:round(float(v),2) for k,v in zip(DIMK,g[4:])})
 cal=CamCalib(ks,g[:4],l,CUM,m)
 cal.score=float(np.mean([pitch.calib_score(F[k],cal,k,sky) for k in ks[::3]]))
 return cal,sol

# ---------- shared camera + line association ----------
def _features(meta,n_line=40):
 # Model features as polylines in metres: straight lines and arcs, same order as pitch.model_segments.
 out=[]
 for kind,*g in pitch.model_segments(meta):
  if kind=='line':a,b=g;out.append(a+(b-a)*np.linspace(0,1,n_line)[:,None])
  else:out.append(g[0][::2])
 return out
def _seg_dist(Q,P):
 # Distance from points Q (m,2) to the polyline P (n,2).
 A,B=P[:-1],P[1:];D=B-A;L2=np.maximum((D*D).sum(1),1e-9)
 t=np.clip(((Q[:,None,:]-A[None])*D[None]).sum(2)/L2[None],0,1)
 C=A[None]+t[...,None]*D[None];return np.sqrt(((Q[:,None,:]-C)**2).sum(2)).min(1)
def _proj_feats(feats,cam):
 out=[]
 for P in feats:
  uv,fr=project(P,*cam);uv[~fr]=np.nan;out.append(uv)
 return out
def _claim(Q,proj,R,sky):
 # Each paint pixel goes to the nearest projected feature within R; at most 250 per feature.
 best=np.full(len(Q),np.inf);who=np.full(len(Q),-1)
 for fi_,uv in enumerate(proj):
  ok=np.isfinite(uv).all(1)
  if ok.sum()<3:continue
  P=uv[ok]
  if not ((P[:,0]>-R)&(P[:,0]<W+R)&(P[:,1]>sky-R)&(P[:,1]<H+R)).any():continue
  lo,hi=P.min(0)-R,P.max(0)+R;sel=np.nonzero((Q[:,0]>=lo[0])&(Q[:,0]<=hi[0])&(Q[:,1]>=lo[1])&(Q[:,1]<=hi[1]))[0]
  if len(sel)==0:continue
  d=_seg_dist(Q[sel],P);upd=(d<R)&(d<best[sel]);best[sel[upd]]=d[upd];who[sel[upd]]=fi_
 claims={}
 for fi_ in np.unique(who[who>=0]):
  idx=np.nonzero(who==fi_)[0]
  if len(idx)<25:continue  # too little paint to say anything about this line
  claims[int(fi_)]=Q[idx[np.linspace(0,len(idx)-1,min(len(idx),250)).astype(int)]]
 return claims
def fit_icp(F,CUM,init,sky,dets=None,step=STEP,radii=(50,25,12),log=print,free_dims=True,max_nfev=60):
 # Shared camera position and lens, per-keyframe rotation and zoom, fitted to paint pixels that
 # are explicitly associated with model lines (re-associated at shrinking radii).
 N=len(F);meta=init.meta;ks=sorted(set(list(range(0,N,step))+[N-1]));K=len(ks)
 L,Wd=meta['length'],meta['width'];DK=('box_depth','box_half','ga_depth','ga_half','spot','width','half_x')
 dd_=pitch.dims(meta);dd_['width']=float(Wd);dd_['half_x']=float(meta.get('half_x',L/2))
 d0=np.array([dd_[k] for k in DK]);dv=d0.copy()
 def meta_of(d):m=dict(meta);m.update(dict(zip(DK,map(float,d))));return m
 feats=_features(meta);NG=4+(len(DK) if free_dims else 0)
 grid=np.array([[x,y] for x in np.arange(0,L+.1,5) for y in np.arange(-Wd/2,Wd/2+.1,5)])
 log('카메라 모델 초기화')
 dec=[decompose(init.img_from_pitch(k)) for k in ks];C0=np.median(np.array([d[2] for d in dec]),0)
 loc=np.array([[*d[1],np.log(d[0])] for d in dec])
 for i,k in enumerate(ks):
  Hk=init.img_from_pitch(k);tgt=warp(grid,Hk);ok=(np.abs(tgt-[CX,CY])<[CX*1.2,CY*1.2]).all(1)&(tgt[:,1]>sky)
  if ok.sum()<6:continue
  def r(p):uv,_=project(grid[ok],C0,p[:3],np.exp(p[3]),0.);return (uv-tgt[ok]).ravel()
  loc[i]=least_squares(r,loc[i],method='lm',max_nfev=200).x
 glob=np.array([*C0,0.])
 Qs=[]
 for k in ks:
  bx=dets[k] if dets is not None else ();m=pitch.line_mask(F[k],sky)
  for b in bx:
   x1,y1,x2,y2=[int(v) for v in b[:4]];m[max(y1-4,0):y2+4,max(x1-4,0):x2+4]=0
  ys,xs=np.nonzero(m[::2,::2]);Qs.append(np.stack([xs*2,ys*2],1).astype(float))
 Hrel=[CUM[ks[i+1]]@np.linalg.inv(CUM[ks[i]]) for i in range(K-1)];WC=.4
 for R in radii:
  feats=_features(meta_of(dv))
  claims=[_claim(Qs[i],_proj_feats(feats,(glob[:3],loc[i,:3],np.exp(loc[i,3]),glob[3])),R,sky) for i in range(K)]
  vis=[]
  for i in range(K-1):
   uv,fr=project(grid,glob[:3],loc[i,:3],np.exp(loc[i,3]),glob[3]);vis.append(np.nonzero(fr&(uv[:,0]>0)&(uv[:,0]<W)&(uv[:,1]>sky)&(uv[:,1]<H))[0])
  rows_l=[];off=0
  for c in claims:n=sum(len(v) for v in c.values());rows_l.append((off,off+n));off+=n
  rows_c=[]
  for i in range(K-1):n=2*len(vis[i]);rows_c.append((off,off+n));off+=n
  nres=off+1+(len(DK) if free_dims else 0)
  def res(p):
   g=p[:4];l=p[NG:].reshape(K,4);out=[]
   fts=_features(meta_of(p[4:NG])) if free_dims else feats
   for i in range(K):
    cam=(g[:3],l[i,:3],np.exp(l[i,3]),g[3])
    for fi_,Q in claims[i].items():
     uv,fr=project(fts[fi_],*cam);uv=uv[fr]
     out.append(np.minimum(_seg_dist(Q,uv),R) if len(uv)>1 else np.full(len(Q),float(R)))
   for i in range(K-1):
    P=grid[vis[i]]
    if len(P)==0:continue
    a,_=project(P,g[:3],l[i,:3],np.exp(l[i,3]),g[3]);b,_=project(P,g[:3],l[i+1,:3],np.exp(l[i+1,3]),g[3])
    out.append(((b-warp(a,Hrel[i]))*WC).ravel())
   out.append(np.array([g[3]*20]))
   if free_dims:out.append((p[4:NG]-d0)/np.array([4,4,4,4,4,8,8.]))
   return np.concatenate(out)
  S=lil_matrix((nres,NG+4*K),dtype=np.int8)
  for i,(a,b) in enumerate(rows_l):S[a:b,:NG]=1;S[a:b,NG+4*i:NG+4+4*i]=1
  for i,(a,b) in enumerate(rows_c):S[a:b,:4]=1;S[a:b,NG+4*i:NG+8+4*i]=1
  S[off,3]=1
  for q in range(NG-4):S[off+1+q,4+q]=1
  log(f'카메라 모델 맞추는 중 (범위 {R}px)')
  p0=np.concatenate([glob]+([dv] if free_dims else [])+[loc.ravel()])
  sol=least_squares(res,p0,jac_sparsity=S,method='trf',loss='soft_l1',f_scale=2.,x_scale='jac',max_nfev=max_nfev)
  glob=sol.x[:4];loc=sol.x[NG:].reshape(K,4)
  if free_dims:dv=sol.x[4:NG]
 mm=meta_of(dv);mm={**mm,**{k:round(float(v),2) for k,v in zip(DK,dv)}}
 cal=CamCalib(ks,glob,loc,CUM,mm)
 r=res(sol.x)[:rows_l[-1][1]];cal.score=float(np.mean(r<3))  # share of claimed paint within 3 px
 cal.info={'claimed':[sum(len(v) for v in c.values()) for c in claims],'within3':cal.score,'C':glob[:3].round(2).tolist(),'k1':float(glob[3]),'dims':{k:round(float(v),2) for k,v in zip(DK,dv)}}
 return cal,sol
