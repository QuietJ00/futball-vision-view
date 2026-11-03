import numpy as np,cv2
from .camera import warp
# One player from one click. Every detection is a node; the path through them is the cheapest
# under motion (in camera-stabilised coordinates, scaled by the player's pixel height) plus
# appearance (torso and leg colour against the clicked player). Extra clicks pin the path.
MAXGAP=15        # frames a player may go undetected (occlusion, a missed box)
GAP_COST=1.      # per missing frame
REACH=.19        # furthest a player can move per frame, in body heights (~9 m/s); box jitter adds .2
APP_W=6.         # weight of appearance distance
APP_OK=.28       # appearance distance at which a box is neither evidence for nor against
# Match costs are relative to APP_OK, so a well-matching box is cheaper than a missing frame
# and the path keeps following the player instead of stopping early.

BANDS=((.1,.35),(.35,.6),(.6,.93))   # chest, waist/shorts, legs as fractions of box height
NB=4*8*8
def features(F,dets):
 # Per box: Lab colour histograms of three body bands, turf pixels masked out. Bands keep a
 # tracksuit apart from a same-colour bib over shorts.
 out=[]
 for fi,D in enumerate(dets):
  img=F[fi];fs=[]
  for x1,y1,x2,y2,_ in D:
   x1,y1,x2,y2=max(int(x1),0),max(int(y1),0),min(int(x2),img.shape[1]),min(int(y2),img.shape[0])
   roi=np.ascontiguousarray(img[y1:y2,x1:x2])
   if roi.size==0 or y2-y1<8:fs.append(np.zeros(NB*len(BANDS),np.float32));continue
   lab=cv2.cvtColor(roi,cv2.COLOR_BGR2LAB);hsv=cv2.cvtColor(roi,cv2.COLOR_BGR2HSV);h=roi.shape[0]
   turf=(hsv[...,0]>28)&(hsv[...,0]<92)&(hsv[...,1]>45);v=[]
   for a,b in BANDS:
    m=np.zeros(turf.shape,np.uint8);m[int(a*h):int(b*h)]=1;m[turf]=0
    hh=cv2.calcHist([lab],[0,1,2],m,[4,8,8],[0,256,0,256,0,256]).ravel();v.append(hh/max(hh.sum(),1))
   fs.append(np.concatenate(v).astype(np.float32))
  out.append(np.array(fs,np.float32).reshape(-1,NB*len(BANDS)))
 return out
def app_dist(a,b):
 # Bhattacharyya distance, averaged over the bands.
 return sum(1-np.sqrt(a[...,k*NB:(k+1)*NB]*b[...,k*NB:(k+1)*NB]).sum(-1) for k in range(len(BANDS)))/len(BANDS)

def stab(CUM,dets):
 # Foot points (box bottom centre) in frame-0 coordinates, plus box heights in the same frame.
 P=[];Hh=[]
 for fi,D in enumerate(dets):
  if len(D)==0:P.append(np.zeros((0,2)));Hh.append(np.zeros(0));continue
  foot=np.stack([(D[:,0]+D[:,2])/2,D[:,3]],1);head=np.stack([(D[:,0]+D[:,2])/2,D[:,1]],1)
  M=np.linalg.inv(CUM[fi]);f0,h0=warp(foot,M),warp(head,M)
  P.append(f0);Hh.append(np.linalg.norm(f0-h0,axis=1))
 return P,Hh

def pick(D,pt):
 # Detection under a click: the box containing it (smallest first), else the nearest box.
 if len(D)==0:return None
 x,y=pt;inside=[(float((b[2]-b[0])*(b[3]-b[1])),i) for i,b in enumerate(D) if b[0]-6<=x<=b[2]+6 and b[1]-10<=y<=b[3]+10]
 if inside:return min(inside)[1]
 c=np.stack([(D[:,0]+D[:,2])/2,(D[:,1]+D[:,3])/2],1);d=np.linalg.norm(c-[x,y],axis=1);i=int(d.argmin())
 return i if d[i]<60 else None

def _dp(frames,P,Hh,FE,ref,start,end=None):
 # Cheapest path over detections in `frames` (ordered, possibly reversed), starting at node
 # `start` in frames[0]. With `end` it must finish on that node in frames[-1]; otherwise it may
 # stop early, paying GAP_COST for every frame left untracked.
 T=len(frames);cost=[None]*T;back=[None]*T;vel=[None]*T
 cost[0]=np.full(len(P[frames[0]]),np.inf);cost[0][start]=0;back[0]=np.full((len(cost[0]),2),-1);vel[0]=np.zeros((len(cost[0]),2))
 app=[APP_W*(app_dist(FE[f],ref)-APP_OK) if len(FE[f]) else np.zeros(0) for f in frames]
 for t in range(1,T):
  f=frames[t];n=len(P[f]);best=np.full(n,np.inf);arg=np.full((n,2),-1);bv=np.zeros((n,2))
  for g in range(1,MAXGAP+2):
   s=t-g
   if s<0:break
   if n==0 or not np.isfinite(cost[s]).any():continue
   ps,hs=P[frames[s]],Hh[frames[s]];hn=np.maximum(hs[:,None],20)
   step=P[f][None]-ps[:,None];d=np.linalg.norm(step,axis=2)/hn
   # Each path carries its own velocity, so two players crossing in opposite directions are
   # told apart by where they were heading, not only by where they are.
   dev=np.linalg.norm(step-vel[s][:,None]*g,axis=2)/hn
   mv=3*(dev/(.08*g+.15))**2
   hr=np.abs(np.log(np.maximum(Hh[f][None],1)/np.maximum(hs[:,None],1)))
   c=cost[s][:,None]+mv+hr*4+GAP_COST*(g-1)
   # Travel beyond sprint reach is impossible however well the colours match.
   c[d>REACH*g+.2]=np.inf;c[hr>.45]=np.inf
   k=c.argmin(0);v=c[k,np.arange(n)]
   upd=v<best;best[upd]=v[upd];arg[upd,0]=s;arg[upd,1]=k[upd]
   bv[upd]=.7*vel[s][k[upd]]+.3*step[k[upd],np.nonzero(upd)[0]]/g
  cost[t]=best+app[t];back[t]=arg;vel[t]=bv
 # Choose the end node.
 if end is not None:
  t,k=T-1,end
  if not np.isfinite(cost[t][k]):return None
 else:
  t,k=-1,-1;bv=np.inf
  for s in range(T):
   if len(cost[s])==0:continue
   j=int(cost[s].argmin());v=cost[s][j]+GAP_COST*(T-1-s)
   if v<bv:bv,t,k=v,s,j
  if t<0:return {}
 path={}
 while t>=0 and k>=0:
  path[frames[t]]=int(k)
  if t==0:break
  s,j=back[t][k];t,k=int(s),int(j)
 return path

def track(F,dets,CUM,anchors,FE=None,log=print):
 # anchors: {frame: (x, y)} clicks on the player. Returns foot points and heights in image
 # coordinates per frame, and a source flag (2 click, 1 detection, 0 interpolated, -1 none).
 N=len(F)
 if FE is None:FE=features(F,dets)
 P,Hh=stab(CUM,dets)
 A={}
 for fi,pt in sorted(anchors.items()):
  k=pick(dets[fi],pt)
  if k is None:raise ValueError(f'{fi}번 프레임 클릭 위치에서 선수를 찾지 못했어요. 선수 몸 위를 다시 눌러 주세요.')
  A[fi]=k
 if not A:raise ValueError('선수를 한 번 클릭해 주세요.')
 # Two clicks the player could not have run between are two different people.
 ks=sorted(A)
 for a,b in zip(ks,ks[1:]):
  d=np.linalg.norm(P[b][A[b]]-P[a][A[a]])/max(Hh[a][A[a]],20)
  if d>REACH*(b-a)*1.3+.6:raise ValueError(f'{a}프레임과 {b}프레임에 클릭한 선수가 서로 달라 보여요 (그 사이에 갈 수 없는 거리예요). 같은 선수를 눌렀는지 확인해 주세요.')
 ref=np.mean([FE[f][k] for f,k in A.items()],0)
 ks=sorted(A);node={f:A[f] for f in ks}
 log('선수 추적 중')
 for a,b in zip(ks,ks[1:]):
  p=_dp(list(range(a,b+1)),P,Hh,FE,ref,A[a],A[b])
  if p is None:p={**_dp(list(range(a,b+1)),P,Hh,FE,ref,A[a]),**_dp(list(range(b,a-1,-1)),P,Hh,FE,ref,A[b])}
  node.update(p)
 node.update(_dp(list(range(ks[-1],N)),P,Hh,FE,ref,A[ks[-1]]))
 node.update(_dp(list(range(ks[0],-1,-1)),P,Hh,FE,ref,A[ks[0]]))
 for f in ks:node[f]=A[f]
 # Fill gaps by interpolating in stabilised coordinates, then map back into each frame.
 src=np.full(N,-1);S=np.full((N,2),np.nan);SH=np.full(N,np.nan);sim=np.full(N,np.nan)
 for f,k in node.items():S[f]=P[f][k];SH[f]=Hh[f][k];src[f]=2 if f in A else 1;sim[f]=1-app_dist(FE[f][k],ref)
 have=np.nonzero(src>=1)[0];lo,hi=have.min(),have.max()
 for j in range(2):S[lo:hi+1,j]=np.interp(np.arange(lo,hi+1),have,S[have,j])
 SH[lo:hi+1]=np.interp(np.arange(lo,hi+1),have,SH[have]);src[lo:hi+1][src[lo:hi+1]<0]=0
 # Ambiguous frames: another box close by that looks about as much like the player. These are
 # where a path can swap onto a team-mate, so the UI asks the user to check them.
 risk=np.zeros(N,bool)
 for f,k in node.items():
  if len(P[f])<2:continue
  d=np.linalg.norm(P[f]-P[f][k],axis=1)/max(Hh[f][k],20);ad=app_dist(FE[f],ref);d[k]=9
  risk[f]=bool(((d<.8)&(ad<ad[k]+.06)).any())
 foot=np.full((N,2),np.nan);height=np.full(N,np.nan)
 for f in range(lo,hi+1):
  foot[f]=warp(S[f],CUM[f])[0];top=warp(S[f]-[0,SH[f]],CUM[f])[0];height[f]=np.linalg.norm(foot[f]-top)
 return {'foot':foot,'height':height,'src':src,'sim':sim,'risk':risk,'anchors':sorted(A),'range':(int(lo),int(hi))}

def _runs(mask,minlen):
 out=[];i=0;n=len(mask)
 while i<n:
  if mask[i]:
   j=i
   while j<n and mask[j]:j+=1
   if j-i>=minlen:out.append((i,j-1))
   i=j
  else:i+=1
 return out
def report(tr):
 # Where the track is weakest, so the UI can ask for an extra click there. Short ambiguous
 # spells are merged when they sit within half a second of each other.
 src,risk=tr['src'],tr['risk'].copy();lo,hi=tr['range'];N=len(src)
 r=np.convolve(risk.astype(float),np.ones(15),'same')>0
 check=[(a,b,'겹침') for a,b in _runs(r,6)]+[(a,b,'안 보임') for a,b in _runs(src==0,10)]
 return {'covered':(lo,hi),'frames':N,'detected':int((src>=1).sum()),'check':sorted(check),
         'lost':([(0,lo-1)] if lo>0 else [])+([(hi+1,N-1)] if hi<N-1 else [])}
