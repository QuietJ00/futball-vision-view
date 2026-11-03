import numpy as np
# Pitch-plane positions of the tracked player, smoothed, then speed in km/h. Foot points bob
# with the stride and jitter more in depth (a pixel of foot error is several times more metres
# toward the camera than across), so depth is smoothed harder than the across-screen axis.
FPS=30
def gsmooth(a,s):
 k=np.arange(-3*s,3*s+1);w=np.exp(-k**2/(2*s*s));w/=w.sum()
 pad=np.pad(a,(3*s,3*s),mode='edge');return np.convolve(pad,w,'valid')
MED=9
def median(a,n):
 r=n//2;pad=np.pad(a,(r,r),mode='edge');return np.median(np.lib.stride_tricks.sliding_window_view(pad,n),axis=1)
def depth_axis(C,fi,p):
 # Pitch direction that moves the foot straight up the image at p: the camera's depth axis.
 a,b=C.to_pitch(fi,[p,(p[0],p[1]-10)]);v=b-a;n=np.linalg.norm(v)
 return v/n if n>0 else np.array([0.,1.])
def analyse(C,foot,lo,hi,end=None):
 # foot: (N,2) image points (nan outside the track). end: last frame counted in the stats
 # (the shot or goal); defaults to the end of the track.
 N=len(foot);XY=np.full((N,2),np.nan)
 for i in range(lo,hi+1):XY[i]=C.to_pitch(i,[foot[i]])[0]
 mid=(lo+hi)//2;dz=depth_axis(C,mid,foot[mid]);dx=np.array([dz[1],-dz[0]])
 u,v=XY[lo:hi+1]@dx,XY[lo:hi+1]@dz
 # A running median first: detector boxes jump for a frame or two when a limb or a neighbour
 # changes the box, and those jumps would read as bursts of speed.
 u,v=gsmooth(median(u,MED),6),gsmooth(median(v,MED),10)
 S=np.full((N,2),np.nan);S[lo:hi+1]=np.outer(u,dx)+np.outer(v,dz)
 V=np.full((N,2),np.nan);V[lo:hi+1]=np.gradient(S[lo:hi+1],axis=0)*FPS
 kmh=np.hypot(V[:,0],V[:,1])*3.6
 dist=np.full(N,np.nan);dist[lo:hi+1]=np.concatenate([[0],np.cumsum(np.hypot(*np.diff(S[lo:hi+1],axis=0).T))])
 end=hi if end is None else int(min(max(end,lo+1),hi))
 # The smoothing window is lopsided near the track ends, so the top speed skips 0.25 s there.
 a,b=lo+8,max(lo+9,end-2);top_f=a+int(np.nanargmax(kmh[a:b+1])) if b>a else lo
 run=float(dist[end]);secs=(end-lo)/FPS
 return {'xy':XY,'xys':S,'kmh':kmh,'dist':dist,'top':float(kmh[top_f]),'top_f':int(top_f),'run':run,
         'avg':run/secs*3.6 if secs>0 else 0.,'end':end,'lo':lo,'hi':hi}
