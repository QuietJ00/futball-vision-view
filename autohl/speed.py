import numpy as np
# Pitch-plane positions of the tracked player, smoothed, then speed in km/h. Foot points bob
# with the stride and jitter more in depth (a pixel of foot error is several times more metres
# toward the camera than across), so depth is smoothed harder than the across-screen axis.
FPS=30


def measure_window(xy, observed, first, last, fps=FPS):
    """Estimate average ground velocity from observed positions over a short interval."""
    positions = np.asarray(xy[first:last+1], dtype=float)
    keep = np.asarray(observed[first:last+1], dtype=bool) & np.isfinite(positions).all(axis=1)
    times = np.arange(len(positions))/fps
    if len(times) < 16 or keep.sum() < .75*len(times) or not (keep[0] and keep[-1]):
        raise ValueError('속도를 추정하려면 시작과 끝을 포함한 연속 관측점이 더 필요해요.')
    gaps = np.diff(np.flatnonzero(keep))
    if gaps.max() > max(4, round(.17*fps)):
        raise ValueError('선수 발 위치를 확인할 수 없는 구간이 너무 길어요.')
    fit_time = times[keep] - times.mean()
    design = np.column_stack((np.ones_like(fit_time), fit_time))
    target = positions[keep]
    weights = np.ones(len(target))
    for _ in range(5):
        root = np.sqrt(weights)[:, None]
        intercept, velocity = np.linalg.lstsq(design*root, target*root, rcond=None)[0]
        residual = np.linalg.norm(target - intercept - fit_time[:, None]*velocity, axis=1)
        scale = max(float(np.median(residual)) * 1.4826, .06)
        weights = np.minimum(1., 1.5*scale/np.maximum(residual, 1e-9))
    mps = float(np.linalg.norm(velocity))
    return {'speed_kmh': 3.6*mps, 'duration_s': float(times[-1]),
            'distance_m': mps*float(times[-1]), 'observed_frames': int(keep.sum()),
            'median_fit_residual_m': float(np.median(residual))}


def _track_model(dt, order, q):
    """One axis of a kinematic model: position and velocity, optionally acceleration."""
    if order == 2:
        F = np.array([[1., dt], [0., 1.]])
        Q = q**2*np.array([[dt**3/3, dt**2/2], [dt**2/2, dt]])
    else:
        F = np.array([[1., dt, dt**2/2], [0., 1., dt], [0., 0., 1.]])
        Q = q**2*np.array([[dt**5/20, dt**4/8, dt**3/6], [dt**4/8, dt**3/3, dt**2/2], [dt**3/6, dt**2/2, dt]])
    return np.kron(np.eye(2), F), np.kron(np.eye(2), Q)


def ground_track(xy, observed, covariance, fps=FPS, order=2, q=2., gate=16.):
    """Forward-backward Kalman smoothing of ground positions in metres.

    Differencing positions turns one pixel of foot noise into tens of km/h, because a pixel can
    be a quarter of a metre in depth. This weighs each frame by how well the camera sees it, along
    each ground direction separately, and limits how fast motion can change. Returns position,
    velocity and velocity covariance per frame, and the log-likelihood of each observation given
    the frames before it, which measures how well a setting predicts unseen data.
    """
    n = len(xy)
    F, Q = _track_model(1/fps, order, q)
    M = np.zeros((2, 2*order))
    M[0, 0] = M[1, order] = 1.
    good = np.asarray(observed, bool) & np.isfinite(xy).all(1) & np.isfinite(covariance).all(axis=(1, 2))
    first, last = int(np.argmax(good)), int(n-1-np.argmax(good[::-1]))
    x = np.zeros(2*order)
    x[[0, order]] = xy[first]
    P = np.diag(np.tile([1., 25.]+[25.]*(order-2), 2))
    size = 2*order
    filtered, predicted = np.full((n, size), np.nan), np.full((n, size), np.nan)
    filtered_cov, predicted_cov = np.zeros((n, size, size)), np.zeros((n, size, size))
    score = np.zeros(n)
    for k in range(first, last+1):
        if k > first:
            x, P = F @ x, F @ P @ F.T + Q
        predicted[k], predicted_cov[k] = x, P
        if good[k]:
            S = M @ P @ M.T + covariance[k]
            innovation = xy[k] - M @ x
            distance = innovation @ np.linalg.solve(S, innovation)
            score[k] = -.5*(min(distance, gate) + np.log(np.linalg.det(S)))
            if distance < gate:  # a jumped box is not a sprint
                K = P @ M.T @ np.linalg.inv(S)
                x, P = x + K @ innovation, (np.eye(size) - K @ M) @ P
        filtered[k], filtered_cov[k] = x, P
    state, cov = filtered.copy(), filtered_cov.copy()
    for k in range(last-1, first-1, -1):
        G = filtered_cov[k] @ F.T @ np.linalg.inv(predicted_cov[k+1])
        state[k] = filtered[k] + G @ (state[k+1] - predicted[k+1])
        cov[k] = filtered_cov[k] + G @ (cov[k+1] - predicted_cov[k+1]) @ G.T
    pick = [1, order+1]
    return {'xy': state[:, [0, order]], 'velocity': state[:, pick], 'velocity_cov': cov[:, pick][:, :, pick],
            'score': score, 'first': first, 'last': last}


def instant_speed(xy, observed, covariance, fps=FPS, order=2, q=1., gate=16.):
    """Speed in km/h at each frame with its standard deviation, from ground_track.

    Noise across the direction of travel can only lengthen a velocity vector, so a plain
    magnitude reads high. That part of the variance is removed.
    """
    track = ground_track(xy, observed, covariance, fps, order, q, gate)
    velocity, variance = track['velocity'], track['velocity_cov']
    magnitude = np.linalg.norm(velocity, axis=1)
    unit = velocity/np.maximum(magnitude, 1e-6)[:, None]
    across = np.c_[-unit[:, 1], unit[:, 0]]
    along_var = np.maximum(np.einsum('ni,nij,nj->n', unit, variance, unit), 0.)
    across_var = np.maximum(np.einsum('ni,nij,nj->n', across, variance, across), 0.)
    kmh = 3.6*np.sqrt(np.maximum(magnitude**2-across_var, 0.))
    sigma = 3.6*np.sqrt(along_var)
    outside = (np.arange(len(xy)) < track['first']) | (np.arange(len(xy)) > track['last'])
    kmh[outside], sigma[outside] = np.nan, np.nan
    return {'kmh': kmh, 'sigma': sigma, 'xy': track['xy']}


def sustained_top(kmh, width=11):
    """Top speed held over a short span. The highest single frame of a noisy curve reads high
    simply because it was picked as the highest."""
    best, frame = np.nan, None
    half = width//2
    for k in range(half, len(kmh)-half):
        span = kmh[k-half:k+half+1]
        if np.isfinite(span).sum() == width and not np.median(span) <= best:
            best, frame = float(np.median(span)), k
    return best, frame


def player_speed(C, foot, observed, first, last, fps=FPS, pixels=3., hold=3):
    """Instantaneous speed of a tracked player from pitch coordinates.

    Speed is reported only where the frame and its neighbours have pitch coordinates and the
    player was seen nearby. Elsewhere the filter would be guessing, so the frame stays NaN.
    """
    count = len(foot)
    xy, covariance, seen = np.full((count, 2), np.nan), np.full((count, 2, 2), np.nan), np.zeros(count, bool)
    known = np.zeros(count, bool)
    for k in range(first, last+1):
        known[k] = C.supports(k)
        if known[k] and np.isfinite(foot[k]).all():
            xy[k] = C.to_pitch(k, foot[k:k+1])[0]
            covariance[k] = foot_covariance(C, k, foot[k], pixels)
            seen[k] = bool(observed[k])
    result = instant_speed(xy, seen, covariance, fps)
    trusted = np.array([known[max(0, k-hold):k+hold+1].all() and seen[max(0, k-hold-1):k+hold+2].sum() >= hold+2
                        for k in range(count)])
    kmh, sigma = result['kmh'].copy(), result['sigma'].copy()
    kmh[~trusted], sigma[~trusted] = np.nan, np.nan
    path = result['xy']
    steps = np.linalg.norm(np.diff(path, axis=0), axis=1)
    steps[~(trusted[:-1] & trusted[1:])] = 0.
    top, frame = sustained_top(kmh)
    return {'kmh': kmh, 'sigma': sigma, 'xy': path, 'top_frame': frame,
            'top_kmh': None if frame is None else top,
            'top_sigma': None if frame is None else float(sigma[frame]),
            'distance_m': float(np.nansum(steps)), 'trusted_frames': int(trusted.sum())}


def foot_covariance(C, frame, foot, pixels=3.):
    """Ground covariance of a foot point seen with this many pixels of image noise."""
    foot = np.asarray(foot, float)
    points = np.stack((foot, foot+[1., 0.], foot+[0., 1.]))
    p, a, b = C.to_pitch(frame, points)
    J = np.column_stack((a-p, b-p))
    return pixels**2 * J @ J.T


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
 if hasattr(C,'supports') and not C.supports(lo,hi):
  raise ValueError('선수 추적 구간의 경기장 좌표를 확인할 수 없어요.')
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
