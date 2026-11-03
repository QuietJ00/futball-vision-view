import json,os,threading,time,traceback,uuid
import numpy as np
from . import video,camera,detect,track,pitch,speed
from .video import ROOT
# One job = one clip in its own folder. Heavy stages (decode, camera, detection) run once when
# the job is created; tracking, calibration and rendering re-run cheaply as the user adjusts.
JOBS=os.path.join(ROOT,'jobs')
_lock=threading.Lock()

class Job:
 def __init__(self,jid):
  self.id=jid;self.dir=os.path.join(JOBS,jid);self.path=os.path.join(self.dir,'job.json')
  self.state=json.load(open(self.path)) if os.path.exists(self.path) else {}
 def f(self,name):return os.path.join(self.dir,name)
 def save(self):
  with _lock:
   tmp=self.path+'.tmp';json.dump(self.state,open(tmp,'w'),ensure_ascii=False,indent=1);os.replace(tmp,self.path)
 def log(self,msg):
  self.state['message']=msg;self.state.setdefault('log',[]).append([time.strftime('%H:%M:%S'),msg]);self.state['log']=self.state['log'][-60:];self.save()
 def set(self,**kw):self.state.update(kw);self.save()
 # Lazy loaders for the cached stage outputs.
 def frames(self):return video.load(self.f('frames.npy'))
 def cam(self):z=np.load(self.f('camera.npz'));return z['cum'],int(z['sky'])
 def dets(self):return detect.load(self.f('dets.npz'))
 def calib(self):
  if not os.path.exists(self.f('calib.npz')):return None
  z=np.load(self.f('calib.npz'),allow_pickle=True);meta=json.loads(str(z['meta']))
  cal={'bases':[(int(k),H,0,None) for k,H in zip(z['ks'],z['Hs'])],'meta':meta,'mode':str(z['mode'])}
  return pitch.Calib(cal,self.cam()[0])
 def save_calib(self,cal):
  np.savez(self.f('calib.npz'),ks=np.array([b[0] for b in cal['bases']]),Hs=np.array([b[1] for b in cal['bases']]),
           meta=json.dumps(cal['meta'],ensure_ascii=False),mode=cal['mode'])

def new_job(cfg,upload=None):
 jid=time.strftime('%m%d-%H%M%S-')+uuid.uuid4().hex[:4];j=Job(jid);os.makedirs(j.dir)
 if upload:
  ext=os.path.splitext(upload.filename or '')[1].lower() or '.mp4';src=j.f('upload'+ext);upload.save(src);cfg['source_file']=src
 j.state={'id':jid,'cfg':cfg,'status':'preparing','message':'대기 중','created':time.time()};j.save()
 threading.Thread(target=prepare,args=(jid,),daemon=True).start()
 return j

def prepare(jid):
 j=Job(jid);c=j.state['cfg']
 try:
  start=video.parse_time(c['start']) if c.get('start') not in (None,'') else None
  end=video.parse_time(c['end']) if c.get('end') not in (None,'') else None
  if c.get('url'):
   if start is None or end is None:raise ValueError('유튜브 링크는 시작·끝 시간을 넣어 주세요.')
   src=video.download(c['url'],start,end,j.dir,log=j.log);ss,ee=None,None
  else:src=c['source_file'];ss,ee=start,end
  n=video.decode(src,j.f('frames.npy'),ss,ee,log=j.log);F=j.frames()
  j.log(f'프레임 {n}장 ({n/video.FPS:.1f}초)');j.set(frames=n)
  # Match clock shown on the scorebug: an explicit value, else the clip's start in the source.
  c['clock_start']=video.parse_time(c['clock_start']) if c.get('clock_start') else (start or 0.);j.set(cfg=c)
  cum,inl,sky=camera.solve(F,log=j.log);np.savez(j.f('camera.npz'),cum=cum,inl=inl,sky=sky)
  if len(inl)>1 and np.min(inl[1:])<40:j.log('주의: 화면이 크게 바뀌는 구간이 있어요(컷 전환?). 한 컷만 잘라 주세요.')
  D=detect.run(F,log=j.log);detect.save(j.f('dets.npz'),D)
  cal=None
  if c.get('venue') in pitch.venues():
   cal=pitch.auto(F,cum,c['venue'],sky,log=j.log,boxes=D)
  if cal:j.save_calib(cal);j.set(calib={'mode':'auto','score':cal['score'],'confirmed':False})
  else:j.set(calib={'mode':None,'confirmed':False})
  j.log('준비 완료');j.set(status='input')
 except Exception as e:
  traceback.print_exc();j.set(status='error',error=str(e));j.log('오류: '+str(e))

def venue_meta(j):
 c=j.state['cfg'];vs=pitch.venues()
 if c.get('venue') in vs:return vs[c['venue']]
 return {'name':c.get('venue_name') or '새 구장','length':float(c.get('length') or 100),'width':float(c.get('width') or 65)}

def manual_calib(jid,frame,points):
 j=Job(jid);F=j.frames();cum,sky=j.cam();meta=venue_meta(j)
 cal=pitch.manual(F,cum,int(frame),[(n,(float(x),float(y))) for n,x,y in points],meta,sky,log=j.log)
 j.save_calib(cal);j.set(calib={'mode':'manual','score':cal['score'],'confirmed':False,'points':points,'frame':int(frame)})
 return cal

def marks_key(s):
 # What a track was computed from; the UI compares it to know when a re-track is due.
 return json.dumps([sorted((int(k),[round(float(x),1) for x in v]) for k,v in s.get('anchors',{}).items()),s.get('event_frame')])
def track_fresh(j):
 t=j.state.get('track');return bool(t) and t.get('marks')==marks_key(j.state) and os.path.exists(j.f('track.npz'))
def run_track(jid):
 j=Job(jid);s=j.state;F=j.frames();cum,_=j.cam();D=j.dets()
 anchors={int(k):tuple(v) for k,v in s.get('anchors',{}).items()}
 tr=track.track(F,D,cum,anchors,log=j.log);rep=track.report(tr)
 np.savez(j.f('track.npz'),foot=tr['foot'],height=tr['height'],src=tr['src'],risk=tr['risk'],lo=tr['range'][0],hi=tr['range'][1])
 C=j.calib();res={'report':rep}
 if C is not None:
  ev=s.get('event_frame');lo,hi=tr['range']
  st=speed.analyse(C,tr['foot'],lo,hi,ev if ev is not None else None)
  np.savez(j.f('speed.npz'),**{k:v for k,v in st.items()})
  # Amateur sprints top out around 30-34 km/h; anything past 36 means the track jumped bodies.
  if st['top']>36:res['warn']=f"최고 속도 {st['top']:.1f} km/h는 사람이 내기 어려운 값이에요. {st['top_f']}프레임 근처에서 추적이 다른 선수로 넘어갔는지 확인해 주세요."
  res['stats']={'top':round(st['top'],1),'top_f':st['top_f'],'run':round(st['run'],1),'avg':round(st['avg'],1),'lo':lo,'hi':hi,'end':st['end']}
 res['marks']=marks_key(s)
 j.set(track=res,status='tracked');return res

def load_track(j):
 z=np.load(j.f('track.npz'));return {'foot':z['foot'],'height':z['height'],'src':z['src'],'risk':z['risk'],'range':(int(z['lo']),int(z['hi']))}
def load_speed(j):
 z=np.load(j.f('speed.npz'));st={k:z[k] for k in z.files}
 for k in ('top','run','avg'):st[k]=float(st[k])
 for k in ('top_f','end','lo','hi'):st[k]=int(st[k])
 return st

def plain_stats(tr,ev):
 # Stand-in for speed.analyse when the pitch is not calibrated: the clip plays with the player
 # marked and the event card, but no speed, distance or trail on the ground.
 lo,hi=tr['range'];N=len(tr['foot']);end=hi if ev is None else int(min(max(ev,lo+1),hi))
 nan=np.full(N,np.nan);return {'xy':np.full((N,2),np.nan),'xys':np.full((N,2),np.nan),'kmh':nan,'dist':nan,'top':0.,'top_f':int(lo),
  'run':0.,'avg':0.,'end':end,'lo':int(lo),'hi':int(hi)}
def render(jid,nospeed=False):
 from .render import Highlight
 j=Job(jid)
 try:
  j.set(status='rendering')
  if not track_fresh(j):
   j.log('선수 추적 중');run_track(jid)
   j=Job(jid);j.set(status='rendering')  # reload: run_track saved new results through its own Job
  F=j.frames();tr=load_track(j);c=dict(j.state['cfg']);c['event_frame']=j.state.get('event_frame')
  C=None if nospeed else j.calib()
  if C is not None and os.path.exists(j.f('speed.npz')):st=load_speed(j)
  else:C=None;st=plain_stats(tr,c['event_frame'])
  h=Highlight(F,C,tr,st,c);out=j.f('highlight.mp4')
  # Still: the top-speed moment, or (without speed) the event card once it is fully on screen.
  h.frame(st['top_f'] if C is not None else min(h.ev+14,len(F)-1)).save(j.f('still.jpg'),quality=92)
  h.encode(out,log=j.log)
  j.log('완성');j.set(status='done',output='highlight.mp4',nospeed=C is None)
 except Exception as e:
  traceback.print_exc();j.set(status='error',error=str(e));j.log('오류: '+str(e))
def start_render(jid,nospeed=False):threading.Thread(target=render,args=(jid,nospeed),daemon=True).start()

def save_spot(jid,vid,name):
 # Keep this clip's calibrated frames as references so later clips from here calibrate alone.
 j=Job(jid);F=j.frames();cum,_=j.cam();C=j.calib();meta=venue_meta(j);n=len(F)
 cal={'bases':[(int(k),C.img_from_pitch(int(k),i),0,None) for i,k in enumerate(C.ks)],'meta':meta}
 keep={k:meta[k] for k in ('length','width','half_x','surface') if k in meta}
 pitch.save_venue(vid,name,keep,F,cum,cal,sorted(set([0,n//3,2*n//3,n-1])))

def list_jobs():
 if not os.path.isdir(JOBS):return []
 out=[]
 for d in sorted(os.listdir(JOBS),reverse=True):
  p=os.path.join(JOBS,d,'job.json')
  if os.path.exists(p):
   try:s=json.load(open(p));out.append({'id':d,'status':s.get('status'),'cfg':s.get('cfg',{}),'message':s.get('message')})
   except Exception:pass
 return out
