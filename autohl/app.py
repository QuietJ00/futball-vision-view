import io,os
import cv2,numpy as np
from flask import Flask,jsonify,request,send_file,abort,render_template
from . import pipeline,pitch,ui,video
from .pipeline import Job
# Local web app: upload or link a clip, click your player, check, render.
app=Flask(__name__,template_folder='templates',static_folder='static')
app.config['MAX_CONTENT_LENGTH']=4*1024**3

def job_or_404(jid):
 j=Job(jid)
 if not j.state:abort(404)
 return j

@app.get('/')
def index():return render_template('index.html')
@app.get('/job/<jid>')
def job_page(jid):job_or_404(jid);return render_template('job.html',jid=jid)

@app.get('/api/venues')
def api_venues():return jsonify([{'id':k,**v} for k,v in pitch.venues().items()])
@app.get('/api/colors')
def api_colors():return jsonify({k:'#%02x%02x%02x'%v for k,v in ui.COLORS.items()})
@app.get('/api/jobs')
def api_jobs():return jsonify(pipeline.list_jobs())
@app.post('/api/jobs')
def api_new():
 f=request.form;up=request.files.get('file')
 if up is not None and not up.filename:up=None
 if not up and not f.get('url'):return jsonify(error='영상 파일을 올리거나 유튜브 링크를 넣어 주세요.'),400
 length,width=f.get('length'),f.get('width')
 if not f.get('venue') and (length or width):
  try:
   length,width=float(length),float(width)
   if not np.isfinite([length,width]).all() or min(length,width)<=0:raise ValueError
  except (TypeError,ValueError):return jsonify(error='구장 크기를 입력하려면 실제 길이와 폭을 모두 적어 주세요.'),400
 cfg={'url':f.get('url') or None,'start':f.get('start',''),'end':f.get('end',''),
      'number':f.get('number','').strip(),'name':f.get('name','').strip(),
      'team':{'club':f.get('team_club',''),'squad':f.get('team_squad',''),'color':f.get('team_color','blue')},
      'opp':{'club':f.get('opp_club',''),'squad':f.get('opp_squad',''),'color':f.get('opp_color','yellow')},
      'event':f.get('event','goal'),'venue':f.get('venue') or None,'length':length,'width':width,
      'show_clock':f.get('show_clock')=='on','clock_start':f.get('clock','').strip()}
 j=pipeline.new_job(cfg,up);return jsonify(id=j.id)

@app.get('/api/job/<jid>')
def api_job(jid):
 j=job_or_404(jid);s=dict(j.state);m=pipeline.venue_meta(j);s['landmarks']=list(pitch.landmarks(m['length'],m['width'],m).keys())
 s['track_fresh']=pipeline.track_fresh(j);return jsonify(s)

@app.get('/api/job/<jid>/frame/<int:fi>.jpg')
def api_frame(jid,fi):
 # Frame preview at half size with optional overlays: d = detections, t = track, p = pitch lines.
 j=job_or_404(jid);F=j.frames();fi=min(max(fi,0),len(F)-1);im=np.ascontiguousarray(F[fi]).copy();ov=request.args.get('ov','')
 if 'p' in ov:
  C=j.calib()
  if C is not None and C.supports(fi):
   for l in pitch.model_lines(C.meta):
    p=C.to_img(fi,l);ok=np.isfinite(p).all(1)&(np.abs(p).max(1)<6000)
    if ok.sum()>1:cv2.polylines(im,[np.int32(p[ok])],False,(60,60,255),3,cv2.LINE_AA)
 if 'd' in ov and os.path.exists(j.f('dets.npz')):
  D=j.dets()[fi]
  for b in D.astype(int):cv2.rectangle(im,tuple(b[:2]),tuple(b[2:4]),(255,255,255),2,cv2.LINE_AA)
 if 't' in ov and os.path.exists(j.f('track.npz')):
  tr=pipeline.load_track(j);ft,h=tr['foot'],tr['height']
  for k in range(max(0,fi-30),fi):
   if np.isfinite(ft[k]).all() and np.isfinite(ft[k+1]).all():
    cv2.line(im,tuple(np.int32(ft[k])),tuple(np.int32(ft[k+1])),(255,160,60),3,cv2.LINE_AA)
  if np.isfinite(ft[fi]).all():
   x,y=ft[fi];hh=h[fi];col=(0,200,255) if tr['risk'][fi] else (255,160,60)
   cv2.rectangle(im,(int(x-hh*.33),int(y-hh)),(int(x+hh*.33),int(y+4)),col,3,cv2.LINE_AA)
 for k,(x,y) in j.state.get('anchors',{}).items():
  if int(k)==fi:cv2.circle(im,(int(x),int(y)),10,(80,255,80),3,cv2.LINE_AA)
 if 'p' in ov:
  for n,x,y in j.state.get('calib_points',{}).get(str(fi),[]):cv2.circle(im,(int(x),int(y)),9,(0,255,255),3,cv2.LINE_AA)
 im=cv2.resize(im,(960,540),interpolation=cv2.INTER_AREA);ok,buf=cv2.imencode('.jpg',im,[cv2.IMWRITE_JPEG_QUALITY,85])
 return send_file(io.BytesIO(buf.tobytes()),mimetype='image/jpeg')

@app.post('/api/job/<jid>/anchors')
def api_anchors(jid):
 j=job_or_404(jid);b=request.get_json();upd={}
 if 'anchors' in b:upd['anchors']={str(int(k)):[float(v[0]),float(v[1])] for k,v in b['anchors'].items()}
 if 'event_frame' in b:upd['event_frame']=None if b['event_frame'] is None else int(b['event_frame'])
 j.set(**upd);return jsonify(ok=True)

@app.post('/api/job/<jid>/track')
def api_track(jid):
 j=job_or_404(jid)
 try:return jsonify(pipeline.run_track(jid))
 except ValueError as e:return jsonify(error=str(e)),400

@app.post('/api/job/<jid>/calib')
def api_calib(jid):
 j=job_or_404(jid);b=request.get_json()
 if b.get('action')=='confirm':
  c=dict(j.state.get('calib') or {});c['confirmed']=True;j.set(calib=c);return jsonify(ok=True)
 if b.get('action')=='points':
  try:cal=pipeline.manual_calib(jid,int(b['frame']),b['points'])
  except ValueError as e:return jsonify(error=str(e)),400
  return jsonify(ok=True,score=cal['score'],coverage=float(np.mean(cal['valid'])))
 if b.get('action')=='save_spot':
  vid=b.get('venue') or j.state['cfg'].get('venue') or 'venue'
  pipeline.save_spot(jid,vid,b.get('name') or pipeline.venue_meta(j).get('name','구장'));return jsonify(ok=True)
 return jsonify(error='?'),400

@app.get('/api/job/<jid>/field/<int:fi>')
def api_field_frame(jid,fi):
 j=job_or_404(jid);C=j.calib()
 if C is None or not C.supports(fi):
  return jsonify(frame=fi,valid=False)
 H=C.img_from_pitch(fi);inverse=np.linalg.inv(H)
 result={'frame':fi,'time_s':fi/video.FPS,'valid':True,'mode':C.mode,
         'confirmed':bool((j.state.get('calib') or {}).get('confirmed')),
         'length_m':C.meta['length'],'width_m':C.meta['width'],
         'projection_model':'per_frame' if C.matrix is not None else 'nearest_reference',
         'field_to_image':H.tolist(),'image_to_field':inverse.tolist()}
 if C.paint_score is not None:result['paint_support']=float(C.paint_score[fi])
 if 'x' in request.args or 'y' in request.args:
  x=request.args.get('x',type=float);y=request.args.get('y',type=float)
  if x is None or y is None or not (0<=x<1920 and 0<=y<1080):
   return jsonify(error='1920×1080 프레임의 x와 y를 입력해 주세요.'),400
  world=inverse@[x,y,1.]
  if abs(world[2])>1e-8:
   p=world[:2]/world[2]
   if 0<=p[0]<=C.meta['length'] and abs(p[1])<=C.meta['width']/2:
    result['ground_xy_m']=p.tolist()
    jacobian=(inverse[:2,:2]-p[:,None]*inverse[2,:2])/world[2]
    result['local_m_per_px']=float(np.linalg.norm(jacobian,ord=2))
  result.setdefault('ground_xy_m',None)
 return jsonify(result)

@app.post('/api/job/<jid>/render')
def api_render(jid):
 j=job_or_404(jid);b=request.get_json(silent=True) or {};nospeed=bool(b.get('nospeed'))
 if j.state.get('status')=='rendering':return jsonify(ok=True)
 if not j.state.get('anchors'):return jsonify(error='3단계에서 선수를 한 번 이상 클릭해 주세요.',need='player'),400
 if not nospeed and not (j.state.get('calib') or {}).get('confirmed'):return jsonify(error='2단계에서 경기장 라인을 확인해 주세요. 라인 없이 만들려면 "속도 없이 만들기"를 누르세요.',need='calib'),400
 pipeline.start_render(jid,nospeed);return jsonify(ok=True)

@app.get('/api/job/<jid>/video')
def api_video(jid):
 j=job_or_404(jid);p=j.f('highlight.mp4')
 if not os.path.exists(p):abort(404)
 return send_file(p,mimetype='video/mp4',conditional=True,download_name=f"highlight_{jid}.mp4",as_attachment=request.args.get('dl')=='1')
@app.get('/api/job/<jid>/still.jpg')
def api_still(jid):
 j=job_or_404(jid);p=j.f('still.jpg')
 if not os.path.exists(p):abort(404)
 return send_file(p,mimetype='image/jpeg')

# ---------- labeling (training data for the pitch model) ----------
import json as _json
LABELS=os.path.join(pipeline.ROOT,'work','labels','manual.json')
def _labels():
 try:return _json.load(open(LABELS))
 except Exception:return {}
def _save_labels(d):
 os.makedirs(os.path.dirname(LABELS),exist_ok=True);tmp=LABELS+'.tmp';_json.dump(d,open(tmp,'w'),ensure_ascii=False,indent=1);os.replace(tmp,LABELS)
_fits={}
@app.get('/label')
def label_page():return render_template('label.html')
@app.get('/api/label/list')
def label_list():
 L=_labels();out=[]
 for j in pipeline.list_jobs():
  J=Job(j['id'])
  if not os.path.exists(J.f('frames.npy')):continue
  c=j['cfg'];out.append({'id':j['id'],'frames':J.state.get('frames',0),'title':f"{c.get('start') or ''}–{c.get('end') or ''} {(c.get('url') or '업로드')[-11:]}",'labels':sorted(map(int,L.get(j['id'],{}).keys()))})
 m=pitch.venues()['noksi'];lm=pitch.landmarks(m['length'],m['width'],m)
 return jsonify(jobs=out,landmarks=[[k,v[0],v[1]] for k,v in lm.items()],pitch={'length':m['length'],'width':m['width']},total=sum(len(v) for v in L.values()))
@app.post('/api/label/fit')
def label_fit():
 b=request.get_json();j=job_or_404(b['jid']);m=pitch.venues()['noksi']
 try:Hp=pitch.from_points(int(b['frame']),[(n,(float(x),float(y))) for n,x,y in b['points']],m)
 except ValueError as e:return jsonify(error=str(e)),400
 _fits[(b['jid'],int(b['frame']))]=Hp;return jsonify(ok=True)
@app.post('/api/label/save')
def label_save():
 b=request.get_json();L=_labels();L.setdefault(b['jid'],{})
 if b.get('delete'):L[b['jid']].pop(str(int(b['frame'])),None)
 else:L[b['jid']][str(int(b['frame']))]=b['points']
 _save_labels(L);return jsonify(ok=True,total=sum(len(v) for v in L.values()))
@app.get('/api/label/frame/<jid>/<int:fi>.jpg')
def label_frame(jid,fi):
 j=job_or_404(jid);F=j.frames();fi=min(max(fi,0),len(F)-1);im=np.ascontiguousarray(F[fi]).copy()
 H=_fits.get((jid,fi))
 if H is None:
  pts=_labels().get(jid,{}).get(str(fi))
  if pts:
   try:H=pitch.from_points(fi,[(n,(float(x),float(y))) for n,x,y in pts],pitch.venues()['noksi'])
   except ValueError:H=None
 if H is not None:
  for l in pitch.model_lines(pitch.venues()['noksi']):
   p=cv2.perspectiveTransform(np.float32(l).reshape(-1,1,2),H).reshape(-1,2);ok=np.isfinite(p).all(1)&(np.abs(p).max(1)<6000)
   if ok.sum()>1:cv2.polylines(im,[np.int32(p[ok])],False,(255,60,255),3,cv2.LINE_AA)
 im=cv2.resize(im,(1280,720),interpolation=cv2.INTER_AREA);ok,buf=cv2.imencode('.jpg',im,[cv2.IMWRITE_JPEG_QUALITY,88])
 return send_file(io.BytesIO(buf.tobytes()),mimetype='image/jpeg')

def main():
 port=int(os.environ.get('PORT',8765))
 print(f'\n  하이라이트 메이커: http://127.0.0.1:{port}\n')
 app.run(host='127.0.0.1',port=port,threaded=True,debug=False)
if __name__=='__main__':main()
