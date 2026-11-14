import subprocess
import cv2,numpy as np
from PIL import Image,ImageDraw
from functools import lru_cache
from . import ui
from .ui import fnt,spaced,spaced_w,ink_text,cap,unit,value,ease,ramp,over,P,W,H,INK,INK2,WHITE,MUTED,AMBER,GOLD
from .video import FF,FPS
# Player highlight template: one marked player with live speed, a running-stats lower third,
# then an event card (GOAL / ASSIST / SPRINT) with top and average speed.
EVENT_WORD={'goal':'GOAL','assist':'ASSIST','sprint':'SPRINT'}
STING=14

class Highlight:
 def __init__(self,F,C,tr,st,cfg):
  self.F,self.C,self.st,self.cfg=F,C,st,cfg
  self.N=len(F);self.lo,self.hi=st['lo'],st['hi']
  self.foot=tr['foot'];self.height=self._smooth_height(tr['height'])
  self.KMH,self.DIST,self.XYs=st['kmh'],st['dist'],st['xys']
  self.num=str(cfg.get('number','')).strip() or '?'
  t,o=cfg['team'],cfg['opp']
  self.tc,self.oc=ui.COLORS.get(t.get('color'),ui.BLU),ui.COLORS.get(o.get('color'),ui.YEL)
  self.tname=f"{t.get('club','')} {t.get('squad','')}".strip() or '우리 팀'
  self.oname=f"{o.get('club','')} {o.get('squad','')}".strip() or '상대 팀'
  self.event=cfg.get('event','goal');self.word=EVENT_WORD.get(self.event,'GOAL')
  self.ev=int(cfg['event_frame']) if cfg.get('event_frame') is not None else st['end']
  self.ev=min(max(self.ev,self.lo+1),self.N-1)
  self.SPMAX=max(35.,self.st['top']*1.15)
  self.speed=C is not None  # without a calibrated pitch: marker and event card only
  self.clock0=cfg.get('clock_start') if cfg.get('show_clock',True) else None
 def _smooth_height(self,h):
  out=h.copy();ok=np.isfinite(h)
  if ok.sum()>5:
   from .speed import gsmooth
   idx=np.nonzero(ok)[0];out[idx]=gsmooth(h[idx],8)
  return out
 def tracked(self,fi):return self.lo<=fi<=self.hi and np.isfinite(self.foot[fi]).all()
 def clock(self,fi):
  if self.clock0 is None:return None
  s=self.clock0+fi/FPS;return f'{int(s//60):02d}:{int(s%60):02d}'
 def v(self,fi):return int(min(max(fi,self.lo),self.st['end']))

 # ---------- ground ----------
 def trail(self,layer,fi):
  a=max(self.lo,fi-40);b=min(fi,self.hi)
  if b-a<2:return
  pts=self.C.to_img(fi,self.XYs[a:b+1]);m=np.zeros((H,W),np.uint8);g=np.zeros((H,W),np.uint8)
  for i in range(1,len(pts)):
   k=i/len(pts);cv2.line(g,P(pts[i-1]),P(pts[i]),int(60*k),8,cv2.LINE_AA);cv2.line(m,P(pts[i-1]),P(pts[i]),int(255*k**1.3),3,cv2.LINE_AA)
  light=tuple(int(c+(255-c)*.45) for c in self.tc)
  over(layer,g,self.tc);over(layer,m,light)
 def disc(self,layer,fi):
  t=np.linspace(0,2*np.pi,48)
  if self.speed:
   c=self.C.to_pitch(fi,[self.foot[fi]])[0];ring=self.C.to_img(fi,np.stack([c[0]+.9*np.cos(t),c[1]+.9*np.sin(t)],1))
  else:
   # No pitch model: a flat ellipse sized from the player's height in the image.
   x,y=self.foot[fi];r=self.height[fi]*.34;ring=np.stack([x+r*np.cos(t),y+r*.3*np.sin(t)],1)
  if not np.isfinite(ring).all() or np.abs(ring).max()>5000:return
  m=np.zeros((H,W),np.uint8);cv2.fillPoly(m,[np.int32(ring*16)],70,cv2.LINE_AA,shift=4);cv2.polylines(m,[np.int32(ring*16)],True,230,2,cv2.LINE_AA,shift=4)
  over(layer,m,self.tc)
 def pbox(self,fi):
  x,y=self.foot[fi];h=self.height[fi];return (int(x-h*.32),int(y-h-6),int(x+h*.32),int(y+6))

 # ---------- UI ----------
 @lru_cache(None)
 def speed_chip(self,txt):
  f=fnt('din',27);fu=fnt('heavy',12);nw=f.getbbox(txt)[2]-f.getbbox(txt)[0];uw=spaced_w('KM/H',fu,1);w=int(10+nw+6+uw+10)
  im=Image.new('RGBA',(w,30));d=ImageDraw.Draw(im);d.rectangle((0,0,w-1,29),fill=(*INK,235))
  ink_text(d,(10,15),txt,f,WHITE);unit(d,(10+nw+6,15+cap(f,txt)/2),'KM/H',fu,MUTED)
  return im
 def tag(self,uiL,fi,speed=True):
  x,y=self.foot[fi];top=y-self.height[fi];t=ui.tag_sprite(self.num,self.tc,fg=ui.text_on(self.tc));bx,by=x-19,top-44
  uiL.alpha_composite(t,(round(bx),round(by)))
  if speed and np.isfinite(self.KMH[fi]):uiL.alpha_composite(self.speed_chip(f'{self.KMH[fi]:.1f}'),(round(bx+40),round(by)))
 def spark(self,fi,w=250,h=46):
  lo,end=self.lo,self.st['end'];n=self.v(fi)-lo+1;span=max(end-lo,1)
  m=np.zeros((h*4,w*4),np.uint8);k=np.nan_to_num(self.KMH[lo:lo+n])
  xs=np.arange(n)/span*(w-8)*4+16;ys=(h-6-k/self.SPMAX*(h-12))*4
  if n>1:cv2.polylines(m,[np.int32(np.stack([xs,ys],1))],False,255,8,cv2.LINE_AA)
  im=Image.new('RGBA',(w,h),(*GOLD,255));im.putalpha(Image.fromarray(m).resize((w,h),Image.Resampling.LANCZOS))
  d=ImageDraw.Draw(im);d.line((4,h-6,w-4,h-6),fill=(255,255,255,60),width=1)
  cx,cy=xs[-1]/4,ys[-1]/4;dot=ui.ss_icon(12,12,WHITE,lambda dd,k:dd.ellipse((k,k,11*k,11*k),fill=255));im.alpha_composite(dot,(int(cx-6),int(cy-6)))
  return im
 def num_block(self,d,w,h,y_num,y_sub):
  fg=ui.text_on(self.tc);d.rectangle((0,0,w,h-1),fill=(*self.tc,255))
  ink_text(d,(w/2,y_num),self.num,fnt('din',56 if len(self.num)<3 else 44),fg,'m')
  spaced(d,(w/2,y_sub),self.cfg['team'].get('squad','') or self.cfg['team'].get('club',''),fnt('heavy',14) if not any(ord(c)>127 for c in self.cfg['team'].get('squad','')) else fnt('kr',14),fg,2,'m')
 def run_panel(self,fi):
  R1,R2=26,62;im=Image.new('RGBA',(880,96));d=ImageDraw.Draw(im)
  self.num_block(d,96,96,42,78)
  d.rectangle((96,0,560,95),fill=(*INK,240));spaced(d,(120,R1),'SPRINT SPEED',fnt('heavy',16),MUTED,2)
  v=self.v(fi);value(d,(120,R2),f'{np.nan_to_num(self.KMH[v]):.1f}','km/h',fnt('din',54),fnt('demi',19),WHITE)
  im.alpha_composite(self.spark(fi),(296,R2-23))
  top=np.nanmax(self.KMH[self.lo:v+1]) if v>self.lo else 0.;tf=self.st['top_f'];hot=tf<=fi<tf+40
  d.rectangle((560,0,879,95),fill=(*(AMBER if hot else INK2),245))
  lab,val,un=(INK,INK,INK) if hot else (MUTED,WHITE,MUTED)
  for x,l,n,u in [(584,'TOP SPEED',f'{top:.1f}','km/h'),(744,'DISTANCE',f'{np.nan_to_num(self.DIST[v]):.0f}','m')]:
   spaced(d,(x,R1),l,fnt('heavy',15),lab,2);value(d,(x,R2),n,u,fnt('din',44),fnt('demi',18),val,un)
  return im
 @lru_cache(None)
 def event_panel(self):
  Hh,R1,R2=128,46,88;fg=fnt('din',100);gb=fg.getbbox(self.word);wc=int(gb[2]-gb[0]+2*44)
  fn,fs,fl,fv,fu=fnt('kr',30),fnt('demi',22),fnt('heavy',15),fnt('din',50),fnt('demi',17)
  pname=self.cfg.get('name','').strip();name=f"#{self.num}  {pname or self.tname}"
  ck=self.clock(self.ev);sub=f"{ck+'  ·  ' if ck else ''}{self.st['run']:.0f} m RUN" if self.speed else (ck or self.tname)
  subf=fnt('kr',20) if any(ord(c)>127 for c in sub) else fs
  colA=int(max(fn.getlength(name),subf.getlength(sub)))
  stats=[('TOP SPEED',f"{self.st['top']:.1f}",AMBER),('AVG SPEED',f"{self.st['avg']:.1f}",WHITE)] if self.speed else []
  cols=[int(max(spaced_w(l,fl,2),fv.getbbox(n)[2]+6+fu.getlength('km/h'))) for l,n,_ in stats]
  x0=128+wc;wd=28+colA+sum(44+c for c in cols)+28
  im=Image.new('RGBA',(x0+wd,Hh));d=ImageDraw.Draw(im)
  d.rectangle((0,0,127,Hh-1),fill=(*self.tc,255));cr=ui.crest(96,self.cfg.get('crest') or ui.CREST)
  if cr:im.alpha_composite(cr,(16,16))
  else:ink_text(d,(64,Hh/2),self.num,fnt('din',72),ui.text_on(self.tc),'m')
  d.rectangle((128,0,x0-1,Hh-1),fill=(*WHITE,250));ink_text(d,(128+wc/2,Hh/2),self.word,fg,INK,'m')
  d.rectangle((x0,0,x0+wd-1,Hh-1),fill=(*INK,242))
  x=x0+28;ink_text(d,(x,R1),name,fn,WHITE);ink_text(d,(x,R2),sub,subf,MUTED);x+=colA
  for (l,n,c),cw in zip(stats,cols):
   x+=44;d.line((x-22,22,x-22,Hh-22),fill=(255,255,255,40),width=1)
   spaced(d,(x,R1),l,fl,MUTED,2);value(d,(x,R2),n,'km/h',fv,fu,c);x+=cw
  return im

 # ---------- frame ----------
 def frame(self,fi):
  bgr=np.ascontiguousarray(self.F[fi]);base=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
  ground=Image.new('RGBA',(W,H));uiL=Image.new('RGBA',(W,H));on=self.tracked(fi)
  live=fi<=self.ev+20
  if on:
   if live and self.speed:self.trail(ground,fi)
   self.disc(ground,fi)
  comp=np.array(Image.alpha_composite(Image.fromarray(base).convert('RGBA'),ground).convert('RGB'))
  if on:ui.restore(comp,base,[ui.turf_key(base,ground),ui.key_mask(bgr,self.pbox(fi),True)])
  if on:self.tag(uiL,fi,speed=self.speed and live and fi>=self.lo+6)
  uiL.alpha_composite(ui.scorebug(self.tname,self.oname,self.tc,self.oc,self.clock(fi),self.cfg.get('crest') or ui.CREST),(48,40))
  ev=self.ev
  if fi<ev:
   a=ramp(fi,self.lo+8,self.lo+18);out=ramp(fi,ev-8,ev)
   if fi>=self.lo and self.speed:
    im=self.run_panel(fi)
    if out>0:t=ease(out);uiL.alpha_composite(im.crop((int(im.width*t),0,im.width,im.height)),(48+int(im.width*t),936))
    else:ui.wipe(uiL,im,(48,936),a)
  else:ui.wipe(uiL,self.event_panel(),(48,904),ramp(fi,ev,ev+10))
  img=Image.alpha_composite(Image.fromarray(comp).convert('RGBA'),uiL)
  if self.event=='goal' and ev<=fi<ev+5:img=Image.blend(img,Image.new('RGBA',(W,H),(255,255,255,255)),.5*(1-(fi-ev)/5))
  return img.convert('RGB')
 def timeline(self):
  # Intro sting, the clip, a freeze on the last frame if the event card would be cut short,
  # then the outro sting.
  tl=[(0,('in',i/(STING-1))) for i in range(STING)]+[(i,None) for i in range(self.N)]
  hold=max(0,int(2.2*FPS)-(self.N-1-self.ev));tl+=[(self.N-1,None)]*hold
  return tl+[(self.N-1,('out',i/(STING-1))) for i in range(STING)]
 def render(self,item):
  fi,s=item;img=self.frame(fi)
  if s:img=ui.stinger(img,s[1],s[0]=='out',word='HIGHLIGHT' if s[0]=='in' else self.word)
  return img
 def encode(self,out,log=print):
  tl=self.timeline()
  cmd=[FF,'-y','-loglevel','error','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r',str(FPS),'-i','-','-f','lavfi','-i','anullsrc=r=44100:cl=stereo','-shortest','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p','-colorspace','bt709','-color_primaries','bt709','-color_trc','bt709','-c:a','aac','-movflags','+faststart',out]
  p=subprocess.Popen(cmd,stdin=subprocess.PIPE);cache={}
  for n,item in enumerate(tl):
   key=item[0] if item[1] is None else None
   if key is not None and key in cache:img=cache[key]
   else:
    img=self.render(item)
    if item[1] is None and item[0]==self.N-1:cache[key]=img
   p.stdin.write(img.tobytes())
   if n%30==0:log(f'영상 만드는 중 {n}/{len(tl)}')
  p.stdin.close()
  if p.wait()!=0:raise RuntimeError('영상 인코딩 실패')
