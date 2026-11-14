import subprocess
import cv2,numpy as np
from PIL import Image,ImageDraw
from functools import lru_cache
from . import ui,track
from .ui import fnt,spaced,ink_text,ease,ramp,fade,over,P,W,H,INK,INK2,WHITE,MUTED,BRAND,AMBER,GO,RED
from .video import FF,FPS
# Offside check at the pass: the second-last defender's line and the attacker's line on the
# pitch (constant X), the gap between them measured along the touchline, and a VAR-style verdict.
DEFC=(40,170,255);ATTC=(255,64,96)
HOLD=96;STING=14;RESULT_AFTER=42

def snap(D,pt):
 # Foot point (box bottom centre) and box of the detection under a click.
 k=track.pick(D,pt)
 if k is None:return None,None
 b=D[k];return np.array([(b[0]+b[2])/2,b[3]]),b[:4]

def height_model(D):
 # Standing height in pixels grows linearly with distance below the horizon: h = a*(y - y0).
 # Fit it to every detected player in the frame (median slopes, robust to sitting/partial boxes).
 if len(D)<3:return lambda y:np.full_like(np.asarray(y,float),60.)
 y=D[:,3];h=D[:,3]-D[:,1];A=np.stack([y,np.ones_like(y)],1)
 sol,*_=np.linalg.lstsq(A,h,rcond=None);a,b=sol
 r=np.abs(A@sol-h);keep=r<np.percentile(r,70)+1
 if keep.sum()>=3:(a,b),*_=np.linalg.lstsq(A[keep],h[keep],rcond=None)
 return lambda yy:np.maximum(a*np.asarray(yy,float)+b,8.)

def analyse(C,dets,kf,clicks,meta):
 # clicks: {'att':(x,y), 'def':(x,y), 'pas':(x,y) optional}, all at the kick frame kf.
 if hasattr(C,'supports') and not C.supports(kf):
  raise ValueError('패스 순간의 경기장 좌표를 확인할 수 없어요.')
 D=dets[kf];out={'kick':int(kf)}
 for r in ('att','def','pas'):
  if clicks.get(r) is None:continue
  f,b=snap(D,clicks[r])
  if f is None:
   if r=='pas':continue
   raise ValueError({'att':'공격수','def':'수비수'}[r]+' 위치에서 선수를 찾지 못했어요. 선수 몸 위를 다시 눌러 주세요.')
  out[r]={'foot':f.tolist(),'box':[float(v) for v in b],'xy':C.to_pitch(kf,[f])[0].tolist()}
 if 'att' not in out or 'def' not in out:raise ValueError('공격수와 최종 수비수를 모두 찍어 주세요.')
 L=meta['length'];hx=meta.get('half_x',L/2)
 xa,ya=out['att']['xy'];xd,yd=out['def']['xy']
 # The defending side's goal is the end the defender stands nearer to.
 side=-1 if xd<hx else 1;out['side']=side
 ahead=(xa-xd)*side           # > 0: attacker nearer the goal line than the defender
 own_half=(xa-hx)*side<0
 behind_ball=False
 if 'pas' in out:behind_ball=(xa-out['pas']['xy'][0])*side<0
 out['gap']=float(ahead);out['own_half']=bool(own_half);out['behind_ball']=bool(behind_ball)
 out['verdict']='offside' if ahead>0 and not own_half and not behind_ball else 'onside'
 return out

class Offside:
 def __init__(self,F,C,dets,res,tracks,cfg):
  self.F,self.C,self.res,self.cfg=F,C,res,cfg;self.N=len(F);self.kf=res['kick'];self.tracks=tracks
  self.side=res['side'];self.D=dets[self.kf];self.hm=height_model(self.D)
  t,o=cfg['team'],cfg['opp']
  self.tc,self.oc=ui.COLORS.get(t.get('color'),ui.BLU),ui.COLORS.get(o.get('color'),ui.YEL)
  self.tname=f"{t.get('club','')} {t.get('squad','')}".strip() or '우리 팀'
  self.oname=f"{o.get('club','')} {o.get('squad','')}".strip() or '상대 팀'
  self.anum=str(cfg.get('number','')).strip() or 'ATT';self.pnum=str(cfg.get('passer','')).strip()
  self.clock0=cfg.get('clock_start') if cfg.get('show_clock',True) else None
  self.verdict=res['verdict'];self.word='OFFSIDE' if self.verdict=='offside' else 'ONSIDE'
  self.end=min(self.N-1,self.kf+int(4*FPS))
  meta=C.meta;w=meta['width']/2;xa,ya=res['att']['xy'];xd=res['def']['xy'][0]
  # Ground lines at the kick frame, clipped to the visible pitch.
  self.dline=self._line(xd,w);self.aline=self._line(xa,w)
  self.gap_a=C.to_img(self.kf,[(xd,ya)])[0];self.gap_b=C.to_img(self.kf,[(xa,ya)])[0]
  tick=C.to_img(self.kf,[(xd,ya-.6),(xd,ya+.6)]);self.tick=(tick[1]-tick[0])/2
  self.masks=None
 def _line(self,x,w):
  ys=np.linspace(-w,w,160);p=self.C.to_img(self.kf,np.stack([np.full_like(ys,x),ys],1))
  ok=np.isfinite(p).all(1)&(p[:,0]>-200)&(p[:,0]<W+200)&(p[:,1]>0)&(p[:,1]<H+50)
  p=p[ok];return p if len(p)>=2 else None
 def clock(self,fi,precise=False):
  if self.clock0 is None:return None
  s=self.clock0+fi/FPS;return f'{int(s//60):02d}:{s%60:05.2f}' if precise else f'{int(s//60):02d}:{int(s%60):02d}'

 # ---------- ground ----------
 def gline(self,layer,pts,color,th=3,glow=10,op=1.,t=1.):
  if pts is None or op<=0 or t<=0:return
  n=max(2,int(len(pts)*t));q=pts[:n];m=np.zeros((H,W),np.uint8)
  # Draw from the near touchline outward, as broadcast lines are sketched.
  if glow:cv2.polylines(m,[np.int32(q*16)],False,56,glow,cv2.LINE_AA,shift=4)
  cv2.polylines(m,[np.int32(q*16)],False,255,th,cv2.LINE_AA,shift=4);over(layer,m,color,op)
 def plane(self,layer,color,rise,op=1.):
  # Translucent vertical wall standing on the offside line, one standing height tall.
  if rise<=0 or op<=0 or self.dline is None:return
  p=self.dline;ht=self.hm(p[:,1])*1.05*rise;top=p-np.stack([np.zeros_like(ht),ht],1)
  m=np.zeros((H,W),np.uint8);poly=np.concatenate([p,top[::-1]])
  cv2.fillPoly(m,[np.int32(poly*16)],70,cv2.LINE_AA,shift=4)
  cv2.polylines(m,[np.int32(top[::4]*16)],False,120,1,cv2.LINE_AA,shift=4);over(layer,m,color,op)
 def bracket(self,layer,t,op):
  if t<=0:return
  a,b=self.gap_a,self.gap_b;m=np.zeros((H,W),np.uint8);e=a+(b-a)*t
  cv2.line(m,P(a),P(e),255,2,cv2.LINE_AA)
  for q in ([a,b] if t>=1 else [a]):cv2.line(m,P(q-self.tick),P(q+self.tick),255,2,cv2.LINE_AA)
  over(layer,m,WHITE,op*.95)
 def player_masks(self):
  # Every player near the drawn lines stays in front of them; mattes are cut once (frozen frame).
  if self.masks is None:
   bgr=np.ascontiguousarray(self.F[self.kf]);ms=[]
   pts=np.concatenate([x for x in (self.dline,self.aline) if x is not None]) if (self.dline is not None or self.aline is not None) else np.zeros((0,2))
   for b in self.D:
    x1,y1,x2,y2=b[:4];cx=(x1+x2)/2
    near=len(pts) and np.min(np.abs(pts[:,0]-cx)+np.maximum(0,np.abs(pts[:,1]-(y1+y2)/2)-(y2-y1)))<(x2-x1)*2+40
    if near or abs(cx-self.gap_a[0])<120:
     h=y2-y1;ms.append(ui.key_mask(bgr,(int(x1-h*.1),int(y1-4),int(x2+h*.1),int(y2+6)),True))
   self.masks=ms
  return self.masks

 # ---------- sprites ----------
 def tag(self,uiL,fi,role,num,color):
  tr=self.tracks.get(role)
  if tr is None or not np.isfinite(tr['foot'][fi]).all():return
  x,y=tr['foot'][fi];h=tr['height'][fi]
  t=ui.tag_sprite(num,color,fg=ui.text_on(color));uiL.alpha_composite(t,(round(x-19),round(y-h-44)))
 def chip(self,text,bg,fg,size=17,sub=None):
  f=fnt('heavy',size);w=int(sum(f.getlength(c) for c in text)+2*(len(text)-1))+22;sw=0
  if sub:fs=fnt('din',size+6);sw=int(fs.getlength(sub))+20
  im=Image.new('RGBA',(w+sw,size+15));d=ImageDraw.Draw(im);d.rectangle((0,0,w-1,size+14),fill=(*bg,250));spaced(d,(11,(size+15)/2),text,f,fg,2)
  if sub:d.rectangle((w,0,w+sw-1,size+14),fill=(*INK,235));ink_text(d,(w+10,(size+15)/2),sub,fs,WHITE)
  return im
 @lru_cache(None)
 def icon(self,ok,color,n=45,k=4):
  s=n*k;m=Image.new('L',(s,s));d=ImageDraw.Draw(m)
  d.ellipse((k,k,s-k-1,s-k-1),outline=255,width=4*k)
  if ok:
   d.line((11*k,23*k,19*k,31*k,34*k,14*k),fill=255,width=5*k,joint='curve')
   for p in [(11,23),(34,14)]:d.ellipse((p[0]*k-2.5*k,p[1]*k-2.5*k,p[0]*k+2.5*k,p[1]*k+2.5*k),fill=255)
  else:
   for a,b in [((14,14),(31,31)),((31,14),(14,31))]:d.line((a[0]*k,a[1]*k,b[0]*k,b[1]*k),fill=255,width=5*k)
  im=Image.new('RGBA',(n,n),(*color,255));im.putalpha(m.resize((n,n),Image.Resampling.LANCZOS));return im
 def var_panel(self,state,h):
  im=Image.new('RGBA',(800,96));d=ImageDraw.Draw(im)
  d.rectangle((0,0,96,95),fill=(*BRAND,255));ink_text(d,(48,40),'VAR',fnt('din',50),WHITE,'m');spaced(d,(48,76),'REVIEW',fnt('heavy',13),WHITE,2,'m')
  d.rectangle((96,0,520,95),fill=(*INK,240));spaced(d,(122,32),'OFFSIDE CHECK',fnt('heavy',32),WHITE,2)
  f=fnt('demi',21);x=123
  if self.pnum:
   x+=ink_text(d,(x,68),f'PASS  #{self.pnum}',f,MUTED)+12
   im.alpha_composite(ui.ss_icon(26,12,MUTED,lambda d,k:[d.rectangle((0,5*k,19*k,7*k),fill=255),d.polygon([(25*k,6*k),(17*k,k),(17*k,11*k)],fill=255)]),(int(x),62));x+=36
   x+=ink_text(d,(x,68),f'#{self.anum}',f,MUTED)+26
  else:x+=ink_text(d,(x,68),f'#{self.anum}',f,MUTED)+26
  ck=self.clock(self.kf,True);ink_text(d,(x,68),'KICK POINT  '+(ck or f'{self.kf/FPS:05.2f}s'),f,MUTED)
  if state=='checking':
   d.rectangle((520,0,799,95),fill=(*INK2,245));d.rectangle((520,0,525,95),fill=(*AMBER,255))
   spaced(d,(548,36),'CHECKING'+'.'*(1+int(h/6)%3),fnt('heavy',30),AMBER,3)
   ink_text(d,(549,68),'POSSIBLE OFFSIDE',fnt('demi',19),MUTED)
   d.rectangle((520,91,799,95),fill=(255,255,255,30));d.rectangle((520,91,520+279*min(h/56,1),95),fill=(*AMBER,255))
  elif state=='complete':
   d.rectangle((520,0,799,95),fill=(*WHITE,250));spaced(d,(660,48),'CHECK COMPLETE',fnt('heavy',25),INK,2,'m')
  else:
   off=self.verdict=='offside';c=RED if off else GO;fg=WHITE if off else INK
   word=self.word;fw=fnt('din',58 if off else 62);ww=fw.getbbox(word)[2]-fw.getbbox(word)[0]
   cw=26+45+16+ww+26;im=im.crop((0,0,520+cw,96));d=ImageDraw.Draw(im)
   d.rectangle((520,0,520+cw-1,95),fill=(*c,255));im.alpha_composite(self.icon(not off,fg),(546,26))
   ink_text(d,(546+45+16,48),word,fw,fg)
  return im
 def kick_point(self,uiL,h,op):
  if 'pas' not in self.res or op<=0:return
  x,y=self.res['pas']['foot'];m=np.zeros((H,W),np.uint8)
  cv2.ellipse(m,P((x,y)),(19,8),0,0,360,240,2,cv2.LINE_AA)
  r=(h%24)/24;cv2.ellipse(m,P((x,y)),(int(19+26*r),int(8+11*r)),0,0,360,int(200*(1-r)),2,cv2.LINE_AA)
  over(uiL,m,WHITE,op)
  c=self.chip('KICK POINT',BRAND,WHITE,16,self.clock(self.kf,True));cx=min(max(x+40,10),W-c.width-10);cy=min(max(y-60,10),H-200)
  uiL.alpha_composite(fade(c,op),(int(cx),int(cy)))
 def gap_chip(self):
  g=abs(self.res['gap']);txt=f'{g:.2f} m' if g<10 else f'{g:.1f} m'
  return self.chip('AHEAD' if self.res['gap']>0 else 'BEHIND',ATTC if self.res['gap']>0 else DEFC,WHITE if self.res['gap']>0 else INK,15,txt)
 def line_label(self,uiL,pts,text,color,fg,op,right):
  if pts is None or op<=0:return
  # Label where the line crosses the lower part of the screen, outside the line.
  k=int(np.argmin(np.abs(pts[:,1]-min(880,pts[:,1].max()-10))));q=pts[k];c=self.chip(text,color,fg,15)
  x=q[0]+18 if right else q[0]-c.width-18;uiL.alpha_composite(fade(c,op),(int(min(max(x,8),W-c.width-8)),int(q[1]-c.height/2)))

 # ---------- frame ----------
 def grade(self,rgb,k):
  if k<=0:return rgb
  g=rgb.astype(np.float32);l=g.mean(2,keepdims=True);g=g+(l-g)*.28*k;g*=1-.1*k
  yy,xx=np.mgrid[0:H,0:W];vig=(np.clip((((xx-W/2)/(W/2))**2+((yy-H/2)/(H/2))**2-.5)/1.5,0,1))[...,None]
  g*=1-.22*k*vig;return np.clip(g,0,255).astype(np.uint8)
 def frame(self,fi,hold=None,after=None):
  frz=hold is not None;h=hold or 0
  k=0 if not frz else ease(h/6)*(1-ramp(h,HOLD-8,HOLD))
  bgr=np.ascontiguousarray(self.F[fi]);raw=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);base=self.grade(raw,k)
  ground=Image.new('RGBA',(W,H));uiL=Image.new('RGBA',(W,H));out=1-ramp(h,HOLD-10,HOLD) if frz else 1
  if frz:
   self.gline(ground,self.dline,DEFC,3,10,out,ease(ramp(h,10,26)))
   self.gline(ground,self.aline,ATTC,3,12,out,ease(ramp(h,18,32)))
   self.bracket(ground,ease(ramp(h,40,50)),out)
  comp=np.array(Image.alpha_composite(Image.fromarray(base).convert('RGBA'),ground).convert('RGB'))
  if frz:
   ms=self.player_masks();ui.restore(comp,base,ms)
   pl=Image.new('RGBA',(W,H));self.plane(pl,DEFC,ease(ramp(h,26,44)),out)
   comp=np.array(Image.alpha_composite(Image.fromarray(comp).convert('RGBA'),pl).convert('RGB'));ui.restore(comp,base,ms)
  self.tag(uiL,fi,'att',self.anum,ATTC);self.tag(uiL,fi,'def','DEF',DEFC)
  if self.pnum:self.tag(uiL,fi,'pas',self.pnum,self.tc)
  if frz:
   self.kick_point(uiL,h,ease(ramp(h,4,12))*out)
   lo=ease(ramp(h,34,46))*out
   self.line_label(uiL,self.dline,'OFFSIDE LINE',DEFC,INK,lo,self.side>0)
   self.line_label(uiL,self.aline,f'#{self.anum}',ATTC,WHITE,lo,self.side<0)
   go=ease(ramp(h,48,56))*out
   if go>0:
    c=self.gap_chip();m=(self.gap_a+self.gap_b)/2;uiL.alpha_composite(fade(c,go),(int(min(max(m[0]-c.width/2,8),W-c.width-8)),int(min(m[1]+22,H-c.height-150))))
  uiL.alpha_composite(ui.scorebug(self.tname,self.oname,self.tc,self.oc,self.clock(fi),self.cfg.get('crest') or ui.CREST),(48,40))
  if frz:
   st='checking' if h<56 else 'complete' if h<66 else 'result';ui.wipe(uiL,self.var_panel(st,h),(48,936),ramp(h,2,12))
  elif after is not None and after<RESULT_AFTER+8:
   im=self.var_panel('result',0)
   if after<RESULT_AFTER:uiL.alpha_composite(im,(48,936))
   else:t=ease(ramp(after,RESULT_AFTER,RESULT_AFTER+8));uiL.alpha_composite(im.crop((int(im.width*t),0,im.width,im.height)),(48+int(im.width*t),936))
  img=Image.alpha_composite(Image.fromarray(comp).convert('RGBA'),uiL)
  if frz and h<6:img=Image.blend(img,Image.new('RGBA',(W,H),(255,255,255,255)),.45*(1-h/6))
  return img.convert('RGB')
 def timeline(self):
  # Build-up from up to 4 s before the pass, the freeze, then up to 4 s after it.
  a=max(0,self.kf-int(4*FPS))
  tl=[(a,None,None,('in',i/(STING-1))) for i in range(STING)]
  tl+=[(i,None,None,None) for i in range(a,self.kf)]+[(self.kf,hh,None,None) for hh in range(HOLD+1)]
  tl+=[(i,None,i-self.kf-1,None) for i in range(self.kf+1,self.end+1)]
  return tl+[(self.end,None,999,('out',i/(STING-1))) for i in range(STING)]
 def render(self,item):
  fi,hold,after,st=item;img=self.frame(fi,hold,after)
  if st:img=ui.stinger(img,st[1],st[0]=='out',word='VAR CHECK' if st[0]=='in' else self.word)
  return img
 def still(self):return self.frame(self.kf,72)
 def encode(self,out,log=print):
  tl=self.timeline()
  cmd=[FF,'-y','-loglevel','error','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r',str(FPS),'-i','-','-f','lavfi','-i','anullsrc=r=44100:cl=stereo','-shortest','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p','-colorspace','bt709','-color_primaries','bt709','-color_trc','bt709','-c:a','aac','-movflags','+faststart',out]
  p=subprocess.Popen(cmd,stdin=subprocess.PIPE);cache={}
  for n,item in enumerate(tl):
   key=item if item[3] is None else None
   img=self.render(item)
   p.stdin.write(img.tobytes())
   if n%30==0:log(f'영상 만드는 중 {n}/{len(tl)}')
  p.stdin.close()
  if p.wait()!=0:raise RuntimeError('영상 인코딩 실패')
