import os
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
from functools import lru_cache
# Broadcast graphics primitives shared by every template (same look as the hand-made edits).
W,H=1920,1080
INK=(9,13,22);INK2=(24,31,44);WHITE=(244,247,250);MUTED=(150,163,182)
BRAND=(112,84,255);YEL=(206,255,64);BLU=(46,118,255);RED=(255,64,96)
AMBER=(255,186,48);GO=(38,218,128);GOLD=(255,216,120)
COLORS={'blue':BLU,'yellow':YEL,'red':RED,'green':GO,'white':WHITE,'purple':BRAND,'orange':AMBER}
DIN='/System/Library/Fonts/Supplemental/DIN Condensed Bold.ttf';AVN='/System/Library/Fonts/Avenir Next Condensed.ttc'
KR='/System/Library/Fonts/AppleSDGothicNeo.ttc'
CREST=os.path.join(os.path.dirname(os.path.abspath(__file__)),'assets','crest.png')

@lru_cache(None)
def fnt(kind,n):
 if kind=='din':return ImageFont.truetype(DIN,n)
 if kind=='kr':return ImageFont.truetype(KR,n,index=14)
 return ImageFont.truetype(AVN,n,index={'heavy':8,'bold':0,'demi':2,'med':5}[kind])
def ease(t):t=min(max(t,0.),1.);return 1-(1-t)**3
def ease_io(t):t=min(max(t,0.),1.);return 4*t**3 if t<.5 else 1-(-2*t+2)**3/2
def ramp(h,a,b):return min(max((h-a)/(b-a),0.),1.)
def P(q):return tuple(np.rint(q).astype(int))
def spaced(d,xy,s,font,fill,track=2,anchor='l'):
 # Letter-spaced caps; y is the caps' visual centre.
 x,y=xy;w=sum(font.getlength(c) for c in s)+track*(len(s)-1);b=font.getbbox(s)
 if anchor=='m':x-=w/2
 for c in s:d.text((x,y-(b[1]+b[3])/2),c,font=font,fill=fill);x+=font.getlength(c)+track
 return w
def spaced_w(t,f,track):return sum(f.getlength(c) for c in t)+track*(len(t)-1)
def ink_text(d,xy,s,font,fill,ax='l',ay='m'):
 # Place by the glyphs' ink box; ax l/m/r, ay 'm' centres the ink on y, 'b' puts its bottom on y.
 x,y=xy;b=font.getbbox(s)
 ox={'l':x-b[0],'m':x-(b[0]+b[2])/2,'r':x-b[2]}[ax];oy=y-(b[1]+b[3])/2 if ay=='m' else y-b[3]
 d.text((ox,oy),s,font=font,fill=fill);return b[2]-b[0]
def cap(f,t):b=f.getbbox(t);return b[3]-b[1]
def unit(d,xy,t,f,fill,track=1):
 x,y=xy;spaced(d,(x,y-cap(f,t)/2),t,f,fill,track)
def value(d,xy,num,u,f,fu,fill,ufill=MUTED):
 # Number with its unit; the unit's bottom sits on the number's baseline.
 x,y=xy;w=ink_text(d,(x,y),num,f,fill);bot=y+cap(f,num)/2
 ink_text(d,(x+w+6,bot),u,fu,ufill,'l','b');return w+6+fu.getlength(u)

# ---------- compositing ----------
def over(layer,mask,color,op=1.):
 if op<=0:return
 s=Image.new('RGBA',(W,H),(*color,255));s.putalpha(Image.fromarray((mask*op).astype(np.uint8) if op<1 else mask,'L'));layer.alpha_composite(s)
def poly_mask(pts):
 m=np.zeros((H,W),np.uint8);cv2.fillPoly(m,[np.rint(np.array(pts,float)*16).astype(np.int32)],255,cv2.LINE_AA,shift=4);return m
def key_mask(frame,box,shoes=False,pad=22):
 # Player matte for drawing ground graphics underneath: GrabCut seeded by distance from the
 # local turf colour. frame is BGR.
 x1,y1,x2,y2=max(box[0],1),max(box[1],1),min(box[2],W-1),min(box[3],H-1)
 if x2-x1<12 or y2-y1<12:return None
 X1,Y1,X2,Y2=max(x1-pad,0),max(y1-pad,0),min(x2+pad,W),min(y2+pad,H)
 roi=np.ascontiguousarray(frame[Y1:Y2,X1:X2]);h,w=roi.shape[:2]
 lab=cv2.cvtColor(cv2.GaussianBlur(roi,(3,3),0),cv2.COLOR_BGR2LAB)
 inner=np.zeros((h,w),bool);inner[y1-Y1:y2-Y1,x1-X1:x2-X1]=True
 px=lab.reshape(-1,3).astype(np.float32);g=px[~inner.reshape(-1)];mu=g.mean(0);ic=np.linalg.inv(np.cov(g.T)+np.eye(3)*4)
 dd=px-mu;dist=np.sqrt(np.einsum('ij,jk,ik->i',dd,ic,dd)).reshape(h,w)
 m=np.full((h,w),cv2.GC_BGD,np.uint8);m[inner]=cv2.GC_PR_BGD;m[inner&(dist>3.2)]=cv2.GC_PR_FGD;m[inner&(dist>5.5)]=cv2.GC_FGD
 if shoes:
  low=np.zeros_like(inner);low[int(y1-Y1+(y2-y1)*.7):y2-Y1,x1-X1:x2-X1]=True
  m[low&(lab[...,0]>215)&(lab[...,2]<158)]=cv2.GC_FGD
 bg=np.zeros((1,65));fg=np.zeros((1,65))
 try:cv2.grabCut(roi,m,None,bg,fg,5,cv2.GC_INIT_WITH_MASK)
 except cv2.error:return None
 mm=((m==1)|(m==3)).astype(np.uint8)*255;mm=cv2.morphologyEx(mm,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
 n,lb,st,_=cv2.connectedComponentsWithStats(mm)
 if n<2:return None
 big=st[1:,cv2.CC_STAT_AREA].max();keep=np.isin(lb,[k for k in range(1,n) if st[k,cv2.CC_STAT_AREA]>max(25,big*.08)])
 return X1,Y1,cv2.GaussianBlur(keep.astype(np.float32),(3,3),0)[...,None]
def turf_key(base,ground,pad=12):
 # Anyone standing on the ground graphics: inside their bounding box, pixels far from the
 # local turf colour are restored above the ground layer. base is RGB.
 a=np.array(ground.getchannel('A'));ys,xs=np.nonzero(a)
 if len(xs)==0:return None
 x1,y1=max(xs.min()-pad,0),max(ys.min()-pad-60,0);x2,y2=min(xs.max()+pad,W),min(ys.max()+pad,H)
 roi=base[y1:y2,x1:x2];lab=cv2.cvtColor(cv2.GaussianBlur(roi,(3,3),0),cv2.COLOR_RGB2LAB).astype(np.float32)
 med=np.median(lab.reshape(-1,3),0);mad=np.median(np.abs(lab.reshape(-1,3)-med),0)*1.48+3
 d=np.sqrt((((lab-med)/mad)**2).sum(2))
 m=((d>4.2).astype(np.uint8))*255;m=cv2.morphologyEx(m,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
 n,lb,st,_=cv2.connectedComponentsWithStats(m);keep=np.isin(lb,[k for k in range(1,n) if st[k,cv2.CC_STAT_AREA]>40 and st[k,cv2.CC_STAT_HEIGHT]>10])
 return x1,y1,cv2.GaussianBlur(keep.astype(np.float32),(3,3),0)[...,None]
def restore(comp,base,masks):
 for k in masks:
  if k is None:continue
  X1,Y1,a=k;h,w=a.shape[:2];sl=(slice(Y1,Y1+h),slice(X1,X1+w))
  comp[sl]=(base[sl]*a+comp[sl]*(1-a)).astype(np.uint8)

# ---------- sprites ----------
@lru_cache(None)
def crest(n,path=CREST):
 # Club crest cropped to its circle; the source may have a white square background.
 if not path or not os.path.exists(path):return None
 im=Image.open(path).convert('RGBA');a=np.array(im);ink=(a[...,:3].min(2)<235)&(a[...,3]>20)
 ys,xs=np.where(ink)
 if len(xs)==0:return None
 cx,cy=(xs.min()+xs.max())/2,(ys.min()+ys.max())/2;r=max(xs.max()-xs.min(),ys.max()-ys.min())/2
 im=im.crop((int(cx-r),int(cy-r),int(cx+r),int(cy+r)));s=im.width*4
 m=Image.new('L',(s,s));ImageDraw.Draw(m).ellipse((6,6,s-7,s-7),fill=255)
 im.putalpha(Image.fromarray(np.minimum(np.array(im.getchannel('A')),np.array(m.resize(im.size,Image.Resampling.LANCZOS)))))
 return im.resize((n,n),Image.Resampling.LANCZOS)
def ss_icon(w,h,color,draw,k=4):
 m=Image.new('L',(w*k,h*k));draw(ImageDraw.Draw(m),k)
 im=Image.new('RGBA',(w,h),(*color,255));im.putalpha(m.resize((w,h),Image.Resampling.LANCZOS));return im
@lru_cache(None)
def tag_sprite(num,color,w=38,h=30,k=4,fg=INK):
 im=Image.new('RGBA',(w*k,(h+9)*k));d=ImageDraw.Draw(im)
 d.rounded_rectangle((0,0,w*k-1,h*k-1),radius=4*k,fill=(*color,255));d.polygon([((w/2-6)*k,(h-1)*k),((w/2+6)*k,(h-1)*k),(w/2*k,(h+8)*k)],fill=(*color,255))
 f=fnt('din',(29 if len(num)<3 else 24)*k);b=f.getbbox(num);d.text((w*k/2-(b[0]+b[2])/2,h*k/2-(b[1]+b[3])/2),num,font=f,fill=fg)
 return im.resize((w,h+9),Image.Resampling.LANCZOS)
def fade(im,op):
 if op>=1:return im
 im=im.copy();im.putalpha(im.getchannel('A').point(lambda v:int(v*op)));return im
def wipe(ui,im,xy,t):
 if t<=0:return
 w=max(1,int(im.width*ease(t)));ui.alpha_composite(im.crop((0,0,w,im.height)),xy)
def text_on(c):
 # Dark text on light fills (yellow, white, green), white on the rest.
 return INK if (.299*c[0]+.587*c[1]+.114*c[2])>150 else WHITE
@lru_cache(None)
def scorebug(home,away,home_c,away_c,clock,crest_path=CREST):
 # home/away: display names. Both sides share font, size and baseline.
 im=Image.new('RGBA',(900,92));d=ImageDraw.Draw(im)
 f=fnt('kr',24);ref=f.getbbox('FC안암 YB');ty=24-(ref[1]+ref[3])/2;cr=crest(34,crest_path);x=0
 for name,col in [(home,home_c),(away,away_c)]:
  lx=x+17+(44 if cr else 0);cw=int(lx-x+f.getlength(name)+20)
  d.rectangle((x,0,x+cw,47),fill=(*INK,238));d.rectangle((x,0,x+5,47),fill=(*col,255))
  if cr:im.alpha_composite(cr,(x+17,7))
  d.text((lx,ty),name,font=f,fill=WHITE);x+=cw
 if clock:d.rectangle((x,0,x+104,47),fill=(*INK2,240));ink_text(d,(x+52,24),clock,fnt('din',34),WHITE,'m');x+=105
 return im.crop((0,0,x,92))
def stinger(img,p,outro=False,word='REPLAY'):
 im=img.convert('RGBA');sk=320
 e=ease_io(p);edge=-sk-120+e*(W+2*sk+240)
 band=lambda a,b:poly_mask([(a,H),(a+sk,0),(b+sk,0),(b,H)])
 if outro:
  ink=band(-sk-10,edge);layers=[(ink,INK),(band(edge-4,edge+68),BRAND),(band(edge+64,edge+78),WHITE)]
  cx=W/2-(W+2*sk+240)*(1-e)
 else:
  ink=band(edge,W+10);layers=[(band(edge-78,edge-60),WHITE),(band(edge-64,edge+4),BRAND),(ink,INK)]
  cx=W/2+(W+2*sk+240)*e
 for m,c in layers:over(im,m,c)
 t=Image.new('RGBA',(W,H));td=ImageDraw.Draw(t);w=spaced(td,(cx,H/2-6),word,fnt('heavy',132),WHITE,14,'m')
 td.rectangle((cx-w/2,H/2+70,cx-w/2+140,H/2+80),fill=(*BRAND,255))
 t.putalpha(Image.fromarray(np.minimum(np.array(t.getchannel('A')),ink)));im.alpha_composite(t)
 return im.convert('RGB')
