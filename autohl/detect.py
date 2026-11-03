import os
import numpy as np
from .video import ROOT
# Person boxes in every frame. On a wide amateur shot players are only ~50-70 px tall, so
# the detector runs at full 1920 width with a low confidence floor; tracking filters the rest.
MODEL=os.path.join(ROOT,'work/models/yolo26m.pt')
_model=None
def model():
 global _model
 if _model is None:
  from ultralytics import YOLO
  import torch
  _model=YOLO(MODEL if os.path.exists(MODEL) else 'yolo26m.pt');_model.to('mps' if torch.backends.mps.is_available() else 'cpu')
 return _model
def run(F,log=print,batch=4):
 m=model();N=len(F);out=[]
 for i in range(0,N,batch):
  imgs=[np.ascontiguousarray(F[j]) for j in range(i,min(N,i+batch))]
  for r in m.predict(imgs,imgsz=1920,classes=[0],conf=.12,iou=.6,verbose=False,device=m.device):
   b=r.boxes;out.append(np.concatenate([b.xyxy.cpu().numpy(),b.conf.cpu().numpy()[:,None]],1).astype(np.float32))
  if i%60<batch:log(f'선수 검출 {i}/{N}')
 return out
def save(path,dets):
 np.savez(path,boxes=np.concatenate(dets) if dets else np.zeros((0,5),np.float32),counts=np.array([len(d) for d in dets]))
def load(path):
 z=np.load(path);idx=np.concatenate([[0],np.cumsum(z['counts'])]);b=z['boxes']
 return [b[idx[i]:idx[i+1]] for i in range(len(z['counts']))]
