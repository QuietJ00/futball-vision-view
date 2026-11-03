import json,os,subprocess,sys
import numpy as np
# Clip input: a local file or a YouTube link, cut to [start, end] seconds, decoded once to a
# 1920x1080 BGR memmap so every later stage can random-access frames.
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FF=os.path.join(ROOT,'work/bin/ffmpeg')
if not os.path.exists(FF):
 import imageio_ffmpeg;FF=imageio_ffmpeg.get_ffmpeg_exe()
YTDLP=os.path.join(os.path.dirname(sys.executable),'yt-dlp')
W,H,FPS=1920,1080,30

def parse_time(s):
 # '9:17', '1:02:03' or plain seconds.
 s=str(s).strip();t=0.
 for part in s.split(':'):t=t*60+float(part)
 return t
def fmt_time(t):return f'{int(t//60)}:{t%60:05.2f}'

def download(url,start,end,out_dir,log=print):
 # SDR stream only: HDR footage would need tone mapping and comes out washed-out otherwise.
 os.makedirs(out_dir,exist_ok=True);out=os.path.join(out_dir,'source.mp4')
 cmd=[YTDLP,'-f','bv*[height<=1080][dynamic_range=SDR]+ba/bv*[height<=1080]+ba/b','--download-sections',f'*{fmt_time(start)}-{fmt_time(end)}',
      '--force-keyframes-at-cuts','--ffmpeg-location',os.path.dirname(FF),'--merge-output-format','mp4','-o',out,'--no-playlist',url]
 log('영상 다운로드 중');p=subprocess.run(cmd,capture_output=True,text=True)
 if p.returncode or not os.path.exists(out):raise RuntimeError('다운로드 실패: '+(p.stderr.strip().splitlines() or ['?'])[-1])
 return out

def probe(path):
 p=subprocess.run([FF,'-hide_banner','-i',path],capture_output=True,text=True).stderr
 hdr=any(k in p for k in ('smpte2084','arib-std-b67','bt2020'))
 dur=0.
 for line in p.splitlines():
  if 'Duration:' in line:
   h,m,s=line.split('Duration:')[1].split(',')[0].strip().split(':');dur=int(h)*3600+int(m)*60+float(s)
 return {'hdr':hdr,'duration':dur}

def decode(path,out_npy,start=None,end=None,log=print):
 # Constant 30 fps at 1080p regardless of the source, so speeds are always per 1/30 s.
 info=probe(path);vf=f'fps={FPS},scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2'
 if info['hdr']:vf='zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,format=yuv420p,'+vf
 cmd=[FF,'-hide_banner','-loglevel','error']
 if start is not None:cmd+=['-ss',str(start)]
 cmd+=['-i',path]
 if end is not None:cmd+=['-t',str(end-(start or 0))]
 cmd+=['-vf',vf,'-an','-f','rawvideo','-pix_fmt','bgr24','-']
 n_est=int(((end-(start or 0)) if end is not None else info['duration']-(start or 0))*FPS)+FPS
 if n_est>FPS*90:raise RuntimeError('클립이 너무 길어요. 90초 이하로 잘라 주세요.')
 log('프레임 추출 중');buf=np.lib.format.open_memmap(out_npy+'.tmp.npy','w+',np.uint8,(n_est,H,W,3))
 p=subprocess.Popen(cmd,stdout=subprocess.PIPE);n=0;sz=W*H*3
 while True:
  b=p.stdout.read(sz)
  if len(b)<sz:break
  buf[n]=np.frombuffer(b,np.uint8).reshape(H,W,3);n+=1
 p.wait()
 if n<FPS:raise RuntimeError('영상에서 프레임을 읽지 못했어요.')
 out=np.lib.format.open_memmap(out_npy,'w+',np.uint8,(n,H,W,3));out[:]=buf[:n];out.flush();del buf,out
 os.remove(out_npy+'.tmp.npy')
 return n

def load(out_npy):return np.load(out_npy,mmap_mode='r')
