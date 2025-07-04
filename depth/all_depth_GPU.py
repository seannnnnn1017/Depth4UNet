# all_depth_safeGPU.py
# ------------------------------------------------------------------
#  Depth Estimator with safe GPU fallback (no hard cv2.cuda.cvtColor)
# ------------------------------------------------------------------
import os, glob, time, warnings
import numpy as np
import cv2
from tqdm import tqdm

warnings.filterwarnings("ignore")

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

# ⚙️ GPU 狀態偵測（需同時偵測到裝置 & cvtColor 支援才開啟）
CUDA_DEV_OK   = cv2.cuda.getCudaEnabledDeviceCount() > 0
CUDA_COLOR_OK = hasattr(cv2.cuda, "cvtColor")
USE_GPU = CUDA_DEV_OK and CUDA_COLOR_OK and HAS_CUPY   # 其他 CUDA API 若缺也會自動 fallback

# ---------- OpenCV GPU ⇆ CPU 轉換 ----------
def to_gpu(arr):
    if not USE_GPU or isinstance(arr, cv2.cuda_GpuMat):
        return arr
    g = cv2.cuda_GpuMat(); g.upload(arr); return g

def to_cpu(arr):
    return arr.download() if isinstance(arr, cv2.cuda_GpuMat) else arr

# ---------- 安全色彩轉換 ----------
def safe_cvtColor(img, code):
    """GPU 支援就用 CUDA，否則用 CPU"""
    if USE_GPU:
        try:
            return cv2.cuda.cvtColor(img, code)
        except Exception:
            pass
    return cv2.cvtColor(to_cpu(img), code)

# ---------- GPU / CPU 二擇一 (非色彩轉換) ----------
def gpu_or_cpu(func_cuda, func_cpu, *args, **kw):
    if USE_GPU:
        try:
            return func_cuda(*args, **kw)
        except Exception:
            pass
    return func_cpu(*args, **kw)

# ---------- Robust imread（支援 BigTIFF / Pillow） ----------
def robust_imread(path):
    import tifffile, PIL.Image
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is not None: return img
    try:
        img = tifffile.imread(path)
        if img.ndim == 3 and img.shape[0] in (3,4):
            img = np.transpose(img,(1,2,0))
        return img
    except: pass
    try:
        return np.array(PIL.Image.open(path).convert("RGB"))
    except: return None

# ------------------------------------------------------------------
#  DepthEstimator
# ------------------------------------------------------------------
class DepthEstimator:
    def __init__(self): self.current = ""

    def set_name(self,name): self.current = name

    # ----- CLAHE -----
    def _clahe(self, l, clip):
        if USE_GPU:
            try:
                return cv2.cuda.createCLAHE(clip,(8,8)).apply(l)
            except: pass
        return cv2.createCLAHE(clipLimit=clip,tileGridSize=(8,8)).apply(l)

    # ----- 自適應前處理 -----
    def adaptive(self, img):
        with tqdm(total=4,desc=f"[{self.current}] 前處理",leave=False,ncols=100,ascii=True) as bar:
            gray = safe_cvtColor(img, cv2.COLOR_RGB2GRAY)
            contrast = np.std(to_cpu(gray)); bar.update(2)
            if contrast<30:   out=self.low_contrast(img)
            elif contrast>80: out=self.high_contrast(img)
            else:             out=self.standard(img)
            bar.update(2)
        return out

    def low_contrast(self,img):
        lab = safe_cvtColor(img, cv2.COLOR_RGB2LAB)
        l,a,b = cv2.split(to_cpu(lab))
        l = self._clahe(l,4.0)
        l = (np.power(l/255.0,0.8)*255).astype(np.uint8)
        lab_np=cv2.merge([l,a,b])
        return safe_cvtColor(lab_np, cv2.COLOR_LAB2RGB)

    def high_contrast(self,img):
        img = gpu_or_cpu(lambda i: cv2.cuda.bilateralFilter(i,9,75,75),
                         cv2.bilateralFilter, img,9,75,75)
        img = gpu_or_cpu(lambda i: cv2.cuda.createGaussianFilter(cv2.CV_8U,cv2.CV_8U,(3,3),0.5).apply(i),
                         cv2.GaussianBlur, img,(3,3),0.5)
        return img

    def standard(self,img):
        lab = safe_cvtColor(img, cv2.COLOR_RGB2LAB)
        l,a,b = cv2.split(to_cpu(lab)); l=self._clahe(l,2.0)
        lab_np=cv2.merge([l,a,b])
        return safe_cvtColor(lab_np, cv2.COLOR_LAB2RGB)

    # ----- 多尺度 Gabor 紋理 -----
    def gabor_texture(self,img):
        gray = safe_cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray_np = to_cpu(gray).astype(np.float32)
        scales=[3,7,15,25]; texture=[]
        with tqdm(total=len(scales),desc=f"[{self.current}] Gabor",leave=False,ncols=100,ascii=True) as bar:
            for sc in scales:
                freq=0.1+0.1*(sc/25); angs=np.arange(0,180,30); res=[]
                for a in angs:
                    k=cv2.getGaborKernel((sc*2+1,sc*2+1),sc/3,np.radians(a),2*np.pi*freq,0.5,0,
                                         ktype=cv2.CV_32F)
                    if USE_GPU:
                        try:
                            resp=cv2.cuda.createLinearFilter(cv2.CV_32F,cv2.CV_32F,k).apply(to_gpu(gray_np))
                            res.append(np.abs(to_cpu(resp))); continue
                        except: pass
                    res.append(np.abs(cv2.filter2D(gray_np,cv2.CV_32F,k)))
                texture.append(np.mean(res,axis=0)); bar.update(1)
        final=sum(w*t for w,t in zip([0.4,0.3,0.2,0.1],texture))
        return 255-cv2.normalize(final,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)

    # ----- 陰影分析 (CPU) -----
    def shadow(self,img):
        rgb=to_cpu(img)
        hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV); lab=cv2.cvtColor(rgb,cv2.COLOR_RGB2LAB)
        h,s,v=cv2.split(hsv); l,_,_=cv2.split(lab)
        mask=((v<np.percentile(v,25))&(s<np.percentile(s,40)))|(l<np.percentile(l,20))
        mask=cv2.morphologyEx(mask.astype(np.uint8),cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
        mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN ,np.ones((5,5),np.uint8))
        dist=cv2.distanceTransform((~mask.astype(bool)).astype(np.uint8),cv2.DIST_L2,5)
        depth=cv2.normalize(dist,None,0,255,cv2.NORM_MINMAX).astype(np.uint8); depth[mask.astype(bool)]=200
        return depth

    # ----- 頻域分析 (FFT) -----
    def freq(self,img):
        gray=to_cpu(safe_cvtColor(img, cv2.COLOR_RGB2GRAY)).astype(np.float32)
        if USE_GPU:
            try:
                g=cp.asarray(gray); f=cp.fft.fftshift(cp.fft.fft2(g))
                r=30; rows,cols=g.shape; crow,ccol=rows//2,cols//2
                y,x=cp.ogrid[:rows,:cols]; mask=cp.ones_like(g)
                mask[(y-crow)**2+(x-ccol)**2 <= r*r]=0; f*=mask
                back=cp.abs(cp.fft.ifft2(cp.fft.ifftshift(f)))
                dep=255-cp.asarray(255*(back-back.min())/(back.max()-back.min()))
                return cp.asnumpy(dep).astype(np.uint8)
            except: pass
        # CPU fallback
        f=np.fft.fftshift(np.fft.fft2(gray))
        rows,cols=gray.shape; crow,ccol=rows//2,cols//2
        y,x=np.ogrid[:rows,:cols]; mask=np.ones_like(gray); r=30
        mask[(y-crow)**2+(x-ccol)**2 <= r*r]=0; f*=mask
        back=np.abs(np.fft.ifft2(np.fft.ifftshift(f)))
        return 255-cv2.normalize(back,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)

    # ----- 梯度一致性 -----
    def coherence(self,img):
        gray=to_cpu(safe_cvtColor(img, cv2.COLOR_RGB2GRAY)).astype(np.float32)
        gx=cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3); gy=cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3)
        mag=np.sqrt(gx**2+gy**2); ang=np.arctan2(gy,gx)
        k=np.ones((15,15),np.float32)
        sum_cos=cv2.filter2D(mag*np.cos(ang),-1,k)
        sum_sin=cv2.filter2D(mag*np.sin(ang),-1,k)
        sum_mag=cv2.filter2D(mag,-1,k)+1e-6
        coh=np.sqrt(sum_cos**2+sum_sin**2)/sum_mag
        return 255-cv2.normalize(coh,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)

    # ----- 邊緣保持融合 -----
    def fuse(self,maps,w,ref):
        gray=cv2.cvtColor(to_cpu(ref),cv2.COLOR_RGB2GRAY)
        edges=cv2.dilate(cv2.Canny(gray,30,100),np.ones((3,3),np.uint8))
        ew=[]
        for dm in maps:
            gx=cv2.Sobel(dm,cv2.CV_32F,1,0,ksize=3)
            gy=cv2.Sobel(dm,cv2.CV_32F,0,1,ksize=3)
            ew.append(np.sqrt(gx**2+gy**2)*(edges/255.0))
        tot=np.sum(ew,axis=0); tot[tot==0]=1
        ew=[e/tot for e in ew]
        fused=np.zeros_like(maps[0],np.float32)
        for dm,wi,ei in zip(maps,w,ew):
            fused+=dm.astype(np.float32)*(0.5*wi+0.5*ei)
        return fused.astype(np.uint8)

    # ----- 後處理 -----
    def post(self,dmap,ref):
        den=cv2.bilateralFilter(dmap,9,75,75)
        edges=cv2.Canny(cv2.cvtColor(ref,cv2.COLOR_RGB2GRAY),50,150)/255.0
        enh=den*(1-edges)+dmap*edges
        sm=cv2.GaussianBlur(enh.astype(np.uint8),(3,3),0.5)
        return (enh*edges+sm*(1-edges)).astype(np.uint8)

    # ----- 主流程 -----
    def estimate(self,img_rgb):
        tqdm.write(f"🔍 {self.current}")
        bar=tqdm(total=6,desc=f"[{self.current}] pipeline",ncols=120,ascii=True)
        img_gpu=to_gpu(img_rgb)
        enh=self.adaptive(img_gpu);            bar.update(1)
        tex=self.gabor_texture(enh);           bar.update(1)
        sha=self.shadow(enh);                  bar.update(1)
        fre=self.freq(enh);                    bar.update(1)
        coh=self.coherence(enh);               bar.update(1)
        fused=self.fuse([tex,sha,fre,coh],[0.35,0.25,0.25,0.15],enh)
        depth=self.post(fused,to_cpu(enh));    bar.update(1); bar.close()
        return depth

# ------------------------------------------------------------------
#  批次處理
# ------------------------------------------------------------------
def process_folder(inp,outp):
    exts=['*.jpg','*.jpeg','*.png','*.bmp','*.tif','*.tiff']
    files=sorted({f for e in exts for f in glob.glob(os.path.join(inp,e))+glob.glob(os.path.join(inp,e.upper()))})
    if not files: print("❌ 無圖片"); return
    os.makedirs(outp,exist_ok=True)
    est=DepthEstimator(); ok=fail=0; st=time.time()
    for i,fpath in enumerate(files,1):
        name=os.path.basename(fpath); tqdm.write(f"\n📷 [{i}/{len(files)}] {name}")
        est.set_name(name); img=robust_imread(fpath)
        if img is None: tqdm.write("讀取失敗"); fail+=1; continue
        rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB) if img.ndim==3 else img
        try:
            depth=est.estimate(rgb)
            cv2.imwrite(os.path.join(outp,f"{os.path.splitext(name)[0]}_depth.png"),depth)
            ok+=1
        except Exception as e:
            tqdm.write(f"❌ {e}"); fail+=1
        eta=(time.time()-st)/i*(len(files)-i); tqdm.write(f"⏱️ 剩 {eta/60:.1f} 分")
    print(f"\n🎉 完成 ✔{ok} ✖{fail} | 用時 {(time.time()-st)/60:.1f} 分")

# ------------------------------------------------------------------
if __name__=="__main__":
    INPUT  = r"E:/論文/空拍地圖語意分割/Depth4UNet/AerialImageDataset/test/images"  # 修改為你的輸入資料夾
    OUTPUT = r"depth"                     # 修改為你的輸出資料夾
    if not os.path.exists(INPUT): raise SystemExit("❌ 找不到輸入資料夾")
    process_folder(INPUT,OUTPUT)
