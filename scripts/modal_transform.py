"""
modal_transform.py — VideoRAG Transform + Assemble API (Modal serverless GPU)

Endpoints:
  POST /transform  — FFmpeg 스타일 필터 적용 (즉시 반환)
  POST /assemble   — 전체 장면 조립 (DINOv2 전환 스코어링)

Deploy:
    modal deploy modal_transform.py
"""

import modal

app = modal.App("videorag-transform")

STYLE_FILTERS = {
    "warm":        "colorbalance=rs=0.4:gs=0.1:bs=-0.3:rm=0.3:gm=0.05:bm=-0.2,eq=brightness=0.06:saturation=1.4:contrast=1.1",
    "cool":        "colorbalance=rs=-0.3:gs=-0.05:bs=0.4:rm=-0.2:bm=0.25,eq=saturation=0.85:contrast=1.1",
    "golden_hour": "colorbalance=rs=0.5:gs=0.2:bs=-0.4:rm=0.3:gm=0.1:bm=-0.25,eq=brightness=0.1:saturation=1.6:contrast=1.15",
    "noir":        "hue=s=0,eq=contrast=1.7:brightness=-0.05",
    "cinematic":   "eq=contrast=1.35:saturation=0.7:brightness=-0.04,vignette=angle=PI/4",
    "documentary": "eq=contrast=1.15:saturation=0.75:brightness=0.04",
    "dramatic":    "eq=contrast=1.6:saturation=1.6:brightness=-0.08",
    "night":       "colorbalance=rs=-0.35:gs=-0.15:bs=0.45:rm=-0.2:bm=0.3,eq=brightness=-0.3:contrast=1.3:saturation=0.7",
    "tense":       "colorbalance=rs=-0.2:bs=0.2:rm=-0.1:bm=0.1,eq=contrast=1.5:saturation=0.6:brightness=-0.15",
    "vibrant":     "eq=saturation=2.2:contrast=1.25:brightness=0.04",
}


def _download_dinov2():
    import torch
    torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", verbose=False)
    print("DINOv2 다운로드 완료")


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0", "ffmpeg", "git")
    .pip_install(
        "torch==2.1.2",
        "torchvision==0.16.2",
        "opencv-python-headless",
        "Pillow",
        "requests",
        "fastapi",
        "python-multipart",
        "numpy<2.0",
    )
    .run_function(_download_dinov2)
)


@app.cls(gpu="T4", image=image, scaledown_window=1800, timeout=300)
class TransformModel:

    @modal.enter()
    def load(self):
        import torch
        self.dino = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vits14"
        ).to("cuda").eval()

    def _dino_sim(self, frame_a, frame_b) -> float:
        import torch
        from torchvision import transforms
        from PIL import Image
        import cv2

        tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        def feat(f):
            pil = Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
            t = tf(pil).unsqueeze(0).to("cuda")
            with torch.no_grad():
                v = self.dino(t)
            return v / v.norm(dim=-1, keepdim=True)

        return float((feat(frame_a) * feat(frame_b)).sum())

    def _build_lut(self, source_frame, ref_frame, size=17, strength=0.8):
        import numpy as np
        import cv2
        n = size
        src_lab = cv2.cvtColor(source_frame, cv2.COLOR_BGR2LAB).astype(np.float32)
        ref_lab = cv2.cvtColor(ref_frame,    cv2.COLOR_BGR2LAB).astype(np.float32)
        src_stats = [(src_lab[:,:,c].mean(), src_lab[:,:,c].std()+1e-6) for c in range(3)]
        ref_stats = [(ref_lab[:,:,c].mean(), ref_lab[:,:,c].std()+1e-6) for c in range(3)]

        axis = np.linspace(0, 255, n, dtype=np.float32)
        r_grid, g_grid, b_grid = np.meshgrid(axis, axis, axis, indexing='ij')
        grid_bgr = np.stack([b_grid, g_grid, r_grid], axis=-1)
        grid_flat = grid_bgr.reshape(-1, 1, 3).astype(np.uint8)
        grid_lab  = cv2.cvtColor(grid_flat, cv2.COLOR_BGR2LAB).astype(np.float32)

        for c in range(3):
            s_mean, s_std = src_stats[c]
            r_mean, r_std = ref_stats[c]
            transferred = (grid_lab[:,0,c] - s_mean) * (r_std / s_std) + r_mean
            grid_lab[:,0,c] = grid_lab[:,0,c]*(1-strength) + transferred*strength

        grid_lab = np.clip(grid_lab, 0, 255).astype(np.uint8)
        lut_table = cv2.cvtColor(grid_lab, cv2.COLOR_LAB2BGR).astype(np.float32)
        return lut_table.reshape(n, n, n, 3)

    def _apply_lut_to_video(self, in_path: str, out_path: str, lut_table):
        import numpy as np
        import cv2
        n = lut_table.shape[0]
        cap = cv2.VideoCapture(in_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        tmp = out_path + "_raw.mp4"
        writer = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            img = frame.astype(np.float32) / 255.0
            b_idx = img[:,:,0]*(n-1); g_idx = img[:,:,1]*(n-1); r_idx = img[:,:,2]*(n-1)
            r0 = np.clip(np.floor(r_idx).astype(int), 0, n-2)
            g0 = np.clip(np.floor(g_idx).astype(int), 0, n-2)
            b0 = np.clip(np.floor(b_idx).astype(int), 0, n-2)
            r1, g1, b1 = r0+1, g0+1, b0+1
            rd = (r_idx-r0).reshape(h,w,1); gd = (g_idx-g0).reshape(h,w,1); bd = (b_idx-b0).reshape(h,w,1)
            c000=lut_table[r0,g0,b0]; c001=lut_table[r0,g0,b1]; c010=lut_table[r0,g1,b0]; c011=lut_table[r0,g1,b1]
            c100=lut_table[r1,g0,b0]; c101=lut_table[r1,g0,b1]; c110=lut_table[r1,g1,b0]; c111=lut_table[r1,g1,b1]
            c00=c000*(1-rd)+c100*rd; c01=c001*(1-rd)+c101*rd; c10=c010*(1-rd)+c110*rd; c11=c011*(1-rd)+c111*rd
            c0=c00*(1-gd)+c10*gd;    c1=c01*(1-gd)+c11*gd
            result = np.clip(c0*(1-bd)+c1*bd, 0, 255).astype(np.uint8)
            writer.write(result)

        cap.release(); writer.release()
        import os
        os.system(f'ffmpeg -y -i {tmp} -vcodec libx264 -pix_fmt yuv420p -movflags +faststart {out_path} -loglevel error')
        os.path.exists(tmp) and os.unlink(tmp)

    def _transition(self, seg_a: str, seg_b: str, out: str, mode: str):
        import os
        if mode == "cut":
            os.system(
                f'ffmpeg -y -i {seg_a} -i {seg_b} '
                f'-filter_complex "[0:v][1:v]concat=n=2:v=1" '
                f'-vcodec libx264 -pix_fmt yuv420p -movflags +faststart {out} -loglevel error'
            )
        else:
            os.system(
                f'ffmpeg -y -i {seg_a} -i {seg_b} '
                f'-filter_complex "[0:v][1:v]xfade=transition=fade:duration=0.5:offset=0" '
                f'-vcodec libx264 -pix_fmt yuv420p -movflags +faststart {out} -loglevel error'
            )

    @modal.asgi_app()
    def web(self):
        from fastapi import FastAPI, Request
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import JSONResponse
        import shlex

        api = FastAPI()
        api.add_middleware(
            CORSMiddleware,
            allow_origins=[
                "https://limpark996.github.io",
                "http://localhost:5173",
                "http://localhost:4173",
            ],
            allow_methods=["POST", "GET", "OPTIONS"],
            allow_headers=["*"],
        )

        model = self

        @api.post("/transform")
        async def transform(request: Request):
            import requests as req, base64, tempfile, os

            body = await request.json()
            video_url = body.get("video_url", "")
            style = body.get("style", "cinematic")

            if not video_url:
                return JSONResponse({"success": False, "error": "video_url required"}, status_code=400)

            vf = STYLE_FILTERS.get(style, STYLE_FILTERS["cinematic"])

            try:
                r = req.get(video_url, timeout=30)
                r.raise_for_status()
            except Exception as e:
                return JSONResponse({"success": False, "error": f"download failed: {e}"}, status_code=400)

            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                f.write(r.content)
                in_path = f.name
            out_path = in_path.replace(".mp4", "_out.mp4")

            try:
                ret = os.system(
                    f'ffmpeg -y -i {shlex.quote(in_path)} -vf "{vf}" '
                    f'-vcodec libx264 -pix_fmt yuv420p -movflags +faststart '
                    f'{shlex.quote(out_path)} -loglevel error'
                )
                if ret != 0 or not os.path.exists(out_path):
                    return JSONResponse({"success": False, "error": "ffmpeg filter failed"}, status_code=500)

                with open(out_path, "rb") as f:
                    video_b64 = base64.b64encode(f.read()).decode()

                return JSONResponse({"success": True, "video_b64": video_b64, "style": style})

            except Exception as e:
                return JSONResponse({"success": False, "error": str(e)}, status_code=500)
            finally:
                for p in [in_path, out_path]:
                    try: os.unlink(p)
                    except: pass

        @api.post("/assemble")
        async def assemble(request: Request):
            import cv2, requests as req, base64, tempfile, os, shutil

            body = await request.json()
            scenes = body.get("scenes", [])
            if not scenes:
                return JSONResponse({"success": False, "error": "scenes required"}, status_code=400)

            tmp_dir = tempfile.mkdtemp()
            seg_paths = []

            try:
                for i, scene in enumerate(scenes):
                    seg = os.path.join(tmp_dir, f"seg_{i:02d}.mp4")
                    if scene["decision"] == "use_as_is":
                        r = req.get(scene["video_url"], timeout=30)
                        with open(seg, "wb") as f:
                            f.write(r.content)
                    else:
                        with open(seg, "wb") as f:
                            f.write(base64.b64decode(scene["video_b64"]))
                    seg_paths.append(seg)

                # DreamColour: 첫 번째 클립을 레퍼런스로 색감 통일
                ref_cap = cv2.VideoCapture(seg_paths[0])
                ret_ref, ref_frame = ref_cap.read()
                ref_cap.release()
                if ret_ref and len(seg_paths) > 1:
                    for i, seg in enumerate(seg_paths[1:], 1):
                        src_cap = cv2.VideoCapture(seg)
                        total = int(src_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        src_cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
                        ret_src, src_frame = src_cap.read()
                        src_cap.release()
                        if ret_src:
                            try:
                                lut = model._build_lut(src_frame, ref_frame)
                                cc_path = seg.replace(".mp4", "_cc.mp4")
                                model._apply_lut_to_video(seg, cc_path, lut)
                                if os.path.exists(cc_path):
                                    seg_paths[i] = cc_path
                            except Exception as e:
                                print(f"[DreamColour] 클립 {i} 색보정 실패: {e}")

                transitions = []
                for i in range(len(seg_paths) - 1):
                    cap_a = cv2.VideoCapture(seg_paths[i])
                    cap_b = cv2.VideoCapture(seg_paths[i + 1])
                    total_a = int(cap_a.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap_a.set(cv2.CAP_PROP_POS_FRAMES, max(0, total_a - 1))
                    ret_a, fa = cap_a.read()
                    ret_b, fb = cap_b.read()
                    cap_a.release(); cap_b.release()

                    if ret_a and ret_b:
                        sim = model._dino_sim(fa, fb)
                        mode = "morph" if sim > 0.85 else "crossfade" if sim > 0.6 else "cut"
                    else:
                        mode = "cut"
                    transitions.append(mode)

                current = seg_paths[0]
                for i, (nxt, mode) in enumerate(zip(seg_paths[1:], transitions)):
                    merged = os.path.join(tmp_dir, f"merged_{i:02d}.mp4")
                    model._transition(current, nxt, merged, mode)
                    current = merged

                with open(current, "rb") as f:
                    video_b64 = base64.b64encode(f.read()).decode()

                return JSONResponse({"success": True, "video_b64": video_b64, "transitions": transitions})

            except Exception as e:
                return JSONResponse({"success": False, "error": str(e)}, status_code=500)
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)

        return api
