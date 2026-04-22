"""
modal_transform.py — VideoRAG Transform + Assemble API (Modal serverless GPU)

Endpoints:
  POST /transform  — OpenCV 스타일 필터 적용 (즉시 반환)
  POST /assemble   — 전체 장면 조립 (DINOv2 전환 스코어링)

Deploy:
    modal deploy scripts/modal_transform.py
"""

import modal

app = modal.App("videorag-transform")

# OpenCV 기반 스타일 파라미터
# r/g/b_gain: 채널 곱셈 계수 (1.0 = 변화 없음)
# r/g/b_add: 채널 덧셈 오프셋 (0~255 범위)
# sat: 채도 배율 (1.0 = 변화 없음, HSV S채널 스케일)
# bright: 밝기 오프셋 (-1.0~1.0, 255 곱해서 적용)
# contrast: 대비 배율 (픽셀을 128 기준으로 스트레칭)
STYLE_PARAMS = {
    "warm":          dict(r_gain=1.25, r_add=15,  g_gain=1.0,  g_add=0,   b_gain=0.72, b_add=-8,  sat=1.3,  bright=0,     contrast=1.1),
    "cool":          dict(r_gain=0.78, r_add=-5,  g_gain=0.92, g_add=0,   b_gain=1.22, b_add=12,  sat=0.88, bright=0,     contrast=1.1),
    "golden_hour":   dict(r_gain=1.38, r_add=20,  g_gain=1.08, g_add=5,   b_gain=0.52, b_add=-22, sat=1.6,  bright=0.08,  contrast=1.15),
    "documentary":   dict(r_gain=1.05, r_add=0,   g_gain=1.0,  g_add=0,   b_gain=0.96, b_add=0,   sat=0.72, bright=0.05,  contrast=1.2),
    "dramatic":      dict(r_gain=1.0,  r_add=0,   g_gain=0.95, g_add=0,   b_gain=0.92, b_add=0,   sat=1.8,  bright=-0.08, contrast=1.8),
    "night":         dict(r_gain=0.52, r_add=0,   g_gain=0.65, g_add=0,   b_gain=1.1,  b_add=0,   sat=0.75, bright=-0.28, contrast=1.35),
    "tense":         dict(r_gain=0.72, r_add=-5,  g_gain=0.85, g_add=0,   b_gain=1.05, b_add=5,   sat=0.55, bright=-0.15, contrast=1.55),
    "vibrant":       dict(r_gain=1.0,  r_add=0,   g_gain=1.0,  g_add=0,   b_gain=1.0,  b_add=0,   sat=2.4,  bright=0.04,  contrast=1.35),
    "dawn":          dict(r_gain=0.85, r_add=-5,  g_gain=0.88, g_add=0,   b_gain=1.18, b_add=12,  sat=0.9,  bright=0.08,  contrast=0.95),
    "dusk":          dict(r_gain=1.30, r_add=10,  g_gain=0.85, g_add=-5,  b_gain=0.65, b_add=-15, sat=1.5,  bright=0.05,  contrast=1.1),
    "bleach_bypass": dict(r_gain=1.0,  r_add=0,   g_gain=1.0,  g_add=0,   b_gain=1.0,  b_add=0,   sat=0.3,  bright=-0.05, contrast=1.9),
    "horror":        dict(r_gain=0.80, r_add=0,   g_gain=1.12, g_add=8,   b_gain=0.78, b_add=0,   sat=0.5,  bright=-0.1,  contrast=1.4),
    "arctic":        dict(r_gain=0.78, r_add=-5,  g_gain=0.85, g_add=0,   b_gain=1.28, b_add=22,  sat=0.7,  bright=0.12,  contrast=1.05),
    "sunset":        dict(r_gain=1.42, r_add=25,  g_gain=0.80, g_add=-10, b_gain=0.50, b_add=-25, sat=1.7,  bright=0,     contrast=1.2),
}

# 특수 효과 — 위 파라미터로 표현 불가능한 스타일
SPECIAL_STYLES = {"noir", "cinematic", "vintage", "foggy"}


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

    def _apply_style_frame(self, frame, style: str):
        """OpenCV 프레임에 스타일 적용. BGR uint8 → BGR uint8."""
        import numpy as np
        import cv2

        if style in SPECIAL_STYLES:
            return self._apply_special_frame(frame, style)

        p = STYLE_PARAMS.get(style, STYLE_PARAMS["cinematic"] if "cinematic" not in SPECIAL_STYLES else STYLE_PARAMS["documentary"])

        # BGR 채널 분리
        b, g, r = cv2.split(frame.astype(np.float32))

        # 채널별 gain + add
        r = r * p["r_gain"] + p["r_add"]
        g = g * p["g_gain"] + p["g_add"]
        b = b * p["b_gain"] + p["b_add"]

        # 대비 (128 기준 스트레칭)
        c = p["contrast"]
        if c != 1.0:
            r = (r - 128) * c + 128
            g = (g - 128) * c + 128
            b = (b - 128) * c + 128

        # 밝기
        br = p["bright"] * 255
        r += br; g += br; b += br

        frame_out = cv2.merge([
            np.clip(b, 0, 255).astype(np.uint8),
            np.clip(g, 0, 255).astype(np.uint8),
            np.clip(r, 0, 255).astype(np.uint8),
        ])

        # 채도 (HSV S 채널 스케일)
        if p["sat"] != 1.0:
            hsv = cv2.cvtColor(frame_out, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * p["sat"], 0, 255)
            frame_out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        return frame_out

    def _apply_special_frame(self, frame, style: str):
        import numpy as np
        import cv2

        if style == "noir":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # 대비 강화
            f = gray.astype(np.float32)
            f = (f - 128) * 1.8 + 128 - 15
            gray = np.clip(f, 0, 255).astype(np.uint8)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        elif style == "cinematic":
            # Teal-orange: shadows → teal, highlights → orange
            f = frame.astype(np.float32)
            b, g, r = cv2.split(f)
            # 밝기 마스크 (0=어두움, 1=밝음)
            luma = (0.114 * b + 0.587 * g + 0.299 * r) / 255.0
            # Shadows: 청록 (r↓, g약↑, b↑)
            shadow_mask = np.clip(1.0 - luma * 2, 0, 1)
            r -= shadow_mask * 18
            g += shadow_mask * 6
            b += shadow_mask * 20
            # Highlights: 오렌지 (r↑, g약↑, b↓)
            hi_mask = np.clip((luma - 0.5) * 2, 0, 1)
            r += hi_mask * 20
            g += hi_mask * 8
            b -= hi_mask * 22
            # 전체 채도 낮추고 대비 강화
            frame_out = cv2.merge([
                np.clip(b, 0, 255).astype(np.uint8),
                np.clip(g, 0, 255).astype(np.uint8),
                np.clip(r, 0, 255).astype(np.uint8),
            ])
            hsv = cv2.cvtColor(frame_out, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 0.65, 0, 255)
            hsv[:, :, 2] = np.clip((hsv[:, :, 2].astype(np.float32) - 128) * 1.4 + 128 - 12, 0, 255)
            return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        elif style == "vintage":
            # 세피아 + 페이드
            f = frame.astype(np.float32)
            b, g, r = cv2.split(f)
            # 세피아 매트릭스
            nr = np.clip(r * 0.393 + g * 0.769 + b * 0.189, 0, 255)
            ng = np.clip(r * 0.349 + g * 0.686 + b * 0.168, 0, 255)
            nb = np.clip(r * 0.272 + g * 0.534 + b * 0.131, 0, 255)
            # 페이드: 어두운 부분을 밝게 올림 (faded look)
            nr = nr * 0.85 + 20
            ng = ng * 0.85 + 15
            nb = nb * 0.80 + 10
            return cv2.merge([
                np.clip(nb, 0, 255).astype(np.uint8),
                np.clip(ng, 0, 255).astype(np.uint8),
                np.clip(nr, 0, 255).astype(np.uint8),
            ])

        elif style == "foggy":
            # 채도 낮추기 + 흰색 헤이즈 오버레이
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 0.45, 0, 255)
            desaturated = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
            # 흰색 50% 블렌드로 헤이즈
            haze = np.ones_like(desaturated) * 255
            alpha = 0.42
            result = desaturated * (1 - alpha) + haze * alpha
            return np.clip(result, 0, 255).astype(np.uint8)

        return frame

    def _process_video_opencv(self, in_path: str, out_path: str, style: str):
        """OpenCV로 프레임별 스타일 적용 후 ffmpeg로 h264 재인코딩."""
        import cv2, os
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
            writer.write(self._apply_style_frame(frame, style))

        cap.release()
        writer.release()
        os.system(
            f'ffmpeg -y -i {tmp} -vcodec libx264 -pix_fmt yuv420p '
            f'-movflags +faststart {out_path} -loglevel error'
        )
        os.path.exists(tmp) and os.unlink(tmp)

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

            if style not in STYLE_PARAMS and style not in SPECIAL_STYLES:
                style = "cinematic"

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
                model._process_video_opencv(in_path, out_path, style)
                if not os.path.exists(out_path):
                    return JSONResponse({"success": False, "error": "opencv processing failed"}, status_code=500)

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
