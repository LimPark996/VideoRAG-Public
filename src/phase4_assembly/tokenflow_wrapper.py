"""
tokenflow_wrapper.py — TokenFlow video-to-video 편집 래퍼

TokenFlow (Geyer et al., 2023): https://github.com/omerbt/TokenFlow
  - 기존 영상 클립을 텍스트 프롬프트로 스타일 변환
  - Stable Diffusion + DDIM inversion + attention feature sharing
  - keyframe subsampling으로 실용적인 속도 달성

실행 흐름 (CLI 기반, subprocess):
  1. preprocess.py       : DDIM inversion → 레포 내부 latent 저장
  2. run_tokenflow_pnp.py: TokenFlow PnP 편집 → 프레임 출력

설정값 (고정):
  extract_fps   = 8    원본에서 8fps로 프레임 추출 (30fps → 1/4 subsampling)
  keyframe_freq = 10   추출된 프레임 중 10개마다 keyframe 1개
  n_timesteps   = 50   DDIM steps
  batch_size    = 8    keyframe 배치 크기

5초 클립 기준:
  추출 프레임 = 5 × 8 = 40개
  keyframe    = 40 ÷ 10 = 4개
  DDIM 배치   = ceil(4/8) = 1배치
  예상 소요   ~ 30~60초 (T4)
"""

import os
import sys
import shutil
import logging
import subprocess
import tempfile
import uuid
import yaml
from typing import Optional, List

import cv2

logger = logging.getLogger(__name__)

# ── 고정 설정 ──────────────────────────────────────────────
EXTRACT_FPS    = 8
KEYFRAME_FREQ  = 10
N_TIMESTEPS    = 50
BATCH_SIZE     = 8

# TokenFlow 레포 경로 (Colab 기준)
TOKENFLOW_DIR  = "/content/TokenFlow"
# Stable Diffusion 버전
# preprocess.py 유효값: {1.5, 2.0, 2.1, ControlNet, depth}  ("1.4" 없음)
SD_VERSION     = "1.5"


def _ensure_tokenflow() -> bool:
    """TokenFlow 레포가 없으면 클론.

    run_tokenflow_pnp.py 파일 존재 여부로 판단한다.
    (실제 레포에는 tokenflow_pnp.py가 없고 run_tokenflow_pnp.py가 CLI 진입점)

    Returns:
        True: 사용 가능 / False: 설치 실패
    """
    if os.path.exists(os.path.join(TOKENFLOW_DIR, "run_tokenflow_pnp.py")):
        return True

    # 디렉토리가 이미 존재하면 re-clone 하지 않는다 (이미 수동 clone한 경우 등)
    if os.path.isdir(TOKENFLOW_DIR):
        logger.warning(
            f"[TokenFlow] {TOKENFLOW_DIR} 존재하지만 run_tokenflow_pnp.py 없음 — "
            f"디렉토리는 있으나 파일이 없습니다. 직접 git clone을 확인하세요."
        )
        return False

    logger.info("[TokenFlow] 레포 클론 시작...")
    r = subprocess.run(
        ["git", "clone", "https://github.com/omerbt/TokenFlow.git", TOKENFLOW_DIR],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        logger.error(f"[TokenFlow] git clone 실패: {r.stderr}")
        return False

    # 의존성 설치
    req_path = os.path.join(TOKENFLOW_DIR, "requirements.txt")
    if os.path.exists(req_path):
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "-r", req_path],
            capture_output=True,
        )

    logger.info("[TokenFlow] 설치 완료")
    _patch_tokenflow()
    return True


def _patch_tokenflow():
    """TokenFlow 스크립트의 알려진 호환성 문제를 패치.

    문제: preprocess.py / run_tokenflow_pnp.py 가 SD 모델 로드 시
    revision="fp16" 을 하드코딩하는데, runwayml/stable-diffusion-v1-5
    저장소에서 fp16 브랜치가 삭제되어 404 오류가 발생한다.

    수정: revision="fp16" 인자 전체를 제거한다.
    """
    import re

    targets = ["preprocess.py", "run_tokenflow_pnp.py", "tokenflow_utils.py"]
    for fname in targets:
        fpath = os.path.join(TOKENFLOW_DIR, fname)
        if not os.path.exists(fpath):
            continue
        with open(fpath, "r", encoding="utf-8") as f:
            src = f.read()
        # revision="fp16" 또는 revision='fp16' 패턴 제거
        patched = re.sub(r',\s*revision=["\']fp16["\']', "", src)
        patched = re.sub(r'revision=["\']fp16["\'],\s*', "", patched)
        if patched != src:
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(patched)
            logger.info(f"[TokenFlow] 패치 완료: {fname} (revision='fp16' 제거)")


class TokenFlowEditor:
    """TokenFlow 기반 video-to-video 편집기

    사용법:
        editor = TokenFlowEditor()
        out = editor.edit(
            video_path="clip.mp4",
            prompt="a car driving at night, cinematic lighting",
            output_path="out.mp4",
        )
    """

    def __init__(self, device: Optional[str] = None):
        """
        Args:
            device: "cuda" / "cpu" (None이면 자동 감지)
        """
        import torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._ready = False

    # ─────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────

    def edit(
        self,
        video_path: str,
        prompt: str,
        output_path: str,
        source_prompt: str = "",
    ) -> str:
        """영상을 프롬프트대로 변환.

        Args:
            video_path:    원본 영상 (.mp4)
            prompt:        목표 스타일 프롬프트 (영어)
            output_path:   출력 영상 경로 (.mp4)
            source_prompt: 원본 영상 설명 (DDIM inversion 품질 향상용, 선택)

        Returns:
            output_path (성공) 또는 video_path (실패 시 원본 반환)
        """
        self._ensure_ready()

        with tempfile.TemporaryDirectory() as tmp:
            frames_dir = os.path.join(tmp, "frames")
            edited_dir = os.path.join(tmp, "edited")
            os.makedirs(frames_dir)
            os.makedirs(edited_dir)

            # 1. 프레임 추출 (subsampling)
            orig_fps, frames = self._extract_frames(video_path, frames_dir)
            if not frames:
                logger.error("[TokenFlow] 프레임 추출 실패 — 원본 반환")
                return video_path

            logger.info(
                f"[TokenFlow] 추출 완료: {len(frames)}프레임 "
                f"(원본 {orig_fps:.1f}fps → {EXTRACT_FPS}fps), "
                f"keyframe {len(frames) // KEYFRAME_FREQ + 1}개"
            )

            # 2. TokenFlow 실행
            edited_frames = self._run_tokenflow(
                frames_dir=frames_dir,
                edited_dir=edited_dir,
                prompt=prompt,
                source_prompt=source_prompt,
            )

            if not edited_frames:
                logger.warning("[TokenFlow] 편집 실패 — 원본 반환")
                return video_path

            # 3. 영상 재조립
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            self._reconstruct_video(edited_frames, output_path, fps=EXTRACT_FPS)

        logger.info(f"[TokenFlow] 완료: {output_path}")
        return output_path

    # ─────────────────────────────────────────
    # 내부 메서드
    # ─────────────────────────────────────────

    def _ensure_ready(self):
        """첫 호출 시 TokenFlow 설치 + 호환성 패치 확인"""
        if self._ready:
            return
        if not _ensure_tokenflow():
            raise RuntimeError(
                "TokenFlow 설치 실패. "
                "인터넷 연결 또는 git이 필요합니다."
            )
        # 이미 clone된 상태에서도 패치가 적용되도록 매번 호출
        # (revision="fp16" 제거 — 이미 제거된 경우 re.sub이 no-op)
        _patch_tokenflow()
        self._ready = True

    def _extract_frames(
        self, video_path: str, out_dir: str
    ) -> tuple:
        """원본 영상에서 EXTRACT_FPS fps로 프레임 추출.

        Returns:
            (orig_fps, sorted frame path list)
        """
        cap = cv2.VideoCapture(video_path)
        orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 몇 프레임마다 1개 추출할지 계산
        step = max(1, round(orig_fps / EXTRACT_FPS))

        saved = []
        idx = 0
        frame_no = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_no % step == 0:
                path = os.path.join(out_dir, f"{idx:05d}.png")
                cv2.imwrite(path, frame)
                saved.append(path)
                idx += 1
            frame_no += 1

        cap.release()
        logger.info(
            f"[TokenFlow._extract_frames] "
            f"총 {total_frames}프레임 중 {len(saved)}프레임 추출 "
            f"(step={step}, orig_fps={orig_fps:.1f})"
        )
        return orig_fps, sorted(saved)

    def _run_tokenflow(
        self,
        frames_dir: str,
        edited_dir: str,
        prompt: str,
        source_prompt: str,
    ) -> List[str]:
        """TokenFlow PnP 편집 실행 (subprocess 방식).

        실제 TokenFlow 레포는 tokenflow_pnp.py 모듈이 아니라
        CLI 스크립트(preprocess.py + run_tokenflow_pnp.py)로 구성되어 있다.
        따라서 Python import 대신 subprocess로 각 스크립트를 순서대로 호출한다.

        단계:
          1. preprocess.py   : DDIM inversion → 레포 내부 inversion_dir에 latent 저장
          2. run_tokenflow_pnp.py: TokenFlow PnP → edited_dir에 프레임 저장

        Returns:
            편집된 프레임 경로 리스트 (오름차순), 실패 시 빈 리스트
        """
        try:
            preprocess_script  = os.path.join(TOKENFLOW_DIR, "preprocess.py")
            tokenflow_script   = os.path.join(TOKENFLOW_DIR, "run_tokenflow_pnp.py")

            if not os.path.exists(preprocess_script):
                logger.error(f"[TokenFlow] preprocess.py 없음: {preprocess_script}")
                return []
            if not os.path.exists(tokenflow_script):
                logger.error(f"[TokenFlow] run_tokenflow_pnp.py 없음: {tokenflow_script}")
                return []

            # preprocess.py : argparse 개별 인자 방식 (--config 미지원)
            # run_tokenflow_pnp.py : --config yaml 방식 (OmegaConf)
            #
            # ★ 핵심 제약: preprocess.py 내부에서 os.path.basename(data_path) 를 사용해
            #   data/{basename}/{i:05d}.jpg 형태로 프레임을 로드한다.
            #   따라서 절대 경로를 넘기면 data/frames/ 같은 경로가 되어 FileNotFoundError 발생.
            #   → TOKENFLOW_DIR/data/tf_{uid}/ 에 프레임을 .jpg 로 복사하고,
            #     상대 경로 "data/tf_{uid}" 를 인자로 전달한다.

            uid = uuid.uuid4().hex[:8]
            # rel_data : preprocess.py 내부에서 basename()이 적용되므로
            #   depth-1 경로("data/tf_{uid}")의 basename = "tf_{uid}" 가 된다.
            #   스크립트는 data/tf_{uid}/{i:05d}.jpg 를 찾으므로 정상.
            # rel_output: run_tokenflow_pnp.py 가 basename() 을 적용할 수도 있으므로
            #   슬래시 없는 단일 이름으로 설정해 basename() 여부와 무관하게 동일 폴더가 되도록 한다.
            rel_data   = f"data/tf_{uid}"    # basename → "tf_{uid}" → data/tf_{uid}/ 정상
            rel_output = f"tf_{uid}_out"     # basename → "tf_{uid}_out" → TOKENFLOW_DIR 바로 아래

            tf_data_dir   = os.path.join(TOKENFLOW_DIR, rel_data)
            tf_output_dir = os.path.join(TOKENFLOW_DIR, rel_output)
            os.makedirs(tf_data_dir,   exist_ok=True)
            os.makedirs(tf_output_dir, exist_ok=True)

            try:
                # ── 프레임 복사 (PNG → JPG) ────────────────────────────
                # preprocess.py 는 {i:05d}.jpg 패턴으로 로드하므로 반드시 .jpg 확장자
                import cv2
                src_frames = sorted([
                    os.path.join(frames_dir, f)
                    for f in os.listdir(frames_dir)
                    if f.lower().endswith((".png", ".jpg", ".jpeg"))
                ])
                n_frames = len(src_frames)
                if n_frames == 0:
                    logger.error(f"[TokenFlow] frames_dir 에 이미지 없음: {frames_dir}")
                    return []

                for idx, src in enumerate(src_frames):
                    dst = os.path.join(tf_data_dir, f"{idx:05d}.jpg")
                    img = cv2.imread(src)
                    if img is None:
                        logger.warning(f"[TokenFlow] 프레임 읽기 실패: {src}")
                        continue
                    cv2.imwrite(dst, img, [cv2.IMWRITE_JPEG_QUALITY, 95])

                logger.info(
                    f"[TokenFlow] 프레임 복사 완료: {n_frames}장 → {tf_data_dir}"
                )

                # ── Step 1: preprocess.py (argparse 개별 인자) ───────────
                # --data_path / --save_dir 모두 상대 경로로 전달 (cwd=TOKENFLOW_DIR)
                logger.info(
                    f"[TokenFlow] Step1 preprocess 시작 "
                    f"(n_timesteps={N_TIMESTEPS}, sd={SD_VERSION}, "
                    f"n_frames={n_frames}, device={self.device})"
                )
                r1 = subprocess.run(
                    [
                        sys.executable, preprocess_script,
                        "--data_path",        rel_data,
                        "--save_dir",         rel_data,   # latents → rel_data/latents/
                        "--sd_version",       SD_VERSION,
                        "--steps",            str(N_TIMESTEPS),
                        "--batch_size",       str(BATCH_SIZE),
                        "--n_frames",         str(n_frames),
                        "--inversion_prompt", source_prompt or prompt,
                    ],
                    capture_output=True, text=True,
                    cwd=TOKENFLOW_DIR,
                )
                if r1.stdout:
                    for line in r1.stdout.strip().splitlines():
                        logger.info(f"[preprocess] {line}")
                if r1.stderr:
                    for line in r1.stderr.strip().splitlines():
                        logger.warning(f"[preprocess] {line}")

                if r1.returncode != 0:
                    logger.error(
                        f"[TokenFlow] preprocess.py 실패 (returncode={r1.returncode})"
                    )
                    return []

                # ── Step 2: run_tokenflow_pnp.py (--config yaml) ─────────
                # data_path / output_path 도 TOKENFLOW_DIR 기준 상대 경로
                cfg_path = os.path.join(TOKENFLOW_DIR, f"cfg_tf_{uid}.yaml")
                tokenflow_cfg = {
                    "data_path":         rel_data,
                    "output_path":       rel_output,
                    "prompt":            prompt,
                    "negative_prompt":   "blurry, low quality, distorted",
                    "sd_version":        SD_VERSION,
                    "n_timesteps":       N_TIMESTEPS,
                    "n_inversion_steps": N_TIMESTEPS,
                    "n_frames":          n_frames,
                    "latents_path":      f"{rel_data}/latents",
                    "keyframe_freq":     KEYFRAME_FREQ,
                    "batch_size":        BATCH_SIZE,
                    "guidance_scale":    7.5,
                    "seed":              42,
                    # PnP (Plug-and-Play) 주입 임계값
                    # pnp_attn_t: 전체 timestep 중 몇 % 까지 attention feature 를 원본에서 주입할지
                    # pnp_f_t   : 전체 timestep 중 몇 % 까지 spatial feature 를 원본에서 주입할지
                    "pnp_attn_t":        0.5,
                    "pnp_f_t":           0.8,
                    "device":            self.device,
                }
                with open(cfg_path, "w") as f:
                    yaml.dump(tokenflow_cfg, f)

                logger.info(
                    f"[TokenFlow] Step2 run_tokenflow_pnp 시작 "
                    f"(keyframe_freq={KEYFRAME_FREQ}, batch_size={BATCH_SIZE})"
                )
                logger.info(f"[TokenFlow] 프롬프트: {prompt[:100]}")

                r2 = subprocess.run(
                    [sys.executable, tokenflow_script, "--config", cfg_path],
                    capture_output=True, text=True,
                    cwd=TOKENFLOW_DIR,
                )
                if r2.stdout:
                    for line in r2.stdout.strip().splitlines():
                        logger.info(f"[tokenflow] {line}")
                if r2.stderr:
                    for line in r2.stderr.strip().splitlines():
                        logger.warning(f"[tokenflow] {line}")

                if r2.returncode != 0:
                    logger.error(
                        f"[TokenFlow] run_tokenflow_pnp.py 실패 (returncode={r2.returncode})"
                    )
                    return []

                # ── 출력 프레임 수집 → edited_dir 로 복사 ────────────────
                os.makedirs(edited_dir, exist_ok=True)
                raw_edited: List[str] = []
                for root, _, files in os.walk(tf_output_dir):
                    for fname in sorted(files):
                        if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                            raw_edited.append(os.path.join(root, fname))
                raw_edited.sort()

                edited: List[str] = []
                for i, src in enumerate(raw_edited):
                    ext = os.path.splitext(src)[1]
                    dst = os.path.join(edited_dir, f"{i:05d}{ext}")
                    shutil.copy2(src, dst)
                    edited.append(dst)

                logger.info(f"[TokenFlow._run_tokenflow] 편집 완료: {len(edited)}프레임")
                return edited

            finally:
                # ── 임시 디렉터리 정리 ─────────────────────────────────
                for tmp_path in [tf_data_dir, tf_output_dir]:
                    if os.path.exists(tmp_path):
                        shutil.rmtree(tmp_path, ignore_errors=True)
                cfg_file = os.path.join(TOKENFLOW_DIR, f"cfg_tf_{uid}.yaml")
                if os.path.exists(cfg_file):
                    os.remove(cfg_file)

        except Exception as e:
            logger.error(f"[TokenFlow._run_tokenflow] 예외 발생: {e}", exc_info=True)
            return []

    def _reconstruct_video(
        self,
        frame_paths: List[str],
        output_path: str,
        fps: float = EXTRACT_FPS,
    ):
        """편집된 프레임들을 mp4로 재조립 (ffmpeg).

        ffmpeg를 우선 사용하고 실패 시 OpenCV VideoWriter로 폴백.
        """
        if not frame_paths:
            return

        # ── ffmpeg 경로 ────────────────────────────────────────
        # 프레임이 0-padded 숫자 파일명(%05d.png)이면 ffmpeg의 image2 demuxer 사용
        first = os.path.basename(frame_paths[0])
        frames_dir = os.path.dirname(frame_paths[0])
        name_part, ext = os.path.splitext(first)

        pattern = os.path.join(frames_dir, f"%0{len(name_part)}d{ext}")

        r = subprocess.run(
            [
                "ffmpeg", "-y",
                "-framerate", str(fps),
                "-i", pattern,
                "-c:v", "libx264", "-preset", "ultrafast",
                "-pix_fmt", "yuv420p", "-an",
                "-loglevel", "error",
                output_path,
            ],
            capture_output=True,
        )

        if r.returncode == 0 and os.path.exists(output_path):
            logger.info(
                f"[TokenFlow._reconstruct_video] ffmpeg 재조립 완료: {output_path}"
            )
            return

        # ── OpenCV 폴백 ────────────────────────────────────────
        logger.warning("[TokenFlow._reconstruct_video] ffmpeg 실패 → OpenCV 폴백")
        sample = cv2.imread(frame_paths[0])
        if sample is None:
            return
        h, w = sample.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        for p in frame_paths:
            f = cv2.imread(p)
            if f is not None:
                writer.write(f)
        writer.release()
        logger.info(
            f"[TokenFlow._reconstruct_video] OpenCV 재조립 완료: {output_path}"
        )
