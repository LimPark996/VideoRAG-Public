"""
tokenflow_wrapper.py — TokenFlow video-to-video 편집 래퍼

TokenFlow (Geyer et al., 2023): https://github.com/omerbt/TokenFlow
  - 기존 영상 클립을 텍스트 프롬프트로 스타일 변환
  - Stable Diffusion + DDIM inversion + attention feature sharing
  - keyframe subsampling으로 실용적인 속도 달성

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
import logging
import shutil
import tempfile
import subprocess
from typing import Optional, List

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ── 고정 설정 ──────────────────────────────────────────────
EXTRACT_FPS    = 8
KEYFRAME_FREQ  = 10
N_TIMESTEPS    = 50
BATCH_SIZE     = 8

# TokenFlow 레포 경로 (Colab 기준)
TOKENFLOW_DIR  = "/content/TokenFlow"
# Stable Diffusion 모델 (HF Hub ID)
SD_MODEL_ID    = "CompVis/stable-diffusion-v1-4"


def _ensure_tokenflow() -> bool:
    """TokenFlow 레포가 없으면 클론.

    Returns:
        True: 사용 가능 / False: 설치 실패
    """
    if os.path.exists(os.path.join(TOKENFLOW_DIR, "tokenflow")):
        return True

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
    return True


def _add_tokenflow_to_path():
    """TokenFlow 소스를 sys.path에 추가"""
    if TOKENFLOW_DIR not in sys.path:
        sys.path.insert(0, TOKENFLOW_DIR)


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
        """첫 호출 시 TokenFlow 설치 + 모델 로드"""
        if self._ready:
            return
        if not _ensure_tokenflow():
            raise RuntimeError(
                "TokenFlow 설치 실패. "
                "인터넷 연결 또는 git이 필요합니다."
            )
        _add_tokenflow_to_path()
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
        """TokenFlow PnP 편집 실행.

        TokenFlow의 TokenFlow 클래스를 직접 호출한다.
        실패 시 빈 리스트 반환.

        Returns:
            편집된 프레임 경로 리스트 (오름차순)
        """
        try:
            # TokenFlow 레포의 tokenflow 패키지 import
            from tokenflow_pnp import TokenFlow  # type: ignore

            config = {
                # 경로
                "data_path":    frames_dir,
                "output_path":  edited_dir,
                # 프롬프트
                "prompt":        prompt,
                "source_prompt": source_prompt or prompt,
                # 모델
                "model_path":    SD_MODEL_ID,
                "device":        self.device,
                # 핵심 파라미터
                "n_timesteps":   N_TIMESTEPS,
                "keyframe_freq": KEYFRAME_FREQ,
                "batch_size":    BATCH_SIZE,
                # 이미지 크기 (TokenFlow 권장: 512)
                "image_size":    [512, 512],
                # 가이던스
                "guidance_scale": 7.5,
                "n_inversion_steps": N_TIMESTEPS,
            }

            logger.info(
                f"[TokenFlow._run_tokenflow] 시작 "
                f"(keyframe_freq={KEYFRAME_FREQ}, n_timesteps={N_TIMESTEPS}, "
                f"batch_size={BATCH_SIZE}, device={self.device})"
            )
            logger.info(f"[TokenFlow._run_tokenflow] 프롬프트: {prompt[:100]}")

            model = TokenFlow(config)
            model.run_pnp()

            # 출력 프레임 수집
            edited = sorted([
                os.path.join(edited_dir, f)
                for f in os.listdir(edited_dir)
                if f.endswith((".png", ".jpg"))
            ])
            logger.info(f"[TokenFlow._run_tokenflow] 편집 완료: {len(edited)}프레임")
            return edited

        except Exception as e:
            logger.error(f"[TokenFlow._run_tokenflow] 실패: {e}", exc_info=True)
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
