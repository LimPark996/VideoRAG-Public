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
  extract_fps   = 4    원본에서 4fps로 프레임 추출 (30fps → 1/7.5 subsampling)
  keyframe_freq = 4    추출된 프레임 중 4개마다 keyframe 1개
  n_timesteps   = 30   DDIM steps (50→30: 초반 픽셀 깨짐 완화)
  batch_size    = 8    keyframe 배치 크기

  변경 이유:
    - extract_fps 8 + keyframe_freq 10 조합은 7초 이하 짧은 클립에서
      keyframe이 5개 미만으로 줄어 보간이 뭉개지고 앞 구간이 반복되는 문제 발생
    - extract_fps 4 + keyframe_freq 4 로 변경하면 7초 클립 기준
      keyframe 7개로 균등하게 확보되어 안정적으로 동작
    - n_timesteps 50 → 30: DDIM 초반 구간 불안정으로 인한 픽셀 깨짐 완화

7초 클립 기준:
  추출 프레임 = 7 × 4 = 28개
  keyframe    = 28 ÷ 4 = 7개
  DDIM 배치   = ceil(7/8) = 1배치
  예상 소요   ~ 20~40초 (T4)
"""

import os
import sys
import re
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
EXTRACT_FPS    = 4
KEYFRAME_FREQ  = 4
N_TIMESTEPS    = 30
BATCH_SIZE     = 8

# TokenFlow 레포 경로 (Colab 기준)
TOKENFLOW_DIR  = "/content/TokenFlow"
# Stable Diffusion 버전
# preprocess.py 유효값: {1.5, 2.0, 2.1, ControlNet, depth}  ("1.4" 없음)
SD_VERSION     = "1.5"


def _sanitize_for_path(text: str) -> str:
    """
    [문제 2 수정] 프롬프트 문자열을 폴더명에 안전하게 쓸 수 있도록 정제.

    TokenFlow가 output_path 뒤에 프롬프트를 폴더명으로 붙이는데,
    슬래시(/) · 백슬래시 · 공백 · 특수문자가 포함되면 경로가 깨진다.
    → 영문자·숫자·하이픈·언더스코어만 남기고 나머지는 '_'로 치환.
    """
    sanitized = re.sub(r'[^\w\-]', '_', text)
    # 연속 언더스코어 압축 & 앞뒤 정리
    sanitized = re.sub(r'_+', '_', sanitized).strip('_')
    return sanitized[:80]  # 폴더명이 너무 길어지지 않도록 80자 제한


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


def _cleanup_tokenflow_dirs(uid: str, rel_output: str):
    """
    [문제 6 수정] TokenFlow 실행 중 생성된 임시 디렉터리를 안전하게 정리.

    finally 블록 외부에서도 호출 가능하도록 분리.
    glob으로 rel_output 접두어 디렉터리 전체 삭제.
    """
    import glob as _glob

    targets = []

    # 프레임 복사용 임시 폴더
    tf_data_dir = os.path.join(TOKENFLOW_DIR, "data", f"tf_{uid}")
    if os.path.exists(tf_data_dir):
        targets.append(tf_data_dir)

    # run_tokenflow_pnp.py 출력 폴더 (접두어 glob)
    for path in _glob.glob(os.path.join(TOKENFLOW_DIR, f"{rel_output}*")):
        targets.append(path)

    # yaml config 파일
    cfg_file = os.path.join(TOKENFLOW_DIR, f"cfg_tf_{uid}.yaml")
    if os.path.exists(cfg_file):
        targets.append(cfg_file)

    for t in targets:
        try:
            if os.path.isdir(t):
                shutil.rmtree(t, ignore_errors=True)
            elif os.path.isfile(t):
                os.remove(t)
        except Exception as e:
            logger.warning(f"[TokenFlow] 임시 파일 정리 실패 (무시): {t} — {e}")


def _check_preprocess_args() -> bool:
    """
    [문제 7 수정] preprocess.py가 --n_frames 인자를 지원하는지 사전 확인.

    지원하지 않으면 해당 인자를 건너뛰도록 플래그를 반환한다.
    """
    script = os.path.join(TOKENFLOW_DIR, "preprocess.py")
    if not os.path.exists(script):
        return False
    with open(script, "r", encoding="utf-8") as f:
        src = f.read()
    return "--n_frames" in src or "n_frames" in src


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

        원본 fps가 EXTRACT_FPS보다 낮은 경우 (예: MSR-VTT 3fps 영상),
        ffmpeg로 먼저 30fps로 보간한 뒤 추출한다.
        → 원본 fps 그대로 추출하면 프레임 수가 너무 적어 TokenFlow 품질 저하.

        Returns:
            (orig_fps, sorted frame path list)
        """
        # 원본 fps 확인
        cap = cv2.VideoCapture(video_path)
        orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 원본 fps가 EXTRACT_FPS보다 낮으면 원본 fps 그대로 사용
        # ffmpeg 보간(가짜 프레임 생성)은 TokenFlow 품질을 오히려 낮추므로 사용하지 않음
        # → 3fps 7초 영상 = 21개 진짜 프레임이 30fps 보간 후 추출한 27개 가짜 프레임보다 나음
        effective_fps = min(EXTRACT_FPS, orig_fps)
        step = max(1, round(orig_fps / effective_fps))
        logger.info(
            f"[TokenFlow._extract_frames] 원본 fps={orig_fps:.1f}, "
            f"effective_fps={effective_fps:.1f}, step={step}"
        )

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

            uid = uuid.uuid4().hex[:8]
            rel_data   = f"data/tf_{uid}"
            # [문제 2 수정] 프롬프트를 경로 안전 문자열로 정제한 뒤 rel_output에 사용
            safe_prompt = _sanitize_for_path(prompt)
            rel_output = f"tf_{uid}_out_{safe_prompt}"

            tf_data_dir = os.path.join(TOKENFLOW_DIR, rel_data)
            os.makedirs(tf_data_dir, exist_ok=True)

            try:
                # ── 프레임 복사 (PNG → JPG) ────────────────────────────
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
                logger.info(
                    f"[TokenFlow] Step1 preprocess 시작 "
                    f"(n_timesteps={N_TIMESTEPS}, sd={SD_VERSION}, "
                    f"n_frames={n_frames}, device={self.device})"
                )

                preprocess_cmd = [
                    sys.executable, preprocess_script,
                    "--data_path",        rel_data,
                    "--save_dir",         rel_data,
                    "--sd_version",       SD_VERSION,
                    "--steps",            str(N_TIMESTEPS),
                    "--save_steps",       str(N_TIMESTEPS),  # run_tokenflow_pnp의 n_timesteps와 동일하게 맞춤
                    "--batch_size",       str(BATCH_SIZE),   # → 같은 timestep 목록으로 latent 저장/요청 일치
                    "--n_frames",         str(n_frames),
                    "--inversion_prompt", source_prompt or prompt,
                ]

                r1 = subprocess.run(
                    preprocess_cmd,
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
                    "latents_path":      rel_data,
                    "keyframe_freq":     KEYFRAME_FREQ,
                    "batch_size":        BATCH_SIZE,
                    "guidance_scale":    7.5,
                    "seed":              42,
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
                # run_tokenflow_pnp.py는 편집된 프레임을 img_ode/ 폴더에만 저장한다.
                # fps_10/, fps_20/, fps_30/ 은 mp4 파일이므로 프레임 탐색 대상에서 제외.
                # ** 재귀 탐색을 쓰면 중복 수집으로 프레임이 뻥튀기되어 영상이 반복됨.
                os.makedirs(edited_dir, exist_ok=True)
                import glob as _glob

                # run_tokenflow_pnp.py는 output_path 아래에 여러 단계의 하위 폴더를
                # 자동으로 만들고 그 안의 img_ode/ 에 프레임을 저장한다.
                # rel_output 접두어로 시작하는 폴더를 재귀 탐색하여 img_ode/ 를 찾는다.
                raw_edited: List[str] = []
                img_ode_dirs = [
                    d for d in _glob.glob(
                        os.path.join(TOKENFLOW_DIR, f"{rel_output}*", "**", "img_ode"),
                        recursive=True,
                    )
                    if os.path.isdir(d)
                ]
                for img_ode_dir in img_ode_dirs:
                    candidates = sorted(
                        f for f in _glob.glob(os.path.join(img_ode_dir, "*"))
                        if os.path.isfile(f) and f.lower().endswith((".png", ".jpg", ".jpeg"))
                    )
                    if candidates:
                        raw_edited = candidates
                        logger.info(
                            f"[TokenFlow] 출력 프레임 수집: {img_ode_dir} ({len(raw_edited)}개)"
                        )
                        break

                if not raw_edited:
                    logger.error(
                        f"[TokenFlow] 출력 프레임 없음 — img_ode 탐색 경로: "
                        f"{os.path.join(TOKENFLOW_DIR, rel_output + '*')}/**/img_ode/"
                    )
                    return []

                edited: List[str] = []
                for i, src in enumerate(raw_edited):
                    ext = os.path.splitext(src)[1]
                    dst = os.path.join(edited_dir, f"{i:05d}{ext}")
                    shutil.copy2(src, dst)
                    edited.append(dst)

                logger.info(f"[TokenFlow._run_tokenflow] 편집 완료: {len(edited)}프레임")
                return edited

            finally:
                # [문제 6 수정] 전용 함수로 분리하여 subprocess 비정상 종료 시에도 확실히 정리
                _cleanup_tokenflow_dirs(uid, rel_output)

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