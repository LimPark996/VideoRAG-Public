"""
⑧e morph_transition.py — Optical Flow 기반 Morph 전환 생성

출처:
- OpenCV Farneback Optical Flow:
  "Two-Frame Motion Estimation Based on Polynomial Expansion"
  (Farnebäck, 2003)
  → Dense optical flow 추정으로 모든 픽셀의 움직임 벡터를 계산

- Frame Interpolation via Flow Warping:
  광학 흐름 벡터를 이용해 두 프레임 사이의 중간 프레임을 합성.
  각 중간 시점 t (0~1)에서:
    - frame_A를 t만큼 forward warp
    - frame_B를 (1-t)만큼 backward warp
    - 두 결과를 가중 합산 (blend)

역할:
  Phase 4 Adaptive Transition Logic에서 유사도 < 0.3인 경우
  시각적으로 매우 다른 두 장면 사이를 부드럽게 연결.
  단순 Crossfade(투명도 블렌딩)와 달리, 물체의 움직임 방향을
  고려한 Morphing으로 시각적 불연속(Flicker)을 해소.
"""

import logging
from typing import List, Optional

import numpy as np
import cv2

logger = logging.getLogger(__name__)


class MorphTransition:
    """Optical Flow 기반 Morph 전환 프레임 생성기

    generate_morph_frames(frame_a, frame_b, n_frames) → List[np.ndarray]
    """

    def __init__(
        self,
        flow_params: Optional[dict] = None,
    ):
        """
        Args:
            flow_params: Farneback optical flow 파라미터 (None이면 기본값)
                기본값은 빠른 추론을 위해 pyr_scale=0.5, levels=3으로 설정.
                품질 우선이면 levels=5, winsize=21로 올릴 수 있음.
        """
        self.flow_params = flow_params or {
            'pyr_scale': 0.5,   # 피라미드 스케일 (0.5 = 절반씩 축소)
            'levels': 3,        # 피라미드 단계 수 (3 = 빠름, 5 = 정밀)
            'winsize': 15,      # 윈도우 크기 (클수록 큰 움직임 포착)
            'iterations': 3,    # 반복 횟수
            'poly_n': 5,        # 다항식 근사 이웃 크기
            'poly_sigma': 1.2,  # 다항식 근사 가우시안 표준편차
            'flags': 0,
        }

    def generate_morph_frames(
        self,
        frame_a: np.ndarray,
        frame_b: np.ndarray,
        n_frames: int = 15,
        output_size: Optional[tuple] = None
    ) -> List[np.ndarray]:
        """두 프레임 사이의 Morph 전환 프레임을 생성

        프로세스:
          1) frame_a, frame_b를 그레이스케일 변환
          2) Farneback Dense Optical Flow 계산 (A→B 방향의 움직임 벡터)
          3) n_frames개의 중간 시점 t ∈ (0, 1)에 대해:
             - remap으로 frame_a를 t*flow만큼 이동 (forward warp)
             - remap으로 frame_b를 (1-t)*(-flow)만큼 이동 (backward warp)
             - 가중 합산: result = (1-t)*warped_a + t*warped_b
          4) 최종 프레임에 가우시안 블러 경량 적용 (artifact 감소)

        Args:
            frame_a: 시작 프레임 [H, W, 3] uint8 BGR
            frame_b: 끝 프레임 [H, W, 3] uint8 BGR
            n_frames: 생성할 중간 프레임 수 (기본 15 = 0.6초 @ 25fps)
            output_size: 출력 해상도 (w, h). None이면 frame_a 크기 유지

        Returns:
            List[np.ndarray] — 중간 프레임 리스트 [n_frames, H, W, 3] uint8 BGR
        """
        # 크기 맞추기
        if frame_a.shape != frame_b.shape:
            h, w = frame_a.shape[:2]
            frame_b = cv2.resize(frame_b, (w, h))

        if output_size:
            frame_a = cv2.resize(frame_a, output_size)
            frame_b = cv2.resize(frame_b, output_size)

        h, w = frame_a.shape[:2]

        # Step 1: 그레이스케일 변환
        gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
        gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)

        # Step 2: Farneback Dense Optical Flow (A→B)
        # flow[y, x] = (dx, dy) — 각 픽셀이 A에서 B로 이동하는 벡터
        try:
            flow = cv2.calcOpticalFlowFarneback(
                gray_a, gray_b, None,
                **self.flow_params
            )
        except Exception as e:
            logger.warning(f"Optical flow 계산 실패: {e} → 단순 crossfade 폴백")
            return self._crossfade_fallback(frame_a, frame_b, n_frames)

        # remap용 기본 좌표 그리드
        # map_x[y, x] = x, map_y[y, x] = y (항등 매핑)
        y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)

        # Step 3: 중간 프레임 생성
        morph_frames = []
        for i in range(n_frames):
            t = (i + 1) / (n_frames + 1)  # (0, 1) 범위, 양 끝 미포함

            # Forward warp: frame_a를 t*flow만큼 이동
            map_x_a = x_coords + t * flow[:, :, 0]
            map_y_a = y_coords + t * flow[:, :, 1]
            warped_a = cv2.remap(
                frame_a, map_x_a, map_y_a,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101
            )

            # Backward warp: frame_b를 (1-t)*(-flow)만큼 이동
            map_x_b = x_coords - (1 - t) * flow[:, :, 0]
            map_y_b = y_coords - (1 - t) * flow[:, :, 1]
            warped_b = cv2.remap(
                frame_b, map_x_b, map_y_b,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101
            )

            # 가중 합산
            blended = cv2.addWeighted(
                warped_a, 1.0 - t,
                warped_b, t,
                0
            )

            # 경량 가우시안 블러로 warp artifact 완화
            blended = cv2.GaussianBlur(blended, (3, 3), 0.5)

            morph_frames.append(blended)

        logger.info(
            f"Morph 프레임 생성: {n_frames}장 ({w}x{h})"
        )
        return morph_frames

    def _crossfade_fallback(
        self,
        frame_a: np.ndarray,
        frame_b: np.ndarray,
        n_frames: int
    ) -> List[np.ndarray]:
        """Optical flow 실패 시 단순 crossfade 폴백"""
        frames = []
        for i in range(n_frames):
            t = (i + 1) / (n_frames + 1)
            blended = cv2.addWeighted(frame_a, 1.0 - t, frame_b, t, 0)
            frames.append(blended)
        return frames
