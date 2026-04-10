"""
⑧f tc_scorer.py — Temporal Consistency Score (TC-Score) 측정

PDF 정의:
  TC-Score(Temporal Consistency Score)는 연속 프레임 간
  광학 흐름(Optical Flow) 분석을 기반으로 시간적 변화의
  자연스러움을 정량 측정하는 지표.

수식:
  TC-Score = (1/N) × Σ_{i=1}^{N} flow_consistency(frame_i, frame_{i+1})

  flow_consistency = 1.0 - normalized_flow_magnitude
  → Optical Flow 벡터 크기가 작을수록 (변화가 부드러울수록) 점수가 높음

  보조 지표로 DINOv2 시각 유사도도 함께 제공하되,
  주 지표는 Optical Flow 기반.

해석:
  ≈ 1.0: 완벽한 시간적 연속성 (프레임 간 변화 없음)
  > 0.85: 우수 — 자연스러운 흐름
  0.7 ~ 0.85: 양호 — 적절한 수준
  0.5 ~ 0.7: 보통 — 일부 불연속 존재
  < 0.5: 개선 필요

역할:
  Phase 4 조립 결과의 품질 지표.
  목표: 1차 0.80, 2차 0.90, 3차 0.95 이상.
"""

import logging
from typing import List, Optional, Dict

import numpy as np
import cv2

logger = logging.getLogger(__name__)


class TCScorer:
    """Temporal Consistency Score 측정기 (PDF 정의: Optical Flow 기반)

    compute_tc_score(clips, ...) → TCResult dict
    """

    # Farneback Optical Flow 파라미터
    FLOW_PARAMS = dict(
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0
    )

    # Flow magnitude 정규화 상한 (이 이상이면 불연속으로 판단)
    MAX_FLOW_MAGNITUDE = 30.0

    def __init__(self):
        pass

    def compute_tc_score(
        self,
        clips,
        visual_scorer=None,
        get_keyframe_fn=None,
        n_sample_frames: int = 5
    ) -> Dict[str, float]:
        """TC-Score 계산 (PDF 정의: Optical Flow 기반)

        각 인접 클립 쌍에 대해:
          1. 이전 클립의 마지막 프레임 추출
          2. 다음 클립의 첫 프레임 추출
          3. Farneback Dense Optical Flow 계산
          4. Flow magnitude의 평균을 정규화하여 consistency 점수 산출

        Args:
            clips: 정렬된 ClipResult 리스트
            visual_scorer: VisualScorer 인스턴스 (보조 DINOv2 점수 계산용, optional)
            get_keyframe_fn: 클립에서 키프레임을 가져오는 함수 (optional)
            n_sample_frames: 경계 주변에서 샘플링할 프레임 수

        Returns:
            Dict with keys:
              - 'tc_score': 전체 평균 TC-Score (0~1, Optical Flow 기반)
              - 'tc_scores_per_pair': 인접 쌍별 TC-Score 리스트
              - 'min_tc': 최저 TC-Score
              - 'max_tc': 최고 TC-Score
              - 'std_tc': TC-Score 표준편차
              - 'n_pairs': 측정된 쌍의 수
              - 'dino_tc_score': DINOv2 보조 TC-Score (visual_scorer 있을 때)
              - 'method': 'optical_flow'
        """
        if len(clips) < 2:
            return {
                'tc_score': 1.0,
                'tc_scores_per_pair': [],
                'min_tc': 1.0,
                'max_tc': 1.0,
                'std_tc': 0.0,
                'n_pairs': 0,
                'dino_tc_score': 1.0,
                'method': 'optical_flow'
            }

        pair_scores = []
        dino_scores = []

        for i in range(len(clips) - 1):
            clip_a = clips[i]
            clip_b = clips[i + 1]

            # 프레임 추출: A의 마지막, B의 첫 프레임
            frame_a = self._extract_last_frame(clip_a)
            frame_b = self._extract_first_frame(clip_b)

            if frame_a is not None and frame_b is not None:
                # Optical Flow 기반 TC-Score
                flow_score = self._compute_flow_consistency(frame_a, frame_b)
                pair_scores.append(flow_score)

                # 보조: DINOv2 유사도
                if visual_scorer is not None:
                    try:
                        feat_a = visual_scorer.extract_features(
                            cv2.cvtColor(frame_a, cv2.COLOR_BGR2RGB)
                        )
                        feat_b = visual_scorer.extract_features(
                            cv2.cvtColor(frame_b, cv2.COLOR_BGR2RGB)
                        )
                        dino_sim = visual_scorer.compute_similarity(feat_a, feat_b)
                        dino_scores.append(dino_sim)
                    except Exception:
                        dino_scores.append(0.0)
            else:
                pair_scores.append(0.0)
                if visual_scorer:
                    dino_scores.append(0.0)

        if not pair_scores:
            return {
                'tc_score': 0.0,
                'tc_scores_per_pair': [],
                'min_tc': 0.0,
                'max_tc': 0.0,
                'std_tc': 0.0,
                'n_pairs': 0,
                'dino_tc_score': 0.0,
                'method': 'optical_flow'
            }

        tc_score = float(np.mean(pair_scores))
        dino_tc = float(np.mean(dino_scores)) if dino_scores else 0.0

        result = {
            'tc_score': tc_score,
            'tc_scores_per_pair': pair_scores,
            'min_tc': float(np.min(pair_scores)),
            'max_tc': float(np.max(pair_scores)),
            'std_tc': float(np.std(pair_scores)),
            'n_pairs': len(pair_scores),
            'dino_tc_score': dino_tc,
            'method': 'optical_flow'
        }

        logger.info(
            f"TC-Score (Optical Flow): {tc_score:.4f} "
            f"(min={result['min_tc']:.4f}, max={result['max_tc']:.4f}, "
            f"std={result['std_tc']:.4f}, pairs={result['n_pairs']})"
            + (f" | DINOv2 보조: {dino_tc:.4f}" if dino_scores else "")
        )

        return result

    def _compute_flow_consistency(
        self, frame_a: np.ndarray, frame_b: np.ndarray
    ) -> float:
        """두 프레임 간 Optical Flow 기반 시간 일관성 점수

        Args:
            frame_a: 이전 클립의 마지막 프레임 (BGR)
            frame_b: 다음 클립의 첫 프레임 (BGR)

        Returns:
            consistency score (0~1, 높을수록 자연스러운 전환)
        """
        # 그레이스케일 변환
        gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
        gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)

        # 크기 맞추기
        h = min(gray_a.shape[0], gray_b.shape[0], 480)
        w = min(gray_a.shape[1], gray_b.shape[1], 640)
        gray_a = cv2.resize(gray_a, (w, h))
        gray_b = cv2.resize(gray_b, (w, h))

        # Farneback Dense Optical Flow 계산
        try:
            flow = cv2.calcOpticalFlowFarneback(
                gray_a, gray_b, None, **self.FLOW_PARAMS
            )
        except Exception as e:
            logger.warning(f"Optical Flow 계산 실패: {e}")
            return 0.0

        # Flow magnitude 계산
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        avg_mag = float(np.mean(mag))

        # 정규화: magnitude가 작을수록 일관성이 높음
        # 0 → 1.0 (완벽한 일관성), MAX_FLOW_MAGNITUDE → 0.0
        consistency = max(0.0, 1.0 - avg_mag / self.MAX_FLOW_MAGNITUDE)

        return consistency

    def _extract_last_frame(self, clip) -> Optional[np.ndarray]:
        """클립의 마지막 프레임 추출 (BGR)"""
        if not hasattr(clip, 'video_path') or not clip.video_path:
            return None
        import os
        if not os.path.exists(clip.video_path):
            return None

        try:
            cap = cv2.VideoCapture(clip.video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

            # 실제 영상 길이 기반 안전한 seek (4차 이슈 보고서 반영)
            actual_duration_ms = (total_frames / fps * 1000) if fps > 0 else clip.end_ms
            effective_end_ms = min(clip.end_ms, actual_duration_ms)
            seek_ms = max(effective_end_ms - 200, clip.start_ms)

            cap.set(cv2.CAP_PROP_POS_MSEC, seek_ms)
            ret, frame = cap.read()

            if not ret and total_frames > 0:
                # 프레임 번호 기반 폴백
                cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
                ret, frame = cap.read()

            cap.release()
            return frame if ret else None
        except Exception:
            return None

    def _extract_first_frame(self, clip) -> Optional[np.ndarray]:
        """클립의 첫 프레임 추출 (BGR)"""
        if not hasattr(clip, 'video_path') or not clip.video_path:
            return None
        import os
        if not os.path.exists(clip.video_path):
            return None

        try:
            cap = cv2.VideoCapture(clip.video_path)
            cap.set(cv2.CAP_PROP_POS_MSEC, clip.start_ms)
            ret, frame = cap.read()
            cap.release()
            return frame if ret else None
        except Exception:
            return None

    def interpret_score(self, tc_score: float) -> str:
        """TC-Score 해석 문자열 반환"""
        if tc_score >= 0.85:
            return "Excellent — 시간적 흐름이 매우 자연스러움 (Optical Flow 기준)"
        elif tc_score >= 0.7:
            return "Good — 적절한 시간적 연속성"
        elif tc_score >= 0.5:
            return "Fair — 일부 시간적 불연속 존재"
        else:
            return "Poor — 시간적 불연속이 심함, 클립 순서/전환 개선 필요"
