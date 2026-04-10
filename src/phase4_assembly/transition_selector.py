"""
⑧b transition_selector.py — 트랜지션 타입 결정 로직

출처: 자체 설계 (Adaptive Transition Logic)
  - DINOv2 cosine similarity 기반 3단계 분기 (PDF 원안)
  - 임계값은 영상 편집 관행과 시지각 연구를 참고하여 설정

PDF 설계:
  거리 ≤ 0.2 → Morph, 0.2~0.5 → Crossfade, > 0.5 → Cut
  (DINOv2 임베딩 거리 기반)

  코사인 유사도로 변환하면:
  sim >= 0.6 → Cut (직접 전환, 시각적으로 유사한 장면)
  0.3 ~ 0.6 → Crossfade (0.5초)로 부드럽게
  sim < 0.3 → Morph (0.48초, 12프레임)로 시각 불연속 해결
"""

import logging
from typing import List

from ..data_models import TransitionDecision, TransitionType

logger = logging.getLogger(__name__)


class TransitionSelector:
    """트랜지션 타입 자동 결정기 (PDF 원안: 순수 DINOv2 기반)

    select(sim_score) → TransitionDecision
    """

    def __init__(
        self,
        cut_threshold: float = 0.6,
        crossfade_threshold: float = 0.3,
        crossfade_duration: float = 0.5,
        morph_duration: float = 0.48
    ):
        self.cut_threshold = cut_threshold
        self.crossfade_threshold = crossfade_threshold
        self.crossfade_duration = crossfade_duration
        self.morph_duration = morph_duration

    def select(
        self,
        sim_score: float,
        from_clip_id: str,
        to_clip_id: str
    ) -> TransitionDecision:
        """DINOv2 유사도 기반 트랜지션 타입 결정 (PDF 원안)

        순수 DINOv2 코사인 유사도만으로 결정. Scene 메타데이터 미참조.

        Args:
            sim_score: DINOv2 cosine similarity (0~1)
            from_clip_id: 이전 클립 ID
            to_clip_id: 다음 클립 ID

        Returns:
            TransitionDecision
        """
        if sim_score >= self.cut_threshold:
            decision = TransitionDecision(
                transition_type=TransitionType.CUT,
                similarity_score=sim_score,
                duration_sec=0.0,
                from_clip_id=from_clip_id,
                to_clip_id=to_clip_id
            )
        elif sim_score >= self.crossfade_threshold:
            decision = TransitionDecision(
                transition_type=TransitionType.CROSSFADE,
                similarity_score=sim_score,
                duration_sec=self.crossfade_duration,
                from_clip_id=from_clip_id,
                to_clip_id=to_clip_id
            )
        else:
            decision = TransitionDecision(
                transition_type=TransitionType.MORPH,
                similarity_score=sim_score,
                duration_sec=self.morph_duration,
                from_clip_id=from_clip_id,
                to_clip_id=to_clip_id
            )

        logger.debug(
            f"트랜지션: {from_clip_id} → {to_clip_id}: "
            f"sim={sim_score:.3f} → "
            f"{decision.transition_type.value} ({decision.duration_sec}s)"
        )
        return decision

    def select_all(
        self,
        similarities: List[float],
        clip_ids: List[str]
    ) -> List[TransitionDecision]:
        """연속 클립 쌍에 대해 트랜지션 일괄 결정 (PDF 원안: DINOv2만 사용)

        Args:
            similarities: 인접 클립 유사도 리스트 (len = N-1)
            clip_ids: 클립 ID 리스트 (len = N)

        Returns:
            List[TransitionDecision] (len = N-1)
        """
        assert len(similarities) == len(clip_ids) - 1

        decisions = []
        for i, sim in enumerate(similarities):
            decision = self.select(sim, clip_ids[i], clip_ids[i + 1])
            decisions.append(decision)

        # 통계 로깅
        type_counts = {}
        for d in decisions:
            t = d.transition_type.value
            type_counts[t] = type_counts.get(t, 0) + 1

        logger.info(f"트랜지션 결정 완료: {type_counts}")
        return decisions
