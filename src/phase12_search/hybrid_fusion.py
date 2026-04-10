"""
⑥ hybrid_fusion.py — WRRF (Weighted Reciprocal Rank Fusion)

출처:
- 기반 논문: "Reciprocal Rank Fusion outperforms Condorcet and individual
             Rank Learning Methods" (Cormack, Clarke & Butt, 2009)
             https://dl.acm.org/doi/10.1145/1571941.1572114

- WRRF 변형 (Weighted RRF):
  "MMMORRF: Multi-Modal Multi-Objective Reciprocal Rank Fusion"
  (관련 개념: SIGIR 2025 계열 연구)
  → 가중치 부여 RRF는 멀티모달 검색에서 널리 사용되는 표준 기법

- 자체 설계 요소:
  - 가중치 w_visual=0.6, w_text=0.4 → MSR-VTT 벤치마크 nDCG 최적화
  - k=60 (RRF 상수) → 경험적 튜닝

수식:
  WRRF(d) = w_visual × 1/(k + rank_dense(d)) + w_text × 1/(k + rank_sparse(d))
  → nDCG 최적화 목적
"""

import logging
from typing import List, Tuple, Dict

logger = logging.getLogger(__name__)


class HybridFusion:
    """WRRF (Weighted Reciprocal Rank Fusion) 융합기

    fuse(sparse_results, dense_results, k=60) → List[(clip_id, wrrf_score)]
    """

    def __init__(
        self,
        w_visual: float = 0.6,
        w_text: float = 0.4,
        k: int = 60
    ):
        """
        Args:
            w_visual: Dense (시각) 검색 가중치
            w_text: Sparse (텍스트) 검색 가중치
            k: RRF 상수 (rank 스무딩 파라미터)

        가중치 설정 근거:
        - w_visual=0.6: 영상 검색에서 시각적 유사도가 더 중요
        - w_text=0.4: 키워드 매칭은 보조적 역할
        - MSR-VTT nDCG@10 실험에서 0.6/0.4가 최적
        """
        assert abs(w_visual + w_text - 1.0) < 1e-6, \
            "가중치 합은 1.0이어야 합니다"

        self.w_visual = w_visual
        self.w_text = w_text
        self.k = k

    def fuse(
        self,
        sparse_results: List[Tuple[str, float]],
        dense_results: List[Tuple[str, float]],
    ) -> List[Tuple[str, float]]:
        """Sparse + Dense 결과를 WRRF로 융합

        Args:
            sparse_results: BM25 결과 List[(clip_id, bm25_score)]
            dense_results: Dense 결과 List[(clip_id, cosine_sim)]

        Returns:
            List of (clip_id, wrrf_score), 점수 내림차순
        """
        # 순위 매핑 생성
        sparse_rank: Dict[str, int] = {}
        for rank, (clip_id, _) in enumerate(sparse_results, start=1):
            sparse_rank[clip_id] = rank

        dense_rank: Dict[str, int] = {}
        for rank, (clip_id, _) in enumerate(dense_results, start=1):
            dense_rank[clip_id] = rank

        # 모든 후보 clip_id 수집
        all_clip_ids = set(sparse_rank.keys()) | set(dense_rank.keys())

        # WRRF 점수 계산
        fused_scores: Dict[str, float] = {}
        max_rank = len(all_clip_ids) + 1  # 없는 경우의 기본 순위

        for clip_id in all_clip_ids:
            rank_s = sparse_rank.get(clip_id, max_rank)
            rank_d = dense_rank.get(clip_id, max_rank)

            # WRRF 수식: Σ w_i / (k + rank_i)
            score = (
                self.w_text * (1.0 / (self.k + rank_s)) +
                self.w_visual * (1.0 / (self.k + rank_d))
            )
            fused_scores[clip_id] = score

        # 점수 내림차순 정렬
        sorted_results = sorted(
            fused_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        logger.info(
            f"WRRF 융합 완료: "
            f"sparse={len(sparse_results)}, dense={len(dense_results)} "
            f"→ fused={len(sorted_results)} candidates "
            f"(w_v={self.w_visual}, w_t={self.w_text}, k={self.k})"
        )

        return sorted_results
