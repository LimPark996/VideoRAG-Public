"""
metrics.py — Recall@K, nDCG, TC-Score 계산

출처:
- Recall@K, nDCG: 정보 검색 표준 지표
  "Introduction to Information Retrieval" (Manning, Raghavan, Schütze, 2008)

- TC-Score (Transition Coherence Score): 자체 설계
  DINOv2 feature 기반 인접 클립 유사도의 평균
  → 영상 합성 결과의 시각적 자연성 수치화
"""

import numpy as np
from typing import List, Set, Optional


def compute_recall_at_k(
    retrieved: List[str],
    relevant: Set[str],
    k: int = 10
) -> float:
    """Recall@K 계산

    Args:
        retrieved: 검색된 clip_id 리스트 (순서대로)
        relevant: 정답 clip_id 집합
        k: 상위 K개까지 평가

    Returns:
        Recall@K (0.0 ~ 1.0)
    """
    if not relevant:
        return 0.0

    retrieved_at_k = set(retrieved[:k])
    hits = retrieved_at_k & relevant
    return len(hits) / len(relevant)


def compute_ndcg(
    retrieved: List[str],
    relevant: Set[str],
    k: int = 10
) -> float:
    """nDCG@K (Normalized Discounted Cumulative Gain) 계산

    Args:
        retrieved: 검색된 clip_id 리스트 (순서대로)
        relevant: 정답 clip_id 집합
        k: 상위 K개까지 평가

    Returns:
        nDCG@K (0.0 ~ 1.0)
    """
    if not relevant:
        return 0.0

    # DCG 계산
    dcg = 0.0
    for i, doc in enumerate(retrieved[:k]):
        rel = 1.0 if doc in relevant else 0.0
        dcg += rel / np.log2(i + 2)  # i+2 because i starts at 0

    # Ideal DCG (모든 정답이 상위에 있는 경우)
    ideal_rels = sorted([1.0] * min(len(relevant), k), reverse=True)
    idcg = sum(r / np.log2(i + 2) for i, r in enumerate(ideal_rels))

    return dcg / idcg if idcg > 0 else 0.0


def compute_tc_score(
    similarities: List[float]
) -> float:
    """TC-Score (Transition Coherence Score) 계산

    설계: 자체 설계
    인접 클립 DINOv2 유사도의 가중 평균
    → 값이 높을수록 시각적으로 자연스러운 영상

    Args:
        similarities: 인접 클립 쌍의 DINOv2 cosine similarity 리스트

    Returns:
        TC-Score (0.0 ~ 1.0)

    목표:
        - 프로토타입: >= 0.80
        - 최종: >= 0.95
    """
    if not similarities:
        return 0.0

    # 가중 평균: 앞쪽 트랜지션에 더 높은 가중치 (첫인상 중요)
    weights = np.array([1.0 / (i + 1) for i in range(len(similarities))])
    weights = weights / weights.sum()

    scores = np.array(similarities)
    tc_score = float(np.dot(weights, scores))

    return max(0.0, min(1.0, tc_score))


def compute_hallucination_rate(
    retrieved_captions: List[str],
    query: str,
    threshold: float = 0.3
) -> float:
    """환각율 (의미 불일치) 측정 — 간이 버전

    설계: 자체 설계 (단어 겹침 기반 간이 측정)
    프로덕션에서는 VLM (PRISM/WCS) 기반 정밀 측정

    Args:
        retrieved_captions: 검색된 클립 캡션 리스트
        query: 원본 쿼리
        threshold: 불일치 판정 임계값

    Returns:
        환각율 (0.0 ~ 1.0)
    """
    if not retrieved_captions:
        return 0.0

    query_words = set(query.lower().split())
    hallucinated = 0

    for caption in retrieved_captions:
        caption_words = set(caption.lower().split())
        overlap = query_words & caption_words
        similarity = len(overlap) / max(len(query_words), 1)

        if similarity < threshold:
            hallucinated += 1

    return hallucinated / len(retrieved_captions)
