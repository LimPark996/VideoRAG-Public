"""
⑤b dense_retriever.py — FAISS 기반 Dense ANN 검색

출처:
- FAISS: Meta AI Research (위 vector_store.py 참조)
- Dense Retrieval 개념:
  "Dense Passage Retrieval for Open-Domain Question Answering"
  (Karpukhin et al., 2020) — Meta AI
  https://arxiv.org/abs/2004.04906

역할:
  - InternVideo2 임베딩 공간에서 쿼리 벡터와 가장 유사한 클립 검색
  - FAISS IVFFlat을 통한 ANN (Approximate Nearest Neighbor) 검색
  - 검색 시간: ~80ms (1K clips, IVFFlat, nprobe=10)
"""

import logging
from typing import List, Tuple

import numpy as np

from ..phase0_indexing.embedder import VideoEmbedder
from ..phase0_indexing.vector_store import FAISSVectorStore

logger = logging.getLogger(__name__)


class DenseRetriever:
    """FAISS 기반 Dense 의미 검색기

    search(query_vec, top_k=100) → List[(clip_id, cosine_sim)]
    """

    def __init__(
        self,
        embedder: VideoEmbedder,
        vector_store: FAISSVectorStore
    ):
        """
        Args:
            embedder: 쿼리 인코딩을 위한 임베더
            vector_store: FAISS 벡터 인덱스
        """
        self.embedder = embedder
        self.vector_store = vector_store

    def search(
        self,
        query_vec: np.ndarray,
        top_k: int = 100
    ) -> List[Tuple[str, float]]:
        """Dense 벡터 유사도 검색

        Args:
            query_vec: np.ndarray [1, dim] (이미 인코딩된 쿼리 벡터)
            top_k: 반환할 결과 수

        Returns:
            List of (clip_id, cosine_similarity), 점수 내림차순
        """
        results = self.vector_store.search(query_vec, k=top_k)
        logger.debug(f"Dense 검색 완료: {len(results)} results (top_k={top_k})")
        return results

    def encode_and_search(
        self,
        query: str,
        top_k: int = 100
    ) -> Tuple[np.ndarray, List[Tuple[str, float]]]:
        """쿼리 인코딩 + 검색을 한 번에 수행

        Args:
            query: 자연어 쿼리
            top_k: 반환할 결과 수

        Returns:
            (query_vec, results)
        """
        query_vec = self.embedder.encode_query(query)
        results = self.search(query_vec, top_k)
        return query_vec, results
