"""
③ vector_store.py — FAISS IVFFlat 벡터 인덱스 관리

출처:
- 논문: "Billion-scale similarity search with GPUs"
        (Johnson, Douze & Jégou, 2019) — Meta/Facebook AI Research
        https://arxiv.org/abs/1702.08734
- GitHub: https://github.com/facebookresearch/faiss
- 라이선스: MIT License

프로토타입 선택:
  - FAISS-cuVS (NVIDIA GPU 빌드) → faiss-cpu로 교체
  - 이유: pip install 한 줄로 설치 가능, 1K clips에서 충분히 빠름
  - IVFFlat: nlist=100 (1K clips 기준 최적)
  - nprobe=10: 검색 시 10개 클러스터 탐색 (속도-정확도 트레이드오프)
"""

import os
import logging
from typing import List, Tuple, Optional

import numpy as np

logger = logging.getLogger(__name__)


class FAISSVectorStore:
    """FAISS IVFFlat 벡터 인덱스

    build_index(vecs, ids) → None
    search(query_vec, k) → List[(clip_id, similarity)]
    save(path) / load(path)
    """

    def __init__(
        self,
        dim: int = 512,
        nlist: int = 100,
        nprobe: int = 10,
        metric: str = "cosine"
    ):
        """
        Args:
            dim: 벡터 차원 (InternVideo2 = 512, vision_proj 출력 기준)
            nlist: IVF 클러스터 수 (1K clips → nlist=100 권장)
            nprobe: 검색 시 탐색할 클러스터 수
            metric: 유사도 메트릭 ('cosine' or 'l2')
        """
        import faiss

        self.dim = dim
        self.nlist = nlist
        self.nprobe = nprobe

        # 코사인 유사도를 위해 Inner Product 사용 (벡터 L2 정규화 전제)
        if metric == "cosine":
            quantizer = faiss.IndexFlatIP(dim)
            self.index = faiss.IndexIVFFlat(
                quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT
            )
        else:
            quantizer = faiss.IndexFlatL2(dim)
            self.index = faiss.IndexIVFFlat(quantizer, dim, nlist)

        self.index.nprobe = nprobe

        # clip_id ↔ internal_id 매핑
        self._id_map: List[str] = []
        self._is_trained = False

    def build_index(self, vectors: np.ndarray, clip_ids: List[str]) -> int:
        """벡터 인덱스 구축"""

        assert len(vectors) == len(clip_ids), \
            f"벡터/ID 수 불일치: {len(vectors)} != {len(clip_ids)}"

        actual_dim = vectors.shape[1]
        n_samples = len(vectors)
        effective_nlist = min(self.nlist, n_samples)

        # 차원 불일치 or nlist 조정 → 한 번만 재초기화
        need_reinit = (actual_dim != self.dim) or (effective_nlist != self.nlist)

        if need_reinit:
            if actual_dim != self.dim:
                logger.warning(
                    f"벡터 차원 불일치: {actual_dim} != {self.dim} "
                    f"→ dim={actual_dim}으로 재초기화 (CLIP 폴백 가능성)"
                )
            if effective_nlist != self.nlist:
                logger.info(f"nlist 조정: {self.nlist} → {effective_nlist} (데이터 수: {n_samples})")

            import faiss
            self.dim = actual_dim
            quantizer = faiss.IndexFlatIP(actual_dim)
            self.index = faiss.IndexIVFFlat(
                quantizer, actual_dim, effective_nlist, faiss.METRIC_INNER_PRODUCT
            )
            self.index.nprobe = self.nprobe
            self._is_trained = False

        vectors = vectors.astype(np.float32)

        if not self._is_trained:
            logger.info(f"IVF 학습 중 (nlist={effective_nlist})...")
            self.index.train(vectors)
            self._is_trained = True

        self.index.add(vectors)
        self._id_map = list(clip_ids)

        logger.info(
            f"인덱스 구축 완료: {self.index.ntotal} vectors, "
            f"dim={self.dim}, nlist={effective_nlist}"
        )
        return self.index.ntotal

    def search(
        self,
        query_vec: np.ndarray,
        k: int = 100
    ) -> List[Tuple[str, float]]:
        """벡터 유사도 검색"""

        if self.index.ntotal == 0:
            logger.warning("인덱스가 비어 있습니다")
            return []

        query_vec = query_vec.astype(np.float32)
        if query_vec.ndim == 1:
            # FAISS는 항상 2D 배열을 요구: (1, 512) 형태여야 함
            # 예시: [0.12, -0.34, ..., 0.87]  shape(512,)
            #     → [[0.12, -0.34, ..., 0.87]] shape(1, 512)
            query_vec = query_vec.reshape(1, -1)

        # k가 인덱스 총 벡터 수보다 크면 FAISS 에러 → min()으로 방어
        # 예시: 클립이 30개뿐인데 k=100 요청 → k=30으로 축소
        k = min(k, self.index.ntotal)

        # ─────────────────────────────────────────────────────────────────
        # index.search() 동작 방식 (IVFFlat + Inner Product 기준)
        #
        # Step 1: 쿼리 벡터와 centroid 100개의 내적 계산
        #         → 가장 가까운 nprobe=10개 버킷 선택
        #
        # Step 2: 선택된 10개 버킷 안의 벡터들(≈100개)과 내적 계산
        #         (전체 1000개 중 100개만 탐색 → 10배 빠름)
        #
        # Step 3: 내적 점수 기준 상위 k개 반환
        #
        # 예시: 쿼리 = "a man playing guitar" 임베딩
        #   scores  = [[0.91, 0.87, 0.84, 0.79, ...]]  shape(1, k)
        #   indices = [[42,   7,    318,  55,   ...]]   shape(1, k)
        #              ↑ FAISS 내부 번호
        scores, indices = self.index.search(query_vec, k)

        # scores[0], indices[0]: 배치 차원 제거 (쿼리 1개이므로 0번째만 사용)
        #
        # 예시 변환:
        #   idx=42,  score=0.91 → _id_map[42]  = "video12_scene003", 0.91
        #   idx=7,   score=0.87 → _id_map[7]   = "video3_scene001",  0.87
        #   idx=318, score=0.84 → _id_map[318] = "video44_scene005", 0.84
        #
        # idx == -1 은 FAISS가 "결과 없음"을 나타내는 특수값 → 필터링
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self._id_map):
                results.append((self._id_map[idx], float(score)))

        # 최종 반환 예시 (점수 내림차순 정렬 — FAISS가 이미 정렬해서 줌):
        # [
        #   ("video12_scene003", 0.91),  ← 기타 치는 남자 클립
        #   ("video3_scene001",  0.87),
        #   ("video44_scene005", 0.84),
        #   ...
        # ]
        return results

    def save(self, path: str):
        """인덱스 저장 (FAISS 바이너리 + ID 매핑)"""
        import faiss
        import pickle

        os.makedirs(os.path.dirname(path), exist_ok=True)

        # FAISS 인덱스 저장
        faiss.write_index(self.index, path)

        # ID 매핑 저장
        id_map_path = path + ".ids.pkl"
        with open(id_map_path, "wb") as f:
            pickle.dump(self._id_map, f)

        logger.info(f"인덱스 저장: {path} ({self.index.ntotal} vectors)")

    def load(self, path: str):
        """인덱스 로드"""
        import faiss
        import pickle

        self.index = faiss.read_index(path)
        self.index.nprobe = self.nprobe
        self._is_trained = True

        id_map_path = path + ".ids.pkl"
        if os.path.exists(id_map_path):
            with open(id_map_path, "rb") as f:
                self._id_map = pickle.load(f)

        logger.info(f"인덱스 로드: {path} ({self.index.ntotal} vectors)")

    @property
    def ntotal(self) -> int:
        return self.index.ntotal
