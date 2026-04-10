"""
⑦ reranker.py — ColBERT v2 Late Interaction 재순위화 + PLAID 가속 엔진

출처:
- ColBERT v1 논문:
  "ColBERT: Efficient and Effective Passage Search via Contextualized Late
   Interaction over BERT" (Khattab & Zaharia, 2020) — Stanford
  https://arxiv.org/abs/2004.12832

- ColBERT v2 논문:
  "ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction"
  (Santhanam et al., 2022) — Stanford
  https://arxiv.org/abs/2112.01488

- PLAID 가속 엔진:
  "PLAID: An Efficient Engine for Late Interaction Retrieval"
  (Santhanam et al., 2022) — Stanford
  https://arxiv.org/abs/2205.09707
  → 센트로이드 가지치기 (Centroid Pruning)로 1억 건급 실시간 처리
  → GPU 0.2초 이내 / CPU에서도 유의미한 가속

- 구현체: RAGatouille (ColBERT 래퍼)
  GitHub: https://github.com/bclavie/RAGatouille
  라이선스: Apache 2.0

핵심 수식 (MaxSim):
  score(Q, D) = Σ_{q_i ∈ Q} max_{d_j ∈ D} (E_q_i · E_d_j)
  - E_q [L_q, 128]: 쿼리 토큰 임베딩
  - E_d [L_d, 128]: 문서 토큰 임베딩
  - 토큰 수준 세밀한 상호작용 → "악기 연주" 같은 맥락 포착

PLAID 가속 원리:
  1) 오프라인: 모든 문서 토큰 임베딩을 K-Means로 클러스터링 → 센트로이드 생성
  2) 온라인: 쿼리 토큰과 센트로이드만 먼저 비교 (저렴한 연산)
  3) 근접 센트로이드에 속한 문서 토큰만 실제 MaxSim 계산 (가지치기)
  → 전체 토큰 비교 대비 10~50배 속도 향상, 품질 손실 < 1%
"""

import os
import time
import logging
from typing import List, Tuple, Optional, Dict

import numpy as np

from ..data_models import ClipResult

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# PLAID 가속 엔진 (센트로이드 가지치기)
# ──────────────────────────────────────────────────────────────────────

class PLAIDEngine:
    """PLAID 가속 엔진 — 센트로이드 기반 가지치기로 MaxSim 고속화

    draw.io 아키텍처 Phase 3:
    "PLAID 가속 엔진: 센트로이드 가지치기 이론을 적용하여 GPU 환경에서
     0.2초 이내에 1억 건급의 연산을 실시간으로 처리합니다."

    동작 원리 (2단계):

    ┌─────────────────────────────────────────────────────────────────┐
    │  오프라인 단계 (인덱싱 시 1회)                                     │
    │                                                                 │
    │  모든 문서 토큰 임베딩 [N_docs × L_d, dim]                         │
    │       ↓ K-Means 클러스터링                                       │
    │  센트로이드 행렬 [n_centroids, dim]                                │
    │       + 각 토큰이 속한 센트로이드 ID 기록                            │
    └─────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────┐
    │  온라인 단계 (쿼리 시 매번)                                        │
    │                                                                 │
    │  Step 1: 쿼리 토큰 [L_q, dim] × 센트로이드 [n_centroids, dim]^T   │
    │          → 센트로이드 점수 [L_q, n_centroids]                      │
    │          → 상위 n_probe개 센트로이드만 선택 (가지치기)                  │
    │                                                                 │
    │  Step 2: 선택된 센트로이드에 속한 문서 토큰만 MaxSim 계산              │
    │          → 전체 토큰의 10~20%만 실제 비교                            │
    │          → 10~50배 가속, 품질 손실 < 1%                             │
    └─────────────────────────────────────────────────────────────────┘

    예시 (1K clips, 평균 캡션 토큰 15개):
      - 전체 토큰: 1000 × 15 = 15,000개
      - 센트로이드: 256개 (n_centroids)
      - n_probe=32: 상위 32개 센트로이드 → ~1,800 토큰만 비교
      - 가속비: 15000/1800 ≈ 8.3배

    프로토타입 참고:
      1K clips에서는 브루트포스(~350ms)로도 목표(2.5초) 내이므로
      PLAID는 선택적 활성화. 대규모(10K+) 확장 시 필수.
    """

    def __init__(
        self,
        n_centroids: int = 256,
        n_probe: int = 32,
    ):
        """
        Args:
            n_centroids: K-Means 클러스터 수 (기본 256)
                → 문서 토큰 수의 제곱근 근처가 최적
                  (15K 토큰 → √15000 ≈ 122, 여유 두고 256)
            n_probe: 쿼리당 탐색할 센트로이드 수 (기본 32)
                → n_centroids의 12.5% → 속도/품질 균형점
                → 높이면 품질↑ 속도↓, 낮추면 반대
        """
        self.n_centroids = n_centroids
        self.n_probe = n_probe
        self.centroids = None          # [n_centroids, dim]
        self.token_assignments = None  # 각 토큰의 센트로이드 ID
        self.doc_token_embeddings = None  # [total_tokens, dim]
        self.doc_token_to_doc = None   # 각 토큰이 속한 문서 인덱스
        self._built = False

    def build_index(
        self,
        doc_embeddings_list: List[np.ndarray],
    ):
        """오프라인: 문서 토큰 임베딩으로 센트로이드 인덱스 구축

        Args:
            doc_embeddings_list: List of [L_d_i, dim] — 각 문서의 토큰 임베딩

        예시:
            doc_embeddings_list = [
                np.array([[0.1, 0.2, ...], [0.3, 0.4, ...]]),  # doc0: 2 tokens
                np.array([[0.5, 0.6, ...], [0.7, 0.8, ...], [0.9, 1.0, ...]]),  # doc1: 3 tokens
            ]
            → all_tokens shape: [5, dim]
            → K-Means → 센트로이드 [n_centroids, dim]
            → token_assignments: [2, 2, 0, 1, 0]  (각 토큰의 클러스터 ID)
        """
        start_time = time.perf_counter()

        # 모든 문서 토큰을 하나로 합침
        all_tokens = []
        doc_mapping = []  # 각 토큰이 속한 문서 인덱스

        for doc_idx, doc_emb in enumerate(doc_embeddings_list):
            if doc_emb is not None and len(doc_emb) > 0:
                all_tokens.append(doc_emb)
                doc_mapping.extend([doc_idx] * len(doc_emb))

        if not all_tokens:
            logger.warning("PLAID: 문서 토큰 없음, 인덱스 구축 건너뜀")
            return

        self.doc_token_embeddings = np.vstack(all_tokens).astype(np.float32)
        self.doc_token_to_doc = np.array(doc_mapping, dtype=np.int32)

        n_tokens = len(self.doc_token_embeddings)
        actual_centroids = min(self.n_centroids, n_tokens)

        # K-Means 클러스터링으로 센트로이드 생성
        try:
            from sklearn.cluster import MiniBatchKMeans

            kmeans = MiniBatchKMeans(
                n_clusters=actual_centroids,
                batch_size=min(1024, n_tokens),
                n_init=3,
                max_iter=100,
                random_state=42
            )
            self.token_assignments = kmeans.fit_predict(self.doc_token_embeddings)
            self.centroids = kmeans.cluster_centers_.astype(np.float32)

        except ImportError:
            logger.warning("scikit-learn 미설치 → 랜덤 샘플링 폴백")
            # 폴백: 랜덤 샘플을 센트로이드로 사용
            indices = np.random.choice(n_tokens, actual_centroids, replace=False)
            self.centroids = self.doc_token_embeddings[indices].copy()
            # 각 토큰을 가장 가까운 센트로이드에 할당
            sims = self.doc_token_embeddings @ self.centroids.T
            self.token_assignments = sims.argmax(axis=1)

        self._built = True
        build_time = (time.perf_counter() - start_time) * 1000

        logger.info(
            f"PLAID 인덱스 구축 완료: {n_tokens} tokens → "
            f"{actual_centroids} centroids ({build_time:.0f}ms)"
        )

    def pruned_maxsim(
        self,
        query_embeddings: np.ndarray,
        doc_indices: Optional[List[int]] = None,
    ) -> Dict[int, float]:
        """센트로이드 가지치기 MaxSim 계산

        Args:
            query_embeddings: [L_q, dim] 쿼리 토큰 임베딩
            doc_indices: 계산 대상 문서 인덱스 (None이면 전체)

        Returns:
            Dict[doc_idx, maxsim_score]

        상세 과정:
          1) 쿼리 토큰 × 센트로이드 → [L_q, n_centroids]
          2) 각 쿼리 토큰별 상위 n_probe 센트로이드 선택
          3) 선택된 센트로이드에 속한 토큰만 실제 비교
          4) 문서별 MaxSim 합산
        """
        if not self._built:
            logger.warning("PLAID 인덱스 미구축 → 브루트포스 폴백")
            return self._brute_force_maxsim(query_embeddings, doc_indices)

        start_time = time.perf_counter()

        q_emb = query_embeddings.astype(np.float32)  # [L_q, dim]

        # Step 1: 쿼리 토큰 × 센트로이드 유사도
        # [L_q, dim] × [dim, n_centroids] → [L_q, n_centroids]
        centroid_scores = q_emb @ self.centroids.T

        # Step 2: 각 쿼리 토큰별 상위 n_probe 센트로이드 선택
        # [L_q, n_probe] — 가장 유사한 센트로이드 ID
        n_probe = min(self.n_probe, centroid_scores.shape[1])
        top_centroid_ids = np.argpartition(
            -centroid_scores, n_probe, axis=1
        )[:, :n_probe]

        # 선택된 센트로이드에 속한 토큰 인덱스 수집 (중복 제거)
        selected_centroids = set()
        for row in top_centroid_ids:
            selected_centroids.update(row.tolist())

        # 선택된 센트로이드에 속한 토큰만 필터링
        token_mask = np.isin(self.token_assignments, list(selected_centroids))

        # doc_indices로 추가 필터링
        if doc_indices is not None:
            doc_mask = np.isin(self.doc_token_to_doc, doc_indices)
            token_mask = token_mask & doc_mask

        candidate_token_indices = np.where(token_mask)[0]

        if len(candidate_token_indices) == 0:
            return {}

        # Step 3: 선택된 토큰만으로 MaxSim 계산
        candidate_tokens = self.doc_token_embeddings[candidate_token_indices]  # [M, dim]
        candidate_docs = self.doc_token_to_doc[candidate_token_indices]        # [M]

        # [L_q, dim] × [dim, M] → [L_q, M]
        sim_matrix = q_emb @ candidate_tokens.T

        # Step 4: 문서별 MaxSim 합산
        unique_docs = set(candidate_docs.tolist())
        doc_scores = {}

        for doc_idx in unique_docs:
            doc_token_mask = candidate_docs == doc_idx
            # 이 문서에 속한 토큰들의 유사도만 추출: [L_q, M_doc]
            doc_sims = sim_matrix[:, doc_token_mask]
            # 각 쿼리 토큰별 max → sum
            maxsim = doc_sims.max(axis=1).sum()
            doc_scores[doc_idx] = float(maxsim)

        elapsed = (time.perf_counter() - start_time) * 1000
        total_tokens = len(self.doc_token_embeddings)
        pruned_tokens = len(candidate_token_indices)
        prune_ratio = (1 - pruned_tokens / max(total_tokens, 1)) * 100

        logger.debug(
            f"PLAID MaxSim: {total_tokens} → {pruned_tokens} tokens "
            f"(가지치기 {prune_ratio:.1f}%), {elapsed:.1f}ms"
        )

        return doc_scores

    def _brute_force_maxsim(
        self,
        query_embeddings: np.ndarray,
        doc_indices: Optional[List[int]] = None,
    ) -> Dict[int, float]:
        """폴백: 브루트포스 MaxSim (PLAID 미구축 시)"""
        if self.doc_token_embeddings is None:
            return {}

        q_emb = query_embeddings.astype(np.float32)
        mask = np.ones(len(self.doc_token_embeddings), dtype=bool)

        if doc_indices is not None:
            mask = np.isin(self.doc_token_to_doc, doc_indices)

        tokens = self.doc_token_embeddings[mask]
        docs = self.doc_token_to_doc[mask]

        if len(tokens) == 0:
            return {}

        sim_matrix = q_emb @ tokens.T  # [L_q, M]

        doc_scores = {}
        for doc_idx in set(docs.tolist()):
            doc_mask = docs == doc_idx
            doc_sims = sim_matrix[:, doc_mask]
            maxsim = doc_sims.max(axis=1).sum()
            doc_scores[doc_idx] = float(maxsim)

        return doc_scores

    def save(self, path: str):
        """PLAID 인덱스 저장"""
        import pickle
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "centroids": self.centroids,
                "token_assignments": self.token_assignments,
                "doc_token_embeddings": self.doc_token_embeddings,
                "doc_token_to_doc": self.doc_token_to_doc,
                "n_centroids": self.n_centroids,
                "n_probe": self.n_probe,
            }, f)
        logger.info(f"PLAID 인덱스 저장: {path}")

    def load(self, path: str):
        """PLAID 인덱스 로드"""
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.centroids = data["centroids"]
        self.token_assignments = data["token_assignments"]
        self.doc_token_embeddings = data["doc_token_embeddings"]
        self.doc_token_to_doc = data["doc_token_to_doc"]
        self.n_centroids = data.get("n_centroids", 256)
        self.n_probe = data.get("n_probe", 32)
        self._built = True
        logger.info(
            f"PLAID 인덱스 로드: {path} "
            f"({len(self.doc_token_embeddings)} tokens, "
            f"{len(self.centroids)} centroids)"
        )


# ──────────────────────────────────────────────────────────────────────
# ColBERT v2 재순위화기 + PLAID 통합
# ──────────────────────────────────────────────────────────────────────

class ColBERTReranker:
    """ColBERT v2 Late Interaction 재순위화기 + PLAID 가속

    rerank(query, candidates[:100]) → List[ClipResult] top-20

    PLAID 통합:
      - use_plaid=True: 센트로이드 가지치기로 MaxSim 가속
      - use_plaid=False: 기존 브루트포스 MaxSim (1K에서 충분)
    """

    def __init__(
        self,
        model_name: str = "colbert-ir/colbertv2.0",
        max_doc_length: int = 180,
        use_ragatouille: bool = True,
        use_plaid: bool = True,
        plaid_n_centroids: int = 256,
        plaid_n_probe: int = 32,
    ):
        """
        Args:
            model_name: ColBERT 모델 이름
            max_doc_length: 최대 문서 토큰 길이
            use_ragatouille: RAGatouille 래퍼 사용 여부
            use_plaid: PLAID 센트로이드 가지치기 활성화 여부
            plaid_n_centroids: PLAID 센트로이드 수
            plaid_n_probe: PLAID 쿼리당 탐색 센트로이드 수
        """
        self.model_name = model_name
        self.max_doc_length = max_doc_length
        self.use_ragatouille = use_ragatouille
        self.use_plaid = use_plaid
        self.model = None
        self._loaded = False

        # PLAID 가속 엔진
        self.plaid = PLAIDEngine(
            n_centroids=plaid_n_centroids,
            n_probe=plaid_n_probe,
        )

    def load_model(self):
        """ColBERT 모델 로드"""
        if self._loaded:
            return

        if self.use_ragatouille:
            self._load_ragatouille()
        else:
            self._load_direct()

        self._loaded = True

    def _load_ragatouille(self):
        """RAGatouille 래퍼를 통한 ColBERT 로드"""
        try:
            from ragatouille import RAGPretrainedModel

            logger.info(f"RAGatouille ColBERT 로딩: {self.model_name}")
            self.model = RAGPretrainedModel.from_pretrained(self.model_name)
            self._mode = "ragatouille"
            logger.info("RAGatouille ColBERT 로드 완료")
        except ImportError as e:
            logger.warning(
                f"RAGatouille import 실패: {e}\n"
                f"  → pip install ragatouille colbert-ai 필요\n"
                f"  → 직접 MaxSim 구현으로 폴백"
            )
            self._load_direct()
        except Exception as e:
            logger.warning(f"RAGatouille 로드 실패: {e} → 폴백")
            self._load_direct()

    def _load_direct(self):
        """직접 MaxSim 구현 (폴백) — ColBERT v2 모델 우선 시도"""
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch

            # ColBERT v2 모델을 우선 시도, 실패 시 bert-base-uncased 폴백
            try:
                logger.info(f"ColBERT 직접 로드 시도: {self.model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.bert = AutoModel.from_pretrained(self.model_name)
                logger.info(f"ColBERT v2 직접 로드 성공: {self.model_name}")
            except Exception as e_colbert:
                logger.warning(
                    f"ColBERT v2 직접 로드 실패: {e_colbert} → bert-base-uncased 폴백"
                )
                self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
                self.bert = AutoModel.from_pretrained("bert-base-uncased")

            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.bert = self.bert.to(device).eval()
            self._device = device
            self._mode = "direct"
            logger.info(f"BERT 기반 MaxSim 폴백 로드 완료 (device={device})")
        except Exception as e:
            logger.warning(f"BERT 로드도 실패: {e} → 점수 기반 폴백")
            self._mode = "passthrough"

    def build_plaid_index(
        self,
        doc_embeddings_list: List[np.ndarray],
    ):
        """PLAID 인덱스 사전 구축 (Phase 0에서 호출)

        Args:
            doc_embeddings_list: 각 문서의 토큰 임베딩 리스트
        """
        if self.use_plaid:
            self.plaid.build_index(doc_embeddings_list)

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[str, float]],
        clip_captions: Dict[str, str],
        top_k: int = 20
    ) -> List[ClipResult]:
        """후보 클립을 ColBERT + PLAID로 재순위화

        Args:
            query: 자연어 쿼리
            candidates: WRRF 결과 List[(clip_id, wrrf_score)] (최대 100개)
            clip_captions: clip_id → caption 매핑
            top_k: 반환할 최종 결과 수

        Returns:
            List[ClipResult] top-K, colbert_score 기준 내림차순
        """
        self.load_model()

        if self._mode == "ragatouille":
            return self._rerank_ragatouille(
                query, candidates, clip_captions, top_k
            )
        elif self._mode == "direct":
            return self._rerank_maxsim(
                query, candidates, clip_captions, top_k
            )
        else:
            return self._rerank_passthrough(
                candidates, clip_captions, top_k
            )

    def _rerank_ragatouille(
        self,
        query: str,
        candidates: List[Tuple[str, float]],
        clip_captions: Dict[str, str],
        top_k: int
    ) -> List[ClipResult]:
        """RAGatouille를 이용한 재순위화"""
        docs = []
        doc_ids = []
        for clip_id, _ in candidates:
            caption = clip_captions.get(clip_id, "")
            if caption:
                docs.append(caption)
                doc_ids.append(clip_id)

        if not docs:
            return []

        # RAGatouille rerank
        results = self.model.rerank(query=query, documents=docs, k=top_k)

        reranked = []
        for r in results:
            idx = r.get("result_index", 0)
            if idx < len(doc_ids):
                clip_id = doc_ids[idx]
                reranked.append(ClipResult(
                    clip_id=clip_id,
                    score=float(r.get("score", 0)),
                    caption=docs[idx],
                    video_path="",
                    start_ms=0,
                    end_ms=0,
                    colbert_score=float(r.get("score", 0))
                ))

        logger.info(f"ColBERT 재순위화 완료: {len(candidates)} → {len(reranked)}")
        return reranked[:top_k]

    def _rerank_maxsim(
        self,
        query: str,
        candidates: List[Tuple[str, float]],
        clip_captions: Dict[str, str],
        top_k: int
    ) -> List[ClipResult]:
        """MaxSim 재순위화 (PLAID 가속 적용 가능)

        PLAID 활성화 시:
          1) 쿼리 토큰 임베딩 추출
          2) 각 후보 문서의 토큰 임베딩 추출
          3) PLAID 엔진에 위임 → 센트로이드 가지치기 MaxSim
        PLAID 비활성화 시:
          기존 브루트포스 MaxSim
        """
        import torch

        # 쿼리 토큰 임베딩
        q_tokens = self.tokenizer(
            query, return_tensors="pt",
            padding=True, truncation=True, max_length=32
        ).to(self._device)

        with torch.no_grad():
            q_emb = self.bert(**q_tokens).last_hidden_state  # [1, L_q, 768]

        q_emb_np = q_emb.squeeze(0).cpu().numpy()  # [L_q, 768]

        # PLAID 가속 경로: 사전 구축된 인덱스가 있으면 사용
        if self.use_plaid and self.plaid._built:
            return self._rerank_with_plaid(
                q_emb_np, candidates, clip_captions, top_k
            )

        # 기본 경로: 브루트포스 MaxSim (PLAID 미구축 시)
        scored = []
        for clip_id, wrrf_score in candidates:
            caption = clip_captions.get(clip_id, "")
            if not caption:
                scored.append((clip_id, wrrf_score * 0.5))
                continue

            # 문서 토큰 임베딩
            d_tokens = self.tokenizer(
                caption, return_tensors="pt",
                padding=True, truncation=True,
                max_length=self.max_doc_length
            ).to(self._device)

            with torch.no_grad():
                d_emb = self.bert(**d_tokens).last_hidden_state  # [1, L_d, 768]

            # MaxSim 계산
            sim_matrix = torch.matmul(
                q_emb.squeeze(0),   # [L_q, 768]
                d_emb.squeeze(0).T  # [768, L_d]
            )
            max_sim_per_token = sim_matrix.max(dim=1).values  # [L_q]
            colbert_score = float(max_sim_per_token.sum())

            scored.append((clip_id, colbert_score))

        # 정렬
        scored.sort(key=lambda x: x[1], reverse=True)

        results = []
        for clip_id, score in scored[:top_k]:
            caption = clip_captions.get(clip_id, "")
            results.append(ClipResult(
                clip_id=clip_id,
                score=score,
                caption=caption,
                video_path="",
                start_ms=0,
                end_ms=0,
                colbert_score=score
            ))

        logger.info(
            f"MaxSim 재순위화 완료: {len(candidates)} → {len(results)} "
            f"(PLAID: {'off — 브루트포스' if not self.plaid._built else 'ready'})"
        )
        return results

    def _rerank_with_plaid(
        self,
        q_emb_np: np.ndarray,
        candidates: List[Tuple[str, float]],
        clip_captions: Dict[str, str],
        top_k: int
    ) -> List[ClipResult]:
        """PLAID 가속 MaxSim 재순위화

        센트로이드 가지치기로 실제 비교 토큰 수를 줄여서 가속.
        """
        start = time.perf_counter()

        # 후보 문서 인덱스 목록
        candidate_clip_ids = [cid for cid, _ in candidates]

        # PLAID 엔진에 MaxSim 위임
        doc_scores = self.plaid.pruned_maxsim(
            q_emb_np,
            doc_indices=list(range(len(candidate_clip_ids)))
        )

        # 점수 매핑
        scored = []
        for doc_idx, (clip_id, wrrf_score) in enumerate(candidates):
            score = doc_scores.get(doc_idx, wrrf_score * 0.5)
            scored.append((clip_id, score))

        scored.sort(key=lambda x: x[1], reverse=True)

        results = []
        for clip_id, score in scored[:top_k]:
            caption = clip_captions.get(clip_id, "")
            results.append(ClipResult(
                clip_id=clip_id,
                score=score,
                caption=caption,
                video_path="",
                start_ms=0,
                end_ms=0,
                colbert_score=score
            ))

        elapsed = (time.perf_counter() - start) * 1000
        logger.info(
            f"PLAID MaxSim 재순위화 완료: {len(candidates)} → {len(results)} "
            f"({elapsed:.1f}ms, n_probe={self.plaid.n_probe})"
        )
        return results

    def _rerank_passthrough(
        self,
        candidates: List[Tuple[str, float]],
        clip_captions: Dict[str, str],
        top_k: int
    ) -> List[ClipResult]:
        """폴백: WRRF 점수 그대로 통과"""
        logger.warning("ColBERT 사용 불가 → WRRF 점수로 직접 반환")

        results = []
        for clip_id, score in candidates[:top_k]:
            caption = clip_captions.get(clip_id, "")
            results.append(ClipResult(
                clip_id=clip_id,
                score=score,
                caption=caption,
                video_path="",
                start_ms=0,
                end_ms=0,
                colbert_score=score
            ))

        return results
