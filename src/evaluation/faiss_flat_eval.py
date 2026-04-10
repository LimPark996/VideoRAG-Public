"""
faiss_flat_eval.py — MSR-VTT 1k-A 평가 전용 exact-search FAISS 인덱스

Why this file exists
--------------------
프로덕션 파이프라인(`src/phase0_indexing/vector_store.py`)은 FAISS IVFFlat
(nlist=100, nprobe=10)을 사용한다. 이건 대규모 코퍼스에서의 속도-정확도
트레이드오프를 위한 설계이고, 실서비스에는 올바르지만
**zero-shot retrieval 벤치마크 재현에는 부적절**하다:

1. IVFFlat는 approximate search라 논문의 brute-force 기준선과 직접 비교 불가
2. 1k-A 코퍼스는 1,000개뿐 → exact search가 오히려 더 빠르다 (< 10ms/query)
3. 논문(InternVideo2, Table 24a)은 당연히 exact matching으로 측정됨

따라서 eval 전용으로 `IndexFlatIP`(exact inner product, 코사인과 동치)를
새로 빌드한다. 프로덕션 인덱스는 건드리지 않는다.

Reference
---------
- InternVideo2 paper Table 24a (Supplementary): InternVideo2s2-1B #F=4
  → MSR-VTT T2V zero-shot R@1/5/10 = 51.9 / 74.6 / 81.7
- Our embedder: `OpenGVLab/InternVideo2-Stage2_1B-224p-f4` (num_frames=4)
- Expected delta from paper: ± a few % due to frame sampling strategy,
  video decoding library, and 1k-A split hash differences.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class FlatEvalStore:
    """Exact-search FAISS IndexFlatIP wrapper for benchmark evaluation.

    Drop-in API subset compatible with FAISSVectorStore used in pipeline:
      build_index(vectors, ids) -> int
      search(query_vec, k) -> List[(id, score)]
      save(path) / load(path)

    Differences from production FAISSVectorStore:
      - IndexFlatIP (exact) instead of IndexIVFFlat (approximate)
      - No training step required
      - No nlist/nprobe parameters
      - Results are deterministic and reproducible
    """

    def __init__(self, dim: int = 512):
        import faiss

        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self._id_map: List[str] = []

    # ------------------------------------------------------------------
    # build / search
    # ------------------------------------------------------------------
    def build_index(self, vectors: np.ndarray, ids: List[str]) -> int:
        """Build exact FAISS index from L2-normalized video embeddings.

        Args:
            vectors: np.ndarray [N, dim], MUST be L2-normalized already
                     (VideoEmbedder.encode_clips already does this)
            ids:     list of video_id / clip_id strings, length N

        Returns:
            number of vectors indexed
        """
        assert len(vectors) == len(ids), \
            f"vector/id count mismatch: {len(vectors)} vs {len(ids)}"
        assert vectors.shape[1] == self.dim, \
            f"dim mismatch: got {vectors.shape[1]}, expected {self.dim}"

        vectors = np.ascontiguousarray(vectors, dtype=np.float32)

        # sanity: verify L2 norm is ~1.0 (InternVideo2 get_vid_feat guarantees it,
        # but we double-check because cosine == IP only holds for unit vectors)
        norms = np.linalg.norm(vectors, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-3):
            bad = float(np.abs(norms - 1.0).max())
            logger.warning(
                f"vectors not unit-normalized (max deviation={bad:.4f}). "
                f"re-normalizing for exact cosine search."
            )
            vectors = vectors / np.maximum(norms[:, None], 1e-8)

        self.index.reset()
        self.index.add(vectors)
        self._id_map = list(ids)

        logger.info(
            f"FlatEvalStore built: {self.index.ntotal} vectors, dim={self.dim}"
        )
        return self.index.ntotal

    def search(
        self,
        query_vec: np.ndarray,
        k: int = 10,
    ) -> List[Tuple[str, float]]:
        """Exact top-k search.

        Args:
            query_vec: [1, dim] or [dim], L2-normalized
            k:         top-k to return

        Returns:
            list of (id, similarity) sorted descending
        """
        if query_vec.ndim == 1:
            query_vec = query_vec[None, :]
        query_vec = np.ascontiguousarray(query_vec, dtype=np.float32)

        k = min(k, self.index.ntotal)
        scores, idxs = self.index.search(query_vec, k)

        results: List[Tuple[str, float]] = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0:
                continue
            results.append((self._id_map[idx], float(score)))
        return results

    def full_ranking(self, query_vec: np.ndarray) -> List[Tuple[str, float]]:
        """Return ranking over the entire corpus (for MdR/MnR metrics)."""
        return self.search(query_vec, k=self.index.ntotal)

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        import faiss
        import pickle

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path))
        with open(path.with_suffix(path.suffix + ".ids"), "wb") as f:
            pickle.dump(self._id_map, f)
        logger.info(f"FlatEvalStore saved: {path}")

    def load(self, path: str | Path) -> None:
        import faiss
        import pickle

        path = Path(path)
        self.index = faiss.read_index(str(path))
        with open(path.with_suffix(path.suffix + ".ids"), "rb") as f:
            self._id_map = pickle.load(f)
        self.dim = self.index.d
        logger.info(
            f"FlatEvalStore loaded: {self.index.ntotal} vectors, dim={self.dim}"
        )

    @property
    def size(self) -> int:
        return self.index.ntotal


# ----------------------------------------------------------------------
# Convenience builder — wraps VideoEmbedder + FlatEvalStore end-to-end
# ----------------------------------------------------------------------
def build_flat_index_from_embeddings(
    video_id_to_vec: dict[str, np.ndarray],
    dim: int = 512,
) -> FlatEvalStore:
    """Build a FlatEvalStore from a dict of {video_id: embedding}.

    Typical use inside 03_evaluation.ipynb:

        from src.evaluation.faiss_flat_eval import build_flat_index_from_embeddings
        # video_vecs is produced by encoding 1k-A corpus with VideoEmbedder
        eval_store = build_flat_index_from_embeddings(video_vecs)
        scores = eval_store.full_ranking(query_embedding)
    """
    ids = list(video_id_to_vec.keys())
    vecs = np.stack([video_id_to_vec[i] for i in ids], axis=0)
    if vecs.ndim == 3 and vecs.shape[1] == 1:
        vecs = vecs.squeeze(1)  # [N, 1, D] -> [N, D]

    store = FlatEvalStore(dim=dim)
    store.build_index(vecs, ids)
    return store


# ----------------------------------------------------------------------
# Parity check — verify current setup matches InternVideo2 paper spec
# ----------------------------------------------------------------------
def parity_check(embedder, dense_retriever=None) -> dict:
    """Self-diagnosis: confirm setup matches paper before running eval.

    Returns a dict of check results. Prints a human-readable summary.
    Should be called once at the top of 03_evaluation.ipynb after pipeline init.
    """
    report = {}

    print("=" * 60)
    print("InternVideo2 Parity Check (vs. paper Table 24a)")
    print("=" * 60)

    # 1. model name
    model_name = getattr(embedder, "model_name", "unknown")
    report["model_name"] = model_name
    expected_model = "OpenGVLab/InternVideo2-Stage2_1B-224p-f4"
    ok1 = model_name == expected_model
    print(f"[{'OK' if ok1 else '!!'}] model_name: {model_name}")
    print(f"       expected:    {expected_model}")

    # 2. num_frames
    num_frames = getattr(embedder, "num_frames", None)
    report["num_frames"] = num_frames
    ok2 = num_frames == 4
    print(f"[{'OK' if ok2 else '!!'}] num_frames: {num_frames} (paper #F=4 row uses 4)")

    # 3. embed dim via a test encode
    try:
        test_emb = embedder.encode_query("a person riding a bicycle")
        report["embed_shape"] = list(test_emb.shape)
        report["embed_dim"] = int(test_emb.shape[-1])
        ok3 = test_emb.shape[-1] == 512
        print(f"[{'OK' if ok3 else '!!'}] embed_dim: {test_emb.shape} "
              f"(expected [..., 512] — joint space dim)")

        # 4. L2 norm
        norm = float(np.linalg.norm(test_emb))
        report["l2_norm"] = norm
        ok4 = abs(norm - 1.0) < 1e-3
        print(f"[{'OK' if ok4 else '!!'}] L2 norm:   {norm:.6f} (expected ~1.0)")
    except Exception as e:
        print(f"[!!] embed test failed: {e}")
        ok3 = ok4 = False
        report["embed_error"] = str(e)

    # 5. FAISS index type (if dense_retriever passed)
    if dense_retriever is not None:
        try:
            # pipeline's DenseRetriever wraps FAISSVectorStore
            store = getattr(dense_retriever, "vector_store", None) or \
                    getattr(dense_retriever, "store", None) or dense_retriever
            idx = getattr(store, "index", None)
            idx_type = type(idx).__name__ if idx is not None else "unknown"
            report["faiss_index_type"] = idx_type
            is_exact = "Flat" in idx_type and "IVF" not in idx_type
            print(f"[{'OK' if is_exact else 'WARN'}] FAISS index: {idx_type}")
            if not is_exact:
                print(f"       ⚠️  Production uses IVFFlat (approximate). "
                      f"For eval, rebuild with FlatEvalStore.")
        except Exception as e:
            print(f"[??] FAISS inspection failed: {e}")

    # 6. summary vs paper
    print("-" * 60)
    print("Paper baseline (InternVideo2s2-1B, #F=4, MSR-VTT 1k-A T2V):")
    print("  R@1 = 51.9    R@5 = 74.6    R@10 = 81.7")
    print("Our Dense-only row should land within ±2-3% of these.")
    print("Larger gap → investigate frame sampling / video decoder / split hash.")
    print("=" * 60)

    return report
