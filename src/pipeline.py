"""
⑨ pipeline.py — End-to-End 파이프라인 오케스트레이터

역할: Phase 0~5 전체를 통합 실행하는 최상위 컨트롤러
     사용자 쿼리 → 검색 → 리랭킹 → 영상 합성 → C2PA 서명

설계: 자체 설계 (파이프라인 오케스트레이션 패턴)
"""

import os
import time
import logging
from typing import Optional, Dict, List, Set

from .data_models import (
    ClipResult, VideoRAGResult, LatencyTracker, ClipMeta, CurationState
)
from .input import QueryPreprocessor, PreprocessedQuery
from .phase0_indexing import VideoIndexer, VideoEmbedder, FAISSVectorStore
from .phase12_search import BM25Retriever, DenseRetriever, HybridFusion
from .phase3_reranking import ColBERTReranker, ITMScorer
from .phase4_assembly import VideoAssembler, VisualScorer, TCScorer
from .phase5_c2pa import C2PATagger

logger = logging.getLogger(__name__)


class VideoRAGPipeline:
    """VideoRAG End-to-End 파이프라인 (v2)

    v2 변경사항:
    - search_only(query) → CurationState (PD 큐레이션용 검색 전용)
    - assemble_curated(curation_state) → VideoRAGResult (PD 선별 후 합성)
    - search(query) → VideoRAGResult (기존 호환: 검색+합성 일체형)
    - QueryPreprocessor 통합: 한국어 쿼리 → Papago → en_query 자동 번역
    """

    def __init__(
        self,
        index_dir: str = "index",
        output_dir: str = "output",
        model_dir: str = "models",
        config: Optional[Dict] = None
    ):
        """
        Args:
            index_dir: 인덱스 저장 디렉토리
            output_dir: 출력 디렉토리
            model_dir: 모델 가중치 디렉토리
            config: 추가 설정 (가중치, 임계값, Papago 키 등)
                - papago_client_id, papago_client_secret: Papago API 키
                - w_visual, w_text, rrf_k: WRRF 가중치
                - embed_dim: 임베딩 차원 (기본 512)
        """
        self.index_dir = index_dir
        self.output_dir = output_dir
        self.model_dir = model_dir
        self.config = config or {}

        # v2: QueryPreprocessor (Papago 한→영 번역)
        self.query_preprocessor = QueryPreprocessor(
            papago_client_id=self.config.get("papago_client_id"),
            papago_client_secret=self.config.get("papago_client_secret"),
        )

        # 컴포넌트 초기화
        self.embedder = VideoEmbedder()
        self.vector_store = FAISSVectorStore(
            dim=self.config.get("embed_dim", 512)   # InternVideo2-1B → 512차원
        )
        self.bm25 = BM25Retriever()
        self.dense_retriever = DenseRetriever(self.embedder, self.vector_store)
        self.hybrid_fusion = HybridFusion(
            w_visual=self.config.get("w_visual", 0.6),
            w_text=self.config.get("w_text", 0.4),
            k=self.config.get("rrf_k", 60)
        )
        self.reranker = ColBERTReranker()
        # ITMScorer: load_index() 시 itm_vision_features.pt 존재 여부 확인 후 활성화
        # itm_features.pt가 없으면 None → Phase 3b 건너뜀 (ColBERT만 적용)
        self.itm_scorer: Optional[ITMScorer] = None
        self.assembler = VideoAssembler(output_dir=output_dir)
        self.tc_scorer = TCScorer()
        self.c2pa_tagger = C2PATagger()

        # 클립 메타데이터
        self.clip_metadata: Dict[str, ClipMeta] = {}
        self._loaded = False

        # 실시간 진행 콜백: (phase_name, "start"|"done", elapsed_ms) → None
        self._progress_callback = None

    def load_index(self):
        """사전 구축된 인덱스 로드 (발표 당일 사용)"""
        import pickle

        if self._loaded:
            return

        logger.info("인덱스 로드 시작...")

        # FAISS 인덱스
        faiss_path = os.path.join(self.index_dir, "faiss_ivfflat.index")
        if os.path.exists(faiss_path):
            self.vector_store.load(faiss_path)

        # BM25 인덱스
        bm25_path = os.path.join(self.index_dir, "bm25_index.pkl")
        if os.path.exists(bm25_path):
            self.bm25.load(bm25_path)

        # 클립 메타데이터
        meta_path = os.path.join(self.index_dir, "clip_metadata.pkl")
        if os.path.exists(meta_path):
            with open(meta_path, "rb") as f:
                self.clip_metadata = pickle.load(f)

            # video_path 경로 검증 및 재매핑
            # 인덱싱 시점의 절대경로가 현재 런타임에서 유효하지 않을 수 있음
            # → 저장된 경로가 없으면 index_dir 기준으로 video_dir를 추정해서 fallback
            _video_dir_candidate = os.path.join(
                os.path.dirname(self.index_dir),
                "data", "msrvtt", "videos"
            )
            _broken = 0
            for meta in self.clip_metadata.values():
                if not os.path.exists(meta.video_path):
                    fallback = os.path.join(_video_dir_candidate, f"{meta.clip_id}.mp4")
                    if os.path.exists(fallback):
                        meta.video_path = fallback
                        _broken += 1
            if _broken:
                logger.info(f"video_path 재매핑: {_broken}개 클립 경로 복구 ({_video_dir_candidate})")

        # ITMScorer: itm_vision_features.pt 있을 때만 활성화
        itm_path = os.path.join(self.index_dir, "itm_vision_features.pt")
        if os.path.exists(itm_path):
            self.embedder.load_model()  # iv_model 확보
            self.itm_scorer = ITMScorer(
                iv_model=self.embedder.model,
                itm_features_path=itm_path,
                bs_itm=self.config.get("bs_itm", 32),
            )
            logger.info(f"ITMScorer 활성화: {itm_path}")
        else:
            self.itm_scorer = None
            logger.info(
                f"ITMScorer 비활성화: {itm_path} 없음 "
                f"(indexer.build_itm_vision_features() 실행 후 재시도)"
            )

        self._loaded = True
        logger.info(
            f"인덱스 로드 완료: {self.vector_store.ntotal} vectors, "
            f"{len(self.clip_metadata)} clips  "
            f"ITM={'활성' if self.itm_scorer else '비활성'}"
        )

    def _run_search_pipeline(
        self,
        query: str,
        top_k: int = 20,
    ) -> tuple:
        """검색 파이프라인 공통 로직 (Phase 0~3)

        Returns:
            (en_query, reranked_clips, tracker)
        """
        self.load_index()
        tracker = LatencyTracker()
        cb = self._progress_callback

        def _phase(name):
            """단계 시작 알림 + 타이밍 시작"""
            if cb:
                cb(name, "start", 0)
            tracker.start(name)

        def _done(name):
            """단계 완료 알림 + 타이밍 종료"""
            ms = tracker.stop(name)
            if cb:
                cb(name, "done", ms)
            return ms

        # ─────────────────────────────────────────
        # Phase 0: 쿼리 전처리 (한→영 번역)
        # ─────────────────────────────────────────
        _phase("phase0_preprocess")
        pq: PreprocessedQuery = self.query_preprocessor.preprocess(query)
        en_query = pq.en_query
        self._last_en_query = en_query  # 진단용: Gradio UI에서 접근
        self._last_raw_query = query    # 진단용
        _done("phase0_preprocess")
        logger.info(
            f"[Phase 0] QueryPreprocess: '{query}' → en_query='{en_query}' "
            f"(lang={pq.lang}, {tracker.all_latencies['phase0_preprocess']:.1f}ms)"
        )

        logger.info(f"={'='*60}")
        logger.info(f"VideoRAG 검색 시작: query='{query}' → en_query='{en_query}'")
        logger.info(f"={'='*60}")

        # ─────────────────────────────────────────
        # Phase 1: BM25 Sparse Search (~5ms)
        # ─────────────────────────────────────────
        _phase("phase1_bm25")
        sparse_results = self.bm25.search(en_query, top_n=100)
        _done("phase1_bm25")
        logger.info(
            f"[Phase 1] BM25: {len(sparse_results)} results "
            f"({tracker.all_latencies['phase1_bm25']:.1f}ms)"
        )

        # ─────────────────────────────────────────
        # Phase 1: Dense Retrieval — 임베딩 + FAISS 분리 측정
        # ─────────────────────────────────────────
        _phase("phase1_embed")
        query_vec = self.embedder.encode_query(en_query)
        _done("phase1_embed")

        _phase("phase1_faiss")
        dense_results = self.dense_retriever.search(query_vec, top_k=100)
        _done("phase1_faiss")

        embed_ms = tracker.all_latencies['phase1_embed']
        faiss_ms = tracker.all_latencies['phase1_faiss']
        logger.info(
            f"[Phase 1] Dense: {len(dense_results)} results "
            f"(embed={embed_ms:.1f}ms, faiss={faiss_ms:.1f}ms, total={embed_ms+faiss_ms:.1f}ms)"
        )

        # ─────────────────────────────────────────
        # Phase 2: WRRF Fusion (~5ms)
        # ─────────────────────────────────────────
        _phase("phase2_fusion")
        fused_results = self.hybrid_fusion.fuse(sparse_results, dense_results)
        _done("phase2_fusion")
        logger.info(
            f"[Phase 2] WRRF: {len(fused_results)} fused candidates "
            f"({tracker.all_latencies['phase2_fusion']:.1f}ms)"
        )

        # ─────────────────────────────────────────
        # Phase 3a: ColBERT Reranking (~350ms)
        # ITM이 활성화된 경우 pool을 더 크게 유지 (top_k * 2.5 정도)
        # ─────────────────────────────────────────
        _phase("phase3_reranking")
        _rerank_pool = 100
        # ITM 활성 시 ColBERT에서 더 많이 뽑아 ITM이 선별할 여지 확보
        _colbert_top_k = min(int(top_k * 2.5), _rerank_pool) if self.itm_scorer else top_k
        clip_captions = {
            cid: self.clip_metadata[cid].caption
            for cid, _ in fused_results[:_rerank_pool]
            if cid in self.clip_metadata
        }
        reranked = self.reranker.rerank(
            en_query, fused_results[:_rerank_pool], clip_captions, top_k=_colbert_top_k
        )

        # ClipResult에 메타데이터 보강
        for clip in reranked:
            if clip.clip_id in self.clip_metadata:
                meta = self.clip_metadata[clip.clip_id]
                clip.video_path = meta.video_path
                clip.start_ms = meta.start_ms
                clip.end_ms = meta.end_ms
                clip.keyframe_path = meta.keyframe_path
                clip.scene_id = meta.scene_id

        _done("phase3_reranking")
        logger.info(
            f"[Phase 3a] ColBERT: {len(reranked)} reranked clips "
            f"({tracker.all_latencies['phase3_reranking']:.1f}ms)"
        )

        # ─────────────────────────────────────────
        # Phase 3b: ITM Reranking (선택적 — itm_vision_features.pt 있을 때만)
        # ColBERT top-(top_k*2.5) → ITM으로 최종 top_k 선별
        # ─────────────────────────────────────────
        if self.itm_scorer:
            _phase("phase3b_itm")
            reranked = self.itm_scorer.rerank(en_query, reranked, top_k=top_k)
            _done("phase3b_itm")
            logger.info(
                f"[Phase 3b] ITM: {len(reranked)} final clips "
                f"({tracker.all_latencies['phase3b_itm']:.1f}ms)"
            )
        else:
            # ITM 없으면 ColBERT 결과를 top_k로 자름
            reranked = reranked[:top_k]

        self._last_tracker = tracker  # Gradio UI에서 단계별 레이턴시 접근용

        # 진단용: Gradio UI에서 검색 결과 세부 확인
        self._last_diag = {
            "raw_query": query,
            "en_query": en_query,
            "lang": pq.lang,
            "bm25_count": len(sparse_results),
            "bm25_top3": [(cid, f"{s:.4f}") for cid, s in sparse_results[:3]],
            "dense_top3": [(cid, f"{s:.4f}") for cid, s in dense_results[:3]],
            "fused_top3": [(cid, f"{s:.4f}") for cid, s in fused_results[:3]],
            "reranked_top3": [(c.clip_id, f"{c.score:.4f}") for c in reranked[:3]],
            "itm_active": self.itm_scorer is not None,
        }

        return en_query, reranked, tracker

    # ──────────────────────────────────────────────────────────────────
    # v2 API: search_only() + assemble_curated() (PD 큐레이션 워크플로)
    # ──────────────────────────────────────────────────────────────────

    def search_only(
        self,
        query: str,
        top_k: int = 20,
    ) -> CurationState:
        """검색만 수행 → PD 큐레이션용 CurationState 반환

        PD 워크플로:
          1) search_only(query) → CurationState
          2) PD가 CurationState에서 클립 선별/순서 조정
          3) assemble_curated(curation_state) → VideoRAGResult

        Args:
            query: 자연어 쿼리 (한국어/영어 모두 가능)
            top_k: 최종 반환 클립 수

        Returns:
            CurationState (PD가 선별/순서 조정할 수 있는 상태)
        """
        en_query, reranked, tracker = self._run_search_pipeline(query, top_k)

        logger.info(
            f"search_only 완료: {len(reranked)} clips, "
            f"총 {tracker.total_ms:.1f}ms"
        )

        return CurationState(
            search_results=reranked,
            selected_clip_ids=[c.clip_id for c in reranked],
            ordering=[c.clip_id for c in reranked],
        )

    def assemble_curated(
        self,
        curation_state: CurationState,
        query: str = "",
        sort_mode: str = "manual",
    ) -> VideoRAGResult:
        """PD가 큐레이션한 클립으로 영상 합성

        Args:
            curation_state: PD가 선별/순서 조정한 CurationState
            query: 원본 쿼리 (C2PA 매니페스트에 사용)
            sort_mode: 정렬 모드 ("manual": PD 순서 유지, "auto": perception sort, "hybrid": 혼합)

        Returns:
            VideoRAGResult
        """
        self.load_index()
        tracker = LatencyTracker()

        curated_clips = curation_state.get_curated_clips()

        if not curated_clips:
            logger.warning("assemble_curated: 큐레이션된 클립이 없습니다.")
            return VideoRAGResult(
                video_path="", clips=[], c2pa_manifest={},
                transitions=[], latency_ms=0, phase_latencies={}, tc_score=0.0
            )

        # ── Phase 4: Video Assembly ──────────────────────────────────
        video_path = ""
        transitions = []
        tc_result = {'tc_score': 0.0}

        tracker.start("phase4_assembly")
        try:
            video_path, transitions, assembly_sub_latencies = self.assembler.assemble(
                curated_clips, query, max_clips=len(curated_clips),
                sort_mode=sort_mode
            )
            for sub_key, sub_ms in assembly_sub_latencies.items():
                tracker.all_latencies[sub_key] = sub_ms
        except Exception as e:
            logger.error(f"영상 합성 실패: {e}")
        tracker.stop("phase4_assembly")

        # TC-Score
        tracker.start("phase4_tc_score")
        try:
            tc_result = self.tc_scorer.compute_tc_score(
                curated_clips[:10],
                self.assembler.visual_scorer,
                get_keyframe_fn=self.assembler._get_keyframe
            )
        except Exception as e:
            logger.warning(f"TC-Score 측정 실패: {e}")
        tracker.stop("phase4_tc_score")

        # ── Phase 5: C2PA ────────────────────────────────────────────
        tracker.start("phase5_c2pa")
        manifest = self.c2pa_tagger.build_manifest(query, curated_clips)
        self.c2pa_tagger.sign_manifest(manifest)
        if video_path and os.path.exists(video_path):
            self.c2pa_tagger.embed_c2pa(video_path, manifest)
        tracker.stop("phase5_c2pa")

        result = VideoRAGResult(
            video_path=video_path,
            clips=curated_clips,
            c2pa_manifest=manifest,
            transitions=transitions,
            latency_ms=tracker.total_ms,
            phase_latencies=tracker.all_latencies,
            tc_score=tc_result.get('tc_score', 0.0)
        )

        # v2: 품질 요약 생성 (시연용 대시보드)
        result.quality_summary = self._build_quality_summary(
            result, tc_result, transitions
        )

        logger.info(
            f"assemble_curated 완료: {len(curated_clips)} clips → {video_path}"
        )
        return result

    # ──────────────────────────────────────────────────────────────────
    # 레거시 호환: search() (검색+합성 일체형)
    # ──────────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 20,
        assemble_video: bool = True,
    ) -> VideoRAGResult:
        """End-to-End 파이프라인 실행

        Args:
            query: 자연어 쿼리 (한국어/영어 모두 가능 — 한국어는 Papago로 자동 번역)
            top_k: 최종 반환 클립 수
            assemble_video: 영상 합성 여부 (False면 검색만)

        Returns:
            VideoRAGResult
        """
        en_query, reranked, tracker = self._run_search_pipeline(query, top_k)

        # ─────────────────────────────────────────
        # Phase 4: Video Assembly (~400ms)
        # ─────────────────────────────────────────
        video_path = ""
        transitions = []
        tc_result = {'tc_score': 0.0}

        if assemble_video and reranked:
            tracker.start("phase4_assembly")
            try:
                video_path, transitions, assembly_sub_latencies = self.assembler.assemble(
                    reranked, en_query, max_clips=10
                )
                for sub_key, sub_ms in assembly_sub_latencies.items():
                    tracker.all_latencies[sub_key] = sub_ms
            except Exception as e:
                logger.error(f"영상 합성 실패: {e}")
                video_path = ""
            tracker.stop("phase4_assembly")
            logger.info(
                f"[Phase 4] Assembly: {video_path} "
                f"({tracker.all_latencies.get('phase4_assembly', 0):.1f}ms)"
            )

            # TC-Score 측정
            tracker.start("phase4_tc_score")
            try:
                tc_result = self.tc_scorer.compute_tc_score(
                    reranked[:10],
                    self.assembler.visual_scorer,
                    get_keyframe_fn=self.assembler._get_keyframe
                )
                logger.info(
                    f"[Phase 4] TC-Score: {tc_result['tc_score']:.4f} "
                    f"({self.tc_scorer.interpret_score(tc_result['tc_score'])})"
                )
            except Exception as e:
                logger.warning(f"TC-Score 측정 실패: {e}")
                tc_result = {'tc_score': 0.0}
            tracker.stop("phase4_tc_score")

        # ─────────────────────────────────────────
        # Phase 5: C2PA Tagging (~50ms)
        # ─────────────────────────────────────────
        tracker.start("phase5_c2pa")
        manifest = self.c2pa_tagger.build_manifest(en_query, reranked)
        self.c2pa_tagger.sign_manifest(manifest)

        if video_path and os.path.exists(video_path):
            c2pa_path = self.c2pa_tagger.embed_c2pa(video_path, manifest)

        tracker.stop("phase5_c2pa")
        logger.info(
            f"[Phase 5] C2PA: signed "
            f"({tracker.all_latencies['phase5_c2pa']:.1f}ms)"
        )

        # ─────────────────────────────────────────
        # 최종 결과 구성
        # ─────────────────────────────────────────
        result = VideoRAGResult(
            video_path=video_path,
            clips=reranked,
            c2pa_manifest=manifest,
            transitions=transitions,
            latency_ms=tracker.total_ms,
            phase_latencies=tracker.all_latencies,
            tc_score=tc_result.get('tc_score', 0.0)
        )

        # v2: 품질 요약 생성
        result.quality_summary = self._build_quality_summary(
            result, tc_result, transitions
        )

        logger.info(f"{'='*60}")
        logger.info(f"총 레이턴시: {result.latency_ms:.1f}ms")
        for phase, lat in result.phase_latencies.items():
            logger.info(f"  {phase}: {lat:.1f}ms")
        logger.info(f"{'='*60}")

        return result

    # ──────────────────────────────────────────────────────────────────
    # 유틸리티
    # ──────────────────────────────────────────────────────────────────

    def _build_quality_summary(
        self, result: VideoRAGResult, tc_result: Dict, transitions: List
    ) -> Dict:
        """품질 요약 딕셔너리 생성 (시연용 대시보드)

        Args:
            result: VideoRAGResult
            tc_result: TC-Score 결과 dict
            transitions: 트랜지션 목록

        Returns:
            품질 요약 딕셔너리
        """
        clips = result.clips
        tc_score = tc_result.get("tc_score", 0.0)

        # TC-Score 해석
        if tc_score >= 0.7:
            tc_interpretation = "우수 — 시간적 일관성이 높아 자연스러운 영상 흐름"
        elif tc_score >= 0.5:
            tc_interpretation = "양호 — 대체로 일관되나 일부 전환에서 어색함 가능"
        elif tc_score >= 0.3:
            tc_interpretation = "보통 — 시각적 흐름 개선이 필요할 수 있음"
        else:
            tc_interpretation = "주의 — 클립 간 시간적 일관성이 낮아 검토 필요"

        # 트랜지션 구성
        transition_counts = {}
        for t in transitions:
            ttype = t.transition_type.value if hasattr(t.transition_type, "value") else str(t.transition_type)
            transition_counts[ttype] = transition_counts.get(ttype, 0) + 1

        # 클립별 점수
        clip_scores = [
            {"clip_id": c.clip_id, "score": c.score, "caption": c.caption}
            for c in clips
        ]

        return {
            "tc_score": tc_score,
            "tc_interpretation": tc_interpretation,
            "avg_search_score": (
                sum(c.score for c in clips) / len(clips) if clips else 0
            ),
            "clip_count": len(clips),
            "transition_counts": transition_counts,
            "clip_scores": clip_scores,
            "latency_ms": result.latency_ms,
        }
