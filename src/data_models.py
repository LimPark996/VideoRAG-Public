"""
⓪ data_models.py — 공유 데이터 모델
모든 Phase에서 사용하는 공통 데이터 클래스 정의

코딩 순서: 가장 먼저 작성 (다른 모든 모듈이 의존)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Set, Tuple
from enum import Enum
import time


# ─────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────

class TransitionType(Enum):
    """트랜지션 유형 (DINOv2 유사도 기반 결정)

    출처: 자체 설계 (DINOv2 visual similarity threshold 기반)
    - sim >= 0.6  → Cut (직접 전환, 시각적으로 유사한 장면)
    - 0.3 ~ 0.6   → Crossfade (부드러운 전환)
    - sim < 0.3   → Morph (시각 불연속 해결)
    """
    CUT = "cut"
    CROSSFADE = "crossfade"
    MORPH = "morph"


# ─────────────────────────────────────────────
# Core Data Classes
# ─────────────────────────────────────────────

@dataclass
class ClipMeta:
    """영상 클립 메타데이터 (Phase 0 인덱싱 시 생성, 1 clip = 1 scene)"""
    clip_id: str                    # 고유 클립 ID (e.g., "video0001_scene003")
    video_id: str                   # 원본 영상 ID
    caption: str                    # 영어 캡션 (BM25 + Dense 인덱싱용)
    start_ms: float                 # 클립 시작 시간 (ms)
    end_ms: float                   # 클립 종료 시간 (ms)
    video_path: str                 # 원본 영상 파일 경로
    keyframe_path: Optional[str] = None  # 대표 키프레임 이미지 경로
    duration_ms: Optional[float] = None  # 클립 길이 (ms)
    scene_id: Optional[int] = None       # 소속 Scene ID (Agglomerative Clustering 결과)
    ko_caption: Optional[str] = None       # 한국어 캡션 (UI 표시용)

    def __post_init__(self):
        if self.duration_ms is None:
            self.duration_ms = self.end_ms - self.start_ms


@dataclass
class ClipResult:
    """검색/리랭킹 결과 클립 (Phase 1-3에서 사용)"""
    clip_id: str
    score: float                    # 최종 점수 (WRRF or ColBERT score)
    caption: str
    video_path: str
    start_ms: float
    end_ms: float
    keyframe_path: Optional[str] = None
    scene_id: Optional[int] = None       # 소속 Scene ID (Agglomerative Clustering 결과)

    # 각 단계별 점수 (디버깅 및 시연용)
    bm25_score: Optional[float] = None
    dense_score: Optional[float] = None
    wrrf_score: Optional[float] = None
    colbert_score: Optional[float] = None


@dataclass
class TransitionDecision:
    """트랜지션 결정 결과 (Phase 4)"""
    transition_type: TransitionType
    similarity_score: float         # DINOv2 cosine similarity
    duration_sec: float             # 트랜지션 지속 시간
    from_clip_id: str
    to_clip_id: str

    @staticmethod
    def from_similarity(sim: float, from_id: str, to_id: str) -> 'TransitionDecision':
        """유사도 점수로부터 트랜지션 타입 자동 결정

        Threshold 설계 근거:
        - >= 0.6: 시각적으로 충분히 유사 → Cut (0s)
        - 0.3~0.6: 중간 유사도 → Crossfade (0.5s)
        - < 0.3: 시각적으로 많이 다름 → Morph (1.0s)
        """
        if sim >= 0.6:
            return TransitionDecision(
                transition_type=TransitionType.CUT,
                similarity_score=sim,
                duration_sec=0.0,
                from_clip_id=from_id,
                to_clip_id=to_id
            )
        elif sim >= 0.3:
            return TransitionDecision(
                transition_type=TransitionType.CROSSFADE,
                similarity_score=sim,
                duration_sec=0.5,
                from_clip_id=from_id,
                to_clip_id=to_id
            )
        else:
            return TransitionDecision(
                transition_type=TransitionType.MORPH,
                similarity_score=sim,
                duration_sec=1.0,
                from_clip_id=from_id,
                to_clip_id=to_id
            )


@dataclass
class C2PAManifest:
    """C2PA 출처 추적 메타데이터 (Phase 5)

    출처: C2PA (Coalition for Content Provenance and Authenticity) 표준
    - https://c2pa.org/specifications/
    - ES256 (ECDSA with P-256 and SHA-256) 전자서명
    """
    query: str                      # 원본 사용자 쿼리
    timestamp: str                  # ISO 8601 생성 시간
    source_clips: List[Dict[str, Any]]  # 사용된 클립 목록
    signature: Optional[str] = None      # ES256 서명 (hex)
    manifest_version: str = "1.0"
    generator: str = "VideoRAG-Prototype-v0.1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "c2pa:manifest": {
                "dc:title": f"VideoRAG Result: {self.query}",
                "dc:format": "video/mp4",
                "c2pa:generator": self.generator,
                "c2pa:signature_info": {
                    "alg": "ES256",
                    "signature": self.signature or "unsigned"
                },
                "stds:version": self.manifest_version,
                "xmp:CreateDate": self.timestamp,
                "c2pa:actions": [
                    {
                        "action": "c2pa.searched",
                        "parameters": {"query": self.query}
                    },
                    {
                        "action": "c2pa.assembled",
                        "parameters": {
                            "clip_count": len(self.source_clips),
                            "clips": self.source_clips
                        }
                    }
                ]
            }
        }


@dataclass
class CurationState:
    """PD 큐레이션 상태 관리

    PD가 search_only() 결과에서 클립을 선택/정렬/제외하는 과정을 추적.
    assemble_curated()에 전달하여 최종 영상 합성에 사용.
    """
    search_results: List[ClipResult]          # 원본 검색 결과
    selected_clip_ids: List[str] = field(default_factory=list)  # PD가 선택한 클립 ID
    ordering: List[str] = field(default_factory=list)           # PD가 지정한 순서
    excluded_ids: Set[str] = field(default_factory=set)         # 제외된 클립 ID

    def get_curated_clips(self) -> List[ClipResult]:
        """큐레이션된 클립 목록 반환 (PD 순서 기준)"""
        id_to_clip = {c.clip_id: c for c in self.search_results}
        return [
            id_to_clip[cid] for cid in self.ordering
            if cid in id_to_clip and cid not in self.excluded_ids
        ]


@dataclass
class IndexBuildResult:
    """인덱싱 결과 (Phase 0)"""
    n_clips: int
    n_vectors: int
    faiss_index_path: str
    bm25_index_path: str
    metadata_path: str
    build_time_sec: float


@dataclass
class VideoRAGResult:
    """최종 파이프라인 결과"""
    video_path: str                 # 합성된 결과 영상 경로
    clips: List[ClipResult]         # 사용된 Top-K 클립 목록
    c2pa_manifest: C2PAManifest     # C2PA 출처 추적 정보
    transitions: List[TransitionDecision]  # 트랜지션 정보
    latency_ms: float               # 전체 E2E 레이턴시

    # 단계별 레이턴시 (시연용 타임라인 차트)
    phase_latencies: Dict[str, float] = field(default_factory=dict)

    # TC-Score (시간 일관성 지표, Phase 4 품질 측정)
    tc_score: float = 0.0

    # v2 확장: 품질 요약 (시연용 대시보드)
    quality_summary: Optional[Dict] = None

    def export_timeline(self, format: str = 'premiere') -> str:
        """NLE 타임라인 내보내기"""
        from src.output.timeline_exporter import TimelineExporter
        exporter = TimelineExporter()
        if format == 'premiere':
            return exporter.export_premiere_xml(self.clips, self.transitions)
        elif format == 'davinci':
            return exporter.export_davinci_edl(self.clips, self.transitions)
        elif format == 'fcpxml':
            return exporter.export_fcpxml(self.clips, self.transitions)
        else:
            raise ValueError(f"지원하지 않는 포맷: {format}")

    def export_clips_zip(self) -> str:
        """개별 클립 + 메타데이터 ZIP 내보내기"""
        from src.output.timeline_exporter import TimelineExporter
        return TimelineExporter().export_clips_zip(self.clips)


@dataclass
class LatencyTracker:
    """단계별 레이턴시 측정 유틸리티"""
    _start_times: Dict[str, float] = field(default_factory=dict)
    _latencies: Dict[str, float] = field(default_factory=dict)

    def start(self, phase: str):
        self._start_times[phase] = time.perf_counter()

    def stop(self, phase: str) -> float:
        elapsed = (time.perf_counter() - self._start_times[phase]) * 1000  # ms
        self._latencies[phase] = elapsed
        return elapsed

    @property
    def total_ms(self) -> float:
        return sum(self._latencies.values())

    @property
    def all_latencies(self) -> Dict[str, float]:
        return dict(self._latencies)
