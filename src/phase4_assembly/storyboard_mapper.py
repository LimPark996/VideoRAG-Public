"""
storyboard_mapper.py — Scene Graph 파서 + Storyboard-to-Clip Mapper

PDF 설계:
  "Scene Graph(JSON) 입력 → Context-Aware Greedy Selection 알고리즘으로
   최적 클립 자동 선택 → 타임라인 배치"

이 모듈이 하는 일:
  1. Scene Graph JSON 파싱: 방송사 PD가 보내는 스토리보드 구조를 파싱
  2. 장면별 검색 쿼리 추출: 각 장면의 텍스트 설명 + 속성 요구사항 추출
  3. 장면별 클립 선택: search_fn 콜백을 통해 아카이브 검색 → 최적 클립 선택
  4. 3경로 분기 판정: 검색 결과 품질에 따라 그대로 사용 / 속성 변환 / 신규 생성
  5. PD 인터랙티브 워크플로: TRANSFORM/GENERATE 시 PD가 프롬프트 수정 + 백엔드 선택 + 결과 확인
  6. 타임라인 조립: Scene Graph 순서대로 최종 클립 리스트 반환

동작 흐름:

  A) 자동 모드 (비인터랙티브):
    map_scene_graph(scene_graph) → StoryboardMapping (기존 호환)

  B) PD 인터랙티브 모드 (Gradio UI 연동):
    prepare_scene(req)    → PDReviewRequest  (Step 1: 분석 + 프롬프트 생성)
      ↓ PD가 프롬프트 수정 + 백엔드 선택
    execute_scene(req, prompt, backend) → PDExecutionResult (Step 3: API 실행)
      ↓ PD가 결과 확인
    accept_scene(result)  → MappedScene (승인)
    또는 → 재시도 (execute_scene 다시 호출, 최대 3회)
    또는 → accept_upload(file_path) (직접 업로드)

Scene Graph JSON 스키마:
  {
    "title": "기후변화와 한국 어촌",
    "scenes": [
      {
        "scene_id": 1,
        "description": "새벽녘 항구에 정박된 어선들",
        "duration_sec": 5.0,
        "attributes": {
          "time_of_day": "dawn",
          "season": "winter",
          "mood": "cool",
          "location": "outdoor"
        }
      },
      ...
    ]
  }
"""

import logging
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

from ..data_models import ClipResult
from .inverse_prompt_engine import (
    InversePromptEngine, SceneAttributes, SceneLocation,
    TimeOfDay, Season, Mood
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Scene Graph 데이터 모델
# ─────────────────────────────────────────────

class BranchDecision(Enum):
    """3경로 분기 판정 결과

    PDF 설계: 유사도 기반 3경로 분기
    - USE_AS_IS: 검색 결과가 요구에 충분히 부합 → 그대로 사용
    - TRANSFORM: 콘텐츠는 맞지만 속성(시간대/계절/분위기)이 다름 → 역방향 프롬프트로 변환
    - GENERATE:  적합한 클립이 아카이브에 없음 → Runway로 신규 생성
    """
    USE_AS_IS = "use_as_is"
    TRANSFORM = "transform"
    GENERATE = "generate"


@dataclass
class SceneRequirement:
    """Scene Graph에서 파싱된 개별 장면 요구사항"""
    scene_id: int
    description: str                        # 장면 텍스트 설명 (검색 쿼리로 사용)
    duration_sec: float = 5.0               # 요청된 장면 길이
    # 속성 요구사항 (None이면 해당 속성 변환 불필요)
    target_time: Optional[TimeOfDay] = None
    target_season: Optional[Season] = None
    target_mood: Optional[Mood] = None
    target_location: Optional[SceneLocation] = None  # indoor/outdoor 힌트


@dataclass
class MappedScene:
    """장면 매핑 결과 — 하나의 Scene Graph 장면에 대한 최종 결과"""
    scene_id: int
    requirement: SceneRequirement           # 원본 요구사항
    selected_clip: Optional[ClipResult] = None  # 선택된 클립
    branch: BranchDecision = BranchDecision.USE_AS_IS
    transform_result: Optional[Dict[str, Any]] = None  # 속성 변환 결과
    transformed_video_path: Optional[str] = None        # 변환된 영상 경로
    search_score: float = 0.0               # 최고 검색 점수
    attribute_match_score: float = 0.0      # 속성 일치도 (0~1)
    candidates_count: int = 0               # 검색된 후보 수
    notes: str = ""                         # 처리 메모
    top_candidates: Optional[List[ClipResult]] = None   # 상위 후보 클립 목록 (PD 선택용)


# ─────────────────────────────────────────────
# PD 인터랙티브 워크플로 데이터 모델
# ─────────────────────────────────────────────

@dataclass
class PDReviewRequest:
    """PD에게 보여줄 리뷰 요청 (Step 1→2)

    분석 결과 + 자동 생성된 프롬프트를 PD에게 전달.
    PD는 프롬프트를 수정하고 백엔드를 선택한 뒤 실행을 요청한다.
    """
    scene_id: int
    branch: BranchDecision                  # TRANSFORM 또는 GENERATE
    requirement: SceneRequirement           # 원본 요구사항
    selected_clip: Optional[ClipResult]     # 검색된 클립 (GENERATE면 None)
    auto_prompt: str                        # GPT/규칙 기반 자동 생성 프롬프트
    current_state: Optional[Dict[str, str]] # 현재 클립 속성 (TRANSFORM 시)
    target_state: Dict[str, str]            # 목표 속성
    available_backends: List[str]           # 사용 가능한 백엔드 목록
    search_score: float = 0.0
    attribute_match_score: float = 0.0
    before_keyframe_path: Optional[str] = None  # Before 키프레임 이미지 경로
    top_candidates: Optional[List[ClipResult]] = None   # 상위 후보 클립 목록 (PD 선택용)


@dataclass
class PDExecutionResult:
    """API 실행 결과 — PD에게 보여줄 결과 (Step 3→4)

    PD가 결과를 보고 승인/재시도/직접업로드를 결정한다.
    """
    scene_id: int
    branch: BranchDecision
    success: bool
    output_path: Optional[str]              # 변환/생성된 영상 경로
    backend: str                            # 사용된 백엔드
    prompt: str                             # 사용된 프롬프트
    before_keyframe_path: Optional[str] = None
    after_keyframe_path: Optional[str] = None
    error: Optional[str] = None
    attempt: int = 1                        # 현재 시도 횟수 (최대 3)
    max_attempts: int = 3


@dataclass
class NeighborContext:
    """인접 클립의 시각적 컨텍스트 — GENERATE 분기의 연속성 보장용

    신규 영상 생성(GENERATE) 시 직전 클립의 마지막 프레임을
    Runway API에 직접 전달하여, 새로 만든 영상이
    앞 영상에서 자연스럽게 이어지도록 한다.

    TRANSFORM은 원본 영상 자체가 API 입력이므로 별도의 컨텍스트가 불필요.
    """
    prev_last_frame_path: Optional[str] = None     # 직전 클립 마지막 프레임 이미지 경로 (.jpg)
    prev_video_path: Optional[str] = None          # 직전 클립 영상 경로
    next_scene_description: Optional[str] = None   # 직후 장면의 텍스트 설명
    next_scene_attrs: Optional[Dict[str, str]] = None  # 직후 장면의 목표 속성


@dataclass
class StoryboardMapping:
    """전체 Scene Graph → 클립 매핑 결과"""
    title: str
    scenes: List[MappedScene]
    total_duration_sec: float = 0.0
    clips_used: int = 0
    clips_transformed: int = 0
    clips_generated: int = 0

    @property
    def summary(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "total_scenes": len(self.scenes),
            "total_duration_sec": self.total_duration_sec,
            "branch_stats": {
                "use_as_is": sum(1 for s in self.scenes if s.branch == BranchDecision.USE_AS_IS),
                "transform": sum(1 for s in self.scenes if s.branch == BranchDecision.TRANSFORM),
                "generate": sum(1 for s in self.scenes if s.branch == BranchDecision.GENERATE),
            }
        }


# ─────────────────────────────────────────────
# 3경로 분기 임계값
# ─────────────────────────────────────────────

# 검색 점수 임계값: 이 이상이면 콘텐츠가 적합한 것으로 판정
CONTENT_MATCH_THRESHOLD = 0.35

# 속성 일치도 임계값: 이 이상이면 변환 없이 그대로 사용
ATTRIBUTE_MATCH_THRESHOLD = 0.7

# 검색 점수 최소값: 이 미만이면 아카이브에 적합한 클립 없음 → 신규 생성
CONTENT_MIN_THRESHOLD = 0.15


class StoryboardMapper:
    """Scene Graph 기반 Storyboard-to-Clip Mapper

    PDF 설계의 핵심 모듈:
    - Scene Graph(JSON)을 입력받아 각 장면에 맞는 클립을 자동 선택
    - Context-Aware Greedy Selection으로 최적 클립 매핑
    - 3경로 분기로 그대로 사용 / 속성 변환 / 신규 생성 판정
    - 역방향 프롬프트 엔진을 자동 트리거하여 속성 변환 수행

    사용법:
        mapper = StoryboardMapper(
            search_fn=my_search_function,      # (query: str, top_k: int) -> List[ClipResult]
            inverse_engine=InversePromptEngine(openai_api_key="..."),
        )
        result = mapper.map_scene_graph(scene_graph_json)
    """

    def __init__(
        self,
        search_fn: Callable[[str, int], List[ClipResult]],
        inverse_engine: Optional[InversePromptEngine] = None,
        top_k: int = 20,
        transform_output_dir: str = "output/transformed",
        generate_output_dir: str = "output/generated",
    ):
        """
        Args:
            search_fn: 검색 콜백. (query, top_k) -> List[ClipResult]
                       VideoRAG 파이프라인의 Phase 1-3 검색 결과를 반환해야 함
            inverse_engine: 역방향 프롬프트 엔진 (None이면 변환 불가 → 분기에서 제외)
                            Runway 연동 시 generate_video()로 GENERATE 분기도 처리
            top_k: 장면별 검색 후보 수
            transform_output_dir: 변환된 영상 저장 디렉토리
            generate_output_dir: 생성된 영상 저장 디렉토리
        """
        self.search_fn = search_fn
        self.inverse_engine = inverse_engine or InversePromptEngine()
        self.top_k = top_k
        self.transform_output_dir = transform_output_dir
        self.generate_output_dir = generate_output_dir

    # ─────────────────────────────────────────
    # Step 1: Scene Graph JSON 파싱
    # ─────────────────────────────────────────

    def parse_scene_graph(self, scene_graph: Dict) -> List[SceneRequirement]:
        """Scene Graph JSON을 파싱하여 장면별 요구사항 리스트로 변환

        입력 JSON 스키마:
        {
            "title": "제목",
            "scenes": [
                {
                    "scene_id": 1,
                    "description": "장면 설명 (검색 쿼리로 사용)",
                    "duration_sec": 5.0,
                    "attributes": {
                        "time_of_day": "dawn|morning|afternoon|evening|night",
                        "season": "spring|summer|autumn|winter",
                        "mood": "bright|warm|cool|dark|dramatic|neutral",
                        "location": "indoor|outdoor"
                    }
                },
                ...
            ]
        }

        Args:
            scene_graph: Scene Graph JSON dict

        Returns:
            List[SceneRequirement]
        """
        requirements = []

        scenes_data = scene_graph.get("scenes", [])
        if not scenes_data:
            logger.warning("Scene Graph에 장면이 없습니다.")
            return requirements

        for scene in scenes_data:
            attrs = scene.get("attributes", {})

            # 속성 파싱 — 문자열을 Enum으로 변환
            target_time = self._parse_time(attrs.get("time_of_day"))
            target_season = self._parse_season(attrs.get("season"))
            target_mood = self._parse_mood(attrs.get("mood"))
            target_location = self._parse_location(attrs.get("location"))

            # description 폴백: description → description_ko → 빈 문자열
            description = (
                scene.get("description")
                or scene.get("description_ko", "")
            )

            req = SceneRequirement(
                scene_id=scene.get("scene_id", len(requirements) + 1),
                description=description,
                duration_sec=scene.get("duration_sec", 5.0),
                target_time=target_time,
                target_season=target_season,
                target_mood=target_mood,
                target_location=target_location,
            )
            requirements.append(req)

        logger.info(f"Scene Graph 파싱 완료: {len(requirements)}개 장면")
        return requirements

    # ─────────────────────────────────────────
    # Step 2: 장면별 클립 매핑 (E2E)
    # ─────────────────────────────────────────

    def map_scene_graph(self, scene_graph: Dict) -> StoryboardMapping:
        """Scene Graph 전체를 처리하여 클립 매핑 결과 반환

        전체 흐름:
        1. Scene Graph 파싱 → List[SceneRequirement]
        2. 각 장면마다:
           a. description으로 아카이브 검색
           b. 최적 클립 선택
           c. 3경로 분기 판정 (그대로/변환/생성)
           d. 변환 필요 시 InversePromptEngine 자동 호출
        3. Scene Graph 순서대로 정렬된 결과 반환

        Args:
            scene_graph: Scene Graph JSON dict

        Returns:
            StoryboardMapping
        """
        title = scene_graph.get("title", "Untitled")
        requirements = self.parse_scene_graph(scene_graph)

        mapped_scenes = []
        for i, req in enumerate(requirements):
            # ── 인접 장면 컨텍스트 구성 (연속성 보장) ──
            neighbor_ctx = self._build_neighbor_context(
                mapped_scenes, requirements, i
            )

            mapped = self._map_single_scene(req, neighbor_ctx)
            mapped_scenes.append(mapped)
            logger.info(
                f"장면 {req.scene_id} 매핑 완료: "
                f"branch={mapped.branch.value}, "
                f"clip={mapped.selected_clip.clip_id if mapped.selected_clip else 'None'}, "
                f"score={mapped.search_score:.3f}"
            )

        result = StoryboardMapping(
            title=title,
            scenes=mapped_scenes,
            total_duration_sec=sum(r.duration_sec for r in requirements),
            clips_used=sum(1 for s in mapped_scenes if s.branch == BranchDecision.USE_AS_IS),
            clips_transformed=sum(1 for s in mapped_scenes if s.branch == BranchDecision.TRANSFORM),
            clips_generated=sum(1 for s in mapped_scenes if s.branch == BranchDecision.GENERATE),
        )

        logger.info(
            f"Storyboard 매핑 완료: {result.summary}"
        )
        return result

    # ─────────────────────────────────────────
    # 인접 프레임 컨텍스트 (연속성 보장)
    # ─────────────────────────────────────────

    def _build_neighbor_context(
        self,
        mapped_so_far: List[MappedScene],
        requirements: List[SceneRequirement],
        current_idx: int,
    ) -> NeighborContext:
        """현재 장면의 인접 컨텍스트 구성

        - 직전 장면: 이미 매핑 완료 → 마지막 프레임의 시각적 속성 추출
        - 직후 장면: 아직 미매핑 → Scene Graph의 텍스트 설명 + 목표 속성만 활용

        Args:
            mapped_so_far: 지금까지 매핑 완료된 장면들
            requirements: 전체 장면 요구사항 리스트
            current_idx: 현재 처리 중인 장면 인덱스

        Returns:
            NeighborContext
        """
        ctx = NeighborContext()

        # ── 직전 장면: 마지막 프레임 이미지 + 영상 경로 추출 ──
        if mapped_so_far:
            prev_scene = mapped_so_far[-1]
            frame_path, video_path = self._extract_prev_scene_references(prev_scene)
            ctx.prev_last_frame_path = frame_path
            ctx.prev_video_path = video_path

        # ── 직후 장면: Scene Graph 설명 + 목표 속성 ──
        if current_idx + 1 < len(requirements):
            next_req = requirements[current_idx + 1]
            ctx.next_scene_description = next_req.description
            ctx.next_scene_attrs = {}
            if next_req.target_time:
                ctx.next_scene_attrs["time_of_day"] = next_req.target_time.value
            if next_req.target_season:
                ctx.next_scene_attrs["season"] = next_req.target_season.value
            if next_req.target_mood:
                ctx.next_scene_attrs["mood"] = next_req.target_mood.value

        return ctx

    def _extract_prev_scene_references(
        self, prev_scene: MappedScene
    ) -> Tuple[Optional[str], Optional[str]]:
        """직전 장면의 마지막 프레임 이미지와 영상 경로를 추출

        GENERATE 분기에서 Runway API에 직접 전달할
        참조 이미지(마지막 프레임)와 참조 영상 경로를 반환한다.

        - 변환/생성된 영상이 있으면 그것을 사용
        - 없으면 원본 클립의 영상을 사용
        - 마지막 프레임을 .jpg로 저장하여 경로 반환

        Args:
            prev_scene: 직전 매핑 완료 장면

        Returns:
            (마지막 프레임 이미지 경로, 영상 경로) — 실패 시 (None, None)
        """
        import cv2
        import os

        # 변환/생성된 영상이 있으면 그것을 사용
        video_path = prev_scene.transformed_video_path
        if not video_path and prev_scene.selected_clip:
            video_path = prev_scene.selected_clip.video_path

        if not video_path:
            return None, None

        try:
            # 마지막 프레임 추출 → .jpg로 저장
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames > 1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
            ret, frame = cap.read()
            cap.release()

            if not ret:
                return None, video_path

            # 프레임 이미지를 generate_output_dir에 저장
            os.makedirs(self.generate_output_dir, exist_ok=True)
            frame_path = os.path.join(
                self.generate_output_dir,
                f"prev_scene{prev_scene.scene_id}_last_frame.jpg"
            )
            cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            logger.info(f"직전 장면 {prev_scene.scene_id} 마지막 프레임 저장: {frame_path}")

            return frame_path, video_path
        except Exception as e:
            logger.warning(f"직전 장면 참조 추출 실패: {e}")
            return None, None

    def _map_single_scene(self, req: SceneRequirement, neighbor_ctx: Optional[NeighborContext] = None) -> MappedScene:
        """단일 장면에 대한 클립 매핑 수행

        1. 검색 (search_fn)
        2. 최적 클립 선택
        3. 3경로 분기 판정
        4. 필요 시 속성 변환 실행

        Args:
            req: 장면 요구사항

        Returns:
            MappedScene
        """
        # ── 검색 ──
        candidates = self.search_fn(req.description, self.top_k)

        if not candidates:
            # 후보가 아예 없음 → 신규 생성
            return self._handle_generate(req, reason="검색 결과 없음", neighbor_ctx=neighbor_ctx)

        # ── 최적 클립 선택 (최고 점수 기준) ──
        best_clip = candidates[0]  # search_fn이 점수순 정렬된 결과 반환 가정
        search_score = best_clip.score

        # ── 3경로 분기 판정 ──
        branch, attr_match = self._decide_branch(best_clip, req)

        mapped = MappedScene(
            scene_id=req.scene_id,
            requirement=req,
            selected_clip=best_clip,
            branch=branch,
            search_score=search_score,
            attribute_match_score=attr_match,
            candidates_count=len(candidates),
        )

        if branch == BranchDecision.USE_AS_IS:
            mapped.notes = "검색 결과 그대로 사용 (콘텐츠 + 속성 모두 적합)"

        elif branch == BranchDecision.TRANSFORM:
            # ── 역방향 프롬프트 자동 트리거 (GPT 동적 프롬프트 + Runway API) ──
            transform_result = self._execute_transform(best_clip, req)
            mapped.transform_result = transform_result
            mapped.transformed_video_path = transform_result.get("output_path")
            mapped.notes = (
                f"속성 변환 적용: {transform_result.get('backend', 'unknown')} 백엔드, "
                f"prompt='{transform_result.get('prompt', '')[:80]}...'"
            )

        elif branch == BranchDecision.GENERATE:
            return self._handle_generate(req, reason="검색 점수 미달", neighbor_ctx=neighbor_ctx)

        return mapped

    # ─────────────────────────────────────────
    # 3경로 분기 판정 로직
    # ─────────────────────────────────────────

    def _decide_branch(
    self, clip: ClipResult, req: SceneRequirement,
    frame: Optional[np.ndarray] = None,
    ) -> tuple:
        """3경로 분기 판정

        판정 기준:
        1) search_score < CONTENT_MIN_THRESHOLD → GENERATE (콘텐츠 자체가 안맞음)
        2) search_score >= CONTENT_MATCH_THRESHOLD AND attr_match >= ATTRIBUTE_MATCH_THRESHOLD
           → USE_AS_IS (콘텐츠도 맞고 속성도 맞음)
        3) search_score >= CONTENT_MIN_THRESHOLD AND (콘텐츠는 맞지만 속성이 다름)
           → TRANSFORM (역방향 프롬프트로 속성 변환)

        Args:
            clip: 검색된 최적 클립
            req: 장면 요구사항

        Returns:
            (BranchDecision, attribute_match_score)
        """
        search_score = clip.score

        # 콘텐츠 매칭 실패 → 신규 생성
        if search_score < CONTENT_MIN_THRESHOLD:
            return BranchDecision.GENERATE, 0.0

        # 속성 일치도 계산
        attr_match = self._compute_attribute_match(clip, req, frame=frame)

        # 콘텐츠 OK + 속성 OK → 그대로 사용
        if search_score >= CONTENT_MATCH_THRESHOLD and attr_match >= ATTRIBUTE_MATCH_THRESHOLD:
            return BranchDecision.USE_AS_IS, attr_match

        # 콘텐츠 OK + 속성 불일치 → 변환
        if search_score >= CONTENT_MIN_THRESHOLD:
            # 속성 요구사항이 아예 없으면 (전부 None) 그대로 사용
            has_attr_req = any([req.target_time, req.target_season, req.target_mood])
            if not has_attr_req:
                return BranchDecision.USE_AS_IS, 1.0
            return BranchDecision.TRANSFORM, attr_match

        return BranchDecision.GENERATE, attr_match

    def _compute_attribute_match(
    self, clip: ClipResult, req: SceneRequirement,
    frame: Optional[np.ndarray] = None,
) -> float:
        """클립의 현재 속성과 요구 속성 간의 일치도 계산

        클립의 첫 프레임을 분석하여 현재 속성을 추론하고,
        요구 속성과 비교하여 일치도 (0~1)을 반환한다.

        일치도 = 일치하는 속성 수 / 요구된 속성 수

        Args:
            clip: 검색된 클립
            req: 장면 요구사항

        Returns:
            0.0 ~ 1.0 사이의 속성 일치도
        """
        import cv2

        # 요구 속성이 없으면 완전 일치
        checks = []
        if req.target_time:
            checks.append("time")
        if req.target_season:
            checks.append("season")
        if req.target_mood:
            checks.append("mood")

        if not checks:
            return 1.0

        # 클립의 첫 프레임에서 현재 상태 추론
        try:
            if frame is None:
                cap = cv2.VideoCapture(clip.video_path)
                if clip.start_ms > 0:
                    cap.set(cv2.CAP_PROP_POS_MSEC, clip.start_ms)
                ret, frame = cap.read()
                cap.release()

                if not ret:
                    logger.warning(f"프레임 추출 실패: {clip.clip_id}")
                    return 0.5

            current = self.inverse_engine.analyze_current_state(frame)
        except Exception as e:
            logger.warning(f"속성 분석 실패: {clip.clip_id} — {e}")
            return 0.5

        # 속성별 일치 판정
        matches = 0
        total = len(checks)

        if "time" in checks:
            if current.time_of_day == req.target_time:
                matches += 1
            # 실내 씬이면 시간대 비교가 무의미 → 일치로 처리
            elif current.location == SceneLocation.INDOOR:
                matches += 0.5  # 부분 일치

        if "season" in checks:
            if current.season == req.target_season:
                matches += 1
            elif current.location == SceneLocation.INDOOR:
                matches += 0.5

        if "mood" in checks:
            if current.mood == req.target_mood:
                matches += 1

        return float(matches) / total

    # ─────────────────────────────────────────
    # 속성 변환 실행
    # ─────────────────────────────────────────

    def _execute_transform(
        self, clip: ClipResult, req: SceneRequirement
    ) -> Dict[str, Any]:
        """역방향 프롬프트 엔진을 통한 속성 변환 실행

        InversePromptEngine.transform_clip()을 호출하여 실제 변환 수행.
        실내 씬이면 엔진 내부에서 자동으로 시간대/계절 변환을 건너뛰고
        분위기 변환만 수행한다.

        TRANSFORM은 원본 영상 자체가 Runway의 입력으로 들어가므로
        인접 장면 컨텍스트가 별도로 필요하지 않다.

        Args:
            clip: 변환할 클립
            req: 장면 요구사항

        Returns:
            변환 결과 딕셔너리
        """
        import os
        os.makedirs(self.transform_output_dir, exist_ok=True)

        output_path = os.path.join(
            self.transform_output_dir,
            f"scene{req.scene_id}_{clip.clip_id}_transformed.mp4"
        )

        result = self.inverse_engine.transform_clip(
            video_path=clip.video_path,
            output_path=output_path,
            target_time=req.target_time,
            target_season=req.target_season,
            target_mood=req.target_mood,
            scene_description=req.description,
        )

        return result

    # ─────────────────────────────────────────
    # 신규 생성 처리
    # ─────────────────────────────────────────

    def _handle_generate(
        self, req: SceneRequirement, reason: str,
        neighbor_ctx: Optional[NeighborContext] = None,
    ) -> MappedScene:
        """적합한 클립이 없을 때 Runway API로 신규 영상 생성

        InversePromptEngine.generate_video()를 통해 Runway API로
        신규 영상을 생성한다. GPT-4o-mini가 장면 요구사항을 바탕으로
        최적의 생성 프롬프트를 동적으로 만든다.

        인접 장면 컨텍스트가 있으면 프롬프트에 포함하여
        앞뒤 영상과 시각적 연속성을 보장한다.

        API 연동 불가 시 GENERATE 상태로만 표시하고 수동 처리를 안내한다.

        Args:
            req: 장면 요구사항
            reason: 생성이 필요한 이유
            neighbor_ctx: 인접 장면의 시각적 컨텍스트

        Returns:
            MappedScene (branch=GENERATE)
        """
        import os
        os.makedirs(self.generate_output_dir, exist_ok=True)

        mapped = MappedScene(
            scene_id=req.scene_id,
            requirement=req,
            branch=BranchDecision.GENERATE,
            notes=f"신규 생성 필요: {reason}",
        )

        # GPT 동적 프롬프트 생성 시도 → 폴백: 규칙 기반 프롬프트
        gen_prompt = self._build_generation_prompt_dynamic(req, neighbor_ctx=neighbor_ctx)
        if not gen_prompt:
            gen_prompt = self._build_generation_prompt(req, neighbor_ctx=neighbor_ctx)

        # Runway 백엔드가 설정된 경우 영상 생성 시도
        backend = self.inverse_engine.default_backend
        if backend == "runway":
            try:
                output_path = os.path.join(
                    self.generate_output_dir,
                    f"scene{req.scene_id}_generated.mp4"
                )
                # 직전 클립 참조 정보 추출 (연속성 보장)
                prev_frame = neighbor_ctx.prev_last_frame_path if neighbor_ctx else None
                prev_video = neighbor_ctx.prev_video_path if neighbor_ctx else None

                result = self.inverse_engine.generate_video(
                    prompt=gen_prompt,
                    duration_sec=req.duration_sec,
                    output_path=output_path,
                    backend=backend,
                    prev_frame_path=prev_frame,
                    prev_video_path=prev_video,
                )
                if result.get("success"):
                    mapped.transformed_video_path = result["output_path"]
                    mapped.transform_result = result
                    mapped.notes = (
                        f"{backend.capitalize()} 신규 생성 완료: "
                        f"prompt='{gen_prompt[:80]}...'"
                    )
                    logger.info(f"장면 {req.scene_id} 신규 생성 완료 ({backend})")
                else:
                    mapped.notes = f"신규 생성 실패 ({backend}): {result.get('error', 'unknown')}"
                    logger.error(f"장면 {req.scene_id} 신규 생성 실패: {result.get('error')}")
            except Exception as e:
                logger.error(f"장면 {req.scene_id} 신규 생성 예외: {e}")
                mapped.notes = f"신규 생성 예외: {e}"
        else:
            logger.warning(
                f"장면 {req.scene_id}: 적합한 클립 없고 backend='{backend}'은 "
                f"신규 생성 미지원. 'runway'로 설정하세요."
            )
            mapped.notes = (
                f"신규 생성 필요 (수동 처리): {reason}. "
                f"prompt='{gen_prompt[:80]}...'"
            )

        return mapped

    @staticmethod
    def _format_neighbor_context_for_prompt(neighbor_ctx: Optional[NeighborContext]) -> str:
        """인접 장면 컨텍스트를 GPT 프롬프트용 텍스트로 포맷팅

        GENERATE 분기에서 GPT 프롬프트에 삽입할 연속성 힌트를 생성한다.
        실제 이미지/영상은 API에 직접 전달되므로, 여기서는
        직후 장면 설명과 "직전 프레임이 참조로 첨부된다"는 안내만 포함한다.

        Args:
            neighbor_ctx: 인접 장면 컨텍스트

        Returns:
            포맷팅된 영문 컨텍스트 문자열 (없으면 빈 문자열)
        """
        if not neighbor_ctx:
            return ""

        parts = []

        if neighbor_ctx.prev_last_frame_path or neighbor_ctx.prev_video_path:
            parts.append(
                "\n[VISUAL CONTINUITY — Previous scene reference]\n"
                "The previous scene's last frame/video is attached as a visual reference.\n"
                "Ensure the generated video begins with a similar color palette, "
                "lighting, and atmosphere for a seamless transition."
            )

        if neighbor_ctx.next_scene_description:
            next_info = "\n[VISUAL CONTINUITY — Next scene preview]\n"
            next_info += f"- Description: {neighbor_ctx.next_scene_description}\n"
            if neighbor_ctx.next_scene_attrs:
                next_info += f"- Target attributes: {neighbor_ctx.next_scene_attrs}\n"
            next_info += (
                "The ending frames should gradually prepare for a natural "
                "transition into this next scene."
            )
            parts.append(next_info)

        return "\n".join(parts)

    def _build_generation_prompt_dynamic(
        self, req: SceneRequirement,
        neighbor_ctx: Optional[NeighborContext] = None,
    ) -> Optional[str]:
        """GPT-4o-mini를 사용한 영상 생성 프롬프트 동적 생성

        장면 요구사항(description + attributes)을 GPT에게 전달하여
        Runway에 최적화된 생성 프롬프트를 만든다.
        인접 장면 컨텍스트가 있으면 시각적 연속성 지시를 포함한다.

        Args:
            req: 장면 요구사항
            neighbor_ctx: 인접 장면의 시각적 컨텍스트

        Returns:
            생성 프롬프트 (실패 시 None)
        """
        api_key = self.inverse_engine.openai_api_key
        if not api_key:
            return None

        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
        except (ImportError, Exception):
            return None

        attrs_info = []
        if req.target_time:
            attrs_info.append(f"Time of day: {req.target_time.value}")
        if req.target_season:
            attrs_info.append(f"Season: {req.target_season.value}")
        if req.target_mood:
            attrs_info.append(f"Mood: {req.target_mood.value}")
        if req.target_location:
            attrs_info.append(f"Setting: {req.target_location.value}")

        system_prompt = (
            "You are an expert prompt engineer for AI video generation APIs (Runway Gen-3). "
            "Generate a detailed, visually specific English prompt that will produce "
            "high-quality, broadcast-ready footage.\n\n"
            "Rules:\n"
            "1. Describe the visual scene in vivid detail (subjects, actions, environment, lighting).\n"
            "2. Include camera movement/angle if appropriate.\n"
            "3. Specify atmosphere, color palette, and mood.\n"
            "4. Keep under 150 words.\n"
            "5. Output ONLY the prompt text.\n"
            "6. If adjacent scene context is provided, ensure visual continuity:\n"
            "   - Match the color temperature, brightness, and tone of the previous scene's ending.\n"
            "   - Gradually transition toward the next scene's expected look at the end.\n"
            "   - Avoid abrupt visual jumps in lighting, color palette, or atmosphere."
        )

        user_prompt = (
            f"Generate a video creation prompt for this scene:\n\n"
            f"Scene description: {req.description}\n"
            f"Duration: {req.duration_sec} seconds\n"
        )
        if attrs_info:
            user_prompt += f"Attributes: {', '.join(attrs_info)}\n"

        # 인접 장면 연속성 컨텍스트 추가
        continuity_info = self._format_neighbor_context_for_prompt(neighbor_ctx)
        if continuity_info:
            user_prompt += continuity_info + "\n"

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=200,
                temperature=0.5,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"GPT 생성 프롬프트 생성 실패: {e}")
            return None

    def _build_generation_prompt(
        self, req: SceneRequirement,
        neighbor_ctx: Optional[NeighborContext] = None,
    ) -> str:
        """장면 요구사항으로부터 영상 생성 프롬프트 조립 (규칙 기반 폴백)

        Args:
            req: 장면 요구사항
            neighbor_ctx: 인접 장면의 시각적 컨텍스트

        Returns:
            영문 생성 프롬프트
        """
        parts = [req.description]

        if req.target_time:
            parts.append(f"Time of day: {req.target_time.value}")
        if req.target_season:
            parts.append(f"Season: {req.target_season.value}")
        if req.target_mood:
            parts.append(f"Mood: {req.target_mood.value}")
        if req.target_location:
            parts.append(f"Setting: {req.target_location.value}")

        parts.append(f"Duration: {req.duration_sec} seconds")

        # 인접 장면 연속성 힌트 (규칙 기반 폴백에서는 간략하게)
        if neighbor_ctx and (neighbor_ctx.prev_last_frame_path or neighbor_ctx.prev_video_path):
            parts.append(
                "The previous scene's last frame is provided as visual reference. "
                "Begin with a matching color palette and lighting for seamless continuity"
            )

        parts.append("High quality, cinematic, broadcast-ready footage.")

        return ". ".join(parts)

    # ─────────────────────────────────────────
    # PD 인터랙티브 워크플로
    # ─────────────────────────────────────────

    def prepare_scene(
        self, req: SceneRequirement,
        excluded_clip_ids: Optional[set] = None,
        n_candidates: int = 5,
    ) -> PDReviewRequest:
        """Step 1→2: 장면 분석 + 프롬프트 생성 → PD 리뷰 요청 반환

        검색 → 분기 판정 → 프롬프트 자동 생성까지 수행하고,
        PD가 프롬프트를 수정하고 백엔드를 선택할 수 있도록
        PDReviewRequest를 반환한다.

        USE_AS_IS인 경우에도 상위 후보를 포함하여 반환한다
        (PD가 클립을 선택할 수 있도록).

        Args:
            req: 장면 요구사항
            excluded_clip_ids: 이미 다른 장면에서 선택된 clip_id 집합 (중복 제거)
            n_candidates: PD에게 보여줄 상위 후보 수 (기본 5)

        Returns:
            PDReviewRequest (항상 반환 — PD가 클립을 선택)
        """
        import cv2
        import os

        if excluded_clip_ids is None:
            excluded_clip_ids = set()

        # ── 검색 (중복 제거 적용) ──
        logger.info(
            f"[Scene {req.scene_id}] 검색 쿼리: '{req.description}' "
            f"(excluded: {len(excluded_clip_ids)}개)"
        )
        raw_candidates = self.search_fn(req.description, self.top_k)
        candidates = [c for c in raw_candidates if c.clip_id not in excluded_clip_ids]
        logger.info(
            f"[Scene {req.scene_id}] 검색 결과: {len(raw_candidates)}개 → "
            f"중복 제거 후 {len(candidates)}개 "
            f"(top1: {candidates[0].clip_id if candidates else 'N/A'}, "
            f"score: {candidates[0].score if candidates else 0:.4f})"
        )

        # 상위 n_candidates개 추출 (PD 선택용)
        top_candidates = candidates[:n_candidates]

        best_clip = candidates[0] if candidates else None
        search_score = best_clip.score if best_clip else 0.0

        # ── 프레임 1회 추출 (이후 분기판정 + 상태분석 + 프롬프트 생성에서 재사용) ──
        _frame = None
        _current_attrs = None
        if best_clip:
            try:
                cap = cv2.VideoCapture(best_clip.video_path)
                if best_clip.start_ms > 0:
                    cap.set(cv2.CAP_PROP_POS_MSEC, best_clip.start_ms)
                ret, _frame = cap.read()
                cap.release()
                if not ret:
                    _frame = None
            except Exception as e:
                logger.warning(f"프레임 추출 실패: {e}")

        # ── 분기 판정 ──
        if not best_clip:
            branch = BranchDecision.GENERATE
            attr_match = 0.0
        else:
            branch, attr_match = self._decide_branch(best_clip, req, frame=_frame)

        # ── 현재 상태 분석 (TRANSFORM 시) ──
        current_state_dict = None
        before_keyframe_path = None

        if branch == BranchDecision.TRANSFORM and best_clip and _frame is not None:
            try:
                _current_attrs = self.inverse_engine.analyze_current_state(_frame)
                current_state_dict = {
                    "time_of_day": _current_attrs.time_of_day.value,
                    "season": _current_attrs.season.value,
                    "mood": _current_attrs.mood.value,
                    "location": _current_attrs.location.value,
                    "brightness": f"{_current_attrs.avg_brightness:.0f}",
                    "color_temperature": f"{_current_attrs.color_temperature:.0f}K",
                }
                preview_dir = "output/preview"
                os.makedirs(preview_dir, exist_ok=True)
                before_keyframe_path = os.path.join(
                    preview_dir, f"scene{req.scene_id}_before.jpg"
                )
                cv2.imwrite(before_keyframe_path, _frame)
            except Exception as e:
                logger.warning(f"현재 상태 분석 실패: {e}")
       
        # ── 목표 상태 ──
        target_state = {}
        if req.target_time:
            target_state["time_of_day"] = req.target_time.value
        if req.target_season:
            target_state["season"] = req.target_season.value
        if req.target_mood:
            target_state["mood"] = req.target_mood.value
        if req.target_location:
            target_state["location"] = req.target_location.value

        # ── 프롬프트 자동 생성 ──
        if branch == BranchDecision.TRANSFORM:
            auto_prompt = self._generate_transform_prompt_for_review(
                best_clip, req, current_state_dict,
                frame=_frame, current_attrs=_current_attrs,
            )
        elif branch == BranchDecision.GENERATE:
            auto_prompt = self._build_generation_prompt_dynamic(req)
            if not auto_prompt:
                auto_prompt = self._build_generation_prompt(req)
        else:  # USE_AS_IS
            auto_prompt = ""

        # ── 사용 가능한 백엔드 ──
        available = []
        if self.inverse_engine.runway_api_key:
            available.append("runway")
        available.append("opencv")  # 항상 가능
        available.append("original")  # 원본 그대로 사용

        return PDReviewRequest(
            scene_id=req.scene_id,
            branch=branch,
            requirement=req,
            selected_clip=best_clip,
            auto_prompt=auto_prompt,
            current_state=current_state_dict,
            target_state=target_state,
            available_backends=available,
            search_score=search_score,
            attribute_match_score=attr_match,
            before_keyframe_path=before_keyframe_path,
            top_candidates=top_candidates,
        )

    def _generate_transform_prompt_for_review(
        self,
        clip: ClipResult,
        req: SceneRequirement,
        current_state_dict: Optional[Dict],
        frame: Optional[np.ndarray] = None,
        current_attrs: Optional['SceneAttributes'] = None,
    ) -> str:
        """TRANSFORM용 프롬프트 생성 (PD 리뷰 전 단계)"""
        import cv2

        try:
            if frame is None:
                cap = cv2.VideoCapture(clip.video_path)
                if clip.start_ms > 0:
                    cap.set(cv2.CAP_PROP_POS_MSEC, clip.start_ms)
                ret, frame = cap.read()
                cap.release()
                if not ret:
                    raise RuntimeError("프레임 추출 실패")

            current = current_attrs or self.inverse_engine.analyze_current_state(frame)
            delta = self.inverse_engine.compute_delta(
                current, req.target_time, req.target_season, req.target_mood,
                target_location=req.target_location,
            )
            prompt = self.inverse_engine.generate_transform_prompt(
                delta, current_state=current, scene_description=req.description
            )
            return prompt
        except Exception as e:
            logger.warning(f"TRANSFORM 프롬프트 생성 실패: {e}")

    def execute_scene(
        self,
        review: PDReviewRequest,
        prompt: str,
        backend: str,
        attempt: int = 1,
    ) -> PDExecutionResult:
        """Step 3: PD가 승인한 프롬프트 + 백엔드로 API 실행

        Args:
            review: prepare_scene()에서 반환된 리뷰 요청
            prompt: PD가 수정한 (또는 그대로 승인한) 프롬프트
            backend: PD가 선택한 백엔드 ("runway", "opencv")
            attempt: 현재 시도 횟수 (1~3)

        Returns:
            PDExecutionResult
        """
        import os
        import cv2

        req = review.requirement
        result_base = {
            "scene_id": review.scene_id,
            "branch": review.branch,
            "prompt": prompt,
            "backend": backend,
            "attempt": attempt,
            "max_attempts": 3,
            "before_keyframe_path": review.before_keyframe_path,
        }

        if review.branch == BranchDecision.TRANSFORM:
            # ── TRANSFORM: 기존 클립 변환 ──
            clip = review.selected_clip
            os.makedirs(self.transform_output_dir, exist_ok=True)
            output_path = os.path.join(
                self.transform_output_dir,
                f"scene{req.scene_id}_{clip.clip_id}_v{attempt}.mp4"
            )

            try:
                api_result = self.inverse_engine.apply_transform(
                    video_path=clip.video_path,
                    prompt=prompt,
                    output_path=output_path,
                    backend=backend,
                )

                # After 키프레임 추출
                after_keyframe = None
                if api_result.get("success"):
                    try:
                        preview_dir = "output/preview"
                        os.makedirs(preview_dir, exist_ok=True)
                        after_keyframe = os.path.join(
                            preview_dir,
                            f"scene{req.scene_id}_after_v{attempt}.jpg"
                        )
                        cap = cv2.VideoCapture(output_path)
                        ret, frame = cap.read()
                        cap.release()
                        if ret:
                            cv2.imwrite(after_keyframe, frame)
                    except Exception:
                        pass

                return PDExecutionResult(
                    success=api_result.get("success", False),
                    output_path=api_result.get("output_path"),
                    after_keyframe_path=after_keyframe,
                    error=api_result.get("error"),
                    **result_base,
                )
            except Exception as e:
                return PDExecutionResult(
                    success=False,
                    output_path=None,
                    error=str(e),
                    **result_base,
                )

        else:
            # ── GENERATE: 신규 영상 생성 ──
            os.makedirs(self.generate_output_dir, exist_ok=True)
            output_path = os.path.join(
                self.generate_output_dir,
                f"scene{req.scene_id}_v{attempt}.mp4"
            )

            try:
                api_result = self.inverse_engine.generate_video(
                    prompt=prompt,
                    duration_sec=req.duration_sec,
                    output_path=output_path,
                    backend=backend,
                )

                # 생성 영상 키프레임 추출
                after_keyframe = None
                if api_result.get("success"):
                    try:
                        preview_dir = "output/preview"
                        os.makedirs(preview_dir, exist_ok=True)
                        after_keyframe = os.path.join(
                            preview_dir,
                            f"scene{req.scene_id}_gen_v{attempt}.jpg"
                        )
                        cap = cv2.VideoCapture(output_path)
                        ret, frame = cap.read()
                        cap.release()
                        if ret:
                            cv2.imwrite(after_keyframe, frame)
                    except Exception:
                        pass

                return PDExecutionResult(
                    success=api_result.get("success", False),
                    output_path=api_result.get("output_path"),
                    after_keyframe_path=after_keyframe,
                    error=api_result.get("error"),
                    **result_base,
                )
            except Exception as e:
                return PDExecutionResult(
                    success=False,
                    output_path=None,
                    error=str(e),
                    **result_base,
                )

    def accept_scene(
        self,
        review: PDReviewRequest,
        execution: PDExecutionResult,
    ) -> MappedScene:
        """PD가 결과를 승인 → MappedScene 확정

        Args:
            review: prepare_scene() 결과
            execution: execute_scene() 결과

        Returns:
            확정된 MappedScene
        """
        return MappedScene(
            scene_id=review.scene_id,
            requirement=review.requirement,
            selected_clip=review.selected_clip,
            branch=review.branch,
            transform_result={
                "backend": execution.backend,
                "prompt": execution.prompt,
                "attempt": execution.attempt,
            },
            transformed_video_path=execution.output_path,
            search_score=review.search_score,
            attribute_match_score=review.attribute_match_score,
            candidates_count=0,
            notes=(
                f"PD 승인 (시도 {execution.attempt}/{execution.max_attempts}): "
                f"{execution.backend} 백엔드"
            ),
        )

    def accept_upload(
        self,
        review: PDReviewRequest,
        uploaded_file_path: str,
    ) -> MappedScene:
        """PD가 직접 파일을 업로드하여 장면 확정

        Args:
            review: prepare_scene() 결과
            uploaded_file_path: PD가 업로드한 영상 파일 경로

        Returns:
            확정된 MappedScene
        """
        return MappedScene(
            scene_id=review.scene_id,
            requirement=review.requirement,
            selected_clip=review.selected_clip,
            branch=review.branch,
            transformed_video_path=uploaded_file_path,
            search_score=review.search_score,
            attribute_match_score=review.attribute_match_score,
            candidates_count=0,
            notes="PD 직접 업로드",
        )

    def skip_scene(self, review: PDReviewRequest) -> MappedScene:
        """PD가 장면을 건너뜀 (취소)

        Args:
            review: prepare_scene() 결과

        Returns:
            빈 MappedScene (클립 없음)
        """
        return MappedScene(
            scene_id=review.scene_id,
            requirement=review.requirement,
            branch=review.branch,
            notes="PD 취소 — 장면 건너뜀",
        )

    # ─────────────────────────────────────────
    # 유틸리티: 문자열 → Enum 변환
    # ─────────────────────────────────────────

    @staticmethod
    def _parse_time(value: Optional[str]) -> Optional[TimeOfDay]:
        if not value:
            return None
        mapping = {
            "dawn": TimeOfDay.DAWN, "새벽": TimeOfDay.DAWN,
            "morning": TimeOfDay.MORNING, "아침": TimeOfDay.MORNING,
            "afternoon": TimeOfDay.AFTERNOON, "오후": TimeOfDay.AFTERNOON, "낮": TimeOfDay.AFTERNOON,
            "evening": TimeOfDay.EVENING, "저녁": TimeOfDay.EVENING,
            "night": TimeOfDay.NIGHT, "밤": TimeOfDay.NIGHT,
        }
        return mapping.get(value.lower())

    @staticmethod
    def _parse_season(value: Optional[str]) -> Optional[Season]:
        if not value:
            return None
        mapping = {
            "spring": Season.SPRING, "봄": Season.SPRING,
            "summer": Season.SUMMER, "여름": Season.SUMMER,
            "autumn": Season.AUTUMN, "fall": Season.AUTUMN, "가을": Season.AUTUMN,
            "winter": Season.WINTER, "겨울": Season.WINTER,
        }
        return mapping.get(value.lower())

    @staticmethod
    def _parse_mood(value: Optional[str]) -> Optional[Mood]:
        if not value:
            return None
        mapping = {
            "bright": Mood.BRIGHT, "밝은": Mood.BRIGHT,
            "warm": Mood.WARM, "따뜻한": Mood.WARM,
            "cool": Mood.COOL, "차가운": Mood.COOL,
            "dark": Mood.DARK, "어두운": Mood.DARK,
            "dramatic": Mood.DRAMATIC, "극적인": Mood.DRAMATIC,
            "neutral": Mood.NEUTRAL, "중립": Mood.NEUTRAL,
        }
        return mapping.get(value.lower())

    @staticmethod
    def _parse_location(value: Optional[str]) -> Optional[SceneLocation]:
        if not value:
            return None
        mapping = {
            "indoor": SceneLocation.INDOOR, "실내": SceneLocation.INDOOR,
            "outdoor": SceneLocation.OUTDOOR, "실외": SceneLocation.OUTDOOR,
        }
        return mapping.get(value.lower())


# ─────────────────────────────────────────────
# 헬퍼: 매핑 결과 → assembler 입력 변환
# ─────────────────────────────────────────────

def mapping_to_clip_list(mapping: StoryboardMapping) -> List[ClipResult]:
    """StoryboardMapping 결과를 VideoAssembler.assemble()에 전달할
    ClipResult 리스트로 변환

    Scene Graph 순서가 유지된다.
    변환된 클립은 video_path가 transformed_video_path로 교체된다.

    Args:
        mapping: StoryboardMapper.map_scene_graph() 결과

    Returns:
        Scene Graph 순서대로 정렬된 ClipResult 리스트
    """
    clips = []
    for scene in mapping.scenes:
        if scene.selected_clip is None:
            logger.warning(f"장면 {scene.scene_id}: 클립 미할당 (생성 필요)")
            continue

        clip = scene.selected_clip

        # 변환된 영상이 있으면 video_path를 교체
        if scene.transformed_video_path and scene.branch == BranchDecision.TRANSFORM:
            # ClipResult는 dataclass이므로 새 인스턴스 생성
            from dataclasses import replace
            clip = replace(clip, video_path=scene.transformed_video_path)

        clips.append(clip)

    return clips
