"""
script_parser.py — 대본/큐시트 → Scene Graph JSON 자동 파싱

Phase 3 구현:
  PD가 텍스트 대본(한국어/영어)을 입력하면
  GPT-4o-mini를 통해 Scene Graph JSON으로 자동 변환.
  StoryboardMapper가 각 장면 description을 쿼리로 사용하여
  pipeline.search_only() 호출.

동작 흐름:
  텍스트 대본 → GPT-4o-mini → Scene Graph JSON → StoryboardMapper

Scene Graph 출력 스키마:
  {
    "title": "기후변화와 한국 어촌",
    "scenes": [
      {
        "scene_id": 1,
        "description": "Fishing boats anchored at dawn in a harbor",
        "description_ko": "새벽녘 항구에 정박된 어선들",
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

import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 데이터 모델
# ─────────────────────────────────────────────

@dataclass
class ParsedScene:
    """파싱된 개별 장면"""
    scene_id: int
    description: str           # 영어 설명 (검색 쿼리로 사용)
    description_ko: str        # 한국어 설명 (UI 표시용)
    duration_sec: float = 5.0
    attributes: Dict[str, str] = field(default_factory=dict)


@dataclass
class ScriptParseResult:
    """대본 파싱 결과"""
    title: str
    scenes: List[ParsedScene]
    original_text: str         # 원본 대본 텍스트
    scene_graph_json: Dict     # StoryboardMapper 호환 JSON
    parse_model: str = "gpt-4o-mini"
    token_usage: Optional[Dict] = None

    @property
    def scene_count(self) -> int:
        return len(self.scenes)

    @property
    def total_duration_sec(self) -> float:
        return sum(s.duration_sec for s in self.scenes)


# ─────────────────────────────────────────────
# 프롬프트 템플릿
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """당신은 한국 방송사의 베테랑 PD입니다.
주어진 대본/큐시트 텍스트를 분석하여, 각 장면(Scene)을 구조화된 Scene Graph JSON으로 변환합니다.

규칙:
1. 각 장면은 하나의 영상 클립에 대응합니다.
2. description은 반드시 영어로 작성합니다 (영상 검색에 사용).
3. description_ko는 한국어 원문 또는 한국어 설명입니다.
4. duration_sec는 장면의 예상 길이(초)입니다 (기본 5초).
5. attributes에는 time_of_day, season, mood, location을 포함합니다.
6. 대본이 한국어이면 description은 영어로 번역하고, description_ko에 원문을 보존합니다.
7. 대본이 영어이면 description에 원문을 넣고, description_ko에 한국어 번역을 넣습니다.

반드시 아래 JSON 형식으로만 응답하세요. 다른 텍스트는 포함하지 마세요."""

USER_PROMPT_TEMPLATE = """다음 대본/큐시트를 Scene Graph JSON으로 변환하세요:

---
{script_text}
---

출력 형식:
{{
  "title": "프로그램/영상 제목",
  "scenes": [
    {{
      "scene_id": 1,
      "description": "영어 장면 설명 (15~40 words, 구체적으로)",
      "description_ko": "한국어 장면 설명",
      "duration_sec": 5.0,
      "attributes": {{
        "time_of_day": "dawn|morning|afternoon|evening|night",
        "season": "spring|summer|autumn|winter",
        "mood": "bright|warm|cool|dark|dramatic|neutral",
        "location": "indoor|outdoor"
      }}
    }}
  ]
}}

주의:
- 장면 수는 대본 내용에 맞게 자연스럽게 나누세요 (보통 3~15개).
- description은 영상 검색에 사용되므로 시각적 요소를 구체적으로 기술하세요.
- attributes의 각 필드는 제시된 선택지 중 하나만 사용하세요."""


class ScriptParser:
    """대본/큐시트 → Scene Graph JSON 자동 파싱

    GPT-4o-mini를 사용하여 텍스트 대본을 구조화된
    Scene Graph JSON으로 변환한다.
    StoryboardMapper와 연동하여 장면별 클립 자동 검색에 사용.

    사용법:
        parser = ScriptParser(api_key="sk-...")
        result = parser.parse("오프닝: 서울 도심 전경, 한강 야경으로 시작...")
        scene_graph = result.scene_graph_json
        # → StoryboardMapper.map_scene_graph(scene_graph)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        max_scenes: int = 20,
        default_duration_sec: float = 5.0,
    ):
        """
        Args:
            api_key: OpenAI API key (None이면 OPENAI_API_KEY 환경변수 사용)
            model: 사용할 모델 (기본 gpt-4o-mini)
            max_scenes: 최대 장면 수 (초과 시 잘라냄)
            default_duration_sec: 장면 기본 길이 (초)
        """
        import os
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model
        self.max_scenes = max_scenes
        self.default_duration_sec = default_duration_sec

    def parse(self, script_text: str, title_hint: Optional[str] = None) -> ScriptParseResult:
        """대본 텍스트 → Scene Graph JSON 변환

        Args:
            script_text: 대본/큐시트 텍스트 (한국어 또는 영어)
            title_hint: 제목 힌트 (None이면 GPT가 자동 추론)

        Returns:
            ScriptParseResult
        """
        if not script_text.strip():
            raise ValueError("대본 텍스트가 비어있습니다.")

        # GPT-4o-mini API 호출
        scene_graph, token_usage = self._call_gpt(script_text, title_hint)

        # 파싱 결과 구조화
        scenes = self._parse_scenes(scene_graph)

        # 제목 결정
        title = scene_graph.get("title", title_hint or "Untitled")

        # scene_graph_json (StoryboardMapper 호환)
        scene_graph_json = {
            "title": title,
            "scenes": [
                {
                    "scene_id": s.scene_id,
                    "description": s.description,
                    "duration_sec": s.duration_sec,
                    "attributes": s.attributes,
                }
                for s in scenes
            ]
        }

        result = ScriptParseResult(
            title=title,
            scenes=scenes,
            original_text=script_text,
            scene_graph_json=scene_graph_json,
            parse_model=self.model,
            token_usage=token_usage,
        )

        logger.info(
            f"대본 파싱 완료: '{title}' — {result.scene_count}개 장면, "
            f"총 {result.total_duration_sec:.1f}초"
        )

        return result

    def parse_from_file(self, file_path: str, title_hint: Optional[str] = None) -> ScriptParseResult:
        """파일에서 대본 읽어 파싱

        Args:
            file_path: 대본 파일 경로 (.txt, .md 등)
            title_hint: 제목 힌트

        Returns:
            ScriptParseResult
        """
        with open(file_path, "r", encoding="utf-8") as f:
            script_text = f.read()
        return self.parse(script_text, title_hint)

    def _call_gpt(
        self, script_text: str, title_hint: Optional[str] = None
    ) -> tuple:
        """GPT-4o-mini API 호출

        Args:
            script_text: 대본 텍스트
            title_hint: 제목 힌트

        Returns:
            (scene_graph_dict, token_usage_dict)
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai 패키지가 필요합니다: pip install openai")

        client = OpenAI(api_key=self.api_key)

        user_prompt = USER_PROMPT_TEMPLATE.format(script_text=script_text)
        if title_hint:
            user_prompt += f"\n\n제목 힌트: {title_hint}"

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        logger.info(f"GPT-4o-mini 호출 시작 (대본 길이: {len(script_text)}자)")

        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            response_format={"type": "json_object"},
            max_tokens=4096,
            temperature=0.3,  # 안정적인 구조화 출력을 위해 낮은 temperature
        )

        content = response.choices[0].message.content
        token_usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }

        logger.info(
            f"GPT-4o-mini 응답 완료: "
            f"prompt={token_usage['prompt_tokens']}, "
            f"completion={token_usage['completion_tokens']}"
        )

        try:
            scene_graph = json.loads(content)
        except json.JSONDecodeError as e:
            logger.error(f"JSON 파싱 실패: {e}\n응답 내용: {content[:500]}")
            raise ValueError(f"GPT 응답이 유효한 JSON이 아닙니다: {e}")

        return scene_graph, token_usage

    def _parse_scenes(self, scene_graph: Dict) -> List[ParsedScene]:
        """Scene Graph JSON에서 장면 리스트 파싱

        Args:
            scene_graph: GPT 응답 JSON dict

        Returns:
            List[ParsedScene]
        """
        scenes_data = scene_graph.get("scenes", [])
        if not scenes_data:
            logger.warning("GPT 응답에 장면이 없습니다.")
            return []

        scenes = []
        for i, scene_data in enumerate(scenes_data[:self.max_scenes]):
            attrs = scene_data.get("attributes", {})

            # 속성 값 검증 및 기본값 설정
            validated_attrs = self._validate_attributes(attrs)

            scene = ParsedScene(
                scene_id=scene_data.get("scene_id", i + 1),
                description=scene_data.get("description", ""),
                description_ko=scene_data.get("description_ko", ""),
                duration_sec=scene_data.get("duration_sec", self.default_duration_sec),
                attributes=validated_attrs,
            )
            scenes.append(scene)

        return scenes

    def _validate_attributes(self, attrs: Dict[str, str]) -> Dict[str, str]:
        """속성 값 검증 및 정규화

        유효하지 않은 값은 기본값으로 대체

        Args:
            attrs: 원본 속성 딕셔너리

        Returns:
            검증된 속성 딕셔너리
        """
        valid_values = {
            "time_of_day": {"dawn", "morning", "afternoon", "evening", "night"},
            "season": {"spring", "summer", "autumn", "winter"},
            "mood": {"bright", "warm", "cool", "dark", "dramatic", "neutral"},
            "location": {"indoor", "outdoor"},
        }

        defaults = {
            "time_of_day": "afternoon",
            "season": "spring",
            "mood": "neutral",
            "location": "outdoor",
        }

        validated = {}
        for key, valid_set in valid_values.items():
            value = attrs.get(key, "").lower().strip()
            if value in valid_set:
                validated[key] = value
            else:
                validated[key] = defaults[key]
                if value:
                    logger.warning(
                        f"속성 '{key}' 값 '{value}'이(가) 유효하지 않습니다. "
                        f"기본값 '{defaults[key]}' 사용."
                    )

        return validated

    # ─────────────────────────────────────────────
    # 오프라인 모드: GPT 없이 규칙 기반 파싱
    # ─────────────────────────────────────────────

    def parse_simple(self, script_text: str, title: str = "Untitled") -> ScriptParseResult:
        """간단한 규칙 기반 파싱 (GPT 없이 동작)

        줄바꿈으로 장면을 나누고, 각 줄을 하나의 장면으로 처리.
        GPT API가 없는 환경에서 테스트/데모용으로 사용.

        규칙:
        - 빈 줄로 구분된 각 단락을 하나의 장면으로 처리
        - 번호가 붙어있으면 scene_id로 사용
        - 'scene', '장면', '#' 등의 키워드로 장면 구분

        Args:
            script_text: 대본 텍스트
            title: 제목

        Returns:
            ScriptParseResult
        """
        import re

        lines = script_text.strip().split("\n")
        scenes = []
        current_scene_text = []
        scene_id = 0

        for line in lines:
            line = line.strip()
            if not line:
                # 빈 줄 → 장면 구분
                if current_scene_text:
                    scene_id += 1
                    desc = " ".join(current_scene_text)
                    scenes.append(ParsedScene(
                        scene_id=scene_id,
                        description=desc,
                        description_ko=desc,
                        duration_sec=self.default_duration_sec,
                        attributes={
                            "time_of_day": "afternoon",
                            "season": "spring",
                            "mood": "neutral",
                            "location": "outdoor",
                        },
                    ))
                    current_scene_text = []
            else:
                # 번호 제거 (예: "1.", "1)", "Scene 1:")
                cleaned = re.sub(r'^(\d+[\.\)]\s*|[Ss]cene\s*\d+[:\s]*|장면\s*\d+[:\s]*|#\s*)', '', line)
                if cleaned.strip():
                    current_scene_text.append(cleaned.strip())

        # 마지막 장면 처리
        if current_scene_text:
            scene_id += 1
            desc = " ".join(current_scene_text)
            scenes.append(ParsedScene(
                scene_id=scene_id,
                description=desc,
                description_ko=desc,
                duration_sec=self.default_duration_sec,
                attributes={
                    "time_of_day": "afternoon",
                    "season": "spring",
                    "mood": "neutral",
                    "location": "outdoor",
                },
            ))

        scene_graph_json = {
            "title": title,
            "scenes": [
                {
                    "scene_id": s.scene_id,
                    "description": s.description,
                    "duration_sec": s.duration_sec,
                    "attributes": s.attributes,
                }
                for s in scenes
            ]
        }

        return ScriptParseResult(
            title=title,
            scenes=scenes,
            original_text=script_text,
            scene_graph_json=scene_graph_json,
            parse_model="rule_based",
        )
