"""
inverse_prompt_engine.py — 역방향 프롬프트 기반 속성 변환 엔진

PDF 설계:
  - 역방향 프롬프트 기반 시간대(낮/밤)·계절(봄/겨울) 자동 변환
  - 현재 상태 추론 → 목표 상태와의 차이 분석 → 변환 프롬프트 생성
  - 1차년도: 컬러 그레이딩, 기본 스타일 전이
  - 2차년도: 시간대/계절 완전 변환

동작 흐름:
  1. analyze_current_state(clip) → 현재 영상의 속성(시간대, 계절, 분위기) 추론
  2. compute_delta(current, target) → 변환에 필요한 차이 분석
  3. generate_transform_prompt(delta) → GPT-4o-mini 동적 프롬프트 생성 (Runway용)
  4. apply_transform(clip, prompt, backend) → 실제 변환 실행 (API 호출)

프롬프트 생성 전략:
  - 1순위: GPT-4o-mini가 현재 상태·목표 상태·delta를 보고 상황별 맞춤 프롬프트 생성
  - 2순위: 하드코딩 매핑 테이블 (GPT API 불가 시 폴백)
  - 3순위: 제네릭 템플릿 (매핑 테이블에도 없는 조합)

적용 대상:
  - Runway Gen-3 API: 이미지→영상 변환
  - 폴백: OpenCV 기반 컬러 그레이딩 (API 불가 시)
"""

import os
import json
import logging
import subprocess
import time
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import cv2

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 속성 정의
# ─────────────────────────────────────────────

class TimeOfDay(Enum):
    DAWN = "dawn"           # 새벽
    MORNING = "morning"     # 아침
    AFTERNOON = "afternoon" # 오후
    EVENING = "evening"     # 저녁
    NIGHT = "night"         # 밤

class Season(Enum):
    SPRING = "spring"
    SUMMER = "summer"
    AUTUMN = "autumn"
    WINTER = "winter"

class Mood(Enum):
    BRIGHT = "bright"       # 밝은
    WARM = "warm"           # 따뜻한
    COOL = "cool"           # 차가운
    DARK = "dark"           # 어두운
    DRAMATIC = "dramatic"   # 극적인
    NEUTRAL = "neutral"     # 중립

class SceneLocation(Enum):
    INDOOR = "indoor"
    OUTDOOR = "outdoor"
    UNKNOWN = "unknown"


@dataclass
class SceneAttributes:
    """영상 클립의 현재 속성 (자동 추론 결과)"""
    time_of_day: TimeOfDay = TimeOfDay.AFTERNOON
    season: Season = Season.SUMMER
    mood: Mood = Mood.NEUTRAL
    location: SceneLocation = SceneLocation.UNKNOWN
    avg_brightness: float = 128.0      # 0~255
    avg_saturation: float = 128.0      # 0~255
    color_temperature: float = 5500.0  # 켈빈 (낮은=따뜻, 높은=차가운)
    confidence: float = 0.5            # 추론 확신도 (0~1)


@dataclass
class TransformDelta:
    """현재 상태 → 목표 상태 간의 차이"""
    time_change: Optional[Tuple[TimeOfDay, TimeOfDay]] = None   # (현재, 목표)
    season_change: Optional[Tuple[Season, Season]] = None
    mood_change: Optional[Tuple[Mood, Mood]] = None
    is_indoor: bool = False            # 실내 씬 여부 (True면 시간대/계절 변환 제한)
    brightness_shift: float = 0.0      # 밝기 변화량 (-255 ~ +255)
    saturation_shift: float = 0.0      # 채도 변화량
    temperature_shift: float = 0.0     # 색온도 변화량


# ─────────────────────────────────────────────
# 속성 → 프롬프트 매핑 테이블
# ─────────────────────────────────────────────

TIME_PROMPT_MAP = {
    (TimeOfDay.AFTERNOON, TimeOfDay.NIGHT): (
        "Transform this daytime scene into a nighttime scene. "
        "Add moonlight, streetlights, and a dark blue sky. "
        "Reduce overall brightness and add warm artificial light sources."
    ),
    (TimeOfDay.NIGHT, TimeOfDay.AFTERNOON): (
        "Transform this nighttime scene into a bright afternoon. "
        "Replace artificial lights with natural sunlight. "
        "Brighten the sky to clear blue with warm natural tones."
    ),
    (TimeOfDay.AFTERNOON, TimeOfDay.EVENING): (
        "Transform this afternoon scene into golden hour/sunset. "
        "Add warm orange-pink tones, long shadows, and a setting sun."
    ),
    (TimeOfDay.MORNING, TimeOfDay.NIGHT): (
        "Transform this morning scene to nighttime. "
        "Replace sunrise with moonlight, darken the sky, add city lights."
    ),
    (TimeOfDay.DAWN, TimeOfDay.NIGHT): (
        "Transform this dawn scene into deep night. "
        "Remove sunrise colors, add starry sky and artificial lighting."
    ),
}

SEASON_PROMPT_MAP = {
    (Season.SUMMER, Season.WINTER): (
        "Transform this summer scene into winter. "
        "Add snow on the ground and rooftops. Make trees bare. "
        "Add breath vapor and cold blue tones."
    ),
    (Season.WINTER, Season.SPRING): (
        "Transform this winter scene into spring. "
        "Replace snow with green grass and blooming flowers. "
        "Add cherry blossoms and warm, fresh tones."
    ),
    (Season.SPRING, Season.AUTUMN): (
        "Transform this spring scene into autumn. "
        "Change green leaves to red, orange, and yellow. "
        "Add warm golden light and scattered fallen leaves."
    ),
    (Season.AUTUMN, Season.SUMMER): (
        "Transform this autumn scene into summer. "
        "Replace fall foliage with lush green leaves. "
        "Brighten the sky and add vivid, saturated colors."
    ),
}

# ─────────────────────────────────────────────
# 실내 씬 전용 분위기 변환 프롬프트 매핑
# 실내는 시간대/계절 변환이 무의미하므로 조명·분위기만 변환
# ─────────────────────────────────────────────

INDOOR_MOOD_PROMPT_MAP = {
    (Mood.BRIGHT, Mood.DARK): (
        "Dim the indoor lighting to create a dark, moody atmosphere. "
        "Replace bright overhead lights with low ambient lighting. "
        "Add shadows and reduce overall exposure."
    ),
    (Mood.BRIGHT, Mood.WARM): (
        "Change the indoor lighting from bright fluorescent to warm incandescent. "
        "Add golden tones and soft shadows for a cozy atmosphere."
    ),
    (Mood.NEUTRAL, Mood.WARM): (
        "Add warm candlelight or incandescent tones to this indoor scene. "
        "Shift the color balance toward golden-amber hues."
    ),
    (Mood.NEUTRAL, Mood.COOL): (
        "Shift the indoor lighting to cool blue-white tones. "
        "Add a clinical, modern atmosphere with higher color temperature."
    ),
    (Mood.NEUTRAL, Mood.DRAMATIC): (
        "Add dramatic chiaroscuro lighting to this indoor scene. "
        "Create strong contrast between highlights and deep shadows. "
        "Emphasize key subjects with directional spotlighting."
    ),
    (Mood.WARM, Mood.COOL): (
        "Replace warm indoor lighting with cool fluorescent or daylight tones. "
        "Remove golden hues and shift toward blue-white color balance."
    ),
    (Mood.DARK, Mood.BRIGHT): (
        "Brighten this dark indoor scene with overhead lighting. "
        "Add uniform illumination and raise overall exposure."
    ),
}


class InversePromptEngine:
    """역방향 프롬프트 기반 속성 변환 엔진

    PDF 핵심기술4의 속성 변환 엔진 구현.
    현재 상태를 추론하고, 목표와의 차이를 분석하여,
    Runway API에 전달할 변환 프롬프트를 생성한다.

    프롬프트 생성:
      GPT-4o-mini가 현재 상태·목표 상태·delta 정보를 바탕으로
      상황별 맞춤 변환 프롬프트를 동적으로 생성한다.
      GPT API 불가 시 하드코딩 매핑 테이블로 폴백한다.
    """

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        runway_api_key: Optional[str] = None,
        default_backend: str = "runway",  # "runway", "opencv"
        runway_model: str = "gen4_turbo",  # gen4_turbo(5cr/s), gen4(12cr/s), gen4.5(25cr/s)
    ):
        # 빈 문자열도 None으로 처리 (Colab userdata에서 빈 값이 올 수 있음)
        self.openai_api_key = (openai_api_key or "").strip() or os.environ.get("OPENAI_API_KEY", "").strip() or None
        # Runway SDK는 RUNWAYML_API_SECRET 환경변수를 사용
        self.runway_api_key = (
            (runway_api_key or "").strip()
            or os.environ.get("RUNWAYML_API_SECRET", "").strip()
            or os.environ.get("RUNWAY_API_KEY", "").strip()  # 하위 호환
            or None
        )
        self.default_backend = default_backend
        self.runway_model = runway_model
        self._runway_client = None

    # ─────────────────────────────────────────
    # Step 1: 현재 상태 추론
    # ─────────────────────────────────────────

    def analyze_current_state(self, frame: np.ndarray) -> SceneAttributes:
        """영상 프레임에서 현재 속성을 추론

        1) 실내/실외 분류 (에지 밀도 + 색상 분산 + 하늘 영역 비율)
        2) HSV 분석으로 밝기/채도/색온도 측정
        3) 실외면 시간대/계절 추론, 실내면 시간대/계절을 UNKNOWN 수준으로 두고 분위기만 추론

        Args:
            frame: BGR 이미지 (numpy array)

        Returns:
            SceneAttributes
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        avg_brightness = float(np.mean(v))
        avg_saturation = float(np.mean(s))
        avg_hue = float(np.mean(h))

        # ── 실내/실외 분류기 ──
        location = self._classify_indoor_outdoor(frame, hsv)

        # ── 색온도 추정 (Hue 기반, 실내·실외 공통) ──
        if avg_hue < 15 or avg_hue > 160:  # Red/Orange 영역
            color_temp = 3000.0  # 따뜻한
        elif avg_hue < 40:  # Yellow 영역
            color_temp = 4500.0
        elif avg_hue < 80:  # Green 영역
            color_temp = 5500.0  # 중립
        else:  # Blue 영역
            color_temp = 7000.0  # 차가운

        # ── 시간대 / 계절 추론 ──
        if location == SceneLocation.INDOOR:
            # 실내: 시간대/계절 추론이 무의미 → 중립 기본값
            time_of_day = TimeOfDay.AFTERNOON  # 기본값 (의미 없음)
            season = Season.SPRING             # 기본값 (의미 없음)
            confidence = 0.3                   # 실내는 시간대/계절 확신도 낮음
        else:
            # 실외: 기존 밝기 기반 추론
            if avg_brightness < 50:
                time_of_day = TimeOfDay.NIGHT
            elif avg_brightness < 80:
                time_of_day = TimeOfDay.EVENING
            elif avg_brightness < 120:
                time_of_day = TimeOfDay.DAWN
            elif avg_brightness < 170:
                time_of_day = TimeOfDay.AFTERNOON
            else:
                time_of_day = TimeOfDay.MORNING

            # 계절 추론 (색상 분포 + 채도)
            if avg_saturation < 60 and avg_brightness < 100:
                season = Season.WINTER
            elif avg_hue > 80 and avg_saturation > 100:
                season = Season.SUMMER
            elif 10 < avg_hue < 30 and avg_saturation > 80:
                season = Season.AUTUMN
            else:
                season = Season.SPRING
            confidence = 0.6

        # ── 분위기 추론 (실내·실외 공통) ──
        if avg_brightness > 160 and avg_saturation > 100:
            mood = Mood.BRIGHT
        elif color_temp < 4000:
            mood = Mood.WARM
        elif color_temp > 6500:
            mood = Mood.COOL
        elif avg_brightness < 60:
            mood = Mood.DARK
        else:
            mood = Mood.NEUTRAL

        return SceneAttributes(
            time_of_day=time_of_day,
            season=season,
            mood=mood,
            location=location,
            avg_brightness=avg_brightness,
            avg_saturation=avg_saturation,
            color_temperature=color_temp,
            confidence=confidence
        )

    def _classify_indoor_outdoor(
        self, frame: np.ndarray, hsv: np.ndarray
    ) -> SceneLocation:
        """실내/실외 분류기 (규칙 기반 3-signal 앙상블)

        3가지 신호를 종합하여 판단:
        1. 에지 밀도: 실내는 가구/벽/천장 등 직선 에지가 많고 밀도가 높음
        2. 색상 다양도: 실외는 하늘/자연 등으로 색상이 넓게 분포, 실내는 좁게 분포
        3. 상단 영역 하늘 비율: 프레임 상단 1/3에서 파란색·밝은 영역이 많으면 실외

        각 신호가 '실내' 방향이면 +1, '실외' 방향이면 -1, 합산으로 판정.

        Args:
            frame: BGR 이미지
            hsv: HSV 이미지 (사전 변환됨)

        Returns:
            SceneLocation.INDOOR or OUTDOOR
        """
        indoor_score = 0  # 양수=실내, 음수=실외

        h_full, s_full, v_full = cv2.split(hsv)
        height, width = frame.shape[:2]

        # ── Signal 1: 에지 밀도 ──
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = float(np.sum(edges > 0)) / (height * width)
        # 에지 밀도가 높으면 실내 가능성 (가구, 직선, 텍스트 등)
        if edge_density > 0.15:
            indoor_score += 1
        elif edge_density < 0.08:
            indoor_score -= 1

        # ── Signal 2: 색상 다양도 (Hue 히스토그램 엔트로피) ──
        h_hist = cv2.calcHist([h_full], [0], None, [30], [0, 180])
        h_hist = h_hist.flatten() / (h_hist.sum() + 1e-8)
        hue_entropy = -float(np.sum(h_hist[h_hist > 0] * np.log2(h_hist[h_hist > 0])))
        # 엔트로피 낮음 = 색상 분포 좁음 = 실내 경향
        if hue_entropy < 3.0:
            indoor_score += 1
        elif hue_entropy > 4.0:
            indoor_score -= 1

        # ── Signal 3: 상단 1/3 하늘 영역 비율 ──
        top_third = hsv[:height // 3, :, :]
        top_h, top_s, top_v = cv2.split(top_third)
        # 하늘: Hue 90~130 (파랑), 채도 30+, 밝기 120+
        sky_mask = (
            (top_h >= 90) & (top_h <= 130) &
            (top_s >= 30) & (top_v >= 120)
        )
        sky_ratio = float(np.sum(sky_mask)) / (top_third.shape[0] * top_third.shape[1])
        if sky_ratio > 0.20:
            indoor_score -= 2   # 하늘이 확실히 보이면 실외 (강한 신호)
        elif sky_ratio < 0.03:
            indoor_score += 1   # 하늘이 거의 없으면 실내 경향

        # ── 판정 ──
        if indoor_score >= 2:
            return SceneLocation.INDOOR
        elif indoor_score <= -1:
            return SceneLocation.OUTDOOR
        else:
            return SceneLocation.UNKNOWN

    # ─────────────────────────────────────────
    # Step 2: 차이 분석
    # ─────────────────────────────────────────

    def compute_delta(
        self,
        current: SceneAttributes,
        target_time: Optional[TimeOfDay] = None,
        target_season: Optional[Season] = None,
        target_mood: Optional[Mood] = None,
        target_location: Optional[SceneLocation] = None,
    ) -> TransformDelta:
        """현재 상태와 목표 상태 간의 차이를 계산

        target_location이 명시되면 프레임 분석 결과보다 우선 적용.
        (Scene Graph에서 PD가 outdoor로 지정했으면 분류기의 indoor 판정을 무시)

        실내 씬인 경우:
        - 시간대/계절 변환 요청을 무시하고 분위기(조명) 변환만 수행
        - 로그로 경고 메시지 출력

        Args:
            current: 현재 속성
            target_time: 목표 시간대 (None이면 변환 안함)
            target_season: 목표 계절
            target_mood: 목표 분위기

        Returns:
            TransformDelta
        """
        # Scene Graph의 target_location이 명시되면 분류기 결과를 덮어씀
        if target_location and target_location != SceneLocation.UNKNOWN:
            actual_location = target_location
            if actual_location != current.location:
                logger.info(
                    f"실내/실외 판정 오버라이드: 분류기={current.location.value} "
                    f"→ Scene Graph={actual_location.value}"
                )
        else:
            actual_location = current.location

        is_indoor = (actual_location == SceneLocation.INDOOR)
        delta = TransformDelta(is_indoor=is_indoor)

        if is_indoor:
            # 실내 씬: 시간대/계절 변환은 무의미 → 건너뜀
            if target_time:
                logger.info(
                    f"실내 씬 감지 — 시간대 변환 요청({target_time.value}) 무시. "
                    f"실내에서는 조명/분위기 변환만 적용합니다."
                )
            if target_season:
                logger.info(
                    f"실내 씬 감지 — 계절 변환 요청({target_season.value}) 무시. "
                    f"실내에서는 조명/분위기 변환만 적용합니다."
                )
            # 분위기만 변환 가능
            if target_mood and target_mood != current.mood:
                delta.mood_change = (current.mood, target_mood)
        else:
            # 실외 씬 또는 UNKNOWN: 기존 로직 유지
            if target_time and target_time != current.time_of_day:
                delta.time_change = (current.time_of_day, target_time)
                brightness_targets = {
                    TimeOfDay.NIGHT: 40, TimeOfDay.EVENING: 80,
                    TimeOfDay.DAWN: 100, TimeOfDay.MORNING: 180,
                    TimeOfDay.AFTERNOON: 150
                }
                delta.brightness_shift = (
                    brightness_targets.get(target_time, 128) - current.avg_brightness
                )

            if target_season and target_season != current.season:
                delta.season_change = (current.season, target_season)
                temp_targets = {
                    Season.WINTER: 7000, Season.SPRING: 5500,
                    Season.SUMMER: 5000, Season.AUTUMN: 4000
                }
                delta.temperature_shift = (
                    temp_targets.get(target_season, 5500) - current.color_temperature
                )

            if target_mood and target_mood != current.mood:
                delta.mood_change = (current.mood, target_mood)

        return delta

    # ─────────────────────────────────────────
    # Step 3: 변환 프롬프트 생성
    # ─────────────────────────────────────────

    def generate_transform_prompt(
        self,
        delta: TransformDelta,
        current_state: Optional[SceneAttributes] = None,
        scene_description: Optional[str] = None,
    ) -> str:
        """TransformDelta로부터 Runway용 변환 프롬프트 생성

        프롬프트 생성 우선순위:
          1순위: GPT-4o-mini 동적 생성 (상황별 맞춤 프롬프트)
          2순위: 하드코딩 매핑 테이블 (GPT API 불가 시)
          3순위: 제네릭 템플릿 (매핑 테이블에도 없는 조합)

        Args:
            delta: 변환 차이 정보
            current_state: 현재 영상 속성 (GPT 프롬프트 생성 시 참고)
            scene_description: 장면 설명 텍스트 (GPT 프롬프트에 컨텍스트 제공)
        Returns:
            영문 변환 프롬프트 (Runway API에 전달)
        """
        if not delta.time_change and not delta.season_change and not delta.mood_change:
            return "No transformation needed."

        # ── 1순위: GPT-4o-mini 동적 프롬프트 생성 ──
        gpt_prompt = self._generate_dynamic_prompt(delta, current_state, scene_description)
        if gpt_prompt:
            logger.info(f"GPT 동적 프롬프트 생성 ({'실내' if delta.is_indoor else '실외'}): {gpt_prompt[:100]}...")
            return gpt_prompt

        # ── 2순위: 하드코딩 매핑 테이블 폴백 ──
        logger.info("GPT 프롬프트 생성 실패, 하드코딩 매핑 테이블로 폴백")
        return self._generate_fallback_prompt(delta, scene_description)

    def _generate_dynamic_prompt(
        self,
        delta: TransformDelta,
        current_state: Optional[SceneAttributes] = None,
        scene_description: Optional[str] = None,
    ) -> Optional[str]:
        """GPT-4o-mini를 사용한 상황별 맞춤 변환 프롬프트 동적 생성

        현재 영상의 구체적 상태(밝기, 색온도, 실내/실외)와 목표 상태,
        장면 설명을 모두 고려하여 Runway API에 최적화된 프롬프트를 생성한다.

        Args:
            delta: 변환 차이 정보
            current_state: 현재 영상 속성
            scene_description: 장면 설명 텍스트

        Returns:
            생성된 프롬프트 (실패 시 None)
        """
        if not self.openai_api_key:
            return None

        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.openai_api_key)
        except (ImportError, Exception) as e:
            logger.warning(f"OpenAI 클라이언트 초기화 실패: {e}")
            return None

        # 변환 요약 구성
        changes = []
        if delta.time_change:
            changes.append(f"시간대: {delta.time_change[0].value} → {delta.time_change[1].value}")
        if delta.season_change:
            changes.append(f"계절: {delta.season_change[0].value} → {delta.season_change[1].value}")
        if delta.mood_change:
            changes.append(f"분위기: {delta.mood_change[0].value} → {delta.mood_change[1].value}")

        # 현재 상태 정보 구성
        state_info = ""
        if current_state:
            state_info = (
                f"\n현재 영상 분석 결과:\n"
                f"- 평균 밝기: {current_state.avg_brightness:.0f}/255\n"
                f"- 평균 채도: {current_state.avg_saturation:.0f}/255\n"
                f"- 색온도: {current_state.color_temperature:.0f}K\n"
                f"- 위치: {'실내' if current_state.location == SceneLocation.INDOOR else '실외'}\n"
            )

        scene_info = ""
        if scene_description:
            scene_info = f"\n장면 설명: {scene_description}\n"

        system_prompt = (
            "You are an expert video transformation prompt engineer for Runway Gen-4 Turbo.\n\n"
            "Your job: rewrite a scene to match the director's creative intent.\n"
            "You receive three pieces of information:\n"
            "  - SCENE INTENT: what the director wants this scene to express\n"
            "  - ATTRIBUTE CHANGES: specific visual changes needed (time, season, mood)\n"
            "  - CURRENT STATE: measured properties of the source video\n\n"
            "Rules:\n"
            "1. Start the prompt with the scene intent — describe the desired FINAL scene vividly.\n"
            "2. Then specify what to change: lighting, sky, color palette, atmosphere, temperature.\n"
            "3. Be cinematic — use film terminology (golden hour, chiaroscuro, cool blue wash, etc).\n"
            "4. Mention what to PRESERVE (subjects, composition, camera angle, motion).\n"
            "5. For indoor: only modify lighting/color grading, never sky/weather.\n"
            "6. Under 150 words. Output ONLY the prompt, no explanation."
        )

        user_prompt = (
            f"SCENE INTENT: {scene_description or 'N/A'}\n\n"
            f"ATTRIBUTE CHANGES:\n"
            f"  {chr(10).join(changes)}\n\n"
            f"LOCATION: {'indoor' if delta.is_indoor else 'outdoor'}\n"
            f"{state_info}\n"
            f"Brightness shift: {delta.brightness_shift:.0f}, "
            f"Temperature shift: {delta.temperature_shift:.0f}K\n"
        )

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=300,
                temperature=0.4,
            )
            prompt = response.choices[0].message.content.strip()
            logger.info(f"GPT 동적 프롬프트 생성 완료 ({len(prompt)}자)")
            return prompt
        except Exception as e:
            logger.warning(f"GPT 프롬프트 생성 실패: {e}")
            return None

    def _generate_fallback_prompt(
        self, delta: TransformDelta, scene_description: Optional[str] = None
    ) -> str:
        """하드코딩 매핑 테이블 기반 폴백 프롬프트 생성

        GPT API 불가 시 사용되는 2순위 프롬프트 생성 로직.
        scene_description을 앞에 배치하여 장면 의도를 포함한다.

        Args:
            delta: 변환 차이 정보
            scene_description: 장면 설명 (있으면 프롬프트 앞에 배치)

        Returns:
            영문 변환 프롬프트
        """
        parts = []

        # 장면 의도를 프롬프트 맨 앞에 배치
        if scene_description:
            parts.append(
                f"This scene depicts: {scene_description}. "
                f"Transform the visual style while preserving the scene content."
            )

        if delta.is_indoor:
            if delta.mood_change:
                key = delta.mood_change
                if key in INDOOR_MOOD_PROMPT_MAP:
                    parts.append(INDOOR_MOOD_PROMPT_MAP[key])
                else:
                    parts.append(
                        f"In this indoor scene, change the lighting mood "
                        f"from {key[0].value} to {key[1].value}. "
                        f"Adjust indoor lighting sources, color temperature, "
                        f"and ambient illumination accordingly."
                    )

            if not parts:
                return "No transformation needed."

            prompt = " ".join(parts)
            prompt += (
                " This is an indoor scene — do not modify sky, vegetation, or weather. "
                "Only adjust indoor lighting, color grading, and atmosphere. "
                "Maintain the original room layout, furniture, and subjects."
            )
        else:
            if delta.time_change:
                key = delta.time_change
                if key in TIME_PROMPT_MAP:
                    parts.append(TIME_PROMPT_MAP[key])
                else:
                    parts.append(
                        f"Transform the scene from {key[0].value} to {key[1].value}. "
                        f"Adjust lighting, sky color, and ambient atmosphere accordingly."
                    )

            if delta.season_change:
                key = delta.season_change
                if key in SEASON_PROMPT_MAP:
                    parts.append(SEASON_PROMPT_MAP[key])
                else:
                    parts.append(
                        f"Transform the scene from {key[0].value} to {key[1].value}. "
                        f"Change vegetation, weather, and color palette accordingly."
                    )

            if delta.mood_change:
                parts.append(
                    f"Change the overall mood from {delta.mood_change[0].value} "
                    f"to {delta.mood_change[1].value}. "
                    f"Adjust color grading, contrast, and lighting to match."
                )

            if not parts:
                return "No transformation needed."

            prompt = " ".join(parts)
            prompt += (
                " Maintain the original composition, subjects, and camera angle. "
                "Only modify lighting, colors, and atmospheric elements."
            )

        logger.info(f"폴백 프롬프트 생성 ({'실내' if delta.is_indoor else '실외'}): {prompt[:100]}...")
        return prompt

    # ─────────────────────────────────────────
    # Step 4: 변환 실행
    # ─────────────────────────────────────────

    def apply_transform(
        self,
        video_path: str,
        prompt: str,
        output_path: str,
        backend: Optional[str] = None
    ) -> Dict[str, Any]:
        """변환 프롬프트를 실제 적용

        Args:
            video_path: 원본 영상 경로
            prompt: 변환 프롬프트
            output_path: 출력 영상 경로
            backend: "runway", "tokenflow", "opencv" (None이면 default_backend)

        Returns:
            {"output_path": str, "backend": str, "quality_score": float, "success": bool}
        """
        backend = backend or self.default_backend

        if backend == "runway":
            return self._apply_runway(video_path, prompt, output_path)
        elif backend == "tokenflow":
            return self._apply_tokenflow(video_path, prompt, output_path)
        else:
            return self._apply_opencv_fallback(video_path, prompt, output_path)

    def _apply_tokenflow(
        self, video_path: str, prompt: str, output_path: str
    ) -> Dict[str, Any]:
        """TokenFlow 기반 video-to-video 변환

        extract_fps=8, keyframe_freq=10, n_timesteps=50, batch_size=8 고정.
        5초 클립 기준 약 30~60초 소요 (T4).
        """
        try:
            from .tokenflow_wrapper import TokenFlowEditor

            logger.info(f"[TokenFlow] 변환 시작: {video_path}")
            logger.info(f"[TokenFlow] 프롬프트: {prompt[:120]}")

            editor = TokenFlowEditor()
            result_path = editor.edit(
                video_path=video_path,
                prompt=prompt,
                output_path=output_path,
            )

            success = result_path == output_path and os.path.exists(output_path)
            if success:
                logger.info(f"[TokenFlow] 변환 완료: {output_path}")
                self._backup_to_drive(output_path, "transformed")
            else:
                logger.warning("[TokenFlow] 실패 — 원본 반환")

            return {
                "output_path": result_path,
                "backend": "tokenflow",
                "quality_score": 0.75 if success else 0.0,
                "success": success,
            }
        except Exception as e:
            logger.error(f"[TokenFlow] 예외: {e}")
            return {
                "output_path": video_path,
                "backend": "tokenflow_failed",
                "quality_score": 0.0,
                "success": False,
                "error": str(e),
            }

    def _get_runway_client(self):
        """Runway SDK 클라이언트 지연 초기화"""
        if self._runway_client is None:
            try:
                from runwayml import RunwayML
                # SDK는 RUNWAYML_API_SECRET 환경변수를 자동으로 읽지만,
                # 명시적으로 전달된 키가 있으면 환경변수에 설정
                if self.runway_api_key:
                    os.environ["RUNWAYML_API_SECRET"] = self.runway_api_key
                self._runway_client = RunwayML()
            except ImportError:
                raise RuntimeError(
                    "runwayml 패키지가 필요합니다: pip install runwayml"
                )
        return self._runway_client

    def _apply_runway(
        self, video_path: str, prompt: str, output_path: str
    ) -> Dict[str, Any]:
        """Runway SDK를 통한 영상 변환 (video-to-video)

        원본 영상의 모션/구도를 보존하면서 분위기/스타일만 변환.
        image_to_video와 달리 입력 영상 전체를 base64로 전달하므로
        촬영된 움직임이 그대로 유지됨.

        1. 입력 영상을 base64 data URI로 인코딩
        2. Runway SDK video_to_video.create() → 폴링으로 완료 대기
        3. 결과 영상 다운로드 → output_path + Drive 백업
        """
        try:
            import base64
            import urllib.request

            if not self.runway_api_key:
                raise ValueError("Runway API 키가 설정되지 않았습니다.")

            client = self._get_runway_client()

            # ── Step 1: 원본 영상 정보 확인 + base64 인코딩 ──
            logger.info(f"[Runway 1/4] 입력 영상 로드 중: {video_path}")
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

            src_duration_sec = total_frames / fps if fps > 0 else 5.0
            # Runway video_to_video 출력 길이: 5s 또는 10s 만 허용
            # 원본이 7초 이상이면 10s, 미만이면 5s
            out_duration = 10 if src_duration_sec >= 7.0 else 5

            with open(video_path, "rb") as f:
                video_bytes = f.read()
            video_b64 = base64.b64encode(video_bytes).decode()
            video_uri = f"data:video/mp4;base64,{video_b64}"
            logger.info(
                f"[Runway 1/4] 영상 인코딩 완료 "
                f"({w}x{h}, {src_duration_sec:.1f}s, {len(video_bytes)//1024}KB → {len(video_b64)//1024}KB base64)"
            )

            # ── Step 2: API 요청 전송 ──
            logger.info(
                f"[Runway 2/4] video_to_video 요청 전송 중... "
                f"(model={self.runway_model}, duration={out_duration}s, ratio=1280:720)"
            )
            logger.info(f"[Runway 2/4] 프롬프트: {prompt[:120]}")

            t0 = time.time()
            task = client.video_to_video.create(
                model=self.runway_model,
                prompt_video=video_uri,
                prompt_text=prompt,
                ratio="1280:720",
                duration=out_duration,
            )

            task_id = task.id
            logger.info(f"[Runway 2/4] 태스크 생성됨: task_id={task_id}")

            # ── Step 3: 완료 대기 (폴링 + 로그) ──
            logger.info(f"[Runway 3/4] 영상 생성 대기 중... (보통 30초~3분)")
            task = task.wait_for_task_output()
            elapsed = time.time() - t0
            logger.info(f"[Runway 3/4] 생성 완료! (소요: {elapsed:.0f}초, task_id={task_id})")

            # ── Step 4: 결과 다운로드 + 저장 ──
            download_url = task.output[0]
            logger.info(f"[Runway 4/4] 결과 다운로드: {download_url[:80]}...")

            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            urllib.request.urlretrieve(download_url, output_path)

            file_size = os.path.getsize(output_path) / 1024
            logger.info(f"[Runway 4/4] 저장 완료: {output_path} ({file_size:.0f}KB)")

            # Drive 백업
            self._backup_to_drive(output_path, "transformed")

            return {
                "output_path": output_path,
                "backend": "runway",
                "quality_score": 0.80,
                "success": True,
                "task_id": task_id,
                "elapsed_sec": elapsed,
            }
        except Exception as e:
            logger.error(f"[Runway] 변환 실패: {e}")
            return {
                "success": False,
                "output_path": None,
                "backend": "runway",
                "error": str(e),
                "quality_score": 0.0,
            }

    def _apply_opencv_fallback(
        self, video_path: str, prompt: str, output_path: str
    ) -> Dict[str, Any]:
        """OpenCV 기반 컬러 그레이딩 폴백 (1차년도 수준)

        프롬프트를 파싱하여 밝기/색온도/채도 조정으로 근사 변환.
        AI 생성 품질에는 미치지 못하지만, API 없이도 기본 변환 가능.
        """
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

            # 프롬프트에서 변환 파라미터 추출
            params = self._parse_prompt_to_params(prompt)

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # 밝기 조정 — alpha=1.0 고정, beta만 사용
                # alpha를 건드리면 대비까지 변해 암흑/화이트아웃이 됨
                if params['brightness'] != 0:
                    frame = cv2.convertScaleAbs(
                        frame,
                        alpha=1.0,
                        beta=params['brightness']
                    )

                # 색온도 조정 (간이 구현)
                if params['temperature'] != 0:
                    b, g, r = cv2.split(frame)
                    if params['temperature'] > 0:  # 따뜻하게
                        r = cv2.add(r, int(params['temperature'] * 0.3))
                        b = cv2.subtract(b, int(params['temperature'] * 0.2))
                    else:  # 차갑게
                        b = cv2.add(b, int(abs(params['temperature']) * 0.3))
                        r = cv2.subtract(r, int(abs(params['temperature']) * 0.2))
                    frame = cv2.merge([b, g, r])

                # 채도 조정
                if params['saturation'] != 1.0:
                    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
                    hsv[:, :, 1] *= params['saturation']
                    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
                    frame = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

                writer.write(frame)

            cap.release()
            writer.release()

            # mp4v 컨테이너는 브라우저/Gradio에서 재생 시간이 깨지는 경우가 있음
            # → ffmpeg으로 libx264 재인코딩하여 정상 mp4 생성
            import subprocess as _sp
            _tmp = output_path + ".raw.mp4"
            os.rename(output_path, _tmp)
            _r = _sp.run(
                ["ffmpeg", "-y", "-i", _tmp,
                 "-c:v", "libx264", "-preset", "ultrafast", "-an",
                 output_path, "-loglevel", "error"],
                capture_output=True,
            )
            os.remove(_tmp)
            if _r.returncode != 0:
                # ffmpeg 실패 시 raw 파일 복구
                os.rename(_tmp if os.path.exists(_tmp) else output_path, output_path)
                logger.warning(f"ffmpeg 재인코딩 실패 (returncode={_r.returncode}), raw mp4 사용")

            return {
                "output_path": output_path,
                "backend": "opencv",
                "quality_score": 0.45,
                "success": True,
                "params": params
            }
        except Exception as e:
            logger.error(f"OpenCV 폴백도 실패: {e}")
            return {
                "output_path": video_path,  # 원본 반환
                "backend": "opencv_failed",
                "quality_score": 0.0,
                "success": False,
                "error": f"OpenCV 변환 실패: {e}",
            }

    def _parse_prompt_to_params(self, prompt: str) -> Dict[str, float]:
        """프롬프트 텍스트에서 OpenCV 파라미터를 추출

        brightness: beta값 (픽셀 값 가산). alpha는 1.0 고정 — 암흑 방지.
        temperature: 양수=따뜻(R↑B↓), 음수=차갑(B↑R↓). ±20 이내 권장.
        saturation: 채도 배율. 0.8~1.2 범위 권장.
        """
        params = {'brightness': 0, 'temperature': 0, 'saturation': 1.0}
        prompt_lower = prompt.lower()

        # 밝기 — 극단적 값 금지: -60이면 convertScaleAbs alpha 효과와 합쳐져 암흑
        if 'nighttime' in prompt_lower or 'night' in prompt_lower:
            params['brightness'] = -25   # 어둑하게, 암흑 아님
        elif 'bright' in prompt_lower or 'morning' in prompt_lower:
            params['brightness'] = 15
        elif 'sunset' in prompt_lower or 'golden hour' in prompt_lower:
            params['brightness'] = -10

        # 색온도 — ±20 범위로 제한
        if 'warm' in prompt_lower or 'sunset' in prompt_lower or 'golden' in prompt_lower:
            params['temperature'] = 20
        elif 'cold' in prompt_lower or 'winter' in prompt_lower:
            params['temperature'] = -20
        elif 'cool' in prompt_lower or 'night' in prompt_lower:
            params['temperature'] = -15

        # 채도
        if 'winter' in prompt_lower or 'snow' in prompt_lower:
            params['saturation'] = 0.85
        elif 'summer' in prompt_lower or 'vivid' in prompt_lower:
            params['saturation'] = 1.15
        elif 'autumn' in prompt_lower:
            params['saturation'] = 1.1

        return params

    # ─────────────────────────────────────────
    # 통합 인터페이스
    # ─────────────────────────────────────────

    def transform_clip(
        self,
        video_path: str,
        output_path: str,
        target_time: Optional[TimeOfDay] = None,
        target_season: Optional[Season] = None,
        target_mood: Optional[Mood] = None,
        backend: Optional[str] = None,
        scene_description: Optional[str] = None,
    ) -> Dict[str, Any]:
        """클립 속성 변환 E2E 파이프라인

        1. 현재 상태 추론
        2. 목표와의 차이 계산
        3. GPT 동적 프롬프트 생성 (폴백: 하드코딩 매핑 테이블)
        4. 변환 실행 (Runway → OpenCV 폴백 체인)

        Args:
            video_path: 원본 영상
            output_path: 출력 영상
            target_time: 목표 시간대
            target_season: 목표 계절
            target_mood: 목표 분위기
            backend: 변환 백엔드 (None이면 default_backend)
            scene_description: 장면 설명 텍스트 (GPT 프롬프트 컨텍스트)

        Returns:
            변환 결과 딕셔너리
        """
        # Step 1: 현재 상태 추론 (첫 프레임 기반)
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            return {"success": False, "error": "프레임 추출 실패"}

        current = self.analyze_current_state(frame)
        logger.info(
            f"현재 상태: time={current.time_of_day.value}, "
            f"season={current.season.value}, mood={current.mood.value}, "
            f"brightness={current.avg_brightness:.0f}"
        )

        # Step 2: 차이 분석
        delta = self.compute_delta(current, target_time, target_season, target_mood)

        # 변환 필요 없으면 원본 복사
        if not delta.time_change and not delta.season_change and not delta.mood_change:
            import shutil
            shutil.copy2(video_path, output_path)
            return {
                "output_path": output_path,
                "backend": "none",
                "quality_score": 1.0,
                "success": True,
                "message": "변환 불필요 (현재 상태 = 목표 상태)"
            }

        # Step 3: 프롬프트 생성 (GPT 동적 → 하드코딩 폴백)
        prompt = self.generate_transform_prompt(
            delta,
            current_state=current,
            scene_description=scene_description,
        )

        # Step 4: 변환 실행
        result = self.apply_transform(video_path, prompt, output_path, backend)
        result["current_state"] = {
            "time": current.time_of_day.value,
            "season": current.season.value,
            "mood": current.mood.value
        }
        result["prompt"] = prompt

        return result

    # ─────────────────────────────────────────
    # 신규 영상 생성 (GENERATE 분기용)
    # ─────────────────────────────────────────

    def generate_video(
        self,
        prompt: str,
        duration_sec: float = 5.0,
        output_path: Optional[str] = None,
        backend: Optional[str] = None,
    ) -> Dict[str, Any]:
        """프롬프트로부터 신규 영상 생성 (GENERATE 분기)

        StoryboardMapper의 GENERATE 판정 시 호출되어
        아카이브에 없는 영상을 Runway video-to-video로 새로 생성한다.

        참고: TokenFlow는 기존 영상을 입력으로 받는 video-to-video 모델이므로
        신규 생성을 지원하지 않는다. 기존 클립 변환은 apply_transform() 사용.

        Args:
            prompt: 영상 생성 프롬프트 (영어)
            duration_sec: 생성할 영상 길이 (초)
            output_path: 출력 영상 경로 (None이면 자동 생성)
            backend: "runway" (None이면 default_backend)

        Returns:
            {"success": bool, "output_path": str, "backend": str, ...}
        """
        backend = backend or self.default_backend
        if output_path is None:
            output_path = os.path.join("output/generated", f"gen_{int(time.time())}.mp4")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        if backend == "tokenflow":
            logger.warning(
                "generate_video: TokenFlow는 신규 생성 불가 (video-to-video 전용). "
                "기존 클립 변환은 apply_transform()을 사용하세요."
            )
            return {
                "success": False,
                "output_path": None,
                "backend": "tokenflow",
                "error": "TokenFlow는 신규 영상 생성 불가 — 기존 클립 변환은 apply_transform() 사용",
            }

        if backend == "runway":
            return self._generate_runway(prompt, duration_sec, output_path)

        logger.warning(f"generate_video: '{backend}' 백엔드로는 신규 생성 불가")
        return {
            "success": False,
            "output_path": None,
            "backend": backend,
            "error": f"'{backend}' 백엔드는 신규 영상 생성을 지원하지 않습니다. 'runway'를 사용하세요.",
        }

    def _generate_runway(
        self, prompt: str, duration_sec: float, output_path: str,
    ) -> Dict[str, Any]:
        """Runway SDK로 신규 영상 생성 (video-to-video text-to-video 모드)"""
        try:
            import urllib.request

            if not self.runway_api_key:
                raise ValueError("Runway API 키가 설정되지 않았습니다.")

            client = self._get_runway_client()
            dur = min(int(duration_sec), 10)

            logger.info(
                f"[Runway-Gen 1/3] 신규 생성 시작 "
                f"(model={self.runway_model}, duration={dur}s)"
            )
            logger.info(f"[Runway-Gen 1/3] 프롬프트: {prompt[:120]}")

            kwargs = {
                "model": self.runway_model,
                "prompt_text": prompt,
                "ratio": "1280:720",
                "duration": dur,
            }

            logger.info(f"[Runway-Gen 2/3] API 요청 전송...")
            t0 = time.time()
            task = client.image_to_video.create(**kwargs)
            task_id = task.id
            logger.info(f"[Runway-Gen 2/3] 태스크 생성됨: task_id={task_id}")

            logger.info(f"[Runway-Gen 2/3] 영상 생성 대기 중...")
            task = task.wait_for_task_output()
            elapsed = time.time() - t0
            logger.info(f"[Runway-Gen 2/3] 생성 완료! (소요: {elapsed:.0f}초)")

            download_url = task.output[0]
            logger.info(f"[Runway-Gen 3/3] 다운로드: {download_url[:80]}...")
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            urllib.request.urlretrieve(download_url, output_path)

            file_size = os.path.getsize(output_path) / 1024
            logger.info(f"[Runway-Gen 3/3] 저장 완료: {output_path} ({file_size:.0f}KB)")

            # Drive 백업
            self._backup_to_drive(output_path, "generated")

            return {
                "success": True,
                "output_path": output_path,
                "backend": "runway",
                "task_id": task_id,
                "prompt": prompt,
                "elapsed_sec": elapsed,
            }
        except Exception as e:
            logger.error(f"[Runway-Gen] 생성 실패: {e}")
            return {
                "success": False,
                "output_path": None,
                "backend": "runway",
                "error": str(e),
            }

    # ─────────────────────────────────────────
    # Drive 백업 (Runway 생성물 보존)
    # ─────────────────────────────────────────

    def _backup_to_drive(self, local_path: str, category: str = "transformed"):
        """Runway 생성 영상을 Google Drive에 자동 백업

        Args:
            local_path: 로컬 영상 파일 경로
            category: "transformed" 또는 "generated"
        """
        import shutil

        drive_base = "/content/drive/MyDrive/videorag_prototype/output/runway"
        drive_dir = os.path.join(drive_base, category)

        try:
            os.makedirs(drive_dir, exist_ok=True)
            fname = os.path.basename(local_path)
            drive_path = os.path.join(drive_dir, fname)
            shutil.copy2(local_path, drive_path)
            logger.info(f"[Drive 백업] {drive_path}")
        except Exception as e:
            logger.warning(f"[Drive 백업 실패] {e} (로컬 파일은 유지됨: {local_path})")

    # ─────────────────────────────────────────
    # Preview 모드 (Phase 3 — PD 승인 워크플로)
    # ─────────────────────────────────────────

    def preview_transform(
        self,
        video_path: str,
        target_time: Optional[TimeOfDay] = None,
        target_season: Optional[Season] = None,
        target_mood: Optional[Mood] = None,
        output_dir: Optional[str] = None,
        backend: Optional[str] = None,
    ) -> Dict[str, Any]:
        """키프레임 1장만 변환하여 Before/After 미리보기 반환

        뉴스/다큐에서 속성 변환(낮→밤, 봄→겨울)은 사실 왜곡 논란
        위험이 있으므로, 실제 변환 전에 PD가 결과를 확인하고
        승인하는 미리보기 워크플로를 제공한다.

        Args:
            video_path: 원본 영상 경로
            target_time: 목표 시간대
            target_season: 목표 계절
            target_mood: 목표 분위기
            output_dir: 미리보기 이미지 저장 디렉토리 (None이면 output/preview)
            backend: 변환 백엔드 (preview는 opencv만 사용)

        Returns:
            {
                "success": bool,
                "before_path": str,        # 원본 키프레임 이미지 경로
                "after_path": str,         # 변환된 키프레임 이미지 경로
                "current_state": dict,     # 현재 속성
                "target_state": dict,      # 목표 속성
                "delta_summary": str,      # 변환 요약 텍스트
                "prompt": str,             # 생성된 변환 프롬프트
                "needs_approval": bool,    # PD 승인 필요 여부
                "warning": str,            # 사실 왜곡 경고 (해당 시)
            }
        """
        import os

        if output_dir is None:
            output_dir = "output/preview"
        os.makedirs(output_dir, exist_ok=True)

        # Step 1: 키프레임 추출
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            return {"success": False, "error": "프레임 추출 실패"}

        # Step 2: 현재 상태 추론
        current = self.analyze_current_state(frame)

        # Step 3: 차이 분석
        delta = self.compute_delta(current, target_time, target_season, target_mood)

        # 변환 불필요 시
        if not delta.time_change and not delta.season_change and not delta.mood_change:
            before_path = os.path.join(output_dir, "preview_before.jpg")
            cv2.imwrite(before_path, frame)
            return {
                "success": True,
                "before_path": before_path,
                "after_path": before_path,
                "current_state": {
                    "time": current.time_of_day.value,
                    "season": current.season.value,
                    "mood": current.mood.value,
                    "location": current.location.value,
                },
                "target_state": {
                    "time": (target_time.value if target_time else current.time_of_day.value),
                    "season": (target_season.value if target_season else current.season.value),
                    "mood": (target_mood.value if target_mood else current.mood.value),
                },
                "delta_summary": "변환 불필요 (현재 상태 = 목표 상태)",
                "prompt": "No transformation needed.",
                "needs_approval": False,
                "warning": "",
            }

        # Step 4: 프롬프트 생성 (GPT 동적 → 하드코딩 폴백)
        prompt = self.generate_transform_prompt(
            delta, current_state=current
        )

        # Step 5: Before 이미지 저장
        before_path = os.path.join(output_dir, "preview_before.jpg")
        cv2.imwrite(before_path, frame)

        # Step 6: 키프레임 1장만 변환 (OpenCV 폴백으로 빠르게)
        after_path = os.path.join(output_dir, "preview_after.jpg")
        transformed_frame = self._transform_single_frame(frame, prompt)
        cv2.imwrite(after_path, transformed_frame)

        # Step 7: 경고 메시지 생성
        warning = self._generate_warning(delta, current)
        needs_approval = bool(delta.time_change or delta.season_change)

        # 변환 요약
        delta_parts = []
        if delta.time_change:
            delta_parts.append(
                f"시간대: {delta.time_change[0].value} → {delta.time_change[1].value}"
            )
        if delta.season_change:
            delta_parts.append(
                f"계절: {delta.season_change[0].value} → {delta.season_change[1].value}"
            )
        if delta.mood_change:
            delta_parts.append(
                f"분위기: {delta.mood_change[0].value} → {delta.mood_change[1].value}"
            )
        delta_summary = " | ".join(delta_parts) if delta_parts else "변환 없음"

        logger.info(
            f"Preview 생성 완료: {delta_summary} "
            f"(needs_approval={needs_approval})"
        )

        return {
            "success": True,
            "before_path": before_path,
            "after_path": after_path,
            "current_state": {
                "time": current.time_of_day.value,
                "season": current.season.value,
                "mood": current.mood.value,
                "location": current.location.value,
            },
            "target_state": {
                "time": (target_time.value if target_time else current.time_of_day.value),
                "season": (target_season.value if target_season else current.season.value),
                "mood": (target_mood.value if target_mood else current.mood.value),
            },
            "delta_summary": delta_summary,
            "prompt": prompt,
            "needs_approval": needs_approval,
            "warning": warning,
        }

    def _transform_single_frame(self, frame: np.ndarray, prompt: str) -> np.ndarray:
        """단일 프레임에 OpenCV 기반 변환 적용 (미리보기용)

        Args:
            frame: 원본 프레임 (BGR)
            prompt: 변환 프롬프트

        Returns:
            변환된 프레임 (BGR)
        """
        params = self._parse_prompt_to_params(prompt)
        result = frame.copy()

        # 밝기 조정
        if params['brightness'] != 0:
            result = cv2.convertScaleAbs(
                result,
                alpha=1.0 + params['brightness'] / 255.0,
                beta=params['brightness'] * 0.5
            )

        # 색온도 조정
        if params['temperature'] != 0:
            b, g, r = cv2.split(result)
            if params['temperature'] > 0:  # 따뜻하게
                r = cv2.add(r, int(params['temperature'] * 0.3))
                b = cv2.subtract(b, int(params['temperature'] * 0.2))
            else:  # 차갑게
                b = cv2.add(b, int(abs(params['temperature']) * 0.3))
                r = cv2.subtract(r, int(abs(params['temperature']) * 0.2))
            result = cv2.merge([b, g, r])

        # 채도 조정
        if params['saturation'] != 1.0:
            hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 1] *= params['saturation']
            hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
            result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        return result

    def _generate_warning(self, delta: 'TransformDelta', current: SceneAttributes) -> str:
        """사실 왜곡 경고 메시지 생성

        뉴스/다큐 영상에서 시간대·계절 변환은 사실 왜곡으로
        이어질 수 있으므로 PD에게 경고를 표시한다.

        Args:
            delta: 변환 차이 정보
            current: 현재 속성

        Returns:
            경고 메시지 (빈 문자열이면 경고 없음)
        """
        warnings = []

        if delta.time_change:
            warnings.append(
                f"⚠️ 시간대 변환 ({delta.time_change[0].value} → {delta.time_change[1].value}): "
                f"뉴스/다큐에서는 사실 왜곡으로 인식될 수 있습니다."
            )

        if delta.season_change:
            warnings.append(
                f"⚠️ 계절 변환 ({delta.season_change[0].value} → {delta.season_change[1].value}): "
                f"실제 촬영 시기와 다른 계절로 표현될 수 있습니다."
            )

        if delta.is_indoor and (delta.time_change or delta.season_change):
            warnings.append(
                "ℹ️ 실내 씬에서는 시간대/계절 변환이 제한됩니다. "
                "조명/분위기 변환만 적용됩니다."
            )

        return " ".join(warnings)
