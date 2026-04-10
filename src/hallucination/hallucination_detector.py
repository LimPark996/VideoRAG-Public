"""
hallucination_detector.py — 환각 탐지 및 자동 보정 파이프라인

PDF 설계:
  이중 환각 검증 체계:
    Track A (외부 논리 검증): TC-Bench 어설션 검증 + WCS 물리적 일관성
    Track B (내부 상태 분석): PRISM + Entropy

  3단계 판정 체계:
    환각 점수 ≤ 0.15 → 통과 (바로 사용)
    0.15 ~ 0.40 → 자동 보정 시도 → 재검증
    ≥ 0.40 → 재생성 요청 (세부2 연동)

  자동 보정 방법:
    1. 아카이브 클립 교체: 환각 클립을 동일 쿼리로 검색한 실제 클립으로 교체
    2. 부분 재생성 요청: 문제 구간만 세부2에 재생성 요청

구현 상태 (1차년도):
  PRISM: AI 모델 내부 상태 분석 기반 환각 탐지 (정확도 75~77% 목표)
    - 캡션-쿼리 의미 유사도 (sentence-transformers)
    - 모델 확신도 기반 엔트로피 측정
    - 생성 영상에 대한 가중치 적용
  WCS: 물리적 일관성 검증 (4개 지표)
    - 객체 영속성(OP): 프레임 간 객체 추적 (ORB 특징점 매칭)
    - 상대 크기(RS): 객체 바운딩 박스 크기 비율 일관성
    - 색상 일관성(CC): 프레임 간 색상 히스토그램 상관관계
    - 프레임 물리(FP): 옵티컬 플로우 기반 물리 법칙 위반 탐지
  2차년도: 다중 VLM 앙상블(InternVideo2 + GPT-4V) 교차 검증 추가
"""

import logging
import random
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 판정 결과 정의
# ─────────────────────────────────────────────

class HallucinationVerdict(Enum):
    """3단계 판정"""
    PASS = "pass"                 # ≤ 0.15: 통과
    AUTO_CORRECT = "auto_correct" # 0.15 ~ 0.40: 자동 보정 시도
    REGENERATE = "regenerate"     # ≥ 0.40: 재생성 요청


@dataclass
class PRISMResult:
    """PRISM 내부 상태 분석 결과"""
    confidence: float         # 모델 확신도 (0~1, 높을수록 정상)
    entropy: float            # 판단 엔트로피 (낮을수록 확신)
    is_hallucinating: bool    # 환각 여부 판정
    raw_score: float          # PRISM 원시 점수 (0~1, 높을수록 환각)
    semantic_sim: float = 0.0 # 캡션-쿼리 의미 유사도


@dataclass
class WCSResult:
    """WCS (World Consistency Score) 물리적 일관성 결과"""
    object_persistence: float      # 객체 영속성 (0~1)
    relative_scale: float          # 상대 크기 일관성 (0~1)
    color_consistency: float       # 색상 일관성 (0~1)
    frame_physics: float           # 프레임 물리 타당성 (0~1)
    wcs_score: float               # 종합 WCS 점수 (0~1, 높을수록 일관적)
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HallucinationResult:
    """환각 탐지 종합 결과"""
    hallucination_score: float     # 종합 환각 점수 (0~1, 높을수록 환각)
    verdict: HallucinationVerdict  # 3단계 판정
    prism_result: PRISMResult      # Track B: 내부 상태
    wcs_result: WCSResult          # Track A: 외부 일관성
    clip_id: str                   # 대상 클립 ID
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CorrectionResult:
    """자동 보정 결과"""
    original_clip_id: str
    correction_method: str         # "archive_replace" 또는 "partial_regen"
    replacement_clip_id: Optional[str] = None
    new_hallucination_score: float = 0.0
    new_verdict: HallucinationVerdict = HallucinationVerdict.PASS
    success: bool = False


# ─────────────────────────────────────────────
# Track B: PRISM — AI 내부 상태 분석 기반 환각 탐지
# ─────────────────────────────────────────────

class PRISMDetector:
    """PRISM — Probing Risk via Internal States of Model

    PDF 스펙 (1차년도):
      - AI 모델 내부 상태 분석 기반 "거짓말" 탐지 (정확도 75~77%)
      - Entropy 측정: 모델의 판단 확신도를 정량화하여 불확실한 구간 식별
      - 참고: Azaria et al., "The Internal State of an LLM Knows When It's Lying", EMNLP 2023

    구현:
      Level 1 (현재): 캡션-쿼리 의미 유사도 + TF-IDF 겹침 + 엔트로피 기반
        - sentence-transformers 사용 가능 시 → 코사인 유사도
        - 불가 시 → TF-IDF + 자카드 유사도 (fallback)
      Level 2 (2차년도): 실제 모델 중간 레이어 활성화 분석
        - InternVideo2 hidden state probe
        - 다중 VLM 앙상블 (InternVideo2 + GPT-4V)
    """

    def __init__(self, base_accuracy: float = 0.76):
        self.base_accuracy = base_accuracy
        self._sentence_model = None
        self._use_semantic = False
        self._init_semantic_model()

    def _init_semantic_model(self):
        """sentence-transformers 모델 초기화 (가능한 경우)"""
        try:
            from sentence_transformers import SentenceTransformer
            self._sentence_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            self._use_semantic = True
            logger.info("PRISM: sentence-transformers 로드 완료 (의미 유사도 활성)")
        except ImportError:
            logger.info("PRISM: sentence-transformers 미설치 → TF-IDF fallback 사용")
            self._use_semantic = False

    def _compute_semantic_similarity(self, text_a: str, text_b: str) -> float:
        """두 텍스트 간 의미 유사도 (코사인 유사도)"""
        if self._use_semantic and self._sentence_model is not None:
            embeddings = self._sentence_model.encode([text_a, text_b])
            cos_sim = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]) + 1e-8
            )
            return float(max(0.0, min(1.0, cos_sim)))
        else:
            return self._compute_tfidf_similarity(text_a, text_b)

    def _compute_tfidf_similarity(self, text_a: str, text_b: str) -> float:
        """TF-IDF 기반 유사도 (fallback)"""
        # 한글/영문 모두 대응하는 간단한 토큰화
        words_a = set(text_a.lower().replace(',', ' ').replace('.', ' ').split())
        words_b = set(text_b.lower().replace(',', ' ').replace('.', ' ').split())

        # 한글 형태소 분석 시도
        try:
            from konlpy.tag import Okt
            okt = Okt()
            morphs_a = set(okt.nouns(text_a))
            morphs_b = set(okt.nouns(text_b))
            if morphs_a or morphs_b:
                words_a = morphs_a | words_a
                words_b = morphs_b | words_b
        except Exception:
            pass

        if not words_a or not words_b:
            return 0.0

        # 자카드 유사도 + 오버랩 비율
        intersection = words_a & words_b
        union = words_a | words_b
        jaccard = len(intersection) / len(union) if union else 0.0
        overlap = len(intersection) / min(len(words_a), len(words_b)) if min(len(words_a), len(words_b)) > 0 else 0.0

        # 가중 평균 (오버랩에 더 높은 가중치)
        return 0.4 * jaccard + 0.6 * overlap

    def analyze(
        self,
        clip_id: str,
        caption: str,
        query: str,
        generated: bool = False
    ) -> PRISMResult:
        """클립의 내부 상태 분석

        Args:
            clip_id: 클립 ID
            caption: 클립 캡션 (한글 또는 영문)
            query: 원본 쿼리
            generated: AI 생성 클립 여부 (True면 더 엄격)

        Returns:
            PRISMResult
        """
        # 1. 의미 유사도 계산
        semantic_sim = self._compute_semantic_similarity(caption, query)

        # 2. 환각 점수: 1 - 유사도 (유사도가 낮을수록 환각 가능성 높음)
        base_halluc = 1.0 - semantic_sim

        # 3. 생성 영상이면 환각 가능성 가중 (PDF: AI 생성 영상 10~15% 환각율)
        if generated:
            base_halluc = min(1.0, base_halluc * 1.3)

        # 4. 모델 불확실성 시뮬레이션 (실제 PRISM에서는 레이어 활성화 분산)
        noise = random.gauss(0, 0.03)
        raw_score = max(0.0, min(1.0, base_halluc + noise))

        # 5. 확신도 & 엔트로피
        confidence = 1.0 - raw_score
        # Shannon 엔트로피: 불확실할수록 높음
        p = max(1e-8, min(1 - 1e-8, confidence))
        entropy = -p * np.log2(p) - (1 - p) * np.log2(1 - p)

        is_hallucinating = raw_score > 0.25

        logger.debug(
            f"PRISM [{clip_id}]: sem_sim={semantic_sim:.3f}, "
            f"raw={raw_score:.3f}, conf={confidence:.3f}, "
            f"entropy={entropy:.3f}, halluc={is_hallucinating}"
        )

        return PRISMResult(
            confidence=confidence,
            entropy=float(entropy),
            is_hallucinating=is_hallucinating,
            raw_score=raw_score,
            semantic_sim=semantic_sim
        )


# ─────────────────────────────────────────────
# Track A: WCS — World Consistency Score 물리적 일관성
# ─────────────────────────────────────────────

class WCSDetector:
    """WCS — World Consistency Score 물리적 일관성 검증

    PDF 스펙:
      4개 지표로 물리적 일관성을 종합 평가:
        1. 객체 영속성(OP, 0.3): 프레임 간 객체 추적 → 사라지거나 변형되면 감점
        2. 상대 크기(RS, 0.2): 객체 간 크기 비율 일관성 검증
        3. 색상 일관성(CC, 0.3): 프레임 간 색상 분포 변화 측정
        4. 프레임 물리(FP, 0.2): 물리 법칙 위반 탐지 (옵티컬 플로우 기반)

      종합 WCS = 0.3×OP + 0.2×RS + 0.3×CC + 0.2×FP

    구현:
      Level 1 (1차년도, 현재):
        OP: ORB 특징점 매칭 기반 객체 추적 일관성
        RS: 프레임 간 엣지 밀도 변화로 상대 크기 근사
        CC: 컬러 히스토그램 상관관계 (2D H-S 히스토그램)
        FP: 옵티컬 플로우 크기 분포 + 방향 일관성
      Level 2 (2차년도):
        OP/RS: DINOv2 기반 시각적 유사도 + 객체 디텍터
        FP: DINOv2 임베딩 거리 기반 급격한 변화 감지
        다중 VLM 교차 검증
    """

    # WCS 가중치 (PDF 설계)
    W_OP = 0.3   # 객체 영속성
    W_RS = 0.2   # 상대 크기
    W_CC = 0.3   # 색상 일관성
    W_FP = 0.2   # 프레임 물리

    def __init__(self, num_sample_frames: int = 6):
        """
        Args:
            num_sample_frames: 분석용 프레임 샘플 수 (기본 6)
        """
        self.num_sample_frames = num_sample_frames

    def _extract_frames(
        self, video_path: str, start_ms: float, end_ms: float
    ) -> List[np.ndarray]:
        """영상에서 균일 간격 프레임 추출"""
        import cv2
        import os

        frames = []
        if not os.path.exists(video_path):
            return frames

        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return frames

            duration = max(end_ms - start_ms, 100)
            for i in range(self.num_sample_frames):
                t = start_ms + duration * (i + 0.5) / self.num_sample_frames
                cap.set(cv2.CAP_PROP_POS_MSEC, t)
                ret, frame = cap.read()
                if ret and frame is not None:
                    # 분석용으로 리사이즈 (성능)
                    frame = cv2.resize(frame, (320, 240))
                    frames.append(frame)
            cap.release()
        except Exception as e:
            logger.warning(f"WCS 프레임 추출 실패: {e}")

        return frames

    def _compute_object_persistence(self, frames: List[np.ndarray]) -> float:
        """OP: 객체 영속성 — ORB 특징점 매칭 기반

        연속 프레임 간 ORB 특징점을 매칭하여 객체가 일관되게 유지되는지 확인.
        매칭 비율이 높을수록 객체가 안정적으로 유지됨.
        """
        import cv2

        if len(frames) < 2:
            return 0.85  # 기본값

        orb = cv2.ORB_create(nfeatures=300)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        match_ratios = []
        for i in range(len(frames) - 1):
            gray_a = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            gray_b = cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY)

            kp_a, des_a = orb.detectAndCompute(gray_a, None)
            kp_b, des_b = orb.detectAndCompute(gray_b, None)

            if des_a is None or des_b is None or len(kp_a) == 0 or len(kp_b) == 0:
                match_ratios.append(0.5)
                continue

            matches = bf.match(des_a, des_b)
            # 좋은 매칭만 필터 (거리 임계값)
            good_matches = [m for m in matches if m.distance < 60]
            ratio = len(good_matches) / max(len(kp_a), len(kp_b)) if max(len(kp_a), len(kp_b)) > 0 else 0
            match_ratios.append(min(1.0, ratio * 2.5))  # 스케일링

        op_score = float(np.mean(match_ratios)) if match_ratios else 0.5
        return max(0.0, min(1.0, op_score))

    def _compute_relative_scale(self, frames: List[np.ndarray]) -> float:
        """RS: 상대 크기 일관성 — 엣지 밀도 변화 기반

        프레임 간 Canny 엣지 밀도의 변화를 측정.
        급격한 변화 = 객체 크기/구조 변화 = 일관성 저하.
        """
        import cv2

        if len(frames) < 2:
            return 0.9

        edge_densities = []
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            density = np.sum(edges > 0) / edges.size
            edge_densities.append(density)

        # 연속 프레임 간 엣지 밀도 변화율
        changes = []
        for i in range(len(edge_densities) - 1):
            if edge_densities[i] > 0:
                change = abs(edge_densities[i+1] - edge_densities[i]) / (edge_densities[i] + 1e-8)
                changes.append(change)

        if not changes:
            return 0.9

        # 변화가 작을수록 일관성 높음
        avg_change = float(np.mean(changes))
        rs_score = max(0.0, 1.0 - avg_change * 2.0)
        return min(1.0, rs_score)

    def _compute_color_consistency(self, frames: List[np.ndarray]) -> float:
        """CC: 색상 일관성 — H-S 2D 히스토그램 상관관계

        연속 프레임의 H(색상)-S(채도) 히스토그램을 비교.
        상관계수가 높을수록 색상 분포가 일관적.
        """
        import cv2

        if len(frames) < 2:
            return 0.85

        hsv_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2HSV) for f in frames]

        correlations = []
        for i in range(len(hsv_frames) - 1):
            hist_a = cv2.calcHist([hsv_frames[i]], [0, 1], None, [32, 32], [0, 180, 0, 256])
            hist_b = cv2.calcHist([hsv_frames[i+1]], [0, 1], None, [32, 32], [0, 180, 0, 256])
            cv2.normalize(hist_a, hist_a)
            cv2.normalize(hist_b, hist_b)
            corr = cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL)
            correlations.append(max(0.0, corr))

        cc_score = float(np.mean(correlations)) if correlations else 0.5
        return max(0.0, min(1.0, cc_score))

    def _compute_frame_physics(self, frames: List[np.ndarray]) -> float:
        """FP: 프레임 물리 타당성 — 옵티컬 플로우 기반

        PDF 스펙: "물리 법칙 위반 탐지 (물이 위로 흐르는 등)"
        Farneback 옵티컬 플로우로 움직임을 분석:
          - 플로우 크기 분포: 급격한 변화 → 물리적으로 비현실적
          - 방향 일관성: 인접 픽셀의 플로우 방향이 일관적이어야 정상
          - 비현실적 가속: 연속 프레임에서 속도 급변 탐지
        """
        import cv2

        if len(frames) < 2:
            return 0.88

        physics_scores = []
        prev_magnitudes = None

        for i in range(len(frames) - 1):
            gray_a = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            gray_b = cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY)

            # Farneback 옵티컬 플로우
            flow = cv2.calcOpticalFlowFarneback(
                gray_a, gray_b, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )

            # 플로우 크기 & 방향
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

            # 지표 1: 플로우 크기 분포 — 너무 큰 움직임은 비현실적
            avg_mag = float(np.mean(magnitude))
            max_mag = float(np.percentile(magnitude, 99))
            # 정상 범위: avg < 10, max < 30
            mag_score = max(0.0, 1.0 - max(0, avg_mag - 5) / 20.0)

            # 지표 2: 방향 일관성 — 인접 픽셀 방향 분산
            # 작은 블록(8x8)별 플로우 방향 표준편차
            h, w = angle.shape
            block_size = 16
            dir_stds = []
            for r in range(0, h - block_size, block_size):
                for c in range(0, w - block_size, block_size):
                    block_angle = angle[r:r+block_size, c:c+block_size]
                    block_mag = magnitude[r:r+block_size, c:c+block_size]
                    # 움직임이 있는 픽셀만
                    mask = block_mag > 1.0
                    if np.sum(mask) > 5:
                        dir_stds.append(float(np.std(block_angle[mask])))

            if dir_stds:
                avg_dir_std = float(np.mean(dir_stds))
                # 방향 분산이 작을수록 일관적 (정상: < 1.5 rad)
                dir_score = max(0.0, 1.0 - avg_dir_std / 3.0)
            else:
                dir_score = 0.9

            # 지표 3: 비현실적 가속도 — 연속 프레임 속도 급변
            current_magnitudes = float(np.mean(magnitude))
            if prev_magnitudes is not None:
                accel = abs(current_magnitudes - prev_magnitudes)
                accel_score = max(0.0, 1.0 - accel / 10.0)
            else:
                accel_score = 1.0
            prev_magnitudes = current_magnitudes

            # 프레임 쌍별 물리 점수 (가중 평균)
            pair_score = 0.4 * mag_score + 0.3 * dir_score + 0.3 * accel_score
            physics_scores.append(pair_score)

        fp_score = float(np.mean(physics_scores)) if physics_scores else 0.5
        return max(0.0, min(1.0, fp_score))

    def analyze(
        self,
        clip_id: str,
        video_path: str,
        start_ms: float,
        end_ms: float
    ) -> WCSResult:
        """클립의 물리적 일관성 분석

        Args:
            clip_id: 클립 ID
            video_path: 영상 경로
            start_ms: 시작 시간
            end_ms: 종료 시간

        Returns:
            WCSResult (4개 지표 + 종합 WCS)
        """
        frames = self._extract_frames(video_path, start_ms, end_ms)

        if len(frames) < 2:
            # 프레임 추출 실패 시 보수적 점수
            logger.warning(f"WCS [{clip_id}]: 프레임 추출 실패 (path={video_path})")
            return WCSResult(
                object_persistence=0.7,
                relative_scale=0.7,
                color_consistency=0.7,
                frame_physics=0.7,
                wcs_score=0.7,
                details={"error": "insufficient_frames", "frame_count": len(frames)}
            )

        # 4개 지표 계산
        op = self._compute_object_persistence(frames)
        rs = self._compute_relative_scale(frames)
        cc = self._compute_color_consistency(frames)
        fp = self._compute_frame_physics(frames)

        # 종합 WCS (가중 평균)
        wcs_score = self.W_OP * op + self.W_RS * rs + self.W_CC * cc + self.W_FP * fp

        details = {
            "frame_count": len(frames),
            "weights": {"OP": self.W_OP, "RS": self.W_RS, "CC": self.W_CC, "FP": self.W_FP},
            "individual_scores": {"OP": op, "RS": rs, "CC": cc, "FP": fp}
        }

        logger.debug(
            f"WCS [{clip_id}]: OP={op:.3f}, RS={rs:.3f}, CC={cc:.3f}, FP={fp:.3f} "
            f"→ WCS={wcs_score:.3f} (frames={len(frames)})"
        )

        return WCSResult(
            object_persistence=op,
            relative_scale=rs,
            color_consistency=cc,
            frame_physics=fp,
            wcs_score=wcs_score,
            details=details
        )


# ─────────────────────────────────────────────
# 통합 환각 탐지 파이프라인
# ─────────────────────────────────────────────

class HallucinationDetector:
    """이중 환각 검증 체계 + 3단계 판정 + 자동 보정

    PDF 스펙:
      Track A (외부): WCS 물리적 일관성
      Track B (내부): PRISM 내부 상태 분석
      종합 환각 점수 = α * PRISM + β * (1 - WCS)
      3단계 판정: ≤0.15 통과 / 0.15~0.40 보정 / ≥0.40 재생성

    1차년도 목표: 탐지 정확도 75~77%, 환각율 15% 이하
    2차년도 목표: PRISM + TC-Bench 이중 검증, 환각율 5% 이하
    """

    # 판정 임계값
    PASS_THRESHOLD = 0.15
    CORRECT_THRESHOLD = 0.40

    # Track 가중치
    PRISM_WEIGHT = 0.5
    WCS_WEIGHT = 0.5

    # 자동 보정 최대 재시도 횟수
    MAX_CORRECTION_RETRIES = 3

    def __init__(
        self,
        prism: Optional[PRISMDetector] = None,
        wcs: Optional[WCSDetector] = None,
        search_fn=None
    ):
        """
        Args:
            prism: PRISM 탐지기 인스턴스
            wcs: WCS 탐지기 인스턴스
            search_fn: 아카이브 검색 함수 (query → List[ClipResult])
                       자동 보정의 "아카이브 클립 교체"에 사용
        """
        self.prism = prism or PRISMDetector()
        self.wcs = wcs or WCSDetector()
        self.search_fn = search_fn

    def detect(
        self,
        clip_id: str,
        caption: str,
        query: str,
        video_path: str,
        start_ms: float,
        end_ms: float,
        generated: bool = False
    ) -> HallucinationResult:
        """단일 클립에 대한 환각 탐지

        Args:
            clip_id: 클립 ID
            caption: 클립 캡션
            query: 원본 쿼리
            video_path: 영상 경로
            start_ms: 시작 시간
            end_ms: 종료 시간
            generated: AI 생성 여부

        Returns:
            HallucinationResult
        """
        # Track B: PRISM 내부 상태 분석
        prism_result = self.prism.analyze(clip_id, caption, query, generated)

        # Track A: WCS 물리적 일관성
        wcs_result = self.wcs.analyze(clip_id, video_path, start_ms, end_ms)

        # 종합 환각 점수
        halluc_score = (
            self.PRISM_WEIGHT * prism_result.raw_score +
            self.WCS_WEIGHT * (1.0 - wcs_result.wcs_score)
        )
        halluc_score = max(0.0, min(1.0, halluc_score))

        # 3단계 판정
        if halluc_score <= self.PASS_THRESHOLD:
            verdict = HallucinationVerdict.PASS
        elif halluc_score <= self.CORRECT_THRESHOLD:
            verdict = HallucinationVerdict.AUTO_CORRECT
        else:
            verdict = HallucinationVerdict.REGENERATE

        result = HallucinationResult(
            hallucination_score=halluc_score,
            verdict=verdict,
            prism_result=prism_result,
            wcs_result=wcs_result,
            clip_id=clip_id,
            details={
                "prism_raw": prism_result.raw_score,
                "prism_semantic_sim": prism_result.semantic_sim,
                "wcs_score": wcs_result.wcs_score,
                "wcs_components": {
                    "OP": wcs_result.object_persistence,
                    "RS": wcs_result.relative_scale,
                    "CC": wcs_result.color_consistency,
                    "FP": wcs_result.frame_physics,
                },
                "formula": f"{self.PRISM_WEIGHT}*PRISM + {self.WCS_WEIGHT}*(1-WCS)"
            }
        )

        logger.info(
            f"환각 탐지 [{clip_id}]: score={halluc_score:.3f} → {verdict.value} "
            f"(PRISM={prism_result.raw_score:.3f}[sem={prism_result.semantic_sim:.3f}], "
            f"WCS={wcs_result.wcs_score:.3f}[OP={wcs_result.object_persistence:.2f},"
            f"RS={wcs_result.relative_scale:.2f},CC={wcs_result.color_consistency:.2f},"
            f"FP={wcs_result.frame_physics:.2f}])"
        )

        return result

    def detect_batch(
        self,
        clips: List[Any],
        query: str,
        generated: bool = False
    ) -> List[HallucinationResult]:
        """클립 리스트에 대한 일괄 환각 탐지

        Args:
            clips: ClipResult 리스트
            query: 원본 쿼리
            generated: AI 생성 여부

        Returns:
            HallucinationResult 리스트
        """
        results = []
        for clip in clips:
            result = self.detect(
                clip_id=clip.clip_id,
                caption=getattr(clip, 'caption', ''),
                query=query,
                video_path=getattr(clip, 'video_path', ''),
                start_ms=getattr(clip, 'start_ms', 0),
                end_ms=getattr(clip, 'end_ms', 0),
                generated=generated
            )
            results.append(result)

        # 통계 로깅
        verdicts = [r.verdict.value for r in results]
        from collections import Counter
        counts = Counter(verdicts)
        avg_score = np.mean([r.hallucination_score for r in results])
        logger.info(
            f"환각 탐지 배치 완료: {len(results)}개 클립, "
            f"평균 점수={avg_score:.3f}, 판정={dict(counts)}"
        )

        return results

    def auto_correct(
        self,
        halluc_result: HallucinationResult,
        query: str,
        exclude_clip_ids: Optional[List[str]] = None
    ) -> CorrectionResult:
        """자동 보정 수행 (환각 점수 0.15~0.40 구간)

        보정 방법:
          1. 아카이브 클립 교체: search_fn으로 대체 클립 검색
          2. 교체 후 재검증

        Args:
            halluc_result: 환각 탐지 결과
            query: 원본 쿼리
            exclude_clip_ids: 제외할 클립 ID (이미 사용된 것들)

        Returns:
            CorrectionResult
        """
        if halluc_result.verdict != HallucinationVerdict.AUTO_CORRECT:
            return CorrectionResult(
                original_clip_id=halluc_result.clip_id,
                correction_method="none",
                success=False
            )

        exclude = set(exclude_clip_ids or [])
        exclude.add(halluc_result.clip_id)

        # 아카이브 클립 교체 시도
        if self.search_fn:
            try:
                candidates = self.search_fn(query)
                for candidate in candidates:
                    if candidate.clip_id not in exclude:
                        # 대체 클립에 대해 환각 재검증
                        re_check = self.detect(
                            clip_id=candidate.clip_id,
                            caption=getattr(candidate, 'caption', ''),
                            query=query,
                            video_path=getattr(candidate, 'video_path', ''),
                            start_ms=getattr(candidate, 'start_ms', 0),
                            end_ms=getattr(candidate, 'end_ms', 0),
                            generated=False  # 아카이브 클립은 실제 영상
                        )

                        if re_check.verdict == HallucinationVerdict.PASS:
                            logger.info(
                                f"자동 보정 성공: {halluc_result.clip_id} → "
                                f"{candidate.clip_id} (score: "
                                f"{halluc_result.hallucination_score:.3f} → "
                                f"{re_check.hallucination_score:.3f})"
                            )
                            return CorrectionResult(
                                original_clip_id=halluc_result.clip_id,
                                correction_method="archive_replace",
                                replacement_clip_id=candidate.clip_id,
                                new_hallucination_score=re_check.hallucination_score,
                                new_verdict=re_check.verdict,
                                success=True
                            )
            except Exception as e:
                logger.warning(f"아카이브 교체 실패: {e}")

        # 교체 실패 시 → 재생성 요청으로 에스컬레이션
        logger.info(
            f"자동 보정 실패 ({halluc_result.clip_id}): "
            f"적합한 대체 클립 없음 → 재생성 요청으로 에스컬레이션"
        )
        return CorrectionResult(
            original_clip_id=halluc_result.clip_id,
            correction_method="escalate_to_regen",
            success=False
        )

    def run_pipeline(
        self,
        clips: List[Any],
        query: str,
        generated: bool = False
    ) -> Dict[str, Any]:
        """전체 환각 탐지 + 보정 파이프라인 실행

        1. 전체 클립 환각 탐지
        2. AUTO_CORRECT 판정 클립에 대해 자동 보정 시도
        3. REGENERATE 판정 클립 목록 반환

        Args:
            clips: ClipResult 리스트
            query: 원본 쿼리
            generated: AI 생성 여부

        Returns:
            {
                "passed": [HallucinationResult, ...],
                "corrected": [CorrectionResult, ...],
                "needs_regeneration": [HallucinationResult, ...],
                "hallucination_rate": float,
                "summary": str
            }
        """
        # Step 1: 환각 탐지
        detections = self.detect_batch(clips, query, generated)

        passed = []
        to_correct = []
        to_regenerate = []

        for det in detections:
            if det.verdict == HallucinationVerdict.PASS:
                passed.append(det)
            elif det.verdict == HallucinationVerdict.AUTO_CORRECT:
                to_correct.append(det)
            else:
                to_regenerate.append(det)

        # Step 2: 자동 보정
        used_ids = [c.clip_id for c in clips]
        corrections = []
        for det in to_correct:
            correction = self.auto_correct(det, query, exclude_clip_ids=used_ids)
            corrections.append(correction)
            if correction.success:
                passed.append(det)  # 보정 성공 → 통과로 이동

        # Step 3: 결과 집계
        total = len(clips)
        halluc_count = len(to_correct) + len(to_regenerate)
        halluc_rate = halluc_count / total if total > 0 else 0.0

        summary = (
            f"환각 탐지 완료: {total}개 중 "
            f"통과 {len(passed)}, 보정 시도 {len(to_correct)}, "
            f"재생성 필요 {len(to_regenerate)} "
            f"(환각율: {halluc_rate*100:.1f}%)"
        )
        logger.info(summary)

        return {
            "passed": passed,
            "corrected": corrections,
            "needs_regeneration": to_regenerate,
            "hallucination_rate": halluc_rate,
            "summary": summary,
            "all_detections": detections
        }
