"""
① shot_detector.py — TransNetV2 기반 Shot 경계 탐지 + Agglomerative Clustering Scene 구성

출처:
- TransNetV2 논문:
  "TransNet V2: An effective deep network architecture for fast shot
   transition detection" (Souček & Lokoč, 2020)
  https://arxiv.org/abs/2008.04838
  GitHub: https://github.com/soCzech/TransNetV2
  라이선스: MIT License

- Agglomerative Clustering:
  scikit-learn AgglomerativeClustering
  https://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering
  → 시공간적으로 유사한 Shot들을 상위 Scene으로 묶어 계층적 구조 구성
  → 인접 Shot 간의 시각적 유사도(히스토그램) + 시간적 근접성을 기반으로 병합

역할:
  1) 영상에서 Shot 경계를 자동 탐지하여 의미 단위 클립으로 분할
  2) Agglomerative Clustering으로 유사 Shot들을 Scene으로 그룹화
프로토타입: TransNetV2가 설치되어 있으면 사용하고, 없으면 균등 분할 폴백
"""

import os
import logging
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field

import numpy as np
import cv2

logger = logging.getLogger(__name__)


@dataclass
class ShotInfo:
    """개별 Shot 정보"""
    shot_id: int
    start_ms: float
    end_ms: float
    scene_id: int = -1  # 소속 Scene ID (-1: 미할당)


@dataclass
class SceneInfo:
    """Scene (Shot 그룹) 정보"""
    scene_id: int
    shots: List[ShotInfo] = field(default_factory=list)

    @property
    def start_ms(self) -> float:
        return min(s.start_ms for s in self.shots) if self.shots else 0.0

    @property
    def end_ms(self) -> float:
        return max(s.end_ms for s in self.shots) if self.shots else 0.0

    @property
    def duration_ms(self) -> float:
        return self.end_ms - self.start_ms


class ShotDetector:
    """TransNetV2 기반 Shot 경계 탐지 + Agglomerative Clustering Scene 구성

    detect_shots(video_path) → List[(start_ms, end_ms)]
    detect_scenes(video_path) → List[SceneInfo]  ← NEW: Shot → Scene 계층 구조
    """

    def __init__(
        self,
        model_dir: Optional[str] = None,
        scene_distance_threshold: float = 0.7,
    ):
        """
        Args:
            model_dir: TransNetV2 모델 가중치 디렉토리
                       None이면 기본 경로 or 폴백 모드
            scene_distance_threshold: Agglomerative Clustering 거리 임계값
                0.0~1.0, 낮을수록 더 잘게 분할 (기본 0.7)
                → 시각적+시간적 유사도가 이 임계값 이하인 Shot들끼리 같은 Scene
        """
        self.model = None
        self.model_dir = model_dir
        self.scene_distance_threshold = scene_distance_threshold
        self._try_load_model()

    def _try_load_model(self):
        """TransNetV2 모델 로드 시도 (없으면 폴백)"""
        try:
            from transnetv2 import TransNetV2
            self.model = TransNetV2(model_dir=self.model_dir)
            logger.info("TransNetV2 모델 로드 성공")
        except ImportError:
            logger.warning(
                "TransNetV2 미설치 → 균등 분할 폴백 모드 사용\n"
                "설치 방법: git clone https://github.com/soCzech/TransNetV2 && "
                "pip install -e TransNetV2/inference/"
            )
        except Exception as e:
            logger.warning(f"TransNetV2 로드 실패: {e} → 폴백 모드")

    def detect_shots(
        self,
        video_path: str,
        threshold: float = 0.5
    ) -> List[Tuple[float, float]]:
        """영상에서 Shot 경계를 탐지

        Args:
            video_path: 입력 영상 파일 경로
            threshold: Shot 경계 판정 임계값 (0~1)

        Returns:
            List of (start_ms, end_ms)
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"영상 파일 없음: {video_path}")

        if self.model is not None:
            return self._detect_with_transnet(video_path, threshold)
        else:
            return self._detect_fallback(video_path)

    def _detect_with_transnet(
        self,
        video_path: str,
        threshold: float
    ) -> List[Tuple[float, float]]:
        """TransNetV2를 사용한 Shot 경계 탐지"""
        import torch

        # TransNetV2 추론
        # ─────────────────────────────────────────────────────────────────
        # predict_video()는 영상을 48×27 픽셀로 리사이즈한 프레임을 읽어서
        # 각 프레임이 "Shot 경계일 확률"을 계산해 돌려줌
        #
        # 예시: 30초짜리 영상 (25fps → 750 프레임)
        #
        # video_frames      → shape (750, 27, 48, 3)   ← 실제 픽셀 데이터 (거의 안 씀)
        # single_frame_preds → shape (750,)             ← 핵심! 프레임별 경계 확률
        #                       [0.02, 0.01, 0.03, ..., 0.97, 0.02, ...]
        #                                                  ↑
        #                                               이 프레임이 Shot 경계일 확률 97%
        #                                               (확 튀는 지점이 Shot 경계일 가능성 높음)
        # all_frame_preds   → shape (750, 2)            ← 앙상블용 (잘 안 씀)
        video_frames, single_frame_preds, all_frame_preds = \
            self.model.predict_video(video_path)

        # 프레임 인덱스 → ms 변환을 위해 fps 확인
        # 예시: cv2가 25.0fps 반환
        fps = self._get_fps(video_path)

        # threshold=0.5 기준으로 경계 확률 > 0.5인 지점을 잘라서
        # 연속된 프레임 구간(= Shot)으로 묶어줌
        #
        # 예시: single_frame_preds 중 경계 확률이 높은 지점이 3곳이라면
        #
        # segments = [
        #   (  0, 187),   ← 0번~187번 프레임이 하나의 Shot
        #   (188, 412),   ← 188번~412번 프레임이 하나의 Shot
        #   (413, 749),   ← 413번~749번 프레임이 하나의 Shot
        # ]
        segments = self.model.predictions_to_scenes(
            single_frame_preds, threshold=threshold
        )

        shots = []
        for start_frame, end_frame in segments:
            # 프레임 번호 → 밀리초 변환
            # 예시: start_frame=188, fps=25.0
            #   start_ms = (188 / 25.0) * 1000 = 7,520 ms  (= 7.52초 지점)
            #   end_ms   = (412 / 25.0) * 1000 = 16,480 ms (= 16.48초 지점)
            start_ms = (start_frame / fps) * 1000
            end_ms = (end_frame / fps) * 1000
            shots.append((start_ms, end_ms))

        # 최종 결과 예시:
        # shots = [
        #   (    0.0,  7520.0),   ← 0.00초 ~ 7.52초
        #   ( 7520.0, 16480.0),   ← 7.52초 ~ 16.48초
        #   (16480.0, 29960.0),   ← 16.48초 ~ 29.96초
        # ]
        logger.info(f"TransNetV2 탐지 완료: {len(shots)} shots from {video_path}")
        return shots

    def _detect_fallback(
        self,
        video_path: str,
        segment_duration_ms: float = 5000.0
    ) -> List[Tuple[float, float]]:
        """폴백: 균등 시간 분할 (TransNetV2 미설치 시)

        MSR-VTT 클립은 보통 10~30초이므로 5초 단위 분할
        """
        duration_ms = self._get_duration_ms(video_path)

        shots = []
        current = 0.0
        while current < duration_ms:
            end = min(current + segment_duration_ms, duration_ms)
            shots.append((current, end))
            current = end

        logger.info(
            f"폴백 균등 분할: {len(shots)} segments "
            f"({segment_duration_ms}ms each) from {video_path}"
        )
        return shots

    def _get_fps(self, video_path: str) -> float:
        """영상 FPS 가져오기"""
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            return fps if fps > 0 else 25.0
        except Exception:
            return 25.0  # 기본값

    def _get_duration_ms(self, video_path: str) -> float:
        """영상 전체 길이(ms) 가져오기"""
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            cap.release()
            if fps > 0 and frame_count > 0:
                return (frame_count / fps) * 1000
        except Exception:
            pass
        return 30000.0  # 기본 30초

    def extract_keyframe(
        self,
        video_path: str,
        timestamp_ms: float,
        output_path: str
    ) -> str:
        """특정 타임스탬프의 키프레임 추출

        Args:
            video_path: 입력 영상 경로
            timestamp_ms: 추출할 시간 (ms)
            output_path: 저장할 이미지 경로

        Returns:
            저장된 키프레임 경로
        """
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_ms)
        ret, frame = cap.read()
        cap.release()

        if ret:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, frame)
            return output_path
        else:
            raise RuntimeError(
                f"키프레임 추출 실패: {video_path} @ {timestamp_ms}ms"
            )

    # ──────────────────────────────────────────────────────────────────
    # Agglomerative Clustering: Shot → Scene 계층 구조
    # ──────────────────────────────────────────────────────────────────

    def detect_scenes(
        self,
        video_path: str,
        threshold: float = 0.5
    ) -> List[SceneInfo]:
        """Shot 경계 탐지 후 Agglomerative Clustering으로 Scene 구성

        draw.io 아키텍처 Phase 0:
        "TransNetV2 & Agglomerative Clustering: 단순 프레임 추출이 아닌
         샷(Shot) 경계 탐지와 장면(Scene) 구성을 통해 비디오의
         시공간적 일관성을 먼저 파악합니다."

        프로세스:
          1) TransNetV2로 Shot 경계 탐지
          2) 각 Shot의 대표 프레임(중간 지점) 추출
          3) 프레임 간 색상 히스토그램 + 시간적 근접성으로 거리 행렬 구성
          4) Agglomerative Clustering (Ward or Average linkage)으로 계층 병합
          5) distance_threshold 기준으로 최종 Scene 할당

        Args:
            video_path: 입력 영상 파일 경로
            threshold: Shot 경계 판정 임계값 (TransNetV2 용)

        Returns:
            List[SceneInfo] — 각 SceneInfo에 소속 Shot 리스트 포함

        예시 (30초짜리 영상에서 6개 Shot이 3개 Scene으로 묶이는 경우):
            shots = [
                (0, 5000),    ← Shot 0: 실내 인터뷰
                (5000, 8000), ← Shot 1: 실내 다른 앵글
                (8000, 15000),← Shot 2: 야외 전경
                (15000,20000),← Shot 3: 야외 클로즈업
                (20000,25000),← Shot 4: 실내 복귀
                (25000,30000),← Shot 5: 실내 마무리
            ]

            scenes = [
                Scene 0: [Shot 0, Shot 1]         ← 실내 인터뷰 (시각적 유사)
                Scene 1: [Shot 2, Shot 3]         ← 야외 장면 (시각적 유사)
                Scene 2: [Shot 4, Shot 5]         ← 실내 복귀 (시각적 유사)
            ]
        """
        # Step 1: Shot 경계 탐지
        shots_raw = self.detect_shots(video_path, threshold)

        if len(shots_raw) <= 1:
            # Shot이 1개면 Scene도 1개
            shot = ShotInfo(
                shot_id=0,
                start_ms=shots_raw[0][0] if shots_raw else 0,
                end_ms=shots_raw[0][1] if shots_raw else 30000,
                scene_id=0
            )
            return [SceneInfo(scene_id=0, shots=[shot])]

        # Step 2: Shot 객체 생성
        shots = [
            ShotInfo(shot_id=i, start_ms=s, end_ms=e)
            for i, (s, e) in enumerate(shots_raw)
        ]

        # Step 3: 각 Shot의 대표 프레임에서 시각 특징 추출
        visual_features = self._extract_shot_features(video_path, shots)

        # Step 4: 거리 행렬 구성 (시각적 유사도 + 시간적 근접성)
        distance_matrix = self._compute_distance_matrix(shots, visual_features)

        # Step 5: Agglomerative Clustering 수행
        labels = self._cluster_shots(distance_matrix)

        # Step 6: Scene 구성
        scenes = self._build_scenes(shots, labels)

        logger.info(
            f"Scene 구성 완료: {len(shots)} shots → {len(scenes)} scenes "
            f"from {video_path}"
        )
        return scenes

    def _extract_shot_features(
        self,
        video_path: str,
        shots: List[ShotInfo]
    ) -> np.ndarray:
        """각 Shot의 대표 프레임에서 색상 히스토그램 특징 추출

        HSV 색공간의 H(색조)와 S(채도) 채널의 2D 히스토그램을 특징으로 사용.
        밝기(V)는 조명 변화에 민감하므로 제외하여 장면의 색감 구성에 집중.

        예시:
          실내 인터뷰 Shot → 피부톤 + 사무실 색감 → 특정 H-S 분포
          야외 전경 Shot   → 하늘색 + 녹색 잔디  → 다른 H-S 분포
          → 히스토그램 비교로 같은 장소의 Shot들이 가까운 거리를 가짐

        Args:
            video_path: 입력 영상 경로
            shots: Shot 정보 리스트

        Returns:
            np.ndarray [N_shots, feature_dim] — 정규화된 히스토그램 특징
        """
        import cv2

        features = []
        cap = cv2.VideoCapture(video_path)

        for shot in shots:
            # Shot 중간 지점의 프레임 추출
            mid_ms = (shot.start_ms + shot.end_ms) / 2
            cap.set(cv2.CAP_PROP_POS_MSEC, mid_ms)
            ret, frame = cap.read()

            if ret and frame is not None:
                # HSV 변환 후 H-S 채널 2D 히스토그램
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                hist = cv2.calcHist(
                    [hsv], [0, 1], None,
                    [32, 32],          # 32×32 bin (H: 0~180, S: 0~256)
                    [0, 180, 0, 256]
                )
                # 정규화하여 프레임 크기에 무관하게 비교 가능
                hist = cv2.normalize(hist, hist).flatten()  # [1024]
            else:
                # 폴백: 영벡터
                hist = np.zeros(32 * 32, dtype=np.float32)

            features.append(hist)

        cap.release()
        return np.array(features)

    def _compute_distance_matrix(
        self,
        shots: List[ShotInfo],
        visual_features: np.ndarray,
        visual_weight: float = 0.7,
        temporal_weight: float = 0.3
    ) -> np.ndarray:
        """시각적 유사도 + 시간적 근접성 결합 거리 행렬 구성

        두 가지 신호를 결합하는 이유:
        - 시각적 유사도만 쓰면: 멀리 떨어진 비슷한 장면이 하나로 묶임
          (예: 영상 처음과 끝의 인터뷰 장면)
        - 시간적 근접성만 쓰면: 연속이지만 완전히 다른 장면이 묶임
          (예: 실내→야외 전환 직후)
        → 두 신호를 w=0.7/0.3으로 가중 합산하여 균형 잡힌 거리 계산

        Args:
            shots: Shot 정보 리스트
            visual_features: [N, feature_dim] 히스토그램 특징
            visual_weight: 시각적 거리 가중치 (기본 0.7)
            temporal_weight: 시간적 거리 가중치 (기본 0.3)

        Returns:
            np.ndarray [N, N] — 대칭 거리 행렬 (0~1 범위)
        """
        n = len(shots)
        dist = np.zeros((n, n), dtype=np.float64)

        # --- 시각적 거리 (히스토그램 비교) ---
        # cv2.compareHist (Bhattacharyya 거리): 0 = 동일, 1 = 완전 다름
        visual_dist = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(i + 1, n):
                d = cv2.compareHist(
                    visual_features[i].reshape(-1).astype(np.float32),
                    visual_features[j].reshape(-1).astype(np.float32),
                    cv2.HISTCMP_BHATTACHARYYA
                )
                visual_dist[i, j] = d
                visual_dist[j, i] = d

        # --- 시간적 거리 (Shot 중심점 간 시간 차이) ---
        # 전체 영상 길이로 정규화하여 0~1 범위
        total_duration = max(s.end_ms for s in shots) - min(s.start_ms for s in shots)
        total_duration = max(total_duration, 1.0)  # 0 방지

        temporal_dist = np.zeros((n, n), dtype=np.float64)
        centers = [(s.start_ms + s.end_ms) / 2 for s in shots]
        for i in range(n):
            for j in range(i + 1, n):
                d = abs(centers[i] - centers[j]) / total_duration
                temporal_dist[i, j] = d
                temporal_dist[j, i] = d

        # --- 가중 합산 ---
        dist = visual_weight * visual_dist + temporal_weight * temporal_dist
        return dist

    def _cluster_shots(self, distance_matrix: np.ndarray) -> np.ndarray:
        """Agglomerative Clustering으로 Shot들을 Scene으로 묶기

        사전 계산된 거리 행렬을 사용하는 Average Linkage 방식.
        distance_threshold로 자동으로 클러스터 수를 결정.

        Average Linkage를 선택한 이유:
        - Ward: 유클리드 거리만 지원 → 사전계산 거리 행렬 불가
        - Complete: 최대 거리 기준 → 이상치에 민감
        - Single: 최소 거리 기준 → 체인 효과(너무 많이 묶임)
        - Average: 평균 거리 기준 → 균형 잡힌 Scene 구성

        Args:
            distance_matrix: [N, N] 대칭 거리 행렬

        Returns:
            np.ndarray [N] — 각 Shot의 Scene 라벨

        예시:
            distance_matrix 3×3 (Shot 0,1은 가깝고 Shot 2는 멀 때):
            [[0.0, 0.2, 0.8],
             [0.2, 0.0, 0.7],
             [0.8, 0.7, 0.0]]

            threshold=0.7 → labels = [0, 0, 1]
            (Shot 0,1 → Scene 0 / Shot 2 → Scene 1)
        """
        n = distance_matrix.shape[0]

        if n <= 1:
            return np.array([0])

        try:
            from sklearn.cluster import AgglomerativeClustering

            # metric="precomputed" → 사전 계산된 거리 행렬 직접 사용
            clustering = AgglomerativeClustering(
                n_clusters=None,
                metric="precomputed",
                linkage="average",
                distance_threshold=self.scene_distance_threshold,
            )
            labels = clustering.fit_predict(distance_matrix)
            logger.info(
                f"Agglomerative Clustering 완료: "
                f"{n} shots → {len(set(labels))} scenes "
                f"(threshold={self.scene_distance_threshold})"
            )
            return labels

        except ImportError:
            logger.warning(
                "scikit-learn 미설치 → 인접 Shot 병합 폴백 사용\n"
                "설치: pip install scikit-learn"
            )
            return self._cluster_fallback(distance_matrix)

    def _cluster_fallback(self, distance_matrix: np.ndarray) -> np.ndarray:
        """폴백: 인접 Shot 간 거리가 임계값 이하이면 같은 Scene

        scikit-learn이 없을 때 사용하는 간단한 순차 병합 방식.
        인접하지 않은 Shot은 같은 Scene으로 묶지 않음.
        """
        n = distance_matrix.shape[0]
        labels = np.zeros(n, dtype=int)
        current_scene = 0

        for i in range(1, n):
            if distance_matrix[i - 1, i] > self.scene_distance_threshold:
                current_scene += 1
            labels[i] = current_scene

        logger.info(
            f"폴백 클러스터링: {n} shots → {current_scene + 1} scenes"
        )
        return labels

    def _build_scenes(
        self,
        shots: List[ShotInfo],
        labels: np.ndarray
    ) -> List[SceneInfo]:
        """클러스터링 결과를 SceneInfo 객체로 변환

        Args:
            shots: Shot 정보 리스트
            labels: 각 Shot의 Scene 라벨

        Returns:
            List[SceneInfo] — Scene ID 순 정렬, 각 Scene 내부는 시간순
        """
        scenes_dict: Dict[int, List[ShotInfo]] = {}

        for shot, label in zip(shots, labels):
            shot.scene_id = int(label)
            scenes_dict.setdefault(int(label), []).append(shot)

        # Scene 내부 Shot을 시간순 정렬
        scenes = []
        for scene_id in sorted(scenes_dict.keys()):
            shots_in_scene = sorted(
                scenes_dict[scene_id], key=lambda s: s.start_ms
            )
            scenes.append(SceneInfo(scene_id=scene_id, shots=shots_in_scene))

        return scenes
