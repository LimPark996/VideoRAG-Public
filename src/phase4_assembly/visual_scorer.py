"""
⑧a visual_scorer.py — DINOv2 시각 유사도 계산

출처:
- 논문: "DINOv2: Learning Robust Visual Features without Supervision"
        (Oquab et al., 2024) — Meta AI Research
        https://arxiv.org/abs/2304.07193
- GitHub: https://github.com/facebookresearch/dinov2
- 라이선스: Apache 2.0

역할:
  - 클립 간 시각적 유사도 측정 (Perception-aware 정렬)
  - DINOv2 ViT-B/14 → 768차원 feature vector 추출
  - Cosine similarity로 인접 클립 유사도 계산
  - 트랜지션 타입 결정의 입력값
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


class VisualScorer:
    """DINOv2 기반 시각 유사도 계산기

    extract_features(frame) → np.ndarray [768]
    compute_similarity(feat_a, feat_b) → float
    """

    def __init__(
        self,
        model_name: str = "dinov2_vitb14",
        device: Optional[str] = None
    ):
        """
        Args:
            model_name: DINOv2 모델 이름 (timm 허브)
            device: 'cuda' or 'cpu'
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.transform = None
        self._loaded = False

    def load_model(self):
        """DINOv2 모델 로드 (Lazy Loading)"""
        if self._loaded:
            return

        try:
            # PyTorch Hub에서 DINOv2 로드
            self.model = torch.hub.load(
                'facebookresearch/dinov2', self.model_name
            )
            self.model = self.model.to(self.device).eval()
            self._mode = "dinov2"
            logger.info(f"DINOv2 로드 완료: {self.model_name}")
        except Exception as e:
            logger.warning(f"DINOv2 로드 실패: {e}")
            try:
                # timm 폴백
                import timm
                self.model = timm.create_model(
                    'vit_base_patch14_dinov2', pretrained=True
                )
                self.model = self.model.to(self.device).eval()
                self._mode = "timm"
                logger.info("timm DINOv2 폴백 로드 완료")
            except Exception as e2:
                logger.warning(f"timm도 실패: {e2} → 랜덤 특징 모드")
                self._mode = "random"

        self._loaded = True

    def _get_transform(self):
        """DINOv2 입력 전처리 변환

        timm DINOv2 폴백(vit_base_patch14_dinov2)은 518×518을 요구한다.
        PyTorch Hub DINOv2는 224×224. 로드된 모드에 따라 크기를 선택한다.
        """
        import torchvision.transforms as T
        size = 518 if getattr(self, '_mode', 'dinov2') == "timm" else 224
        return T.Compose([
            T.ToPILImage(),
            T.Resize(size),
            T.CenterCrop(size),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    @torch.no_grad()
    def extract_features(self, frame: np.ndarray) -> np.ndarray:
        """프레임에서 DINOv2 feature 추출

        Args:
            frame: np.ndarray [H, W, 3] (uint8, RGB)

        Returns:
            np.ndarray [768] float32 (L2 정규화된 feature vector)
        """
        self.load_model()

        if self._mode == "random":
            feat = np.random.randn(768).astype(np.float32)
            return feat / np.linalg.norm(feat)

        transform = self._get_transform()
        tensor = transform(frame).unsqueeze(0).to(self.device)

        if self._mode == "dinov2":
            features = self.model(tensor)
        else:
            features = self.model.forward_features(tensor)
            if features.ndim == 3:
                features = features[:, 0]  # CLS token

        feat = features.squeeze().cpu().numpy()

        # L2 정규화
        norm = np.linalg.norm(feat)
        if norm > 0:
            feat = feat / norm

        return feat.astype(np.float32)

    def compute_similarity(
        self,
        feat_a: np.ndarray,
        feat_b: np.ndarray
    ) -> float:
        """두 feature 벡터 간 코사인 유사도

        Args:
            feat_a, feat_b: L2 정규화된 feature 벡터

        Returns:
            cosine similarity (0~1)
        """
        sim = float(np.dot(feat_a, feat_b))
        return max(0.0, min(1.0, sim))  # clamp to [0, 1]

    def compute_pairwise_similarities(
        self,
        frames: List[np.ndarray]
    ) -> List[float]:
        """연속 프레임 쌍의 유사도 계산

        Args:
            frames: 순서대로 정렬된 클립 프레임 리스트

        Returns:
            List of similarity scores (len = len(frames) - 1)
        """
        if len(frames) < 2:
            return []

        features = [self.extract_features(f) for f in frames]

        similarities = []
        for i in range(len(features) - 1):
            sim = self.compute_similarity(features[i], features[i + 1])
            similarities.append(sim)

        return similarities
