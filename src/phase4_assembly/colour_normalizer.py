"""
⑧c colour_normalizer.py — DreamColour Training-free 3D LUT 색감 통일

출처:
- DreamColour 논문:
  "DreamColour: Controllable Video Colour Editing without Training"
  GitHub: https://github.com/CHAITron/DreamColour
  → 추가 학습 없이(Training-free) 3D LUT를 활용하여
    여러 클립의 색감을 통일하고 시각적 연속성을 보장

- 3D LUT (Look-Up Table) 기반 색보정:
  영상 후반 작업에서 표준적으로 사용되는 색감 보정 기법
  DaVinci Resolve, Adobe Premiere 등 전문 편집 도구의 핵심 기술

- 원리:
  3D LUT는 입력 RGB → 출력 RGB 매핑 테이블.
  R, G, B 각각 N개 격자점을 잡아 N×N×N 크기의 3차원 격자를 만들고,
  각 격자점에서의 출력 색상을 기록.
  중간 색상은 삼선형 보간(Trilinear Interpolation)으로 계산.

  DreamColour 방식은 레퍼런스 프레임의 색상 분포를 분석하여
  실시간으로 3D LUT를 생성 → 추가 학습 불필요(Training-free).

설계: DreamColour 방식 기반 자체 구현
  - LAB 색공간에서 채널별 분포 분석
  - 최적 수송(Optimal Transport) 근사 색상 매핑
  - RGB 3D LUT 격자 생성 및 삼선형 보간 적용
"""

import os
import logging
from typing import List, Optional, Tuple

import numpy as np
import cv2

logger = logging.getLogger(__name__)


class LUT3D:
    """3D Look-Up Table (색상 매핑 테이블)

    RGB 색공간에서 N×N×N 격자의 입력-출력 색상 매핑을 저장.
    중간 색상은 삼선형 보간으로 계산.

    구조:
      lut_table: np.ndarray [N, N, N, 3] (float32, 0~255)
      - lut_table[r_idx, g_idx, b_idx] = [R_out, G_out, B_out]
      - 격자 크기 N이 클수록 정밀하지만 메모리 사용 증가

    예시 (N=17):
      격자점 수: 17 × 17 × 17 = 4,913개
      메모리: 4,913 × 3 × 4 bytes = ~59KB
      → 매우 가벼운 구조, 실시간 적용 가능

    DaVinci Resolve/Adobe 표준 LUT 크기: 17 또는 33
    """

    def __init__(self, size: int = 17):
        """
        Args:
            size: LUT 격자 크기 (기본 17, 전문가용 33)
        """
        self.size = size
        # 항등 LUT (입력 = 출력)으로 초기화
        self.table = self._create_identity()

    def _create_identity(self) -> np.ndarray:
        """항등 3D LUT 생성 (입력 색상 = 출력 색상)

        각 격자점에서의 입력 RGB 값을 그대로 출력 값으로 설정.
        이 LUT를 적용하면 색상 변화 없음 (No-op).

        예시 (N=3):
          lut[0,0,0] = [0, 0, 0]      ← 검정
          lut[1,1,1] = [127, 127, 127] ← 중간 회색
          lut[2,2,2] = [255, 255, 255] ← 흰색
        """
        n = self.size
        # 0~255 범위를 N등분한 격자점
        axis = np.linspace(0, 255, n, dtype=np.float32)
        # meshgrid로 3D 격자 생성
        r, g, b = np.meshgrid(axis, axis, axis, indexing='ij')
        # [N, N, N, 3] 형태로 스택
        identity = np.stack([r, g, b], axis=-1)
        return identity

    def apply(self, image: np.ndarray) -> np.ndarray:
        """3D LUT를 이미지에 적용 (삼선형 보간)

        삼선형 보간(Trilinear Interpolation):
          입력 RGB가 격자점 사이에 있으면, 인접 8개 격자점의
          출력 값을 거리 기반으로 가중 평균하여 출력 RGB 결정.

          예시: 입력 R=100, 격자점 [0, 85, 170, 255] (N=4)
            → R=100은 85와 170 사이
            → 비율: (100-85)/(170-85) = 0.176
            → 출력 = lut[1] × 0.824 + lut[2] × 0.176

        Args:
            image: [H, W, 3] uint8 BGR

        Returns:
            [H, W, 3] uint8 BGR — LUT 적용된 이미지
        """
        n = self.size
        h, w, _ = image.shape

        # BGR → float [0, 1] 범위로 정규화
        img = image.astype(np.float32) / 255.0

        # 격자 인덱스 계산 (0 ~ n-1 범위)
        # B, G, R 채널 (OpenCV는 BGR)
        b_idx = img[:, :, 0] * (n - 1)
        g_idx = img[:, :, 1] * (n - 1)
        r_idx = img[:, :, 2] * (n - 1)

        # 하한/상한 인덱스
        r0 = np.clip(np.floor(r_idx).astype(int), 0, n - 2)
        g0 = np.clip(np.floor(g_idx).astype(int), 0, n - 2)
        b0 = np.clip(np.floor(b_idx).astype(int), 0, n - 2)
        r1, g1, b1 = r0 + 1, g0 + 1, b0 + 1

        # 보간 비율 (0~1)
        rd = (r_idx - r0).reshape(h, w, 1)
        gd = (g_idx - g0).reshape(h, w, 1)
        bd = (b_idx - b0).reshape(h, w, 1)

        # 8개 꼭짓점에서 LUT 값 가져오기
        c000 = self.table[r0, g0, b0]
        c001 = self.table[r0, g0, b1]
        c010 = self.table[r0, g1, b0]
        c011 = self.table[r0, g1, b1]
        c100 = self.table[r1, g0, b0]
        c101 = self.table[r1, g0, b1]
        c110 = self.table[r1, g1, b0]
        c111 = self.table[r1, g1, b1]

        # 삼선형 보간 (R → G → B 순)
        c00 = c000 * (1 - rd) + c100 * rd
        c01 = c001 * (1 - rd) + c101 * rd
        c10 = c010 * (1 - rd) + c110 * rd
        c11 = c011 * (1 - rd) + c111 * rd

        c0 = c00 * (1 - gd) + c10 * gd
        c1 = c01 * (1 - gd) + c11 * gd

        result = c0 * (1 - bd) + c1 * bd

        return np.clip(result, 0, 255).astype(np.uint8)


class ColourNormalizer:
    """DreamColour Training-free 3D LUT 색감 통일기

    draw.io 아키텍처 Phase 4:
    "DreamColour 색감 통일: Training-free 3D LUT
     → 추가 학습 없이 색상 적이
     → 전문 편집 수준 통일감"

    normalize(clip_paths) → List[corrected_paths]
    """

    def __init__(
        self,
        reference_method: str = "first",
        lut_size: int = 17,
        strength: float = 0.8,
    ):
        """
        Args:
            reference_method: 레퍼런스 프레임 선택 방법
                - "first": 첫 번째 클립을 레퍼런스로
                - "mean": 전체 평균 색감을 레퍼런스로
            lut_size: 3D LUT 격자 크기 (17 = 표준, 33 = 고정밀)
            strength: 색보정 강도 (0.0 = 원본, 1.0 = 완전 보정)
        """
        self.reference_method = reference_method
        self.lut_size = lut_size
        self.strength = strength

    def normalize(
        self,
        clip_paths: List[str],
        output_dir: str = "output/color_corrected"
    ) -> List[str]:
        """클립들의 색감을 3D LUT로 통일

        프로세스:
          1) 레퍼런스 프레임 선택 (첫 번째 클립 or 평균)
          2) 각 클립의 대표 프레임에서 색상 분포 분석
          3) 레퍼런스 ↔ 타겟 간 3D LUT 생성 (Training-free)
          4) LUT를 클립 전체 프레임에 적용

        Args:
            clip_paths: 입력 클립 파일 경로 리스트
            output_dir: 보정된 클립 저장 디렉토리

        Returns:
            보정된 클립 파일 경로 리스트
        """
        if len(clip_paths) <= 1:
            return clip_paths

        os.makedirs(output_dir, exist_ok=True)

        # 레퍼런스 프레임 추출
        ref_frame = self._get_reference_frame(clip_paths)

        corrected_paths = []
        for i, path in enumerate(clip_paths):
            if not os.path.exists(path):
                corrected_paths.append(path)
                continue

            try:
                out_path = os.path.join(
                    output_dir, f"cc_{i:03d}_{os.path.basename(path)}"
                )

                # 3D LUT 생성 + 적용
                target_frame = self._get_mid_frame(path)
                lut = self._build_3d_lut(target_frame, ref_frame)
                self._apply_lut_to_clip(path, lut, out_path)

                corrected_paths.append(out_path)
            except Exception as e:
                logger.warning(f"3D LUT 색보정 실패: {path}: {e}")
                corrected_paths.append(path)  # 원본 사용

        logger.info(f"DreamColour 3D LUT 색감 통일 완료: {len(corrected_paths)} clips")
        return corrected_paths

    def normalize_frames(
        self,
        frames: List[np.ndarray]
    ) -> List[np.ndarray]:
        """프레임 리스트의 색감 통일 (인메모리, 3D LUT 적용)

        Args:
            frames: [N, H, W, 3] (uint8, BGR)

        Returns:
            색보정된 프레임 리스트
        """
        if len(frames) <= 1:
            return frames

        ref_frame = frames[0]
        corrected = [ref_frame]

        for frame in frames[1:]:
            lut = self._build_3d_lut(frame, ref_frame)
            corrected.append(lut.apply(frame))

        return corrected

    def _build_3d_lut(
        self,
        source: np.ndarray,
        reference: np.ndarray
    ) -> LUT3D:
        """Training-free 3D LUT 생성 (DreamColour 방식)

        DreamColour의 핵심: 학습 없이 통계적 색상 매핑으로 LUT 생성

        프로세스:
          1) source와 reference를 LAB 색공간으로 변환
          2) 각 채널(L, A, B)의 평균/표준편차 계산
          3) 채널별 선형 변환 계수 도출
             (source 분포 → reference 분포로 매핑하는 affine 변환)
          4) 이 변환을 3D LUT 격자점에 사전 계산하여 기록
          5) 실제 적용 시에는 LUT 조회 + 삼선형 보간만 수행 → 초고속

        히스토그램 매칭과의 차이:
          - 히스토그램 매칭: 픽셀 단위로 매번 계산 → 느림
          - 3D LUT: 격자점만 사전 계산 → 적용 시 조회만 → 빠름
          - 3D LUT: R/G/B 채널 간 상호작용 반영 → 더 자연스러움

        Args:
            source: 타겟 프레임 [H, W, 3] uint8 BGR
            reference: 레퍼런스 프레임 [H, W, 3] uint8 BGR

        Returns:
            LUT3D 객체 (적용 준비 완료)
        """
        lut = LUT3D(size=self.lut_size)
        n = self.lut_size

        # LAB 변환
        src_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
        ref_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB).astype(np.float32)

        # 채널별 통계 (mean, std)
        src_stats = [(src_lab[:, :, c].mean(), src_lab[:, :, c].std() + 1e-6)
                     for c in range(3)]
        ref_stats = [(ref_lab[:, :, c].mean(), ref_lab[:, :, c].std() + 1e-6)
                     for c in range(3)]

        # 변환 계수: y = (x - src_mean) * (ref_std / src_std) + ref_mean
        # 이것을 RGB → LAB → 변환 → BGR 순으로 LUT 격자점에 적용

        # LUT 격자점의 RGB 값
        axis = np.linspace(0, 255, n, dtype=np.float32)
        r_grid, g_grid, b_grid = np.meshgrid(axis, axis, axis, indexing='ij')

        # 격자점 [N³, 3] BGR 이미지로 변환하여 LAB 변환
        grid_bgr = np.stack([b_grid, g_grid, r_grid], axis=-1)  # [N,N,N,3]
        grid_shape = grid_bgr.shape
        grid_flat = grid_bgr.reshape(-1, 1, 3).astype(np.uint8)

        grid_lab = cv2.cvtColor(grid_flat, cv2.COLOR_BGR2LAB).astype(np.float32)

        # LAB 채널별 색상 전이 적용
        for c in range(3):
            s_mean, s_std = src_stats[c]
            r_mean, r_std = ref_stats[c]

            # strength로 보정 강도 조절
            # strength=1.0: 완전 보정, 0.0: 원본 유지
            transfer = (grid_lab[:, 0, c] - s_mean) * (r_std / s_std) + r_mean
            original = grid_lab[:, 0, c]
            grid_lab[:, 0, c] = original * (1 - self.strength) + transfer * self.strength

        # LAB → BGR 역변환
        grid_lab = np.clip(grid_lab, 0, 255).astype(np.uint8)
        grid_bgr_out = cv2.cvtColor(grid_lab, cv2.COLOR_LAB2BGR).astype(np.float32)

        # LUT 테이블에 기록 (BGR 출력)
        lut.table = grid_bgr_out.reshape(grid_shape)

        return lut

    def _apply_lut_to_clip(
        self,
        input_path: str,
        lut: LUT3D,
        output_path: str
    ):
        """3D LUT를 비디오 클립 전체에 적용

        LUT가 이미 사전 계산되어 있으므로 프레임별 적용은
        테이블 조회 + 삼선형 보간만 수행 → 프레임당 수 ms.
        """
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            corrected = lut.apply(frame)
            out.write(corrected)

        cap.release()
        out.release()

    def _get_reference_frame(self, clip_paths: List[str]) -> np.ndarray:
        """레퍼런스 프레임 추출"""
        if self.reference_method == "first":
            cap = cv2.VideoCapture(clip_paths[0])
            ret, frame = cap.read()
            cap.release()
            if ret:
                return frame

        if self.reference_method == "mean":
            return self._compute_mean_reference(clip_paths)

        # 폴백: 첫 번째 클립의 중간 프레임
        return self._get_mid_frame(clip_paths[0])

    def _compute_mean_reference(self, clip_paths: List[str]) -> np.ndarray:
        """전체 클립의 평균 색감을 레퍼런스로 사용

        각 클립의 중간 프레임을 추출하여 평균을 계산.
        개별 클립의 편향을 줄여서 더 중립적인 레퍼런스 생성.
        """
        frames = []
        for path in clip_paths:
            frame = self._get_mid_frame(path)
            if frame is not None:
                # 모든 프레임을 동일 크기로 리사이즈
                frame = cv2.resize(frame, (224, 224))
                frames.append(frame.astype(np.float32))

        if not frames:
            return np.zeros((224, 224, 3), dtype=np.uint8)

        mean_frame = np.mean(frames, axis=0).astype(np.uint8)
        return mean_frame

    @staticmethod
    def _get_mid_frame(video_path: str) -> Optional[np.ndarray]:
        """비디오 중간 시점의 프레임 추출"""
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
        ret, frame = cap.read()
        cap.release()
        return frame if ret else np.zeros((224, 224, 3), dtype=np.uint8)

    def export_lut_cube(self, lut: LUT3D, output_path: str):
        """3D LUT를 .cube 파일로 내보내기 (DaVinci Resolve/Premiere 호환)

        .cube 형식은 업계 표준 3D LUT 포맷.
        이 파일을 전문 편집 소프트웨어에서 직접 로드 가능.

        Args:
            lut: LUT3D 객체
            output_path: .cube 파일 경로
        """
        n = lut.size
        with open(output_path, 'w') as f:
            f.write(f"# DreamColour Training-free 3D LUT\n")
            f.write(f"# VideoRAG Prototype\n")
            f.write(f"LUT_3D_SIZE {n}\n")
            f.write(f"DOMAIN_MIN 0.0 0.0 0.0\n")
            f.write(f"DOMAIN_MAX 1.0 1.0 1.0\n")
            f.write(f"\n")

            for b_idx in range(n):
                for g_idx in range(n):
                    for r_idx in range(n):
                        # .cube 형식은 RGB 순서, 0~1 범위
                        bgr = lut.table[r_idx, g_idx, b_idx]
                        r_val = bgr[2] / 255.0  # B → R
                        g_val = bgr[1] / 255.0
                        b_val = bgr[0] / 255.0  # R → B
                        f.write(f"{r_val:.6f} {g_val:.6f} {b_val:.6f}\n")

        logger.info(f"3D LUT 내보내기: {output_path} ({n}×{n}×{n})")
