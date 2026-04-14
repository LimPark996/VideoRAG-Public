"""
④ indexer.py — Phase 0 오케스트레이터

역할: Shot 탐지 → 임베딩 → FAISS 인덱스 + BM25 인덱스 구축
     모든 Phase 0 컴포넌트를 조율하는 오케스트레이터

설계: 자체 설계 (파이프라인 패턴)
"""

import os
import time
import json
import pickle
import logging
from typing import List, Optional, Dict
from tqdm import tqdm

import numpy as np
import pandas as pd
import cv2

from ..data_models import ClipMeta, IndexBuildResult
from .shot_detector import ShotDetector
from .embedder import VideoEmbedder
from .vector_store import FAISSVectorStore
from ..phase12_search.bm25_retriever import (
    SpacyLemmatizerTokenizer,
    KoreanMorphTokenizer,
)

logger = logging.getLogger(__name__)


class VideoIndexer:
    """Phase 0 오프라인 인덱싱 오케스트레이터

    build_index(video_dir, metadata_csv) → IndexBuildResult
    """

    def __init__(
        self,
        shot_detector: Optional[ShotDetector] = None,
        embedder: Optional[VideoEmbedder] = None,
        vector_store: Optional[FAISSVectorStore] = None,
        index_dir: str = "index",
        keyframe_dir: str = "data/msrvtt/keyframes",
        num_frames: int = 4,
        tokenizer_mode: str = "spacy",
    ):
        """
        Args:
            shot_detector: Shot 탐지기 (기본: ShotDetector())
            embedder: 영상 임베더 (기본: VideoEmbedder())
            vector_store: FAISS 벡터 저장소 (기본: FAISSVectorStore())
            index_dir: 인덱스 파일 저장 디렉토리
            keyframe_dir: 키프레임 이미지 저장 디렉토리
            num_frames: 클립당 샘플링 프레임 수
            tokenizer_mode: BM25 토크나이저 선택
                - "spacy": SpacyLemmatizerTokenizer (영어 spaCy Lemmatizer, 기본값)
                - "korean": KoreanMorphTokenizer (Okt 형태소 분석, 레거시)
                - "whitespace": 단순 공백 분할 (디버깅용)
        """
        self.shot_detector = shot_detector or ShotDetector()
        self.embedder = embedder or VideoEmbedder()
        self.vector_store = vector_store or FAISSVectorStore()
        self.index_dir = index_dir
        self.keyframe_dir = keyframe_dir
        self.num_frames = num_frames  # 클립당 샘플링할 프레임 수
        self.tokenizer_mode = tokenizer_mode

        # ✅ 토크나이저 모드에 따라 적절한 토크나이저 선택 (v2: spacy 기본)
        if tokenizer_mode == "spacy":
            self._bm25_tokenizer = SpacyLemmatizerTokenizer()
            logger.info("BM25 토크나이저: SpacyLemmatizerTokenizer (영어)")
        elif tokenizer_mode == "korean":
            self._bm25_tokenizer = KoreanMorphTokenizer()
            logger.info("BM25 토크나이저: KoreanMorphTokenizer (Okt 형태소 분석, 레거시)")
        else:
            self._bm25_tokenizer = None
            logger.info("BM25 토크나이저: whitespace (단순 공백 분할)")

        # 클립 메타데이터 저장소
        self.clip_metadata: Dict[str, ClipMeta] = {}

    def build_index(
        self,
        video_dir: str,
        metadata_csv: Optional[str] = None,
        max_clips: Optional[int] = None,
    ) -> IndexBuildResult:
        """전체 인덱싱 파이프라인 실행

        파이프라인 연결 순서:
            ShotDetector → Indexer
            1) metadata_csv(MSR-VTT 원본)에서 caption 필드 직접 사용
            2) 각 MSR-VTT 영상에 ShotDetector 실행 → shot 단위 클립 생성
            3) BM25 인덱스: caption → spaCy Lemmatizer로 구축
            4) FAISS 인덱스: caption → InternVideo2 임베딩

        Args:
            video_dir: 영상 파일 디렉토리 (MSR-VTT 샘플)
            metadata_csv: MSR-VTT 메타데이터 CSV 경로 (caption/sentence 열 포함)
            max_clips: 최대 클립 수 제한 (테스트용)

        Returns:
            IndexBuildResult
        """
        start_time = time.perf_counter()
        os.makedirs(self.index_dir, exist_ok=True)
        os.makedirs(self.keyframe_dir, exist_ok=True)

        # ── Step 1: Shot 탐지 → 클립 생성 ──────────────────────────────
        if metadata_csv and os.path.exists(metadata_csv):
            # 레거시: CSV 기반 (하위 호환)
            print(f'\n📄 [Step 1] metadata_csv 발견 → CSV에서 클립 메타데이터 로드 (레거시)')
            print(f'   경로: {metadata_csv}')
            clips = self._load_metadata_csv(metadata_csv, video_dir)
        else:
            print(f'\n🎬 [Step 1] 영상 파일에서 ShotDetector로 클립 생성')
            clips = self._build_clips_from_videos(video_dir)

        if max_clips:
            clips = clips[:max_clips]
            print(f'   ⚠️  max_clips={max_clips} 제한 적용 → {len(clips)}개로 축소')

        print(f'\n✅ [Step 1 완료] 총 {len(clips)}개 클립 준비 (MSR-VTT 원본 캡션 사용)')
        logger.info(f'총 {len(clips)}개 클립 준비 완료')

        # Step 2~6: 공통 인덱스 구축 로직
        return self._build_from_clips(clips, start_time)

    def _get_video_duration_ms(self, video_path: str) -> float:
        """영상 파일에서 실제 재생 길이(ms)를 cv2로 직접 계산

        MSR-VTT CSV의 'start time'/'end time' 열은 부정확한 값이 다수 존재하므로,
        영상 파일에서 fps × frame_count로 직접 산출한다.

        Args:
            video_path: 영상 파일 경로

        Returns:
            재생 길이(ms). 파일이 없거나 읽기 실패 시 30000.0(30초) 반환.
        """
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            cap.release()
            if fps > 0 and frame_count > 0:
                return (frame_count / fps) * 1000.0
        except Exception as e:
            logger.debug(f'영상 길이 계산 실패: {video_path}: {e}')
        return 30000.0  # 기본 30초

    def _load_metadata_csv(
        self,
        csv_path: str,
        video_dir: str
    ) -> List[ClipMeta]:
        """MSR-VTT 메타데이터 CSV 로드 + scene_id 할당

        변경사항 (v3):
          CSV의 'start time'/'end time' 열은 사용하지 않는다.
          MSR-VTT의 각 클립은 독립 영상 파일이므로, start_ms=0.0으로 고정하고
          end_ms(=duration)는 cv2로 영상 파일에서 직접 계산한다.
          CSV의 시간 값이 부정확한 사례가 다수 발견되어 직접 계산 방식으로 교체.

          CSV에 scene_id 열이 있으면 그대로 사용.
          없으면 같은 video_id를 공유하는 클립들을 같은 scene으로 간주하여
          자동으로 scene_id를 부여합니다.
          MSR-VTT에서는 각 클립이 서로 다른 영상이므로
          video_id 기반 scene_id가 곧 "출처 영상 그룹"을 의미합니다.
        """
        df = pd.read_csv(csv_path)
        print(f'   CSV 로드 완료: {len(df)}행, 열={list(df.columns)}')

        clips = []
        path_inferred_count = 0
        path_from_csv_count = 0
        duration_computed_count = 0  # cv2로 직접 계산한 건수

        # scene_id 자동 할당: video_id별 고유 번호 부여
        has_scene_col = 'scene_id' in df.columns
        video_id_to_scene = {}
        scene_counter = 0

        for _, row in df.iterrows():
            clip_id = str(row.get('clip_id', row.get('video_id', '')))

            # video_path 처리: CSV에 있으면 사용, 없으면 {clip_id}.mp4 추정
            if 'video_path' in df.columns and pd.notna(row.get('video_path')):
                video_file = row['video_path']
                path_from_csv_count += 1
            else:
                video_file = f'{clip_id}.mp4'
                path_inferred_count += 1

            video_path = os.path.join(video_dir, video_file)

            # scene_id 결정
            if has_scene_col and pd.notna(row.get('scene_id')):
                scene_id = int(row['scene_id'])
            else:
                vid = str(row.get('video_id', clip_id))
                if vid not in video_id_to_scene:
                    video_id_to_scene[vid] = scene_counter
                    scene_counter += 1
                scene_id = video_id_to_scene[vid]

            # ── 시간 범위: 항상 영상 파일에서 직접 계산 ──────────────────
            # MSR-VTT CSV의 'start time'/'end time'은 부정확하고,
            # 'start_ms'/'end_ms' 열이 있어도 0/30000 같은 더미 값이 들어 있음.
            # 따라서 CSV 시간 값은 일체 사용하지 않고,
            # 각 클립 영상 파일에서 cv2로 실제 재생 길이를 직접 측정한다.
            start_ms = 0.0
            end_ms = self._get_video_duration_ms(video_path)
            duration_computed_count += 1

            clip = ClipMeta(
                clip_id=clip_id,
                video_id=str(row.get('video_id', clip_id)),
                caption=str(row.get('caption', row.get('sentence', ''))),
                start_ms=start_ms,
                end_ms=end_ms,
                video_path=video_path,
                scene_id=scene_id
            )
            clips.append(clip)

        # video_path 출처 요약
        print(f'\n   📋 video_path 처리 결과:')
        if path_from_csv_count > 0:
            print(f'      CSV에서 가져옴:  {path_from_csv_count}개  (열: video_path)')
        if path_inferred_count > 0:
            print(f'      경로 추정:       {path_inferred_count}개  (패턴: {{clip_id}}.mp4)')
        if duration_computed_count > 0:
            print(f'      시간 직접 계산:  {duration_computed_count}개  (cv2 fps×frames, CSV 시간 무시)')

        # 샘플 출력 (최대 3개)
        print(f'\n   📌 클립 샘플 (최대 3개):')
        for c in clips[:3]:
            path_exists = '✓ 존재' if os.path.exists(c.video_path) else '✗ 없음'
            print(f'      clip_id={c.clip_id}  video_path={c.video_path}  [{path_exists}]')
            print(f'        caption="{c.caption[:60]}{"..." if len(c.caption) > 60 else ""}"')
            print(f'        구간: {c.start_ms:.0f}ms ~ {c.end_ms:.0f}ms')

        logger.info(f'CSV에서 {len(clips)}개 클립 로드')
        return clips

    def _build_clips_from_videos(self, video_dir: str) -> List[ClipMeta]:
        """비디오 파일들에서 Shot 탐지 + Scene 클러스터링으로 클립 생성

        변경사항 (v3 — scene = clip):
          이전(v2): shot마다 clip 생성 → 3초짜리 조각이 검색 단위
          현재(v3): scene 단위로 clip 생성 → ~30초 의미 단위가 검색·합성 기본 단위
                    PD 관점에서 자연스러운 장면 단위로 검색·합성 가능
        """
        video_files = sorted([
            f for f in os.listdir(video_dir)
            if f.endswith(('.mp4', '.avi', '.mkv', '.webm'))
        ])
        print(f'   영상 파일 발견: {len(video_files)}개  (디렉토리: {video_dir})')

        clips = []
        total_scenes = 0
        for vf in tqdm(video_files, desc='🎬 Shot→Scene 탐지', unit='video'):
            video_path = os.path.join(video_dir, vf)
            video_id = os.path.splitext(vf)[0]
            try:
                # Scene 탐지 (Shot 탐지 + Agglomerative Clustering 포함)
                scenes = self.shot_detector.detect_scenes(video_path)
                total_scenes += len(scenes)
                for scene in scenes:
                    clip = ClipMeta(
                        clip_id=f'{video_id}_scene{scene.scene_id:03d}',
                        video_id=video_id,
                        caption='',
                        start_ms=scene.start_ms,
                        end_ms=scene.end_ms,
                        video_path=video_path,
                        scene_id=scene.scene_id
                    )
                    clips.append(clip)
            except Exception as e:
                logger.warning(f'Scene 탐지 실패: {vf}: {e}')

        print(f'   Scene 탐지 완료: {len(video_files)}개 영상 → {total_scenes}개 Scene(= 클립)')
        return clips

    def _extract_clip_frames(
        self,
        clips: List[ClipMeta]
    ) -> List[List[np.ndarray]]:
        """클립별 균등 간격 num_frames장 프레임 추출

        샘플링 공식: start_ms + step_ms * (i + 0.5)
          - 0.5를 더하는 이유: 각 구간의 중심점을 찍기 위함
          - 예) 0~10000ms, num_frames=4
            step_ms=2500 → 시점: 1250, 3750, 6250, 8750ms
        """
        print(f'\n   샘플링 공식: start_ms + (duration_ms / {self.num_frames}) * (i + 0.5)')
        print(f'   예시) 0~10000ms → step=2500ms → [1250, 3750, 6250, 8750]ms')

        all_clip_frames = []
        dummy_count = 0       # 랜덤 더미로 채운 프레임 수
        pad_count = 0         # 마지막 프레임 복제로 패딩한 클립 수
        thumb_saved_count = 0 # 썸네일 저장 성공 수

        for clip in tqdm(clips, desc='🖼️  프레임 추출', unit='clip'):
            duration_ms = clip.end_ms - clip.start_ms
            step_ms = duration_ms / self.num_frames
            sample_times = [
                clip.start_ms + step_ms * (i + 0.5)
                for i in range(self.num_frames)
            ]

            clip_frames = []
            cap = None
            clip_dummy_frames = 0

            try:
                if os.path.exists(clip.video_path):
                    cap = cv2.VideoCapture(clip.video_path)

                    for idx, t_ms in enumerate(sample_times):
                        cap.set(cv2.CAP_PROP_POS_MSEC, t_ms)
                        ret, frame = cap.read()

                        if ret:
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            clip_frames.append(frame_rgb)
                        else:
                            # 해당 시점 읽기 실패 → 랜덤 더미로 채움
                            logger.debug(
                                f'프레임 읽기 실패: {clip.clip_id} @ {t_ms:.0f}ms'
                            )
                            dummy = np.random.randint(
                                0, 255, (224, 224, 3), dtype=np.uint8
                            )
                            clip_frames.append(dummy)
                            clip_dummy_frames += 1

                    # ── 썸네일 저장: 4장 중 두 번째 프레임 ──────────────
                    # 4장 모두 저장하면 디스크 낭비 → 썸네일 1장만 저장
                    if clip_frames:
                        thumb_idx = min(1, len(clip_frames) - 1)
                        kf_path = os.path.join(
                            self.keyframe_dir, f'{clip.clip_id}.jpg'
                        )
                        # RGB → BGR 변환 후 저장 (OpenCV는 BGR 포맷)
                        save_ok = cv2.imwrite(
                            kf_path,
                            cv2.cvtColor(clip_frames[thumb_idx], cv2.COLOR_RGB2BGR)
                        )
                        if save_ok:
                            clip.keyframe_path = kf_path
                            thumb_saved_count += 1
                        else:
                            logger.debug(f'썸네일 저장 실패: {kf_path}')

            except Exception as e:
                logger.debug(f'프레임 추출 실패: {clip.clip_id}: {e}')
            finally:
                if cap is not None:
                    cap.release()  # VideoCapture 반드시 닫기

            # ── 프레임이 0장이면 전부 더미로 채움 ────────────────────
            if len(clip_frames) == 0:
                clip_frames = [
                    np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                    for _ in range(self.num_frames)
                ]
                clip_dummy_frames = self.num_frames

            # ── num_frames보다 적으면 마지막 프레임 복제로 패딩 ──────
            if len(clip_frames) < self.num_frames:
                pad_count += 1
                while len(clip_frames) < self.num_frames:
                    clip_frames.append(clip_frames[-1])

            dummy_count += clip_dummy_frames
            all_clip_frames.append(clip_frames)

        # ── 추출 결과 요약 ────────────────────────────────────────────
        total_frames = len(clips) * self.num_frames
        print(f'\n   📊 프레임 추출 결과:')
        print(f'      전체 클립:          {len(clips)}개')
        print(f'      목표 프레임:        {total_frames}장  ({len(clips)} × {self.num_frames})')
        print(f'      랜덤 더미 프레임:   {dummy_count}장  (읽기 실패 → 224×224 랜덤 배열로 대체)')
        print(f'      마지막 프레임 패딩: {pad_count}개 클립  (num_frames 미달 시 마지막 프레임 복제)')
        print(f'      썸네일 저장 성공:   {thumb_saved_count}개  (클립당 2번째 프레임, keyframe_dir)')
        print(f'      저장 경로:          {self.keyframe_dir}/')

        return all_clip_frames

    def _build_bm25_index(
        self,
        captions: List[str],
        clip_ids: List[str]
    ) -> str:
        """BM25 인덱스 구축 및 저장

        토크나이저는 self.tokenizer_mode에 따라 결정:
        - "korean": KoreanMorphTokenizer (Okt 형태소 분석)
        - "spacy": SpacyLemmatizerTokenizer (영어 Lemmatizer)
        - "whitespace": 단순 공백 분할

        출처:
        - rank_bm25 라이브러리: https://github.com/dorianbrown/rank_bm25
        - BM25 알고리즘 원본: Robertson & Zaragoza (2009)
          "The Probabilistic Relevance Framework: BM25 and Beyond"
        """
        from rank_bm25 import BM25Okapi

        # ── Step A: 토크나이징 ────────────────────────────────────────
        tokenizer_desc = {
            "korean": "KoreanMorphTokenizer (Okt 형태소 분석)",
            "spacy": "SpacyLemmatizerTokenizer (en_core_web_sm)",
            "whitespace": "WhitespaceTokenizer (단순 공백 분할)"
        }.get(self.tokenizer_mode, self.tokenizer_mode)

        print(f'   [BM25 Step A] 토크나이징 중...')
        print(f'     tokenizer: {tokenizer_desc}')
        print(f'     처리 대상: {len(captions)}개 캡션')

        if self._bm25_tokenizer is not None:
            tokenized = [self._bm25_tokenizer.tokenize(cap) for cap in captions]
        else:
            tokenized = [cap.lower().strip().split() for cap in captions]

        # 샘플 토크나이징 결과 출력 (최대 3개)
        print(f'     📌 토크나이징 샘플 (최대 3개):')
        for i, (cap, tokens) in enumerate(zip(captions[:3], tokenized[:3])):
            print(f'       [{i}] 원문: "{cap[:60]}{"..." if len(cap) > 60 else ""}"')
            print(f'           토큰:  {tokens}')

        avg_tokens = sum(len(t) for t in tokenized) / max(len(tokenized), 1)
        print(f'     평균 토큰 수: {avg_tokens:.1f}개/캡션')

        # ── Step B: BM25 인덱스 구축 ─────────────────────────────────
        print(f'   [BM25 Step B] BM25Okapi 인덱스 구축 중...')
        bm25 = BM25Okapi(tokenized)
        print(f'     완료: {len(captions)}개 문서 인덱싱  (k1=1.5, b=0.75)')

        # ── Step C: 피클 저장 ─────────────────────────────────────────
        bm25_path = os.path.join(self.index_dir, 'bm25_index.pkl')
        print(f'   [BM25 Step C] 인덱스 저장 중: {bm25_path}')
        with open(bm25_path, 'wb') as f:
            pickle.dump({
                'bm25': bm25,
                'clip_ids': clip_ids,
                'captions': captions,
                'tokenized': tokenized,
                'tokenizer_mode': self.tokenizer_mode,
            }, f)
        print(f'     저장 완료: {len(captions)}개 문서, tokenizer={self.tokenizer_mode}')

        logger.info(
            f'BM25 인덱스 저장: {bm25_path} ({len(captions)} docs, '
            f'tokenizer={self.tokenizer_mode})'
        )
        return bm25_path

    def build_index_from_grouper(
        self,
        grouper_result,
        caption_json: Optional[str] = None,
        output_csv: Optional[str] = None,
        max_clips: Optional[int] = None,
    ) -> IndexBuildResult:
        """[DEPRECATED] video_grouper → shot_detector → caption_ko → indexer 전체 파이프라인

        v2에서 VideoGrouper가 제거되어 이 메서드는 더 이상 사용하지 않습니다.
        대신 build_index(video_dir, metadata_json=...) 을 사용하세요.

        Phase 0 전체 플로우를 하나의 메서드로 오케스트레이션:
        1) video_grouper의 GroupedIndexingResult에서 merged video 목록 추출
        2) 각 merged video에 대해 shot_detector.detect_scenes() 실행
        3) caption_ko 출력(JSON)에서 한국어 캡션 병합
        4) BM25 인덱스는 tokenizer_mode에 맞는 토크나이저로 구축

        Args:
            grouper_result: VideoGrouper.group_and_merge() 반환값 (GroupedIndexingResult)
            caption_json: caption_ko 출력 JSON 경로 (한국어 캡션, optional)
            output_csv: 중간 결과 CSV 저장 경로 (optional, 디버깅용)
            max_clips: 최대 클립 수 제한 (테스트용)

        Returns:
            IndexBuildResult
        """
        start_time = time.perf_counter()
        os.makedirs(self.index_dir, exist_ok=True)
        os.makedirs(self.keyframe_dir, exist_ok=True)

        # ── Step 1: merged video에서 Shot → Scene 탐지 ──────────────
        print(f'\n🎬 [Step 1] video_grouper 결과에서 Shot→Scene 탐지')
        print(f'   합쳐진 영상: {len(grouper_result.merged_videos)}개 카테고리')
        print(f'   원본 영상 총: {grouper_result.total_source_videos}개')

        clips = []
        total_scenes = 0

        # 한국어 캡션 로드 (있으면)
        ko_captions: Dict[str, str] = {}
        if caption_json and os.path.exists(caption_json):
            with open(caption_json, 'r', encoding='utf-8') as f:
                ko_captions = json.load(f)
            print(f'   한국어 캡션 로드: {len(ko_captions)}개')

        for merged in grouper_result.merged_videos:
            print(f'\n   📂 카테고리: {merged.category}')
            print(f'      영상: {merged.merged_path}')
            print(f'      원본 영상: {len(merged.source_videos)}개')

            try:
                # shot_detector로 Scene 탐지
                scenes = self.shot_detector.detect_scenes(merged.merged_path)
                total_scenes += len(scenes)
                print(f'      → {len(scenes)}개 Scene 탐지')

                for scene in scenes:
                    # scene 단위로 clip 생성 (v3: scene = clip)
                    scene_mid_ms = (scene.start_ms + scene.end_ms) / 2
                    source_vid = self._find_source_video(
                        scene_mid_ms, merged.source_videos,
                        merged.source_boundaries_ms
                    )
                    clip_id = f'{source_vid}_scene{scene.scene_id:03d}'

                    # 캡션: 한국어 캡션 → 빈 문자열 폴백
                    caption = ko_captions.get(source_vid, '')

                    clip = ClipMeta(
                        clip_id=clip_id,
                        video_id=source_vid,
                        caption=caption,
                        start_ms=scene.start_ms,
                        end_ms=scene.end_ms,
                        video_path=merged.merged_path,
                        scene_id=scene.scene_id,
                    )
                    clips.append(clip)

            except Exception as e:
                logger.warning(f'Scene 탐지 실패 ({merged.category}): {e}')

        print(f'\n✅ [Step 1 완료] {total_scenes}개 Scene(= 클립) → {len(clips)}개 클립')

        # 중간 결과 CSV 저장 (디버깅용)
        if output_csv:
            self._save_clips_csv(clips, output_csv)
            print(f'   중간 CSV 저장: {output_csv}')

        if max_clips:
            clips = clips[:max_clips]
            print(f'   ⚠️  max_clips={max_clips} 제한 적용')

        # ── Step 2~6: 기존 build_index 플로우 재사용 ──────────────────
        return self._build_from_clips(clips, start_time)

    def _find_source_video(
        self,
        time_ms: float,
        source_videos: List[str],
        boundaries: list,
    ) -> str:
        """시간 구간으로 원본 video_id 역매핑

        merged video 내의 특정 시점이 어느 원본 영상에 해당하는지 찾기.
        video_grouper가 기록한 source_boundaries_ms를 이용.
        """
        for vid, (start, end) in zip(source_videos, boundaries):
            if start <= time_ms <= end:
                return vid
        # 폴백: 가장 가까운 영상
        return source_videos[-1] if source_videos else "unknown"

    def _save_clips_csv(self, clips: List[ClipMeta], csv_path: str):
        """클립 메타데이터를 CSV로 저장 (디버깅용)"""
        rows = []
        for c in clips:
            rows.append({
                'clip_id': c.clip_id,
                'video_id': c.video_id,
                'caption': c.caption,
                'start_ms': c.start_ms,
                'end_ms': c.end_ms,
                'video_path': c.video_path,
                'scene_id': c.scene_id,
            })
        df = pd.DataFrame(rows)
        os.makedirs(os.path.dirname(csv_path) if os.path.dirname(csv_path) else '.', exist_ok=True)
        df.to_csv(csv_path, index=False, encoding='utf-8')

    def _build_from_clips(
        self,
        clips: List[ClipMeta],
        start_time: float,
    ) -> IndexBuildResult:
        """클립 목록으로부터 인덱스 구축 (Step 2~6 공통 로직)

        build_index()와 build_index_from_grouper() 양쪽에서 재사용.
        """
        # ── Step 2: 균등 프레임 추출 ──────────────────────────────────
        print(f'\n🖼️  [Step 2] 클립당 {self.num_frames}프레임 균등 추출 시작')
        frames = self._extract_clip_frames(clips)
        print(f'✅ [Step 2 완료] 총 {len(frames)}개 클립 × {self.num_frames}프레임 추출')

        # ── Step 3: 임베딩 벡터 생성 ──────────────────────────────────
        print(f'\n⚡ [Step 3] 임베딩 생성 중... (batch_size=8)')
        embeddings = self.embedder.encode_clips(frames, batch_size=8)
        print(f'✅ [Step 3 완료] 임베딩 shape: {embeddings.shape}  dtype: {embeddings.dtype}')

        # ── Step 4: FAISS 인덱스 구축 ─────────────────────────────────
        print(f'\n🗂️  [Step 4] FAISS 인덱스 구축 중...')
        clip_ids = [c.clip_id for c in clips]
        n_indexed = self.vector_store.build_index(embeddings, clip_ids)
        print(f'✅ [Step 4 완료] FAISS 인덱스에 {n_indexed}개 벡터 등록')

        # ── Step 5: BM25 인덱스 구축 ──────────────────────────────────
        print(f'\n🔍 [Step 5] BM25 인덱스 구축 중...')
        captions = [c.caption for c in clips]
        bm25_path = self._build_bm25_index(captions, clip_ids)
        print(f'✅ [Step 5 완료] BM25 인덱스 저장: {bm25_path}')

        # ── Step 6: 저장 ───────────────────────────────────────────────
        print(f'\n💾 [Step 6] 인덱스 파일 저장 중...')
        faiss_path = os.path.join(self.index_dir, 'faiss_ivfflat.index')
        self.vector_store.save(faiss_path)
        print(f'   FAISS 인덱스 저장: {faiss_path}')

        self.clip_metadata = {c.clip_id: c for c in clips}
        metadata_path = os.path.join(self.index_dir, 'clip_metadata.pkl')
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.clip_metadata, f)
        print(f'   클립 메타데이터 저장: {metadata_path}')
        print(f'   → 총 {len(self.clip_metadata)}개 항목')

        build_time = time.perf_counter() - start_time
        print(f'\n🎉 [전체 완료] {len(clips)}개 클립 인덱싱 | 소요 시간: {build_time:.1f}초')

        result = IndexBuildResult(
            n_clips=len(clips),
            n_vectors=n_indexed,
            faiss_index_path=faiss_path,
            bm25_index_path=bm25_path,
            metadata_path=metadata_path,
            build_time_sec=build_time
        )
        logger.info(
            f'인덱싱 완료: {result.n_clips} clips, '
            f'{result.n_vectors} vectors, '
            f'{result.build_time_sec:.1f}초'
        )
        return result

    def build_itm_vision_features(
        self,
        video_dir: str,
        metadata_csv: Optional[str] = None,
        max_clips: Optional[int] = None,
        batch_size: int = 4,
    ) -> str:
        """ITM용 vision full token 시퀀스 사전 추출 및 저장

        ITC 인덱스(FAISS) 구축 이후 별도 실행하는 단계.
        ColBERT 이후 ITM 재순위에 쓸 vision full token [N, 1025, 1408]을
        미리 계산해두어, 쿼리 타임에 영상 재인코딩 없이 즉시 로드 가능하게 함.

        저장 파일: {index_dir}/itm_vision_features.pt
          → {'clip_ids': List[str], 'features': Tensor[N, 1025, 1408] fp16}

        용량 참고 (fp16):
          1000클립 × 1025 × 1408 × 2bytes ≈ 2.9GB
          콜랩 Drive에 저장 권장 (런타임 재시작 후 재사용 가능)

        Args:
            video_dir: 영상 파일 디렉토리
            metadata_csv: MSR-VTT 메타데이터 CSV (없으면 기존 clip_metadata 사용)
            max_clips: 최대 클립 수 제한 (테스트용)
            batch_size: encode_clips_itm() 배치 크기 (OOM 시 2로 낮춤)

        Returns:
            str: 저장 경로
        """
        import torch

        os.makedirs(self.index_dir, exist_ok=True)

        # ── 클립 목록 확정 ─────────────────────────────────────────────
        # 우선순위: 1) 이미 self.clip_metadata 로드됨 → 재사용
        #           2) metadata_csv 있으면 로드
        #           3) video_dir에서 직접 탐색
        if self.clip_metadata:
            clips = list(self.clip_metadata.values())
            print(f'\n📋 [ITM feature] 기존 clip_metadata 사용: {len(clips)}개 클립')
        elif metadata_csv and os.path.exists(metadata_csv):
            clips = self._load_metadata_csv(metadata_csv, video_dir)
            print(f'\n📄 [ITM feature] metadata_csv에서 로드: {len(clips)}개 클립')
        else:
            clips = self._build_clips_from_videos(video_dir)
            print(f'\n🎬 [ITM feature] 영상에서 클립 탐지: {len(clips)}개 클립')

        if max_clips:
            clips = clips[:max_clips]
            print(f'   ⚠️  max_clips={max_clips} 제한 적용')

        clip_ids = [c.clip_id for c in clips]

        # ── 프레임 추출 ────────────────────────────────────────────────
        print(f'\n🖼️  [ITM feature Step 1] 프레임 추출 중...')
        frames = self._extract_clip_frames(clips)

        # ── ITM vision feature 추출 ────────────────────────────────────
        print(f'\n⚡ [ITM feature Step 2] encode_clips_itm() 실행 중...')
        print(f'   예상 소요: 클립당 약 0.35초 × {len(clips)}클립 ≈ {len(clips)*0.35/60:.1f}분 (T4 기준)')
        vis_feats = self.embedder.encode_clips_itm(frames, batch_size=batch_size)
        # vis_feats: [N, 1025, 1408] fp16 CPU tensor

        # ── 저장 ──────────────────────────────────────────────────────
        itm_path = os.path.join(self.index_dir, 'itm_vision_features.pt')
        print(f'\n💾 [ITM feature Step 3] 저장 중: {itm_path}')
        torch.save({
            'clip_ids': clip_ids,
            'features': vis_feats,   # [N, 1025, 1408] fp16
        }, itm_path)

        mem_gb = vis_feats.element_size() * vis_feats.nelement() / 1e9
        print(f'✅ [ITM feature 완료] {len(clips)}클립 저장')
        print(f'   경로  : {itm_path}')
        print(f'   shape : {tuple(vis_feats.shape)}')
        print(f'   크기  : {mem_gb:.2f}GB (fp16)')

        logger.info(
            f'ITM vision feature 저장: {itm_path} '
            f'({len(clips)} clips, {tuple(vis_feats.shape)}, {mem_gb:.2f}GB)'
        )
        return itm_path

    def load_index(self):
        """사전 구축된 인덱스 로드 (발표 당일 사용)"""
        faiss_path = os.path.join(self.index_dir, 'faiss_ivfflat.index')
        metadata_path = os.path.join(self.index_dir, 'clip_metadata.pkl')

        self.vector_store.load(faiss_path)

        with open(metadata_path, 'rb') as f:
            self.clip_metadata = pickle.load(f)

        logger.info(
            f'인덱스 로드 완료: {self.vector_store.ntotal} vectors, '
            f'{len(self.clip_metadata)} clips'
        )