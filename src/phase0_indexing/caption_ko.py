"""
caption_ko.py — 한글 캡션 생성 파이프라인

MSR-VTT 영상에 대해 GPT-4o-mini Vision API로 한글 캡션을 생성한다.
기존 영문 캡션 JSON과 별도로, 한글 캡션 JSON을 새로 생성하여 저장.

사용법:
    generator = KoreanCaptionGenerator(api_key="sk-...")
    generator.generate_all(
        video_dir="/path/to/videos",
        output_path="/path/to/msrvtt_1000_captions_ko.json",
        sample_list_csv="/path/to/sample_list.csv"  # Optional: 샘플링할 경우
    )

출력 JSON 형식:
    {
        "video0001": "한 남성이 기타를 연주하고 있다",
        "video0002": "부엌에서 요리를 하는 여성",
        ...
    }
"""

import os
import json
import time
import base64
import logging
from typing import Dict, List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class KoreanCaptionGenerator:
    """GPT-4o-mini Vision 기반 한글 캡션 생성기"""

    # 프레임 추출 설정
    N_FRAMES = 8
    FRAME_WIDTH = 512
    FRAME_HEIGHT = 288

    # GPT 프롬프트 (한글 출력 지정)
    SYSTEM_PROMPT = (
        "당신은 영상 분석 전문가입니다. "
        "제공된 영상 프레임들을 분석하여, 이 영상 클립의 내용을 "
        "한국어 한 문장으로 간결하고 정확하게 설명해주세요. "
        "반드시 한국어로만 작성하세요."
    )

    USER_PROMPT = (
        "이 영상의 8개 프레임을 보고, 영상 전체 내용을 "
        "한국어 한 문장(15~30자)으로 설명해주세요. "
        "예시: '한 남성이 무대에서 기타를 연주하고 있다', "
        "'아이들이 공원에서 뛰어놀고 있다'"
    )

    def __init__(self, api_key: Optional[str] = None):
        """
        Args:
            api_key: OpenAI API 키. None이면 환경변수 OPENAI_API_KEY 사용.
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._client = None

    def _get_client(self):
        """OpenAI 클라이언트 지연 초기화"""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
            except ImportError:
                raise RuntimeError("openai 패키지가 필요합니다: pip install openai")
        return self._client

    def extract_frames(self, video_path: str) -> List[str]:
        """영상에서 N_FRAMES개 프레임을 균등 추출 → base64 인코딩

        Args:
            video_path: 영상 파일 경로

        Returns:
            base64 인코딩된 JPEG 이미지 문자열 리스트
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.warning(f"영상 열기 실패: {video_path}")
            return []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            return []

        # 균등 간격 프레임 인덱스
        indices = np.linspace(0, total_frames - 1, self.N_FRAMES, dtype=int)

        frames_b64 = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue

            # 리사이즈
            frame = cv2.resize(frame, (self.FRAME_WIDTH, self.FRAME_HEIGHT))

            # JPEG 인코딩 → base64
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            b64 = base64.b64encode(buffer).decode('utf-8')
            frames_b64.append(b64)

        cap.release()
        return frames_b64

    def generate_caption(self, video_path: str) -> Optional[str]:
        """단일 영상에 대한 한글 캡션 생성

        Args:
            video_path: 영상 파일 경로

        Returns:
            한글 캡션 문자열, 실패 시 None
        """
        frames_b64 = self.extract_frames(video_path)
        if not frames_b64:
            return None

        client = self._get_client()

        # Vision API 호출 메시지 구성
        image_contents = []
        for b64 in frames_b64:
            image_contents.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64}",
                    "detail": "low"
                }
            })

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self.USER_PROMPT},
                    *image_contents
                ]
            }
        ]

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=100,
                temperature=0.3
            )
            caption = response.choices[0].message.content.strip()
            # 따옴표 제거 (GPT가 종종 따옴표로 감쌈)
            caption = caption.strip("'\"")
            return caption
        except Exception as e:
            logger.error(f"캡션 생성 실패 ({video_path}): {e}")
            return None

    def generate_all(
        self,
        video_dir: str,
        output_path: str,
        sample_list_csv: Optional[str] = None,
        max_videos: Optional[int] = None,
        resume: bool = True
    ) -> Dict[str, str]:
        """전체 영상에 대한 한글 캡션 일괄 생성

        Args:
            video_dir: 영상 파일이 있는 디렉토리
            output_path: 출력 JSON 경로
            sample_list_csv: 샘플 목록 CSV (None이면 video_dir 내 전체 영상)
            max_videos: 최대 처리 영상 수 (None이면 전체)
            resume: True이면 기존 JSON에서 이어서 생성

        Returns:
            {video_id: korean_caption} 딕셔너리
        """
        # 기존 결과 로드 (resume 모드)
        captions = {}
        if resume and os.path.exists(output_path):
            with open(output_path, 'r', encoding='utf-8') as f:
                captions = json.load(f)
            logger.info(f"기존 캡션 {len(captions)}개 로드 (resume 모드)")

        # 처리할 영상 목록 구성
        if sample_list_csv and os.path.exists(sample_list_csv):
            import csv
            with open(sample_list_csv, 'r') as f:
                reader = csv.DictReader(f)
                video_ids = [row.get('video_id', row.get('clip_id', '')) for row in reader]
        else:
            # video_dir 내 전체 영상 파일 탐색
            video_ids = []
            for fname in sorted(os.listdir(video_dir)):
                if fname.endswith(('.mp4', '.avi', '.mkv', '.webm')):
                    vid = os.path.splitext(fname)[0]
                    video_ids.append(vid)

        if max_videos:
            video_ids = video_ids[:max_videos]

        logger.info(f"처리 대상: {len(video_ids)}개 영상 (이미 완료: {len(captions)}개)")

        # 캡션 생성 루프
        for idx, vid in enumerate(video_ids):
            if vid in captions:
                continue  # 이미 생성됨 (resume)

            # 영상 파일 경로 찾기
            video_path = None
            for ext in ['.mp4', '.avi', '.mkv', '.webm']:
                candidate = os.path.join(video_dir, f"{vid}{ext}")
                if os.path.exists(candidate):
                    video_path = candidate
                    break

            if video_path is None:
                logger.warning(f"[{idx+1}/{len(video_ids)}] 영상 파일 없음: {vid}")
                continue

            caption = self.generate_caption(video_path)
            if caption:
                captions[vid] = caption
                logger.info(f"[{idx+1}/{len(video_ids)}] {vid}: {caption}")
            else:
                captions[vid] = f"[캡션 생성 실패] {vid}"
                logger.warning(f"[{idx+1}/{len(video_ids)}] {vid}: 캡션 생성 실패")

            # 50개마다 중간 저장
            if (idx + 1) % 50 == 0:
                self._save_json(captions, output_path)
                logger.info(f"중간 저장: {len(captions)}개 완료")

            # Rate limit 방지
            time.sleep(0.5)

        # 최종 저장
        self._save_json(captions, output_path)
        logger.info(f"한글 캡션 생성 완료: {len(captions)}개 → {output_path}")

        return captions

    def _save_json(self, data: dict, path: str):
        """JSON 저장 (ensure_ascii=False로 한글 보존)"""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
