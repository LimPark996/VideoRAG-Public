"""
쿼리 전처리기 — Papago 기반 한→영 번역 + 언어 감지

설계 (v2):
  - 방법 A 채택: 인덱싱 때 영어 캡션 저장, 런타임은 en_query 단일 경로
  - 한국어 쿼리 → Papago API로 영어 번역 → en_query로 통일
  - BM25, Dense, ColBERT 전부 en_query만 사용
  - ko_query 없음 → 한국어 형태소 분석(konlpy) 의존성 제거
"""

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class PreprocessedQuery:
    """전처리된 쿼리"""
    original: str       # 원본 쿼리
    lang: str           # 감지된 언어 ("ko", "en", 기타)
    en_query: str       # 영어 쿼리 (번역 or 원본) — 모든 검색에 사용


class QueryPreprocessor:
    """쿼리 전처리기 (Papago 기반 한→영 번역)

    사용법:
        preprocessor = QueryPreprocessor(papago_client_id, papago_client_secret)
        pq = preprocessor.preprocess("강남역 폭우 장면")
        # pq.en_query = "Gangnam Station heavy rain scene"
    """

    def __init__(
        self,
        papago_client_id: Optional[str] = None,
        papago_client_secret: Optional[str] = None,
    ):
        self.client_id = papago_client_id
        self.client_secret = papago_client_secret

    def _detect_language(self, text: str) -> str:
        """langdetect로 언어 감지"""
        try:
            from langdetect import detect
            return detect(text)
        except ImportError:
            logger.warning("langdetect 미설치: pip install langdetect")
            return "en"
        except Exception:
            return "en"  # 감지 실패 시 영어로 간주

    def _translate_ko_to_en(self, text: str) -> str:
        """Naver Papago API로 한→영 번역"""
        if not self.client_id or not self.client_secret:
            logger.warning("Papago API 키 미설정 → 원본 텍스트 그대로 반환")
            return text

        import requests
        url = "https://naveropenapi.apigw.ntruss.com/nmt/v1/translation"
        headers = {
            "X-NCP-APIGW-API-KEY-ID": self.client_id,
            "X-NCP-APIGW-API-KEY": self.client_secret,
        }
        data = {"source": "ko", "target": "en", "text": text}

        try:
            resp = requests.post(url, headers=headers, data=data, timeout=5)
            resp.raise_for_status()
            translated = resp.json()["message"]["result"]["translatedText"]
            logger.info(f"Papago 번역: '{text}' → '{translated}'")
            return translated
        except Exception as e:
            logger.warning(f"Papago 번역 실패: {e} → 원본 텍스트 반환")
            return text

    def preprocess(self, query: str) -> PreprocessedQuery:
        """쿼리 전처리: 언어 감지 + 필요 시 영어 번역

        Args:
            query: 원본 쿼리 (한국어 또는 영어)

        Returns:
            PreprocessedQuery (en_query 포함)
        """
        lang = self._detect_language(query)

        if lang == "ko":
            en_query = self._translate_ko_to_en(query)
            return PreprocessedQuery(original=query, lang="ko", en_query=en_query)
        else:
            # 영어 or 기타 → 그대로 사용
            return PreprocessedQuery(original=query, lang=lang, en_query=query)
