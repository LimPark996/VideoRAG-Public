"""
⑤a bm25_retriever.py — spaCy Lemmatization BM25 키워드 검색 (Sparse Retrieval)

출처:
- 알고리즘: BM25 (Best Matching 25)
  논문: "The Probabilistic Relevance Framework: BM25 and Beyond"
        (Robertson & Zaragoza, 2009)
        https://doi.org/10.1561/1500000019
  원조: "Okapi at TREC-3" (Robertson et al., 1994)

- spaCy Lemmatizer 작동 원리
Lemmatization이란
단어의 **굴절형(inflected form)을 표제어(lemma)로 되돌리는 작업**이에요. 굴절형이란 문법적 역할에 따라 변형된 형태예요.
```
playing → play   (동사 현재분사)
played  → play   (동사 과거형)
ran     → run    (동사 불규칙 과거형)
better  → good   (형용사 비교급)
```
spaCy가 Lemmatization을 처리하는 방법
```
입력 텍스트
    ↓
토크나이징 (공백/구두점 기준 분리)
    ↓
CNN 기반 POS 태깅 (각 토큰에 품사 부착)
    ↓
Lemmatizer
    ├─ 룩업 테이블에 있으면 → 바로 반환
    ├─ 없으면 → 품사 기반 규칙 적용
    └─ 그래도 없으면 → 원형 그대로 반환
```

- 구현체: rank_bm25 라이브러리
  GitHub: https://github.com/dorianbrown/rank_bm25
  라이선스: Apache 2.0

프로토타입 선택:
  - Elasticsearch → rank_bm25로 교체
  - 이유: 순수 Python, pickle 저장/로드 가능, 코랩 세션 끊김에 안전
  - BPE 토크나이저 → spaCy Lemmatizer로 교체
  - 이유: GPT-2 BPE는 고빈도 단어를 분리하지 않아 어휘 불일치 해결 실패
           spaCy Lemmatizer는 굴절형을 품사 기반으로 정확하게 표제어로 변환

수식:
  score(D, Q) = Σ IDF(qi) × [ f(qi, D) × (k1 + 1) / (f(qi, D) + k1 × (1 - b + b × |D|/avgdl)) ]
  - k1 = 1.5 (기본값), b = 0.75 (문서 길이 정규화)
  - qi: WordNet 표제어 토큰 (굴절형이 아닌 표제어 단위)
"""

import os
import pickle
import logging
from typing import List, Tuple, Optional, Set

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# spaCy 기반 Lemmatization 토크나이저 클래스
# ──────────────────────────────────────────────────────────────────────

class SpacyLemmatizerTokenizer:
    """
        spaCy 기반 Lemmatization 토크나이저 (영어)
        "playing" → "play"
        "ran"      → "run"
        "cooking"  → "cook"
        "better"   → "good"
    """

    def __init__(self):
        self._nlp = None
        self._mode = "spacy_en"
        self._load_tokenizer()

    def _load_tokenizer(self):
        try:
            import spacy
            self._nlp = spacy.load("en_core_web_sm")
            logger.info("토크나이저 로드: spaCy en_core_web_sm")
        except ImportError:
            logger.warning("spaCy 미설치: pip install spacy")
            self._nlp = None
        except OSError:
            logger.warning("모델 미설치: python -m spacy download en_core_web_sm")
            self._nlp = None

    def tokenize(self, text: str) -> List[str]:
        text = text.lower().strip()
        if not text:
            return []

        if self._nlp is None:
            return text.split()

        doc = self._nlp(text)
        return [
            token.lemma_
            for token in doc
            if not token.is_stop
            and not token.is_punct
            and token.lemma_.strip()
        ]


class KoreanMorphTokenizer:
    """한국어 형태소 분석 토크나이저

    konlpy의 Okt(Open Korean Text)를 사용하여 한국어 텍스트를 토큰화.
    명사, 동사, 형용사 등 의미 있는 품사만 추출.

    동작 예시:
        "한 남성이 기타를 연주하고 있다" → ["남성", "기타", "연주"]
        "아이들이 공원에서 뛰어놀고 있다" → ["아이", "공원", "뛰어놀다"]
    """

    # 추출할 품사 태그 (Okt 기준)
    KEEP_POS = {'Noun', 'Verb', 'Adjective', 'Alpha', 'Number'}

    # 한국어 불용어
    STOPWORDS_KO = {
        '것', '수', '등', '때', '중', '안', '좀', '더', '이', '그', '저',
        '및', '또', '또는', '하다', '되다', '있다', '없다', '이다', '아니다',
        '들', '의', '에', '를', '을', '가', '이', '은', '는', '와', '과',
        '로', '으로', '에서', '까지', '부터', '만', '도', '나', '혹은'
    }

    def __init__(self):
        self._okt = None
        self._mode = "okt_ko"
        self._load_tokenizer()

    def _load_tokenizer(self):
        try:
            from konlpy.tag import Okt
            self._okt = Okt()
            logger.info("토크나이저 로드: konlpy Okt (한국어 형태소 분석)")
        except ImportError:
            logger.warning(
                "konlpy 미설치. 한국어 토크나이저 사용 불가. "
                "설치: pip install konlpy (+ JDK 필요)"
            )
            self._okt = None
        except Exception as e:
            logger.warning(f"Okt 초기화 실패: {e}")
            self._okt = None

    def tokenize(self, text: str) -> List[str]:
        """한국어 텍스트 토크나이징

        Args:
            text: 한국어 텍스트

        Returns:
            형태소 분석된 토큰 리스트 (명사, 동사, 형용사 등)
        """
        text = text.strip()
        if not text:
            return []

        if self._okt is None:
            # 폴백: 공백 분할 (형태소 분석 불가 시)
            return [t for t in text.split() if len(t) > 1]

        # 형태소 분석 (품사 태깅)
        morphs = self._okt.pos(text, stem=True)  # stem=True: 어간 추출

        tokens = []
        for word, pos in morphs:
            if pos in self.KEEP_POS and word not in self.STOPWORDS_KO and len(word) > 1:
                tokens.append(word)

        return tokens

# ──────────────────────────────────────────────────────────────────────
# BM25 검색기
# ──────────────────────────────────────────────────────────────────────

class BM25Retriever:
    """spaCy Lemmatization BM25 키워드 검색기

    build(captions, clip_ids)
    search(query, top_n=100) → List[(clip_id, bm25_score)]
    """

    def __init__(self, tokenizer_mode: str = "spacy_lemmatizer"):
        """
        Args:
            tokenizer_mode: 토크나이저 선택
                - "spacy_lemmatizer": spaCy 영어 Lemmatizer (기본)
                - "korean": 한국어 형태소 분석기 (Okt)
                - "whitespace": 단순 공백 분할 (이전 호환용)
                - "bpe": spacy_lemmatizer의 이전 호환 별칭
        """
        self.bm25 = None
        self.clip_ids: List[str] = []
        self.captions: List[str] = []
        self.tokenizer_mode = tokenizer_mode

        if tokenizer_mode == "korean":
            self._tokenizer = KoreanMorphTokenizer()
        elif tokenizer_mode in ("spacy_lemmatizer", "bpe"):
            self._tokenizer = SpacyLemmatizerTokenizer()
        else:
            self._tokenizer = None

    def build(
        self,
        captions: List[str],
        clip_ids: List[str],
    ):
        from rank_bm25 import BM25Okapi

        self.clip_ids = clip_ids
        self.captions = captions

        tokenized = [self._tokenize(cap) for cap in captions]
        self.bm25 = BM25Okapi(tokenized)

        logger.info(
            f"BM25 인덱스 구축 완료: {len(captions)} documents "
            f"(tokenizer={self.tokenizer_mode}, "
            f"mode={self._tokenizer._mode if self._tokenizer else 'whitespace'})"
        )

    def search(
        self,
        query: str,
        top_n: int = 100,
    ) -> List[Tuple[str, float]]:
        """BM25 키워드 검색

        Args:
            query: 자연어 쿼리 (en_query 단일 경로 — 영어만 사용)
            top_n: 반환할 결과 수

        Returns:
            List of (clip_id, bm25_score), 점수 내림차순
        """
        if self.bm25 is None:
            raise RuntimeError("BM25 인덱스가 구축되지 않았습니다. build() 먼저 호출")

        tokens = self._tokenize(query)
        scores = self.bm25.get_scores(tokens)

        top_indices = scores.argsort()[::-1][:top_n]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append((self.clip_ids[idx], float(scores[idx])))

        logger.debug(
            f"BM25 검색 완료: query='{query}', tokens={tokens}, {len(results)} results"
        )
        return results

    def save(self, path: str):
        """인덱스 저장"""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "bm25": self.bm25,
                "clip_ids": self.clip_ids,
                "captions": self.captions,
                "tokenizer_mode": self.tokenizer_mode
            }, f)
        logger.info(f"BM25 인덱스 저장: {path}")

    def load(self, path: str):
        """인덱스 로드"""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.bm25 = data["bm25"]
        self.clip_ids = data["clip_ids"]
        self.captions = data.get("captions", [])
        self.tokenizer_mode = data.get("tokenizer_mode", "whitespace")
        logger.info(f"BM25 인덱스 로드: {path} ({len(self.clip_ids)} docs)")

    def _tokenize(self, text: str) -> List[str]:
        """텍스트 토크나이징 (spaCy Lemmatizer / Korean Morph / whitespace)

        Args:
            text: 입력 텍스트

        Returns:
            토큰 리스트
        """
        if self._tokenizer is not None:
            return self._tokenizer.tokenize(text)
        else:
            return text.lower().strip().split()
