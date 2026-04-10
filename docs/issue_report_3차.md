# VideoRAG 프로토타입 — 이슈 보고서

작성일: 2026-03-15

이 문서는 VideoRAG 프로토타입을 개발하면서 발견된 이슈들을 정리한 것입니다. 각 이슈가 왜 발생했는지, 어떤 증상이 나타났는지, 어떻게 해결했는지(또는 아직 해결 안 된 건 왜 안 됐는지)를 초보자도 이해할 수 있도록 자세히 설명합니다.

---

## 목차

1. [00_setup.ipynb 동기화 불일치 (7건)](#1-00_setupipynb-동기화-불일치-7건)
2. [ColourNormalizer 미연결 문제](#2-colournormalizer-미연결-문제)
3. [Python 3.12 + MoviePy 호환성 크래시](#3-python-312--moviepy-호환성-크래시)
4. [RAGatouille import 실패 (langchain 의존성)](#4-ragatouille-import-실패-langchain-의존성)
5. [01_indexing 영상 Drive 백업 미실행](#5-01_indexing-영상-drive-백업-미실행)

---

## 1. 00_setup.ipynb 동기화 불일치 (7건)

### 상태: 해결 완료

### 이게 뭔 문제야?

`00_setup.ipynb`는 Colab에서 맨 처음 실행하는 노트북입니다. GPU를 확인하고, Google Drive를 연결하고, 필요한 라이브러리를 설치하고, 코드를 불러오는 "환경 세팅" 역할을 합니다. 이 노트북을 먼저 실행해야 01, 02, 03 노트북이 정상 동작합니다.

문제는 01_indexing과 02_demo 노트북을 개발하면서 코드가 많이 바뀌었는데, 00_setup은 그에 맞춰 업데이트가 안 됐다는 것입니다. 01/02/03에서는 새로운 클래스 이름을 쓰고, 새로운 라이브러리를 쓰고, 새로운 모델을 불러오는데, 00_setup에서는 옛날 방식 그대로였습니다. 이러면 Colab 런타임을 처음부터 다시 시작할 때 00_setup을 실행하면 필요한 것들이 빠져 있어서 01/02/03이 에러가 납니다.

### 구체적으로 뭐가 달랐는지 (7가지)

**불일치 1: BPETokenizer → SpacyLemmatizerTokenizer 이름 변경**

BM25 검색에서 쿼리를 토큰화하는 클래스의 이름이 바뀌었습니다. 토큰화란 문장을 단어 단위로 쪼개는 작업입니다. 원래는 `BPETokenizer`라는 이름이었는데, 개발 과정에서 spaCy라는 자연어 처리 라이브러리의 표제어 추출(lemmatization) 방식으로 바꿨습니다. 표제어 추출이란 "playing"을 "play"로, "children"을 "child"로 바꾸는 것입니다. 이렇게 하면 BM25 키워드 검색의 정확도가 올라갑니다.

그래서 클래스 이름도 `SpacyLemmatizerTokenizer`로 바뀌었는데, 00_setup의 import 검증 셀에서는 아직 `BPETokenizer`를 불러오려고 하고 있었습니다.

수정: import 구문을 `SpacyLemmatizerTokenizer`로 변경했습니다.

**불일치 2: GitHub 토큰 획득 방식 (`getpass` → `userdata.get`)**

GitHub에서 비공개 저장소의 코드를 clone할 때 인증 토큰이 필요합니다. 원래는 `getpass`라는 함수를 써서 사용자가 직접 토큰을 타이핑하게 했는데, 01/02 노트북에서는 Colab의 `userdata.get('GITHUB_TOKEN')`을 쓰도록 바뀌었습니다. `userdata.get`은 Colab의 "비밀 키" 기능에서 자동으로 값을 가져오는 방식이라 매번 입력할 필요가 없습니다.

00_setup에서는 아직 `getpass` 방식이었습니다.

수정: `userdata.get('GITHUB_TOKEN')`으로 변경했습니다.

**불일치 3: HF_TOKEN 설정 누락**

InternVideo2 모델을 HuggingFace에서 다운로드하려면 HuggingFace 인증 토큰(`HF_TOKEN`)이 필요합니다. 01_indexing에서는 이 토큰을 설정하고 있었지만, 00_setup에는 이 부분이 아예 없었습니다.

수정: `os.environ["HF_TOKEN"] = userdata.get('HF_TOKEN')` 추가했습니다.

**불일치 4: `open_clip_torch` 패키지 누락**

InternVideo2 모델은 내부적으로 `open_clip`이라는 라이브러리를 사용합니다. open_clip은 OpenAI의 CLIP 모델을 오픈소스로 재구현한 것인데, InternVideo2가 비디오 인코더의 일부 구조를 여기서 가져다 씁니다. 01_indexing에서는 이걸 설치하고 있었는데, 00_setup의 패키지 설치 목록에는 빠져 있었습니다.

수정: `!pip install open_clip_torch` 추가했습니다.

**불일치 5: `easydict` 패키지 누락**

InternVideo2의 설정 파일(config)을 로드할 때 `easydict`라는 라이브러리를 사용합니다. easydict는 파이썬 딕셔너리를 `config.model_name` 같은 점(.) 표기법으로 접근할 수 있게 해주는 유틸리티입니다. 01_indexing에는 있었지만 00_setup에는 없었습니다.

수정: pip install 목록에 `easydict` 추가했습니다.

**불일치 6: `colbert-ai`, `ragatouille` 패키지 누락**

Phase 3 리랭킹에서 ColBERT v2 모델을 사용합니다. ColBERT는 검색 결과의 순서를 더 정밀하게 재조정하는 모델입니다. 이 모델을 쓰려면 `colbert-ai`(ColBERT 엔진)와 `ragatouille`(ColBERT를 쉽게 쓸 수 있게 감싸주는 래퍼 라이브러리)이 필요합니다. 02_demo와 03_evaluation에서는 이것들을 사용하는데, 00_setup에서 설치를 안 하고 있었습니다.

수정: `!pip install -q ragatouille colbert-ai` 추가했습니다.

**불일치 7: InternVideo2 sparse checkout 클론 셀 누락**

InternVideo2는 영상을 512차원 벡터로 변환하는 모델입니다. 이 모델의 코드가 GitHub의 `OpenGVLab/InternVideo` 저장소에 있는데, 저장소 전체를 clone하면 너무 크기 때문에 "sparse checkout"이라는 방법으로 필요한 폴더(`InternVideo2/multi_modality`)만 가져옵니다. 그리고 가져온 코드의 경로를 Python의 `sys.path`에 등록해야 import가 됩니다. 01_indexing에는 이 과정이 있었지만, 00_setup에는 통째로 빠져 있었습니다.

수정: "Step 3.5: InternVideo2 설치" 셀을 새로 추가했습니다.

---

## 2. ColourNormalizer 미연결 문제

### 상태: 해결 완료

### 이게 뭔 문제야?

VideoRAG의 Phase 4는 검색된 영상 클립들을 하나의 영상으로 합치는 "영상 조립" 단계입니다. 여기서 여러 영상을 이어붙일 때 각 영상의 색감이 제각각이면 결과물이 부자연스럽습니다. 예를 들어 클립 A는 따뜻한 노란 톤인데 클립 B는 차가운 파란 톤이면, 이어붙였을 때 갑자기 색이 확 바뀌어서 어색합니다.

이 문제를 해결하기 위해 `ColourNormalizer`라는 클래스를 만들어뒀습니다. DreamColour라는 논문의 방식을 기반으로, 첫 번째 클립의 색감을 기준(레퍼런스)으로 삼아서 나머지 클립들의 색감을 거기에 맞춰주는 기능입니다.

그런데 문제는 이 클래스를 만들어놓고 실제로 호출하는 코드를 안 넣었다는 것입니다.

`assembler.py`의 `__init__` 메서드에서 `self.colour_normalizer = ColourNormalizer()`로 객체를 생성하고 있었지만, `assemble()` 메서드의 어디에서도 이 객체를 사용하지 않았습니다. 그래서 영상 합성 결과물은 각 클립의 원본 색감이 그대로 유지된 채 단순 이어붙이기만 된 상태였습니다.

### 어떻게 해결했는지

처음에는 단순히 `colour_normalizer.normalize(clip_paths)`를 호출하는 방식을 시도했습니다. 이 메서드는 각 클립 영상 파일을 처음부터 끝까지 열어서 프레임 하나하나에 LUT를 적용하고 새 파일로 저장하는 방식인데, 10개 클립을 전부 재인코딩하면 약 142초가 걸립니다. 이건 너무 느립니다.

그래서 방식을 바꿨습니다. "키프레임 기반 LUT 사전 계산 + MoviePy 콜백 적용" 방식입니다.

**Step 3.5 (LUT 사전 계산)**: 각 클립에서 키프레임(대표 프레임) 한 장만 뽑아서, 그 한 장의 색상 분포를 분석합니다. 분석 결과로 3D LUT(Look-Up Table)를 미리 만들어둡니다. LUT란 "이 색상이 들어오면 저 색상으로 바꿔라"라는 변환 테이블입니다. 키프레임 한 장만 분석하면 되니까 클립당 수십 밀리초면 끝납니다.

- 구체적인 과정: 첫 번째 클립의 키프레임을 레퍼런스(기준)로 정합니다. 두 번째 클립부터는 각각의 키프레임을 레퍼런스와 비교해서, "이 클립의 색감을 레퍼런스처럼 만들려면 어떤 변환을 해야 하는가"를 계산합니다. 이 변환 정보가 17x17x17 크기의 3D LUT에 담깁니다.

- LUT 내부 동작: RGB 색공간 전체를 17등분한 격자를 만듭니다(17x17x17 = 4,913개 격자점). 각 격자점에서 "이 입력 색상은 이 출력 색상으로 바꿔라"를 기록합니다. 격자점 사이의 색상은 삼선형 보간(trilinear interpolation)으로 계산합니다. 삼선형 보간이란 3차원 공간에서 인접한 8개 점의 값을 거리 비율로 가중 평균하는 것입니다.

**렌더링 시 적용**: MoviePy의 `fl_image` 기능을 사용합니다. `fl_image`는 영상의 각 프레임에 사용자가 정한 함수를 자동으로 적용해주는 기능입니다. MoviePy가 렌더링을 위해 프레임을 하나씩 읽어올 때마다, 그 프레임에 미리 만들어둔 LUT를 적용합니다. LUT 적용은 테이블 조회(lookup)라서 프레임당 수 밀리초밖에 안 걸립니다.

### 관련 코드 위치

- `src/phase4_assembly/assembler.py` — `assemble()` 메서드의 Step 3.5 부분 (91~111행)
- `src/phase4_assembly/assembler.py` — `_render_video()` 메서드의 DreamColour LUT 콜백 부분 (269~279행)
- `src/phase4_assembly/colour_normalizer.py` — `_build_3d_lut()` 메서드 (258~332행)

---

## 3. Python 3.12 + MoviePy 호환성 크래시

### 상태: 해결 완료

### 이게 뭔 문제야?

ColourNormalizer를 연결한 뒤 02_demo를 Colab에서 실행하면 이런 에러가 납니다:

```
ERROR:src.pipeline:영상 합성 실패: module 'pkgutil' has no attribute 'ImpImporter'
```

이건 ColourNormalizer 자체의 문제가 아니라, MoviePy 라이브러리와 Python 버전 간의 호환성 문제입니다.

**자세한 설명:**

`pkgutil`은 Python 표준 라이브러리에 포함된 모듈로, 패키지(라이브러리)를 찾고 불러오는 내부 메커니즘을 제공합니다. 이 안에 `ImpImporter`라는 클래스가 있었는데, 이것은 Python 2 시절부터 내려온 옛날 방식의 패키지 임포트 메커니즘입니다.

Python 3.12에서 이 `ImpImporter`가 완전히 삭제되었습니다. Python 개발팀이 "이건 너무 오래됐고, 새로운 방식(importlib)이 있으니까 없애자"고 한 것입니다.

문제는 MoviePy 1.0.x 버전이 내부적으로 `pkgutil.ImpImporter`를 참조하는 코드를 가지고 있다는 것입니다. MoviePy가 직접 쓰는 건 아니고, MoviePy가 의존하는 다른 라이브러리(또는 MoviePy 초기화 과정에서 실행되는 코드)가 이걸 참조합니다. Python 3.11까지는 `ImpImporter`가 존재했으니까 문제가 없었는데, Colab 런타임이 Python 3.12로 업데이트되면서 `ImpImporter`가 없어져서 크래시가 난 것입니다.

이 에러는 `from moviepy.editor import VideoFileClip`을 실행하는 순간 발생합니다.

### 어떻게 해결했는지

MoviePy를 import하기 직전에 빈 껍데기 클래스를 만들어서 `pkgutil.ImpImporter` 자리에 넣어줬습니다:

```python
import pkgutil
if not hasattr(pkgutil, 'ImpImporter'):
    pkgutil.ImpImporter = type('ImpImporter', (), {})
```

이 코드가 하는 일을 한 줄씩 설명하면:

1. `import pkgutil` — Python의 pkgutil 모듈을 불러옵니다.
2. `if not hasattr(pkgutil, 'ImpImporter')` — pkgutil에 `ImpImporter`라는 속성이 있는지 확인합니다. Python 3.12에서는 없으므로 True가 됩니다. Python 3.11 이하에서는 있으므로 False가 되어 아무 일도 안 합니다.
3. `type('ImpImporter', (), {})` — 아무 기능도 없는 빈 클래스를 동적으로 만듭니다. `type` 함수의 세 번째 인자 `{}`가 빈 딕셔너리이므로 속성이나 메서드가 하나도 없습니다.
4. `pkgutil.ImpImporter = ...` — 이 빈 클래스를 `pkgutil.ImpImporter`에 대입합니다. 이제 MoviePy가 `pkgutil.ImpImporter`를 참조해도 "없다"는 에러 대신 빈 클래스가 반환됩니다.

이 방식은 일종의 "심(shim)"입니다. 심이란 원래 없는 것을 있는 것처럼 꽂아넣어서 호환성을 맞춰주는 것을 의미합니다. 이건 근본적인 해결(MoviePy를 2.x로 업그레이드)이 아니라 임시 우회입니다. MoviePy 2.0이 나오면 이 심은 제거해도 됩니다.

### 관련 코드 위치

- `src/phase4_assembly/assembler.py` — `_render_video()` 메서드 첫 부분 (234~238행)

---

## 4. RAGatouille import 실패 (langchain 의존성)

### 상태: 미해결 (폴백으로 정상 동작 중)

### 이게 뭔 문제야?

Phase 3 "리랭킹"에서 ColBERT v2라는 모델을 사용합니다. 리랭킹이 뭔지 먼저 설명하겠습니다.

Phase 1-2에서 BM25(키워드 검색)와 Dense Retrieval(의미 벡터 검색)로 1,000개 클립 중 상위 100개를 뽑습니다. 그런데 이 100개의 순서가 완벽하지는 않습니다. BM25는 단어 일치만 보고, Dense Retrieval은 문장 전체의 의미만 봅니다. ColBERT v2는 이 두 가지 사이의 중간 지점에서 동작합니다. "토큰 수준의 세밀한 상호작용"을 봅니다. 예를 들어 쿼리가 "아이가 악기를 연주하는 영상"이면, "아이"와 "어린이"의 유사도, "연주"와 "기타를 치다"의 유사도를 토큰(단어) 하나하나 비교해서 더 정확한 점수를 매깁니다.

이 ColBERT v2를 사용하는 방법이 3가지(3-tier) 있고, 하나가 실패하면 다음 방법을 시도하는 "폴백(fallback)" 구조입니다:

**1단계 — RAGatouille (래퍼 라이브러리)**

RAGatouille은 ColBERT를 쉽게 쓸 수 있게 감싸주는(wrapping) 라이브러리입니다. `model.rerank(query, documents)`처럼 한 줄로 리랭킹을 할 수 있습니다. 내부적으로 ColBERT 엔진과 langchain이라는 또 다른 라이브러리를 사용합니다.

문제: RAGatouille이 `from langchain.retrievers import ...`를 시도하는데, langchain의 최신 버전에서는 이 경로가 바뀌었습니다(`langchain_community`로 분리됨). 그래서 `No module named 'langchain.retrievers'` 에러가 나면서 RAGatouille 자체가 import되지 않습니다.

**2단계 — Direct (직접 MaxSim 구현) ← 현재 이것이 작동 중**

RAGatouille이 안 되면, HuggingFace의 `transformers` 라이브러리로 ColBERT v2 모델(`colbert-ir/colbertv2.0`)을 직접 로드합니다. 그리고 MaxSim이라는 점수 계산 방식을 직접 코드로 구현합니다.

MaxSim이란: 쿼리의 각 토큰에 대해 문서의 모든 토큰과 유사도를 계산하고, 그중 최대값(max)만 취합니다. 이걸 모든 쿼리 토큰에 대해 합산(sum)합니다. 수식으로 쓰면 `score = Σ max(q_i · d_j)`입니다.

현재 이 방식이 잘 작동하고 있습니다. ColBERT v2 모델의 가중치 199개 파라미터가 정상 로드되고, 리랭킹도 정상적으로 수행됩니다.

**3단계 — Passthrough (점수 그대로 통과)**

Direct마저 실패하면 (예: GPU가 없거나 transformers가 설치 안 됐을 때) Phase 2에서 나온 WRRF 점수를 그대로 사용합니다. 리랭킹 없이 그냥 통과시키는 것입니다.

### 왜 아직 안 고쳤는지

2단계(Direct)가 잘 작동하고 있어서, 1단계(RAGatouille)를 고치는 것은 우선순위가 낮습니다. RAGatouille의 langchain 의존성 문제를 고치려면 RAGatouille 자체의 코드를 수정하거나 langchain 버전을 맞춰야 하는데, 이건 외부 라이브러리의 문제라 우리가 직접 고치기 어렵습니다. RAGatouille 개발자가 업데이트해주면 자동으로 해결될 문제입니다.

실행 로그에서 이 폴백이 작동하는 걸 확인할 수 있습니다:

```
WARNING:src.phase3_reranking.reranker:RAGatouille import 실패: No module named 'langchain.retrievers'
  → pip install ragatouille colbert-ai 필요
  → 직접 MaxSim 구현으로 폴백
INFO:src.phase3_reranking.reranker:ColBERT v2 직접 로드 성공: colbert-ir/colbertv2.0
```

### 향후 조치

- RAGatouille이 langchain 호환성을 업데이트하면 자동 해결
- 또는 `pip install langchain-community`를 추가해서 해결할 수 있지만, 의존성이 복잡해질 수 있음
- 현재로서는 Direct 모드가 정상이므로 긴급하지 않음

### 관련 코드 위치

- `src/phase3_reranking/reranker.py` — `_load_ragatouille()` (399~417행), `_load_direct()` (419~445행)

---

## 5. 01_indexing 영상 Drive 백업 미실행

### 상태: 확인됨 (코드는 있으나 실행된 적 없음)

### 이게 뭔 문제야?

01_indexing.ipynb에서 MSR-VTT 데이터셋의 영상 1,000개를 다운로드하고 인덱싱합니다. 이 영상 파일들을 Google Drive에 백업하는 코드가 cell-8에 작성되어 있습니다.

그런데 이 셀이 한 번도 실행된 적이 없습니다. Jupyter 노트북에서는 셀을 실행하면 셀 아래에 출력(output)이 남는데, cell-8에는 출력이 아예 없습니다. 셀 왼쪽의 실행 번호([1], [2] 같은 것)도 비어 있습니다. 이건 셀을 작성만 하고 Shift+Enter를 누르지 않았다는 뜻입니다.

이게 당장 문제가 되는 건 아닙니다. 인덱싱된 결과(FAISS 인덱스, BM25 인덱스, 메타데이터)는 별도로 저장되어 있고, 02_demo와 03_evaluation은 인덱스만 있으면 돌아갑니다. 하지만 Colab 런타임이 초기화되면 로컬에 다운로드한 영상 파일이 날아가기 때문에, 영상 합성(Phase 4) 테스트를 다시 하려면 영상을 다시 다운로드해야 합니다. Drive 백업이 있었으면 다시 다운로드하지 않아도 됐을 것입니다.

### 관련 코드 위치

- `notebooks/01_indexing.ipynb` — cell-8

---

## 전체 파이프라인 구조 요약 (이슈의 맥락)

이슈들이 파이프라인의 어디에서 발생하는지 이해하려면 전체 흐름을 알아야 합니다.

```
Phase 0: 인덱싱 (오프라인, 1회)
  ├── Shot Detection (TransNetV2) — 영상을 장면 단위로 자름
  ├── InternVideo2 — 각 클립을 512차원 벡터로 변환
  ├── FAISS 인덱스 저장 — 벡터 검색용
  └── BM25 인덱스 저장 — 키워드 검색용

Phase 1-2: 검색 (온라인, 매 쿼리)
  ├── BM25 Sparse Search — 키워드 일치 기반
  ├── Dense Retrieval — 의미 벡터 유사도 기반
  └── WRRF Fusion — 두 결과를 가중 합산하여 상위 100개 선정

Phase 3: 리랭킹 (온라인)                    ← 이슈 #4 발생 지점
  └── ColBERT v2 — 토큰 수준 세밀 비교로 100개 → 20개 정밀 선별

Phase 4: 영상 조립 (온라인)                  ← 이슈 #2, #3 발생 지점
  ├── Perception Sort — DINOv2로 클립 순서 최적화
  ├── Transition Selection — 클립 간 유사도로 전환 효과 결정
  ├── DreamColour 3D LUT — 색감 통일 (키프레임 기반)
  └── MoviePy 렌더링 — 최종 영상 생성

Phase 5: C2PA 서명 (온라인)
  └── ES256 디지털 서명 — AI 생성 콘텐츠 출처 증명
```

`00_setup.ipynb`(이슈 #1)는 이 모든 Phase가 실행되기 전의 환경 설정이고, `01_indexing` Drive 백업(이슈 #5)는 Phase 0의 부가 기능입니다.

---

## 이슈 현황 요약표

| # | 이슈 | 위치 | 상태 | 심각도 |
|---|------|------|------|--------|
| 1 | 00_setup 동기화 불일치 (7건) | `00_setup.ipynb` | 해결 완료 | 높음 (환경 설정 깨짐) |
| 2 | ColourNormalizer 미연결 | `assembler.py` | 해결 완료 | 중간 (기능 누락) |
| 3 | Python 3.12 + MoviePy 크래시 | `assembler.py` | 해결 완료 (심 적용) | 높음 (합성 불가) |
| 4 | RAGatouille langchain 의존성 | `reranker.py` | 미해결 (폴백 동작) | 낮음 (대안 작동 중) |
| 5 | 01_indexing Drive 백업 미실행 | `01_indexing.ipynb` | 확인됨 | 낮음 (편의 기능) |

---

## 아직 검증이 필요한 것

이슈 #2와 #3의 수정사항(LUT 사전 계산 + fl_image 콜백 + pkgutil 심)은 코드에 반영되었지만, Colab에서 실제로 02_demo를 다시 실행해서 영상 합성이 성공하는지 아직 확인되지 않았습니다. 다음 Colab 실행 시 검증이 필요합니다.
