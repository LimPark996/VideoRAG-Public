# VideoRAG — AI 기반 영상 검색·변환·합성 PD 워크스테이션

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

방송사 PD가 대본(큐시트)이나 자연어 쿼리를 입력하면 영상 아카이브에서 장면을 검색하고, 속성이 맞지 않으면 색감 변환을 적용하고, 최종 편집 영상까지 만들어주는 시스템. 핵심은 **검색과 생성의 경계를 PD가 직접 제어**한다는 것이다. Google Colab T4에서 1인 개발한 프로토타입이다.

---

## 웹 데모 (GitHub Pages)

**주소:** https://limpark996.github.io/VideoRAG-Public/

전체 검색→합성 파이프라인을 시연하는 React 웹앱. 10개 사전 구성 방송 시나리오, MSR-VTT 163개 클립에서 Top-5 ITM 재순위 결과를 제공한다.

**PD 플로우:**

1. 시나리오 선택 (방송 주제 10개)
2. 장면 탭 선택 → Top-5 ITM 재순위 클립 확인
3. 클립 클릭 → In/Out 구간 설정
4. **USE AS-IS** — 클립 그대로 사용 / **TRANSFORM** — OpenCV 색감 프리셋 18종 적용 (tone 7 · mood 4 · look 7)
5. 각 장면마다 반복 (완료된 장면은 초록 배지)
6. 합성 패널에서 드래그로 장면 순서 변경
7. **합성** → DreamColour 3D LUT 색보정 + DINOv2 전환 스코어링(CUT/CROSSFADE/MORPH) → 최종 영상

> 스타일 변환은 속성 전환(낮→밤, 여름→겨울 분위기) 목적의 색보정이다. SD img2img·TokenFlow는 실제 적용 시 화질 손상이 심하고 클립당 수 분이 걸려 제외했다. OpenCV 방식도 원본 장면 특성에 따라 변화가 미미할 수 있다.

**백엔드:** Modal 서버리스 T4 GPU (`scripts/modal_transform.py`)

```bash
# 배포
modal deploy scripts/modal_transform.py
cd videorag-demo && npm run deploy
```

---

## 기술 스택

### 데모에서 사용하는 기술

| 역할 | 기술 | 출처 |
|---|---|---|
| 영상 임베딩 | InternVideo2-1B (512차원, 4프레임) | Shanghai AI Lab, CVPR 2024 |
| 희소 검색 | BM25 + spaCy 레마타이저 (k1=1.5, b=0.75) | rank_bm25 |
| 밀집 인덱스 | FAISS IVFFlat (nlist=100, nprobe=10) | Meta AI Research |
| 검색 융합 | WRRF (w_visual=0.6, w_text=0.4, k=60) | Cormack 2009 기반 자체 설계 |
| 리랭킹 | ColBERT v2 MaxSim (PLAID centroid pruning) | Stanford, SIGIR/NAACL 2022 |
| 최종 재순위 | ITM (InternVideo2 cross-attention, full 1k 적용) | 자체 통합 |
| 텍스트 임베딩 | InternVideo2 encode_text + mean pooling (ITC collapse 우회) | 자체 수정 |
| 영상 변환 | OpenCV 프레임별 색보정 18종 + FFmpeg 재인코딩 (Modal T4) | OpenCV / FFmpeg |
| 전환 효과 | DINOv2 시각 유사도 (CUT/CROSSFADE/MORPH) | Meta AI Research |
| 색보정 (합성) | DreamColour 3D LUT (LAB 색공간, 첫 클립 기준 통일) | CHAITron/DreamColour |

### 전체 시스템에만 있는 기술

| 역할 | 기술 | 출처 |
|---|---|---|
| 대본 파싱 | GPT-4o-mini → Scene Graph JSON | OpenAI |
| 역프롬프트 | InversePromptEngine (속성→시네마틱 프롬프트, Rule-based) | 자체 설계 |
| AI 변환 | TokenFlow video-to-video / Runway API | TokenFlow / Runway |
| 샷 탐지 | TransNetV2 + Agglomerative Clustering | Souček & Lokoč 2020 |
| 시간 일관성 | TC-Score (Optical Flow 기반) | 자체 설계 |
| 출처 추적 | C2PA + ES256 서명 | C2PA specification |
| 평가 인덱스 | FAISS IndexFlatIP (exact brute-force, Tier 1 전용) | 자체 구현 |

---

## 전체 시스템 (02_demo.ipynb)

전체 기능을 갖춘 Gradio 프로토타입. Google Colab T4에서 실행하는 2-탭 인터페이스다.

### Tab 1: Scene Graph 워크플로

PD가 대본(JSON)을 넣으면 GPT-4o-mini가 장면별 description(영어, 검색용)과 attributes(시간대·계절·분위기·장소)를 추출하고, 시스템이 2경로 분기 판정을 자동 제안한다.

| 분기 | 자동 판정 기준 | 처리 |
|---|---|---|
| **USE_AS_IS** | 검색 점수 ≥ 임계값 + 속성 일치도 ≥ 임계값 | 그대로 사용 |
| **TRANSFORM** | 속성 불일치 또는 검색 점수 낮음 | 역프롬프트 생성 → InversePromptEngine → TokenFlow / Runway AI 변환 |

**역프롬프트(Inverse Prompt):** "저녁→밤으로 바꿔" 같은 추상적 지시 대신, 장면 의도를 포함한 구체적 시네마틱 프롬프트를 Rule-based로 생성한다. 예: "A sprawling cityscape at night, neon signs blazing in electric blue and magenta, deep indigo sky, volumetric haze catching the neon glow"

**PD 액션:** 후보 클립 미리보기 → 클립 선정 → 역프롬프트 확인/수정 → 구간 크롭 → 승인/재시도/건너뛰기/직접 업로드 → 장면 순서 재배치 → 최종 합성 (DINOv2 전환 + DreamColour + C2PA 서명)

### Tab 2: PD 큐레이션 (TBD)

Scene Graph 없이 텍스트 쿼리로 바로 검색 → PD가 클립 선택/제외/순서 변경 → 합성. B-roll 빠르게 뽑을 때 사용.

### 공통 기능

실시간 로그 패널, 단계별 레이턴시 차트, TC-Score(시간 일관성), C2PA 출처 서명.

---

## 벤치마크 (03_evaluation.ipynb)

MSR-VTT 1k-A split(테스트 영상 1,000개)에서 R@1/R@5/R@10을 논문 수치와 대조. FAISS IndexFlatIP(exact brute-force)으로 근사 검색 오차를 제거한다.

| 방법 | R@1 | R@5 | R@10 |
|---|---|---|---|
| InternVideo2-1B #F=4 (논문, ITC+ITM) | 51.9 | 74.6 | 81.7 |
| Ours: full ITM | **41.1** | **65.9** | **76.1** |

**-10.8%p 갭 원인:** ITC 텍스트 임베딩이 cosine ≈ 0.9997로 collapse되어 top-128 필터링을 쓰면 오히려 R@1이 39.5%로 하락한다. 현재는 ITM을 전체 1,000개에 직접 적용(full ITM)한다. collapse 원인(체크포인트 차이, feature pipeline 분기 등)은 미확정. 상세 진단은 `docs/issue_report_8차.md` 참고.

**Tier 1.5 레이턴시 프로파일링:** 7,010개 전체 코퍼스에서 BM25/Dense/Hybrid/Full 4가지 구성의 엔드투엔드 레이턴시를 측정한다.

---

## 아키텍처

```
대본/쿼리
    │
    ▼
[QueryPreprocessor] ─── Papago (ko→en)            [전체시스템]
    │
    ├── 텍스트 쿼리 ──────────────────────────────┐
    ├── 대본 ──→ [ScriptParser/GPT-4o-mini] ──→ Scene Graph
    │                                             │
    ▼                                             ▼
┌─────────── 검색 파이프라인 ───────────┐
│ [BM25] ←→ [Dense(InternVideo2)]      │
│      └──→ [WRRF 융합]                │
│              └──→ [ColBERT 리랭킹]   │
│                   └──→ [ITM 재순위]  │
└──────────────────────────────────────┘
    │ Top-K 후보 클립
    ▼
[StoryboardMapper] ← Scene Graph attributes
    │
    ├── USE_AS_IS ──→ 클립 그대로
    └── TRANSFORM ──→ OpenCV 색보정 18종 ★데모
                      TokenFlow / Runway ★전체시스템
    │
    ▼
★ PD 리뷰 + 장면 순서 드래그 리오더
    │
    ▼
[VideoAssembler]
    DreamColour 3D LUT (LAB 색보정)
    DINOv2 전환 스코어링 (CUT / CROSSFADE / MORPH)
    FFmpeg 렌더링
    │
    ▼
최종 영상
    └── C2PA ES256 서명 ★전체시스템
```

---

## 설계 결정

**왜 2경로 분기?** 딱 맞는 클립은 그대로 쓰고(USE_AS_IS), 속성이 다르거나 점수가 낮으면 변환한다(TRANSFORM). 완전히 새로운 영상 생성은 외부 툴이 더 효율적이다. PD가 매 판정을 검토·오버라이드할 수 있어 AI 자동화와 사람의 편집 판단이 공존한다.

**왜 OpenCV 색보정을 썼나?** 스타일 변환의 목적은 합치기 전 속성 전환(낮→밤, 여름→겨울)이다. SD img2img·TokenFlow는 내용을 예측 불가하게 바꾸고 클립당 수 분이 걸린다 — 실제 적용에서 화질 손상이 심하고 너무 느려 제외했다. OpenCV 프레임별 처리(R/G/B gain·offset, 대비, HSV 채도, 세피아/Teal-Orange 등 특수 효과)는 내용을 유지하면서 색감만 조정한다. 단, 원본 장면 특성에 따라 변화가 미미할 수 있다.

**왜 역프롬프트?** (전체 시스템) 생성 모델에 "저녁을 밤으로"라고 하면 그냥 어두워진다. InversePromptEngine이 장면 의도를 포함한 시네마틱 프롬프트를 생성해 변환 품질을 높인다.

**왜 하이브리드 검색?** BM25는 고유명사·숫자, Dense(InternVideo2)는 의미 유사도를 잡는다. WRRF로 두 장점을 결합하고, ColBERT(PLAID centroid pruning, 10~50x 가속)로 정밀 리랭킹, ITM cross-attention으로 최종 재순위한다. ITC collapse 문제로 dense 채널은 CLS 대신 mean pooling을 사용한다.

**왜 full ITM?** ITC cosine이 ≈0.9997로 collapse되어 top-128 필터링 시 정답을 22.5% 탈락시킨다. ITM을 전체 1,000개에 적용하면(R@1 41.1%) top-128→ITM(R@1 39.5%)보다 낫다.

**왜 C2PA?** AI 변환/생성 클립이 섞인 최종 영상에서 어느 클립이 아카이브 원본이고 어느 것이 AI 생성인지 암호학적으로 증명한다.

**왜 샷 탐지를 데모에서 끄나?** TransNetV2 기반 shot_detector는 클립당 프레임 추출 시간이 누적되어 인덱싱에 수 시간이 걸린다. 데모에서는 클립당 단일 프레임·단일 벡터로 설정한다.

---

## 프로젝트 구조

```
videorag-public/
  src/
    pipeline.py                    # 메인 오케스트레이터
    data_models.py                 # 공용 데이터 모델
    input/
      query_preprocessor.py        # Papago 번역           [전체시스템]
      script_parser.py             # GPT-4o-mini → Scene Graph [전체시스템]
    phase0_indexing/
      shot_detector.py             # TransNetV2 샷 탐지     [전체시스템]
      embedder.py                  # InternVideo2-1B 임베딩
      vector_store.py              # FAISS IVFFlat 인덱스
      indexer.py                   # Phase 0 오케스트레이터
    phase12_search/
      bm25_retriever.py            # BM25 + spaCy
      dense_retriever.py           # FAISS 밀집 검색
      hybrid_fusion.py             # WRRF 융합
    phase3_reranking/
      reranker.py                  # ColBERT v2 MaxSim + PLAID
      itm_scorer.py                # ITM 최종 재순위
    phase4_assembly/
      storyboard_mapper.py         # Scene Graph → 2경로 분기
      inverse_prompt_engine.py     # 역프롬프트 생성        [전체시스템]
      tokenflow_wrapper.py         # TokenFlow 래퍼         [전체시스템]
      assembler.py                 # 영상 어셈블리
      visual_scorer.py             # DINOv2 시각 유사도
      transition_selector.py       # CUT/CROSSFADE/MORPH
      colour_normalizer.py         # DreamColour 3D LUT     [데모: modal_transform.py에 인라인]
      morph_transition.py          # Optical Flow 전환
      tc_scorer.py                 # TC-Score               [전체시스템]
    phase5_c2pa/
      c2pa_tagger.py               # C2PA ES256 서명        [전체시스템]
    evaluation/
      faiss_flat_eval.py           # Exact-search 평가 인덱스
  notebooks/
    00_setup.ipynb                 # 환경 설정
    01_indexing.ipynb              # 오프라인 인덱싱 (7,010개)
    01b_caption_remaining.ipynb    # 나머지 6,010개 캡션 생성
    02_demo.ipynb                  # ★ PD 워크스테이션 (전체 시스템)
    03_evaluation.ipynb            # ★ MSR-VTT 1k-A 벤치마크
  scripts/
    modal_transform.py             # ★ 데모 Transform/Assemble API (Modal 배포용)
  docs/
    인덱싱_검색_과정_정리.md        # 인덱싱·검색 전체 흐름
    기술_출처_정리.md               # 모듈별 논문·라이선스 출처
    issue_report_1차~8차.md        # 단계별 이슈·진단·해결 기록
  data/
    msrvtt/                        # 벤치마크 데이터
    queries/                       # 데모 쿼리셋
```

---

## 빠른 시작 (Colab)

### 사전 준비

- Google Colab (T4 GPU)
- HuggingFace 토큰 (`HF_TOKEN`) — InternVideo2 가중치
- OpenAI API 키 — GPT-4o-mini (캡션, Scene Graph, 역프롬프트)
- (선택) Papago API — 한국어 쿼리 번역
- MSR-VTT 영상 — Google Drive에 MSR-VTT.ZIP (`data/msrvtt/README.md` 참고)

```bash
notebooks/00_setup.ipynb           # 환경 설정
notebooks/01_indexing.ipynb        # 인덱싱 (T4에서 ~30분)
notebooks/02_demo.ipynb            # PD 워크스테이션
notebooks/03_evaluation.ipynb      # 검색 파이프라인 정량 평가
```

---

## 배경

정부 R&D 과제 "대화형 멀티모달 AI 기반 미디어 프로덕션 기술개발"의 세부3 "고속 검색 기반 사실형 영상 합성 기술개발"을 위해 개발됐다. 총괄 과제는 세부1(바이브 편집), 세부2(역프롬프트 영상 생성), 세부3(검색 기반 영상 합성)으로 구성되며, 이 저장소는 세부3 프로토타입이다.

## 라이선스

여러 오픈소스 컴포넌트를 통합한다. 개별 모듈 헤더에서 라이선스와 저작자 표시를 확인할 것.

---

## English Summary

**VideoRAG** is a PD workstation for broadcast production. Given a screenplay or text query, it searches a video archive for matching scenes, applies color-grading transforms for attribute mismatches, and assembles a final edited video — with the PD reviewing every decision.

**Live demo:** https://limpark996.github.io/VideoRAG-Public/
→ 10 pre-computed broadcast scenarios · 163 MSR-VTT clips · Top-5 ITM-reranked results per scene · drag-to-reorder assembly

**Pipeline:** BM25 + InternVideo2 dense retrieval → WRRF fusion → ColBERT rerank → ITM rerank → 2-path routing (USE_AS_IS / TRANSFORM) → DreamColour 3D LUT + DINOv2 transitions → final video

**Demo transform:** OpenCV per-frame color grading (18 presets: tone 7 · mood 4 · look 7). SD img2img and TokenFlow were tested but excluded — too slow and degraded content quality. The full system uses InversePromptEngine + TokenFlow/Runway for proper AI stylization.

**Retrieval results (MSR-VTT 1k-A):** R@1 41.1% / R@5 65.9% / R@10 76.1% (paper: 51.9% / 74.6% / 81.7%). Gap is due to ITC cosine collapse (≈0.9997); full ITM over all 1,000 videos outperforms ITC→top-128→ITM.

**Deploy:**
```bash
modal deploy scripts/modal_transform.py   # Modal backend
cd videorag-demo && npm run deploy        # GitHub Pages frontend
```
