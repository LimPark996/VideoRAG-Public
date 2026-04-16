# VideoRAG — AI 기반 영상 검색·변환·합성 PD 워크스테이션

방송사 PD가 대본(큐시트)이나 자연어 쿼리를 입력하면, 영상 아카이브에서 장면을 검색하고, 속성이 맞지 않으면 AI로 변환하고, 아예 적합한 영상이 없으면 새로 생성해서, 최종 편집 영상까지 자동으로 만들어주는 시스템이다.

핵심은 **검색(Retrieval)과 생성(Generation)의 경계를 PD가 직접 제어한다**는 것이다. 기존 시스템은 검색만 하거나(Google), 생성만 하거나(Runway/Sora), 출처만 추적하는데(Adobe), VideoRAG는 이 세 가지를 하나의 워크플로로 통합한다. PD는 각 장면마다 아카이브 클립을 그대로 쓸지, AI로 변환할지, 새로 생성할지를 직접 결정하고, 프롬프트를 수정하고, 구간을 크롭하고, 최종 합성까지 하나의 인터페이스에서 처리한다.

Google Colab T4에서 1인 개발한 프로토타입이다.

## 02_demo.ipynb — PD 워크스테이션

이 프로젝트의 핵심 산출물. 2-탭 Gradio 인터페이스다.

### Tab 1: Scene Graph 워크플로

PD가 대본(JSON)을 넣으면 GPT-4o-mini가 장면별 Scene Graph를 생성한다. 각 장면의 description(영어, 검색용)과 attributes(시간대, 계절, 분위기, 장소)가 추출되고, 시스템이 아카이브를 검색한 뒤 **장면마다 자동으로 3경로 분기 판정**을 내린다:

| 분기 | 조건 | 처리 |
|---|---|---|
| **USE_AS_IS** | 아카이브 클립이 요구 속성과 일치 | 그대로 사용 |
| **TRANSFORM** | 클립 내용은 맞지만 시간대/계절/분위기가 다름 | 역프롬프트 생성 → TokenFlow 또는 Runway Gen-4 Turbo로 영상 변환 |
| **GENERATE** | 적합한 클립이 아카이브에 없음 | Runway로 텍스트→영상 생성 |

**PD가 매 장면에서 하는 일:**
1. 상위 5개 후보 클립을 영상으로 미리보기
2. 후보 클립 한개 선정
3. TRANSFORM이면 — 생성된 역프롬프트를 확인/직접 수정, 원하는 구간을 슬라이더로 크롭, 백엔드 선택(tokenflow / runway / opencv)
4. 승인 / 재시도 / 건너뛰기 / 직접 파일 업로드 중 선택
5. 모든 장면 완료 후 장면 순서 재배치
6. 최종 합성 → DINOv2 전환 효과 + 색보정 + C2PA 서명된 영상 출력

**역프롬프트(Inverse Prompt)란?**
"저녁→밤으로 바꿔" 같은 추상적 지시가 아니라, 장면 의도(Scene Intent)를 포함한 구체적 시네마틱 프롬프트를 자동 생성하는 것이다. 예를 들어 "네온사인이 빛나는 도시 밤" 장면인데 검색된 클립이 저녁이면, 단순히 "어둡게 해라"가 아니라 "A sprawling cityscape at night, neon signs blazing in electric blue and magenta, deep indigo sky, volumetric haze catching the neon glow" 같은 프롬프트를 GPT-4o-mini가 생성하고, 이걸 TokenFlow나 Runway에 넘긴다.

### Tab 2: PD 큐레이션

Scene Graph 없이 텍스트 쿼리로 바로 검색 → PD가 클립을 직접 선택/제외/순서 변경 → 합성. 빠르게 B-roll을 뽑을 때 사용한다.

**Section C — 클립 변환:** 선택한 클립에 프롬프트를 입력하면 TokenFlow(로컬 SD 기반) 또는 Runway(API 기반)로 스타일 변환한다. 변환 결과를 승인하면 변환 클립이 클립 목록에 추가되어 합성에 사용된다.

**Section D — 새 클립 생성:** 프롬프트만으로 Runway가 신규 영상을 생성한다.

### 공통 기능

실시간 로그 패널, 단계별 레이턴시 차트, TC-Score(시간 일관성) 표시, C2PA 출처 서명, Runway 생성 영상 Drive 자동 백업.

## 03_evaluation.ipynb — MSR-VTT 1k-A 벤치마크

검색 파이프라인의 정량 평가 노트북이다. InternVideo2 통합이 올바른지, 각 컴포넌트가 얼마나 기여하는지를 측정한다.

**Tier 1: 검색 정확도** — MSR-VTT 1k-A split(테스트 영상 1,000개)에서 Dense-only 검색의 R@1, R@5, R@10을 논문 수치(51.9 / 74.6 / 81.7)와 대조한다. FAISS IndexFlatIP(exact brute-force)으로 근사 검색 오차를 제거하고, ±3% 이내 일치 시 모델 통합 정상.

**Tier 1.5: 레이턴시 프로파일링** — 7,010개 전체 코퍼스에서 4가지 구성(BM25 / Dense / Hybrid / Full)의 엔드투엔드 레이턴시를 측정하여, 각 컴포넌트(BM25, FAISS, WRRF 융합, ColBERT)의 비용-편익 트레이드오프를 보여준다.

| 방법 | R@1 | R@5 | R@10 |
|---|---|---|---|
| InternVideo2-1B #F=4 (논문) | 51.9 | 74.6 | 81.7 |
| Ours: Dense-only | _TBD_ | _TBD_ | _TBD_ |

## 아키텍처

```
대본/쿼리
    │
    ▼
[QueryPreprocessor] ─── Papago (ko→en)
    │
    ├── 텍스트 쿼리 ────────────────────────┐
    │                                       │
    ├── 대본 ──→ [ScriptParser] ──→ Scene Graph JSON
    │              (GPT-4o-mini)      (장면별 description + attributes)
    │                                       │
    ▼                                       ▼
┌─────────── 검색 파이프라인 ───────────┐    │
│ [BM25] ←→ [Dense(InternVideo2)]      │    │
│      └──→ [WRRF 융합]               │    │
│              └──→ [ColBERT 리랭킹]    │    │
│                   └──→ [ITM 재순위]  │    │
└──────────────────────────────────────┘    │
    │ 상위 K개 후보 클립                      │
    ▼                                       ▼
[StoryboardMapper] ← Scene Graph attributes
    │
    ├── USE_AS_IS ──→ 클립 그대로 사용
    │
    ├── TRANSFORM ──→ [InversePromptEngine]
    │                   역프롬프트 생성 (GPT-4o-mini)
    │                     ├──→ TokenFlow (로컬, SD 기반 video-to-video)
    │                     └──→ Runway Gen-4 Turbo (API 기반 video-to-video)
    │
    └── GENERATE ───→ Runway Gen-4 Turbo
                       (text-to-video 생성)
    │
    ▼
  ★ PD 리뷰 (승인/수정/재시도/건너뛰기/업로드)
    │
    ▼
[VideoAssembler]
  DINOv2 전환 스코어링 (CUT / CROSSFADE / MORPH)
  DreamColour 3D LUT 색보정
  FFmpeg 렌더링
    │
    ▼
[C2PA Tagger] ── ES256 출처 서명
    │
    ▼
최종 영상 + TC-Score + C2PA 메타데이터
```

## 기술 스택

| 역할 | 기술 | 출처 |
|---|---|---|
| 영상 임베딩 | InternVideo2-1B (512차원, 4프레임) | Shanghai AI Lab, CVPR 2024 |
| 희소 검색 | BM25 + spaCy 레마타이저 | rank_bm25 |
| 밀집 인덱스 | FAISS IVFFlat (코사인/IP) | Meta AI Research |
| 검색 융합 | WRRF (w_visual=0.6, w_text=0.4, k=60) | Cormack 2009 기반 자체 설계 |
| 리랭킹 | ColBERT v2 (MaxSim 브루트포스) | Stanford, NAACL 2022 |
| 최종 재순위 | ITM (Image-Text Matching, InternVideo2 cross-attention) | 자체 통합 |
| 대본 파싱 | GPT-4o-mini → Scene Graph JSON | OpenAI |
| 역프롬프트 | InversePromptEngine (속성→시네마틱 프롬프트) | 자체 설계 |
| 영상 변환 (로컬) | TokenFlow (SD + DDIM inversion, keyframe subsampling 8fps/10f) | Geyer et al. 2023 |
| 영상 변환/생성 (API) | Runway Gen-4 Turbo (video-to-video / text-to-video) | Runway API |
| 전환 효과 | DINOv2 시각 유사도 (CUT/CROSSFADE/MORPH) | Meta AI Research |
| 색보정 | DreamColour 3D LUT | CHAITron/DreamColour |
| 시간 일관성 | TC-Score (Optical Flow 기반) | 자체 설계 |
| 샷 탐지 | TransNetV2 + Agglomerative Clustering | Souček & Lokoč 2020 |
| 출처 추적 | C2PA + ES256 서명 | C2PA specification |
| 평가 인덱스 | FAISS IndexFlatIP (exact, Tier 1 전용) | 자체 구현 |

## 빠른 시작 (Colab)

```bash
# 1. 환경 설정
notebooks/00_setup.ipynb

# 2. 인덱싱 (오프라인, T4에서 ~30분)
notebooks/01_indexing.ipynb → Drive에 저장

# 2b. (최초 1회) 나머지 6010개 영문 캡션 생성
notebooks/01b_caption_remaining.ipynb → ~$7, ~2.5시간

# 3. PD 워크스테이션 (핵심)
notebooks/02_demo.ipynb

# 4. 검색 파이프라인의 정량 평가
notebooks/03_evaluation.ipynb
```

### 사전 준비

- Google Colab (T4 GPU)
- HuggingFace 토큰 (`HF_TOKEN`) — InternVideo2 가중치
- OpenAI API 키 — GPT-4o-mini (캡션, Scene Graph, 역프롬프트)
- (선택) Runway API 키 — TRANSFORM/GENERATE 경로 (없으면 TokenFlow로 로컬 변환)
- (선택) Papago API — 한국어 쿼리 번역
- MSR-VTT 영상 — Google Drive에 MSR-VTT.ZIP (`data/msrvtt/README.md` 참고)

## 프로젝트 구조

```
videorag-public/
  src/
    pipeline.py                  # 메인 오케스트레이터
    data_models.py               # 공용 데이터 모델
    input/
      query_preprocessor.py      # 한국어→영어 번역 (Papago)
      script_parser.py           # 대본 → Scene Graph JSON (GPT-4o-mini)
    phase0_indexing/
      shot_detector.py           # TransNetV2 + Agglomerative Clustering
      embedder.py                # InternVideo2-1B 임베딩
      vector_store.py            # FAISS IVFFlat 인덱스
      indexer.py                 # Phase 0 오케스트레이터
    phase12_search/
      bm25_retriever.py          # BM25 + spaCy 레마타이저
      dense_retriever.py         # FAISS 밀집 검색
      hybrid_fusion.py           # WRRF 융합
    phase3_reranking/
      reranker.py                # ColBERT v2 MaxSim
      itm_scorer.py              # ITM 최종 재순위
    phase4_assembly/
      storyboard_mapper.py       # Scene Graph → 3경로 분기 판정
      inverse_prompt_engine.py   # 역프롬프트 생성 + TokenFlow/Runway 호출
      tokenflow_wrapper.py       # TokenFlow video-to-video 래퍼
      assembler.py               # 영상 어셈블리
      visual_scorer.py           # DINOv2 시각 유사도
      transition_selector.py     # CUT/CROSSFADE/MORPH 자동 선택
      colour_normalizer.py       # DreamColour 3D LUT
      morph_transition.py        # Optical Flow 변형 전환
      tc_scorer.py               # TC-Score
    phase5_c2pa/
      c2pa_tagger.py             # C2PA ES256 서명
    evaluation/
      faiss_flat_eval.py         # Exact-search 평가 인덱스
  notebooks/
    00_setup.ipynb               # 환경 설정
    01_indexing.ipynb            # 오프라인 인덱싱 (7,010개)
    01b_caption_remaining.ipynb  # 나머지 6,010개 캡션 생성
    02_demo.ipynb                # ★ PD 워크스테이션
    03_evaluation.ipynb          # ★ MSR-VTT 1k-A 벤치마크
  docs/                          # 기술 문서 + 이슈 리포트
  data/
    msrvtt/                      # 벤치마크 데이터
    queries/                     # 데모 쿼리셋
```

## 설계 결정

**왜 3경로 분기?** 아카이브만으로는 모든 장면을 충족할 수 없다. 클립이 딱 맞으면 그대로 쓰고(USE_AS_IS), 내용은 맞는데 밤/낮이 다르면 AI로 변환하고(TRANSFORM), 아예 없으면 생성한다(GENERATE). PD가 매 판정을 검토하고 오버라이드할 수 있어서, AI의 자동화와 사람의 편집 판단이 공존한다.

**왜 TokenFlow + Runway 이중 변환 백엔드?** Runway는 API 비용이 발생하고 인터넷이 필요하다. TokenFlow는 Stable Diffusion 기반으로 Colab 로컬에서 실행되어 비용 없이 변환 가능하다. keyframe subsampling(8fps 추출, 10프레임마다 keyframe 1개)으로 5초 클립 기준 T4에서 약 30~60초에 처리한다. 두 백엔드를 같은 인터페이스에서 선택할 수 있어 비용·품질 트레이드오프를 PD가 직접 결정한다.

**왜 역프롬프트?** Runway에 "저녁을 밤으로 바꿔"라고 넣으면 그냥 어두워지기만 한다. InversePromptEngine이 장면 의도를 포함한 시네마틱 프롬프트를 생성해서 변환 품질을 높인다.

**왜 하이브리드 검색?** BM25는 고유명사·숫자 매칭, Dense(InternVideo2)는 의미 유사도. WRRF로 두 장점을 결합하고, ColBERT로 상위 후보를 정밀 리랭킹한 뒤 ITM cross-attention으로 최종 재순위한다.

**왜 C2PA?** AI가 변환/생성한 클립이 섞인 영상의 출처를 암호학적으로 증명한다. 어떤 클립이 아카이브 원본이고, 어떤 클립이 AI가 만든 것인지 추적 가능.

## 배경

이 프로토타입은 정부 R&D 과제 "대화형 멀티모달 AI 기반 미디어 프로덕션 기술개발"의 세부3 "고속 검색 기반 사실형 영상 합성 기술개발"을 위해 개발되었다. 총괄 과제는 세부1(바이브 편집 플랫폼), 세부2(역프롬프트 기반 영상 생성), 세부3(검색 기반 영상 합성)으로 구성되며, 이 저장소는 세부3의 프로토타입 구현이다.

## 라이선스

이 프로젝트는 여러 오픈소스 컴포넌트를 통합한다. 개별 모듈 헤더에서 라이선스와 저작자 표시를 확인할 것.

---

# VideoRAG — AI-Powered Video Retrieval · Transform · Synthesis PD Workstation

A system where a broadcast PD inputs a screenplay or natural-language query, and the system searches the video archive for matching scenes, transforms clips whose attributes don't match via AI, generates entirely new footage when nothing suitable exists, and assembles the final edited video — all in one interface.

The core idea: **the PD controls the boundary between retrieval and generation**. Existing systems only search (Google), only generate (Runway/Sora), or only track provenance (Adobe). VideoRAG integrates all three into a single workflow. For each scene, the PD decides whether to use an archive clip as-is, transform it with AI, or generate from scratch — reviewing prompts, cropping segments, and approving results at every step.

Built as a solo prototype on Google Colab (T4 GPU).

## 02_demo.ipynb — PD Workstation

The core deliverable. A 2-tab Gradio interface.

### Tab 1: Scene Graph Workflow

The PD inputs a screenplay (JSON). GPT-4o-mini generates a per-scene Scene Graph with description (English, for retrieval) and attributes (time_of_day, season, mood, location). The system searches the archive and **automatically routes each scene to one of three paths**:

| Path | Condition | Action |
|---|---|---|
| **USE_AS_IS** | Archive clip matches required attributes | Use directly |
| **TRANSFORM** | Content matches but time/season/mood differs | Generate inverse prompt → TokenFlow or Runway Gen-4 Turbo video transform |
| **GENERATE** | No suitable clip in archive | Runway text-to-video generation |

**What the PD does per scene:**
1. Preview top 5 candidate clips as video
2. Select one clip
3. For TRANSFORM — review/edit the inverse prompt, crop the desired segment with sliders, select backend (tokenflow / runway / opencv)
4. Accept / Retry / Skip / Upload own file
5. After all scenes — reorder scenes
6. Final assembly → DINOv2 transitions + color normalization + C2PA-signed output

**Inverse Prompt:** Not "make it darker" but a concrete cinematic prompt that includes scene intent. For a "neon-lit city night" scene where the retrieved clip is evening, InversePromptEngine generates "A sprawling cityscape at night, neon signs blazing in electric blue and magenta, deep indigo sky, volumetric haze catching the neon glow" — then sends it to TokenFlow or Runway.

### Tab 2: PD Curation

Text query → search → PD manually selects/excludes/reorders clips → assemble. Fast mode for B-roll without Scene Graph.

**Section C — Clip Transform:** Input a prompt for the selected clip and transform it via TokenFlow (local SD-based) or Runway (API-based). Approved results are added to the clip list for assembly.

**Section D — New Clip Generation:** Runway generates new footage from a text prompt only.

### Shared Features

Real-time log panel, per-phase latency chart, TC-Score (temporal consistency), C2PA provenance signing, automatic Drive backup of Runway-generated videos.

## 03_evaluation.ipynb — MSR-VTT 1k-A Benchmark

Quantitative evaluation of the retrieval pipeline.

**Tier 1: Retrieval Accuracy** — Dense-only R@1/R@5/R@10 on MSR-VTT 1k-A (1,000 test videos) vs. paper baseline (51.9/74.6/81.7). Uses FAISS IndexFlatIP (exact brute-force). Match within ±3% confirms correct integration.

**Tier 1.5: Latency Profiling** — End-to-end latency across 4 configurations (BM25/Dense/Hybrid/Full) on the full 7,010-clip corpus.

| Method | R@1 | R@5 | R@10 |
|---|---|---|---|
| InternVideo2-1B #F=4 (paper) | 51.9 | 74.6 | 81.7 |
| Ours: Dense-only | _TBD_ | _TBD_ | _TBD_ |

## Architecture

```
Script/Query
    │
    ▼
[QueryPreprocessor] ─── Papago (ko→en)
    │
    ├── Text query ─────────────────────────┐
    │                                       │
    ├── Script ──→ [ScriptParser] ──→ Scene Graph JSON
    │               (GPT-4o-mini)     (per-scene description + attributes)
    │                                       │
    ▼                                       ▼
┌──────── Retrieval Pipeline ──────────┐    │
│ [BM25] ←→ [Dense (InternVideo2)]     │    │
│      └──→ [WRRF Fusion]             │    │
│              └──→ [ColBERT Rerank]   │    │
│                   └──→ [ITM Rerank] │    │
└──────────────────────────────────────┘    │
    │ Top-K candidate clips                  │
    ▼                                       ▼
[StoryboardMapper] ← Scene Graph attributes
    │
    ├── USE_AS_IS ──→ Use clip directly
    │
    ├── TRANSFORM ──→ [InversePromptEngine]
    │                   Cinematic prompt (GPT-4o-mini)
    │                     ├──→ TokenFlow  (local, SD video-to-video)
    │                     └──→ Runway Gen-4 Turbo (API video-to-video)
    │
    └── GENERATE ───→ Runway Gen-4 Turbo
                       (text-to-video generation)
    │
    ▼
  ★ PD Review (accept/edit/retry/skip/upload)
    │
    ▼
[VideoAssembler]
  DINOv2 transition scoring (CUT/CROSSFADE/MORPH)
  DreamColour 3D LUT color normalization
  FFmpeg rendering
    │
    ▼
[C2PA Tagger] ── ES256 provenance signing
    │
    ▼
Final Video + TC-Score + C2PA Metadata
```

## Tech Stack

| Role | Technology | Source |
|---|---|---|
| Video Embedding | InternVideo2-1B (512-dim, 4 frames) | Shanghai AI Lab, CVPR 2024 |
| Sparse Retrieval | BM25 + spaCy lemmatizer | rank_bm25 |
| Dense Index | FAISS IVFFlat (cosine/IP) | Meta AI Research |
| Retrieval Fusion | WRRF (w_visual=0.6, w_text=0.4, k=60) | Custom (Cormack 2009) |
| Reranking | ColBERT v2 MaxSim (brute-force) | Stanford, NAACL 2022 |
| Final Reranking | ITM (InternVideo2 cross-attention) | Custom integration |
| Script Parsing | GPT-4o-mini → Scene Graph JSON | OpenAI |
| Inverse Prompt | InversePromptEngine (attribute→cinematic prompt) | Custom |
| Video Transform (local) | TokenFlow (SD + DDIM inversion, 8fps/keyframe-10) | Geyer et al. 2023 |
| Video Transform/Gen (API) | Runway Gen-4 Turbo (video-to-video / text-to-video) | Runway API |
| Transition Effects | DINOv2 visual similarity (CUT/CROSSFADE/MORPH) | Meta AI Research |
| Color Normalization | DreamColour 3D LUT | CHAITron/DreamColour |
| Temporal Consistency | TC-Score (Optical Flow) | Custom |
| Shot Detection | TransNetV2 + Agglomerative Clustering | Souček & Lokoč 2020 |
| Provenance | C2PA + ES256 signing | C2PA specification |
| Eval Index | FAISS IndexFlatIP (exact, Tier 1 only) | Custom |

## Quick Start (Colab)

```bash
# 1. Environment setup
notebooks/00_setup.ipynb

# 2. Indexing (offline, ~30 min on T4)
notebooks/01_indexing.ipynb → saves to Drive

# 2b. (One-time) Generate English captions for remaining 6010 videos
notebooks/01b_caption_remaining.ipynb → ~$7, ~2.5 hours

# 3. PD Workstation (core)
notebooks/02_demo.ipynb

# 4. Benchmark evaluation
notebooks/03_evaluation.ipynb
```

### Prerequisites

- Google Colab (T4 GPU)
- HuggingFace token (`HF_TOKEN`) — InternVideo2 weights
- OpenAI API key — GPT-4o-mini (captions, Scene Graph, inverse prompts)
- (Optional) Runway API key — for TRANSFORM/GENERATE paths (falls back to TokenFlow locally)
- (Optional) Papago API — Korean query translation
- MSR-VTT videos — MSR-VTT.ZIP on Google Drive (see `data/msrvtt/README.md`)

## Project Structure

```
videorag-public/
  src/
    pipeline.py                  # Main orchestrator
    data_models.py               # Shared data models
    input/
      query_preprocessor.py      # Korean→English translation (Papago)
      script_parser.py           # Screenplay → Scene Graph JSON (GPT-4o-mini)
    phase0_indexing/
      shot_detector.py           # TransNetV2 + Agglomerative Clustering
      embedder.py                # InternVideo2-1B embedding
      vector_store.py            # FAISS IVFFlat index
      indexer.py                 # Phase 0 orchestrator
    phase12_search/
      bm25_retriever.py          # BM25 + spaCy lemmatizer
      dense_retriever.py         # FAISS dense retrieval
      hybrid_fusion.py           # WRRF fusion
    phase3_reranking/
      reranker.py                # ColBERT v2 MaxSim
      itm_scorer.py              # ITM final reranking
    phase4_assembly/
      storyboard_mapper.py       # Scene Graph → 3-path routing
      inverse_prompt_engine.py   # Inverse prompt + TokenFlow/Runway calls
      tokenflow_wrapper.py       # TokenFlow video-to-video wrapper
      assembler.py               # Video assembly
      visual_scorer.py           # DINOv2 visual similarity
      transition_selector.py     # CUT/CROSSFADE/MORPH selection
      colour_normalizer.py       # DreamColour 3D LUT
      morph_transition.py        # Optical Flow morph transition
      tc_scorer.py               # TC-Score
    phase5_c2pa/
      c2pa_tagger.py             # C2PA ES256 signing
    evaluation/
      faiss_flat_eval.py         # Exact-search eval index
  notebooks/
    00_setup.ipynb               # Environment setup
    01_indexing.ipynb            # Offline indexing (7,010 clips)
    01b_caption_remaining.ipynb  # Caption generation for remaining 6,010
    02_demo.ipynb                # ★ PD Workstation
    03_evaluation.ipynb          # ★ MSR-VTT 1k-A benchmark
  docs/                          # Technical docs + issue reports
  data/
    msrvtt/                      # Benchmark data
    queries/                     # Demo query sets
```

## Design Decisions

**Why 3-path routing?** An archive alone can't cover every scene. If a clip matches, use it (USE_AS_IS). If the content matches but it's daytime instead of night, transform it (TRANSFORM). If nothing exists, generate it (GENERATE). The PD reviews every decision, so AI automation and human editorial judgment coexist.

**Why TokenFlow + Runway dual transform backends?** Runway incurs API costs and requires internet. TokenFlow runs locally on Colab using Stable Diffusion — zero cost. Keyframe subsampling (8fps extraction, 1 keyframe per 10 frames) makes it practical: a 5-second clip processes in ~30–60s on T4. Both backends share the same Gradio interface, letting the PD decide the cost/quality tradeoff per clip.

**Why Inverse Prompt?** Telling Runway "change evening to night" just makes things darker. InversePromptEngine generates cinematic prompts with scene intent, producing far better transforms.

**Why Hybrid Retrieval?** BM25 catches proper nouns and numbers; Dense (InternVideo2) captures semantic similarity. WRRF fuses both, ColBERT reranks the top candidates with token-level precision, and ITM cross-attention performs final reranking using full visual-text alignment.

**Why C2PA?** When the final video mixes archive originals with AI-transformed and AI-generated clips, provenance tracking proves cryptographically which clip is which.

## Background

This prototype was developed for a Korean government R&D project: "Interactive Multimodal AI-based Media Production Technology." The overall project comprises Sub-task 1 (Vibe Editing Platform), Sub-task 2 (Inverse Prompt Video Generation), and Sub-task 3 (Retrieval-based Factual Video Synthesis). This repository is the Sub-task 3 prototype implementation.

## License

This project integrates multiple open-source components. See individual module headers for specific licenses and attributions.
