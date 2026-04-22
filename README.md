# VideoRAG — AI 기반 영상 검색·변환·합성 PD 워크스테이션

방송사 PD가 대본(큐시트)이나 자연어 쿼리를 입력하면, 영상 아카이브에서 장면을 검색하고, 속성이 맞지 않으면 AI로 변환하고, 아예 적합한 영상이 없으면 새로 생성해서, 최종 편집 영상까지 자동으로 만들어주는 시스템이다.

핵심은 **검색(Retrieval)과 생성(Generation)의 경계를 PD가 직접 제어한다**는 것이다. 기존 시스템은 검색만 하거나(Google), 생성만 하거나(Runway), 출처만 추적하는데(Adobe), VideoRAG는 이 세 가지를 하나의 워크플로로 통합한다. PD는 각 장면마다 아카이브 클립을 그대로 쓸지, AI로 변환할지, 새로 생성할지를 직접 결정하고, 프롬프트를 수정하고, 구간을 크롭하고, 최종 합성까지 하나의 인터페이스에서 처리한다.

Google Colab T4에서 1인 개발한 프로토타입이다.

## 웹 데모 (GitHub Pages)

**주소:** https://limpark996.github.io/VideoRAG-Public/

전체 검색→합성 파이프라인을 시연하는 React 웹앱. 10개 사전 구성 방송 시나리오, MSR-VTT 163개 클립에서 Top-5 ITM 재순위 결과를 제공한다.

**PD가 하는 일:**
1. 시나리오 선택 (방송 주제 10개)
2. 장면 탭 선택 → Top-5 ITM 재순위 클립 확인
3. 클립 클릭 → In/Out 구간 설정
4. **USE AS-IS** — 클립 그대로 사용 / **TRANSFORM** — FFmpeg 색감 프리셋 10종 중 하나 적용 (warm / cool / golden hour / noir / cinematic / documentary / dramatic / night / tense / vibrant)
5. 각 장면마다 반복 (완료된 장면은 초록 배지)
6. 합성 패널에서 **드래그로 장면 순서 변경**
7. **합성** → DINOv2 전환 스코어링 (CUT/CROSSFADE/MORPH) + DreamColour 3D LUT 색보정 → 최종 영상

**백엔드:** Modal 서버리스 (DINOv2·DreamColour용 T4 GPU, 변환은 FFmpeg)

## 02_demo.ipynb — PD 워크스테이션 (전체 시스템)

전체 기능을 갖춘 Gradio 프로토타입. Google Colab T4에서 실행하는 2-탭 인터페이스다.

### Tab 1: Scene Graph 워크플로

PD가 대본(JSON)을 넣으면 GPT-4o-mini가 장면별 Scene Graph를 생성한다. 각 장면의 description(영어, 검색용)과 attributes(시간대, 계절, 분위기, 장소)가 추출되고, 시스템이 아카이브를 검색한 뒤 **장면마다 2경로 분기 판정을 자동으로 제안**한다.

| 분기 | 자동 판정 기준 | 처리 |
|---|---|---|
| **USE_AS_IS** | 검색 점수 ≥ 임계값 + 속성 일치도 ≥ 임계값 | 그대로 사용 |
| **TRANSFORM** | 속성 불일치 또는 검색 점수 낮음 | 역프롬프트 생성 → InversePromptEngine → AI 스타일 변환 |

**PD가 매 장면에서 하는 일:**
1. 상위 5개 후보 클립을 영상으로 미리보기
2. 후보 클립 한개 선정
3. TRANSFORM이면 — 생성된 역프롬프트를 확인/직접 수정, 원하는 구간을 슬라이더로 크롭
4. 승인 / 재시도 / 건너뛰기 / 직접 파일 업로드 중 선택
5. 모든 장면 완료 후 장면 순서 재배치
6. 최종 합성 → DINOv2 전환 효과 + DreamColour 색보정 + C2PA 서명된 영상 출력

**역프롬프트(Inverse Prompt)란?**
"저녁→밤으로 바꿔" 같은 추상적 지시가 아니라, 장면 의도(Scene Intent)를 포함한 구체적 시네마틱 프롬프트를 자동 생성하는 것이다. "네온사인이 빛나는 도시 밤" 장면인데 검색된 클립이 저녁이면 "A sprawling cityscape at night, neon signs blazing in electric blue and magenta, deep indigo sky, volumetric haze catching the neon glow" 같은 프롬프트를 Rule Based로 생성한다.

### Tab 2: PD 큐레이션(TBD)

Scene Graph 없이 텍스트 쿼리로 바로 검색 → PD가 클립을 직접 선택/제외/순서 변경 → 합성. 빠르게 B-roll을 뽑을 때 사용한다.

### 공통 기능

실시간 로그 패널, 단계별 레이턴시 차트, TC-Score(시간 일관성) 표시, C2PA 출처 서명.

## 03_evaluation.ipynb — MSR-VTT 1k-A 벤치마크

검색 파이프라인의 정량 평가 노트북이다. InternVideo2 통합이 올바른지, 각 컴포넌트가 얼마나 기여하는지를 측정한다.

**Tier 1: 검색 정확도** — MSR-VTT 1k-A split(테스트 영상 1,000개)에서 R@1, R@5, R@10을 논문 수치와 대조한다. FAISS IndexFlatIP(exact brute-force)으로 근사 검색 오차를 제거한다.

**Tier 1.5: 레이턴시 프로파일링** — 7,010개 전체 코퍼스에서 4가지 구성(BM25 / Dense / Hybrid / Full)의 엔드투엔드 레이턴시를 측정하여, 각 컴포넌트(BM25, FAISS, WRRF 융합, ColBERT)의 비용-편익 트레이드오프를 보여준다.

| 방법 | R@1 | R@5 | R@10 |
|---|---|---|---|
| InternVideo2-1B #F=4 (논문, ITC+ITM) | 51.9 | 74.6 | 81.7 |
| Ours: full ITM | **41.1** | **65.9** | **76.1** |

**논문 대비 -10.8%p 갭 원인:** InternVideo2의 정상 파이프라인은 ITC → top-128 필터링 → ITM 재순위 순서인데, 우리 구현에서 ITC 텍스트 임베딩이 cosine ≈ 0.9997로 collapse되어 top-128 필터링을 쓸 수 없는 상태다. ITC pre-filter를 강제 적용하면 오히려 R@1이 39.5%로 하락하기 때문에 현재는 ITM을 전체 1,000개에 직접 적용(full ITM)한다. collapse의 정확한 원인(체크포인트 차이, vision/text feature mismatch 등)은 미확정이다. 상세 분석은 `docs/issue_report_8차.md` 참고.

**주요 진단 과정 (docs/issue_report_7차, 8차):**
- ITC only R@1 = 3.5% → ITM 없이는 사실상 랜덤
- `itm_head` (Linear 1024→2)가 체크포인트 로딩 시 `unexpected_keys`로 조용히 무시됐음을 발견 → 수동 탑재
- ITM 전체 적용 → R@1 41.1% (+37.6%p)
- CLS → mean pooling 전환 시 ITC recall@128 44.2% → 77.5%로 개선되나, ITM 기준 R@1은 39.5%로 하락 (top-128에서 이미 탈락한 22.5%를 복구 불가)

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
    └── TRANSFORM ──→ FFmpeg 색감 필터 ★데모         [InversePromptEngine → AI변환: 전체시스템]
                        (warm/cool/noir/cinematic 등 10종 프리셋)
    │
    ▼
  ★ PD 리뷰 + 장면 순서 드래그 리오더
    │
    ▼
[VideoAssembler] ★데모
  DreamColour 3D LUT 색보정 (첫 클립 기준 LAB 매핑)
  DINOv2 전환 스코어링 (CUT / CROSSFADE / MORPH)
  FFmpeg 렌더링
    │
    ▼                   [C2PA ES256 서명: 전체시스템]
최종 영상
```

## 기술 스택

> ✓ 데모 활성 · ✗ 데모 비활성 (전체 시스템에는 포함)

| 역할 | 기술 | 출처 | 데모 |
|---|---|---|:---:|
| 영상 임베딩 | InternVideo2-1B (512차원, 4프레임) | Shanghai AI Lab, CVPR 2024 | ✓ |
| 희소 검색 | BM25 + spaCy 레마타이저 (k1=1.5, b=0.75) | rank_bm25 | ✓ |
| 밀집 인덱스 | FAISS IVFFlat (nlist=100, nprobe=10) | Meta AI Research | ✓ |
| 검색 융합 | WRRF (w_visual=0.6, w_text=0.4, k=60) | Cormack 2009 기반 자체 설계 | ✓ |
| 리랭킹 | ColBERT v2 MaxSim (PLAID centroid pruning) | Stanford, SIGIR/NAACL 2022 | ✓ |
| 최종 재순위 | ITM (InternVideo2 cross-attention, full 1k 적용) | 자체 통합 | ✓ |
| 텍스트 임베딩 | InternVideo2 encode_text + mean pooling (ITC collapse 우회) | 자체 수정 | ✓ |
| 영상 변환 | FFmpeg 색감 필터 (warm / cool / golden hour / noir / cinematic 등 10종, Modal 서버리스) | FFmpeg | ✓ |
| 전환 효과 | DINOv2 시각 유사도 (CUT/CROSSFADE/MORPH) | Meta AI Research | ✓ |
| 대본 파싱 | GPT-4o-mini → Scene Graph JSON (데모: pre-computed JSON 사용) | OpenAI | ✗ |
| 역프롬프트 | InversePromptEngine (속성→시네마틱 프롬프트) | 자체 설계 | ✗ |
| 색보정 | DreamColour 3D LUT (assemble 시 첫 클립 기준 LAB 색공간 매핑) | CHAITron/DreamColour | ✓ |
| 시간 일관성 | TC-Score (Optical Flow 기반) — FFmpeg 필터는 결정론적, shot 단위 클립은 원본이 보장 | 자체 설계 | ✗ |
| 샷 탐지 | TransNetV2 + Agglomerative Clustering — 클립당 단일 프레임 사용 | Souček & Lokoč 2020 | ✗ |
| 출처 추적 | C2PA + ES256 서명 | C2PA specification | ✗ |
| 평가 인덱스 | FAISS IndexFlatIP (exact, Tier 1 전용) | 자체 구현 | ✗ |

## 웹 데모 배포

```bash
# Modal 백엔드 배포 (Transform + Assemble API)
modal deploy scripts/modal_transform.py

# 프론트엔드 빌드 + GitHub Pages 배포
cd videorag-demo
npm run deploy
```

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
- (선택) Papago API — 한국어 쿼리 번역
- MSR-VTT 영상 — Google Drive에 MSR-VTT.ZIP (`data/msrvtt/README.md` 참고)

## 프로젝트 구조

```
videorag-public/
  src/
    pipeline.py                  # 메인 오케스트레이터
    data_models.py               # 공용 데이터 모델
    input/
      query_preprocessor.py      # 한국어→영어 번역 (Papago)          [데모 비활성]
      script_parser.py           # 대본 → Scene Graph JSON (GPT-4o-mini) [데모 비활성 — pre-computed]
    phase0_indexing/
      shot_detector.py           # TransNetV2 + Agglomerative Clustering  [데모 비활성 — 클립당 단일 프레임]
      embedder.py                # InternVideo2-1B 임베딩 (ITC: mean pooling, ITM: full token)
      vector_store.py            # FAISS IVFFlat 인덱스
      indexer.py                 # Phase 0 오케스트레이터
    phase12_search/
      bm25_retriever.py          # BM25 + spaCy 레마타이저
      dense_retriever.py         # FAISS 밀집 검색
      hybrid_fusion.py           # WRRF 융합
    phase3_reranking/
      reranker.py                # ColBERT v2 MaxSim + PLAID
      itm_scorer.py              # ITM 최종 재순위 (itm_head 수동 탑재)
    phase4_assembly/
      storyboard_mapper.py       # Scene Graph → 2경로 분기 판정
      inverse_prompt_engine.py   # 역프롬프트 생성                     [데모 비활성]
      tokenflow_wrapper.py       # TokenFlow video-to-video 래퍼       [데모 비활성]
      assembler.py               # 영상 어셈블리
      visual_scorer.py           # DINOv2 시각 유사도
      transition_selector.py     # CUT/CROSSFADE/MORPH 자동 선택
      colour_normalizer.py       # DreamColour 3D LUT                  [데모 비활성]
      morph_transition.py        # Optical Flow 변형 전환
      tc_scorer.py               # TC-Score                            [데모 비활성 — FFmpeg 필터/shot 단위로 불필요]
    phase5_c2pa/
      c2pa_tagger.py             # C2PA ES256 서명                     [데모 비활성]
    evaluation/
      faiss_flat_eval.py         # Exact-search 평가 인덱스
  notebooks/
    00_setup.ipynb               # 환경 설정
    01_indexing.ipynb            # 오프라인 인덱싱 (7,010개)
    01b_caption_remaining.ipynb  # 나머지 6,010개 캡션 생성
    02_demo.ipynb                # ★ PD 워크스테이션
    03_evaluation.ipynb          # ★ MSR-VTT 1k-A 벤치마크
  scripts/
    modal_transform.py           # ★ 데모 Transform/Assemble API (Modal 배포용)
  docs/
    인덱싱_검색_과정_정리.md      # 인덱싱·검색 전체 흐름 (단계별 입출력 명세)
    기술_출처_정리.md             # 모듈별 논문·라이선스 출처
    issue_report_1차~8차.md      # 단계별 이슈·진단·해결 기록
  data/
    msrvtt/                      # 벤치마크 데이터
    queries/                     # 데모 쿼리셋
```

## 설계 결정

**왜 2경로 분기?** 클립이 딱 맞으면 그대로 쓰고(USE_AS_IS), 속성이 다르거나 점수가 낮으면 AI로 변환한다(TRANSFORM). 완전히 새로운 영상을 생성하는 것은 PD가 외부 툴로 직접 하는 것이 더 효율적이다. PD가 매 판정을 검토하고 오버라이드할 수 있어서, AI의 자동화와 사람의 편집 판단이 공존한다.

**왜 데모에서 FFmpeg 필터를 쓰나?** SD img2img나 TokenFlow 같은 생성 모델은 원본 영상의 내용을 예측 불가하게 변형하고 응답이 수 분씩 걸려 데모에 부적합하다. FFmpeg 필터(colorbalance, eq, hue, vignette)는 원본 화질과 내용을 유지하면서 색감·분위기만 즉각 바꾼다. warm / cool / golden hour / noir / cinematic / documentary / dramatic / night / tense / vibrant 10종 프리셋을 제공하고, 단일 선택으로 복합 적용으로 인한 의도치 않은 결과를 방지한다.

**왜 역프롬프트?** (전체 시스템 기준, 데모 비활성) 생성 모델에 "저녁을 밤으로 바꿔"라고 넣으면 그냥 어두워지기만 한다. InversePromptEngine이 장면 의도를 포함한 시네마틱 프롬프트를 생성해서 변환 품질을 높인다.

**왜 하이브리드 검색?** BM25는 고유명사·숫자 매칭, Dense(InternVideo2)는 의미 유사도. WRRF로 두 장점을 결합하고, ColBERT(PLAID centroid pruning으로 10~50x 가속)로 상위 후보를 정밀 리랭킹한 뒤 ITM cross-attention으로 최종 재순위한다. ITC 텍스트 임베딩의 cosine collapse 문제로 dense 채널은 CLS 대신 mean pooling을 사용한다.

**왜 ITM을 전체에 적용하나?** ITC cosine이 ≈0.9997로 collapse되어 top-128 pre-filter를 쓰면 R@1이 오히려 하락한다(39.5% < 41.1%). 평가에서는 full ITM(1,000개 전체)으로 R@1 41.1%를 확정했다. Production에서는 BM25·ColBERT가 병렬 채널로 동작하기 때문에 ITC collapse의 영향이 평가만큼 크지 않다.

**왜 C2PA?** AI가 변환/생성한 클립이 섞인 영상의 출처를 암호학적으로 증명한다. 어떤 클립이 아카이브 원본이고, 어떤 클립이 AI가 만든 것인지 추적 가능.

**왜 샷 탐지를 데모에서 끄나?** TransNetV2 기반 shot_detector를 사용하면 클립당 프레임 추출 시간이 누적되어 인덱싱에 수 시간이 걸린다. 데모 환경에서는 클립당 단일 프레임·단일 벡터로 설정해 속도를 확보한다.

## 배경

이 프로토타입은 정부 R&D 과제 "대화형 멀티모달 AI 기반 미디어 프로덕션 기술개발"의 세부3 "고속 검색 기반 사실형 영상 합성 기술개발"을 위해 개발되었다. 총괄 과제는 세부1(바이브 편집 플랫폼), 세부2(역프롬프트 기반 영상 생성), 세부3(검색 기반 영상 합성)으로 구성되며, 이 저장소는 세부3의 프로토타입 구현이다.

## 라이선스

이 프로젝트는 여러 오픈소스 컴포넌트를 통합한다. 개별 모듈 헤더에서 라이선스와 저작자 표시를 확인할 것.

---

# VideoRAG — AI-Powered Video Retrieval · Transform · Synthesis PD Workstation

A system where a broadcast PD inputs a screenplay or natural-language query, and the system searches the video archive for matching scenes, transforms clips whose attributes don't match via AI, and assembles the final edited video — all in one interface.

The core idea: **the PD controls the boundary between retrieval and generation**. Existing systems only search (Google), only generate (Runway/Sora), or only track provenance (Adobe). VideoRAG integrates all three into a single workflow. For each scene, the PD decides whether to use an archive clip as-is, transform it with AI, or generate from scratch — reviewing prompts, cropping segments, and approving results at every step.

Built as a solo prototype on Google Colab (T4 GPU).

## Web Demo (GitHub Pages)

**Live:** https://limpark996.github.io/VideoRAG-Public/

A React web app demonstrating the full retrieval-to-assembly pipeline. 10 pre-computed broadcast scenarios, each with Top-5 ITM-reranked clips from 163 MSR-VTT clips.

**What the PD does:**
1. Pick a scenario (10 broadcast subjects)
2. Select a scene tab — Top-5 ITM-reranked clips appear
3. Click a clip to open the player; trim In/Out points
4. **USE AS-IS** — use the clip directly; **TRANSFORM** — apply one of 10 FFmpeg color presets (warm / cool / golden hour / noir / cinematic / documentary / dramatic / night / tense / vibrant)
5. Repeat for each scene (badge turns green when decided)
6. **Drag to reorder** scenes in the assemble panel
7. **Assemble** — DINOv2 transition scoring (CUT/CROSSFADE/MORPH) + DreamColour 3D LUT color normalization → final video

**Backend:** Modal serverless (T4 GPU for DINOv2 + DreamColour, FFmpeg for transform)

## 02_demo.ipynb — PD Workstation (Full System)

The full Gradio prototype. A 2-tab interface running on Google Colab T4.

### Tab 1: Scene Graph Workflow

The PD inputs a screenplay (JSON). GPT-4o-mini generates a per-scene Scene Graph with description (English, for retrieval) and attributes (time_of_day, season, mood, location). The system searches the archive and **automatically routes each scene to one of two paths**:

| Path | Condition | Action |
|---|---|---|
| **USE_AS_IS** | Archive clip matches required attributes | Use directly |
| **TRANSFORM** | Attribute mismatch or low search score | Generate inverse prompt → InversePromptEngine → AI stylization |

**What the PD does per scene:**
1. Preview top 5 candidate clips as video
2. Select one clip
3. For TRANSFORM — review/edit the inverse prompt, crop the desired segment with sliders
4. Accept / Retry / Skip / Upload own file
5. After all scenes — reorder scenes
6. Final assembly → DINOv2 transitions + DreamColour color normalization + C2PA-signed output

**Inverse Prompt:** Not "make it darker" but a concrete cinematic prompt that includes scene intent. For a "neon-lit city night" scene where the retrieved clip is evening, InversePromptEngine generates "A sprawling cityscape at night, neon signs blazing in electric blue and magenta, deep indigo sky, volumetric haze catching the neon glow" via rule-based generation.

### Tab 2: PD Curation (TBD)

Text query → search → PD manually selects/excludes/reorders clips → assemble. Fast mode for B-roll without Scene Graph.

### Shared Features

Real-time log panel, per-phase latency chart, TC-Score (temporal consistency), C2PA provenance signing.

## 03_evaluation.ipynb — MSR-VTT 1k-A Benchmark

Quantitative evaluation of the retrieval pipeline.

**Tier 1: Retrieval Accuracy** — R@1/R@5/R@10 on MSR-VTT 1k-A (1,000 test videos) vs. paper baseline. Uses FAISS IndexFlatIP (exact brute-force) to eliminate approximate search error.

**Tier 1.5: Latency Profiling** — End-to-end latency across 4 configurations (BM25/Dense/Hybrid/Full) on the full 7,010-clip corpus.

| Method | R@1 | R@5 | R@10 |
|---|---|---|---|
| InternVideo2-1B #F=4 (paper, ITC+ITM) | 51.9 | 74.6 | 81.7 |
| Ours: full ITM | **41.1** | **65.9** | **76.1** |

**Gap analysis (-10.8%p):** The paper pipeline runs ITC → top-128 filter → ITM rerank. Our ITC text embeddings collapse to cosine ≈ 0.9997, making top-128 filtering counterproductive (R@1 drops to 39.5% when forced). Current evaluation uses full ITM over all 1,000 videos. Root cause of ITC collapse (checkpoint mismatch, vision/text feature pipeline divergence) is unconfirmed. See `docs/issue_report_8차.md` for full diagnosis.

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
    └── TRANSFORM ──→ FFmpeg color grading ★demo      [InversePromptEngine → AI transform: full system]
                        (warm/cool/noir/cinematic, 10 presets)
    │
    ▼
  ★ PD Review + drag-to-reorder scenes
    │
    ▼
[VideoAssembler] ★demo
  DreamColour 3D LUT color normalization (LAB-space mapping from first clip)
  DINOv2 transition scoring (CUT/CROSSFADE/MORPH)
  FFmpeg rendering
    │
    ▼                   [C2PA ES256 signing: full system]
Final Video
```

## Tech Stack

> ✓ Active in demo · ✗ Disabled in demo (present in full system)

| Role | Technology | Source | Demo |
|---|---|---|:---:|
| Video Embedding | InternVideo2-1B (512-dim, 4 frames) | Shanghai AI Lab, CVPR 2024 | ✓ |
| Sparse Retrieval | BM25 + spaCy lemmatizer (k1=1.5, b=0.75) | rank_bm25 | ✓ |
| Dense Index | FAISS IVFFlat (nlist=100, nprobe=10) | Meta AI Research | ✓ |
| Retrieval Fusion | WRRF (w_visual=0.6, w_text=0.4, k=60) | Custom (Cormack 2009) | ✓ |
| Reranking | ColBERT v2 MaxSim + PLAID centroid pruning | Stanford, SIGIR/NAACL 2022 | ✓ |
| Final Reranking | ITM (InternVideo2 cross-attention, full 1k) | Custom integration | ✓ |
| Text Embedding | InternVideo2 encode_text + mean pooling (ITC collapse workaround) | Custom | ✓ |
| Video Transform | FFmpeg color grading (10 presets: warm / cool / golden hour / noir / cinematic / etc., Modal serverless) | FFmpeg | ✓ |
| Transition Effects | DINOv2 visual similarity (CUT/CROSSFADE/MORPH) | Meta AI Research | ✓ |
| Script Parsing | GPT-4o-mini → Scene Graph JSON (demo: pre-computed JSON) | OpenAI | ✗ |
| Inverse Prompt | InversePromptEngine (attribute→cinematic prompt) | Custom | ✗ |
| Color Normalization | DreamColour 3D LUT (assemble 시 첫 클립 기준 LAB 색공간 매핑) | CHAITron/DreamColour | ✓ |
| Temporal Consistency | TC-Score (Optical Flow) — FFmpeg filters are deterministic; shot-level clips guarantee consistency via source | Custom | ✗ |
| Shot Detection | TransNetV2 + Agglomerative Clustering — one frame per clip | Souček & Lokoč 2020 | ✗ |
| Provenance | C2PA + ES256 signing | C2PA specification | ✗ |
| Eval Index | FAISS IndexFlatIP (exact, Tier 1 only) | Custom | ✗ |

## Web Demo Deployment

```bash
# Deploy Modal backend (Transform + Assemble API)
modal deploy scripts/modal_transform.py

# Build frontend + deploy to GitHub Pages
cd videorag-demo
npm run deploy
```

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
- (Optional) Papago API — Korean query translation
- MSR-VTT videos — MSR-VTT.ZIP on Google Drive (see `data/msrvtt/README.md`)

## Project Structure

```
videorag-public/
  src/
    pipeline.py                  # Main orchestrator
    data_models.py               # Shared data models
    input/
      query_preprocessor.py      # Korean→English translation (Papago)    [demo: disabled]
      script_parser.py           # Screenplay → Scene Graph JSON (GPT-4o-mini) [demo: disabled — pre-computed]
    phase0_indexing/
      shot_detector.py           # TransNetV2 + Agglomerative Clustering  [demo: disabled — one frame per clip]
      embedder.py                # InternVideo2-1B embedding (ITC: mean pooling, ITM: full token)
      vector_store.py            # FAISS IVFFlat index
      indexer.py                 # Phase 0 orchestrator
    phase12_search/
      bm25_retriever.py          # BM25 + spaCy lemmatizer
      dense_retriever.py         # FAISS dense retrieval
      hybrid_fusion.py           # WRRF fusion
    phase3_reranking/
      reranker.py                # ColBERT v2 MaxSim + PLAID
      itm_scorer.py              # ITM final reranking (itm_head manually loaded)
    phase4_assembly/
      storyboard_mapper.py       # Scene Graph → 2-path routing
      inverse_prompt_engine.py   # Inverse prompt generation              [demo: disabled]
      tokenflow_wrapper.py       # TokenFlow video-to-video wrapper       [demo: disabled]
      assembler.py               # Video assembly
      visual_scorer.py           # DINOv2 visual similarity
      transition_selector.py     # CUT/CROSSFADE/MORPH selection
      colour_normalizer.py       # DreamColour 3D LUT                    [demo: disabled]
      morph_transition.py        # Optical Flow morph transition
      tc_scorer.py               # TC-Score                              [demo: disabled — FFmpeg filters are deterministic; shot-level clips guarantee consistency]
    phase5_c2pa/
      c2pa_tagger.py             # C2PA ES256 signing                    [demo: disabled]
    evaluation/
      faiss_flat_eval.py         # Exact-search eval index
  notebooks/
    00_setup.ipynb               # Environment setup
    01_indexing.ipynb            # Offline indexing (7,010 clips)
    01b_caption_remaining.ipynb  # Caption generation for remaining 6,010
    02_demo.ipynb                # ★ PD Workstation
    03_evaluation.ipynb          # ★ MSR-VTT 1k-A benchmark
  scripts/
    modal_transform.py           # ★ Demo Transform/Assemble API (Modal deploy)
  docs/
    인덱싱_검색_과정_정리.md      # End-to-end indexing/search flow (step-by-step I/O spec)
    기술_출처_정리.md             # Per-module paper/license attribution
    issue_report_1차~8차.md      # Incremental issue diagnosis and resolution logs
  data/
    msrvtt/                      # Benchmark data
    queries/                     # Demo query sets
```

## Design Decisions

**Why 2-path routing?** If a clip matches, use it (USE_AS_IS). If attributes don't match or the score is low, transform it (TRANSFORM). Generating entirely new footage is better done with dedicated tools outside this system. The PD reviews every decision, so AI automation and human editorial judgment coexist.

**Why FFmpeg filters for the demo transform?** Generative models (SD img2img, TokenFlow) unpredictably alter video content and take several minutes per clip — unsuitable for an interactive demo. FFmpeg filters (colorbalance, eq, hue, vignette) preserve original content and quality while instantly changing color tone and mood. 10 presets are provided (warm / cool / golden hour / noir / cinematic / documentary / dramatic / night / tense / vibrant), each mapping to a deterministic filter chain. Single-select UI prevents unintended combinations.

**Why Inverse Prompt?** (full system; disabled in demo) Telling a generative model "change evening to night" just makes things darker. InversePromptEngine generates cinematic prompts with scene intent, producing far better transforms.

**Why Hybrid Retrieval?** BM25 catches proper nouns and numbers; Dense (InternVideo2) captures semantic similarity. WRRF fuses both, ColBERT (PLAID centroid pruning, 10–50x faster than brute-force) reranks the top candidates with token-level precision, and ITM cross-attention performs final reranking. ITC text embeddings use mean pooling instead of CLS to work around the cosine collapse issue.

**Why full ITM instead of ITC pre-filter?** ITC cosine similarity collapses to ≈0.9997, so top-128 filtering discards correct answers at 22.5% rate. Applying ITM to all 1,000 videos (R@1 41.1%) outperforms ITC→top-128→ITM (R@1 39.5%). In production, ITC collapse matters less because dense retrieval is a parallel channel alongside BM25 and ColBERT, not an exclusive gate.

**Why C2PA?** When the final video mixes archive originals with AI-transformed and AI-generated clips, provenance tracking proves cryptographically which clip is which.

**Why is shot detection disabled in demo?** Using TransNetV2 shot_detector causes indexing to take several hours due to per-clip frame extraction overhead. The demo uses one frame and one vector per clip for speed.

## Background

This prototype was developed for a Korean government R&D project: "Interactive Multimodal AI-based Media Production Technology." The overall project comprises Sub-task 1 (Vibe Editing Platform), Sub-task 2 (Inverse Prompt Video Generation), and Sub-task 3 (Retrieval-based Factual Video Synthesis). This repository is the Sub-task 3 prototype implementation.

## License

This project integrates multiple open-source components. See individual module headers for specific licenses and attributions.
