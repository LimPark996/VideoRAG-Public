# VideoRAG — AI-Powered Video Search, Transform & Assembly Workstation

> A PD workstation for broadcast production. Input a screenplay or natural language query → retrieve matching scenes from a video archive → apply color grading when attributes don't match → assemble a final edited video. The PD reviews and controls every decision.

**Live demo:** https://limpark996.github.io/VideoRAG-Public/  
Prototype built on Google Colab T4.

---

## Overview

MSR-VTT is just a benchmark dataset — the pipeline can attach to any video archive. Re-run `01_indexing.ipynb` with domain footage to adapt it, though retrieval quality may vary by domain and may need additional tuning.

What this system has concretely demonstrated: **retrieval accuracy (R@1 44.4%)** and a modular, swappable pipeline. Color grading adjusts mood/atmosphere; assembly uses CUT edits. Each stage is independently measurable and replaceable — this is a prototype, not a finished product.

| Domain | What this pipeline offers | Limitations |
|--------|--------------------------|-------------|
| Broadcast / news archives | Narrow candidate clips via natural language instead of timecode or filename lookup | Color grading cannot perfectly replicate dramatic attribute shifts (day→night, etc.) |
| Stock video platforms | Semantic scene grouping and candidate recommendation as an internal tool | Retrieval quality varies by domain |
| Ad / short-form production | Script-based candidate retrieval + segment editing workflow | Not fully automated — editor involvement required |

---

## Web Demo (GitHub Pages)

**URL:** https://limpark996.github.io/VideoRAG-Public/

React web app demonstrating the full search → assembly pipeline. 10 pre-configured broadcast scenarios · 163 MSR-VTT clips · Top-5 ITM-reranked results per scene.

**PD workflow:**

1. Select a broadcast scenario (10 topics available)
2. Pick a scene tab → review Top-5 ITM-reranked clips
3. Click a clip → set In/Out crop points
4. Check routing recommendation — Canvas API analyzes the mid-frame of the crop range, estimating brightness (time of day) and saturation/green ratio (season), then compares against required scene attributes to recommend **USE AS-IS** or **TRANSFORM**. Note: single-frame analysis can be inaccurate when lighting or color varies significantly within a clip.
5. **USE AS-IS** → clip used directly · **TRANSFORM** → apply one of 18 OpenCV color presets (7 tone · 4 mood · 7 look)
6. Repeat per scene (completed scenes show a green badge)
7. Assemble → FFmpeg multi-input concat (single pass) → final video

> SD img2img and TokenFlow were evaluated but excluded — content degradation was too severe and per-clip latency too high. OpenCV grading may also appear subtle depending on source footage characteristics.

**Backend:** Modal serverless T4 GPU (`scripts/modal_transform.py`)

```bash
modal deploy scripts/modal_transform.py   # backend
cd videorag-demo && npm run deploy        # GitHub Pages frontend
```

---

## Tech Stack

### Demo pipeline

| Role | Technology | Source |
|------|-----------|--------|
| Video embedding | InternVideo2-1B (512-dim, 4 frames) | Shanghai AI Lab, CVPR 2024 |
| Sparse retrieval | BM25 + spaCy lemmatizer (k1=1.5, b=0.75) | rank_bm25 |
| Dense index | FAISS IVFFlat (nlist=100, nprobe=10) | Meta AI Research |
| Retrieval fusion | WRRF (w_visual=0.6, w_text=0.4, k=60) | Based on Cormack 2009, custom design |
| Reranking | ColBERT v2 MaxSim (brute-force; sufficient at 7K scale) | Stanford, SIGIR/NAACL 2022 |
| Final reranking | ITM (InternVideo2 cross-attention, applied to full 1k) | Custom integration |
| Text embedding | InternVideo2 encode_text + mean pooling (ITC collapse workaround) | Custom modification |
| Video transform | OpenCV per-frame color grading (18 presets) + FFmpeg re-encode via Modal T4 | OpenCV / FFmpeg |
| Transitions | FFmpeg CUT — DINOv2-based auto transitions (CUT/CROSSFADE/MORPH) are full-system only; excluded from demo due to GPU cold start (60–120s) | FFmpeg / DINOv2 (Meta AI) |

### Full system only

| Role | Technology | Source |
|------|-----------|--------|
| Script parsing | GPT-4o-mini → Scene Graph JSON | OpenAI |
| Inverse prompt | InversePromptEngine (attributes → cinematic prompt, rule-based) | Custom design |
| AI stylization | TokenFlow video-to-video / Runway API / SD img2img *(evaluated; excluded — degradation + speed)* | TokenFlow / Runway / Stable Diffusion |
| Color grading | DreamColour 3D LUT (Reinhard colour transfer → auto-generated 3D LUT) | CHAITron/DreamColour |
| Transition selection | DINOv2 visual similarity → auto CUT/CROSSFADE/MORPH (`visual_scorer.py`) | DINOv2: Meta AI + custom logic |
| Shot detection | TransNetV2 + Agglomerative Clustering | Souček & Lokoč 2020 |
| Temporal consistency | TC-Score (Optical Flow-based) | Custom design |
| Provenance | C2PA + ES256 signing | C2PA specification |
| Evaluation index | FAISS IndexFlatIP (exact brute-force, Tier 1) | Custom implementation |

---

## Benchmark — MSR-VTT 1k-A

Evaluated on MSR-VTT 1k-A split (1,000 test videos). FAISS IndexFlatIP (exact brute-force) eliminates approximate search error.

| Method | R@1 | R@5 | R@10 |
|--------|-----|-----|------|
| InternVideo2-1B #F=4 (paper, ITC+ITM) | 51.9 | 74.6 | 81.7 |
| **Ours: full ITM** | **44.4** | **66.3** | **75.8** |

**Gap (−10.8%p):** ITC text embeddings collapse to cosine ≈ 0.9997 across all pairs, making top-128 pre-filtering effectively random — 22.5% of ground truths drop out at this stage, pushing R@1 down to 39.5%. Skipping ITC pre-filter and running ITM directly over all 1,000 videos recovers 4.9%p. Root cause (checkpoint mismatch, feature pipeline branching, etc.) is unconfirmed. See `docs/issue_report_8th.md` for full diagnosis.

**Tier 1.5 latency profiling:** End-to-end latency measured across 4 configurations (BM25 / Dense / Hybrid / Full) on the 7,010-video corpus.

---

## Architecture

```
Script / Query
    │
    ▼
[QueryPreprocessor] ── Papago (ko→en)                     [Full system]
    │
    ├── text query ─────────────────────────────────────┐
    ├── script ──→ [ScriptParser / GPT-4o-mini] ──→ Scene Graph
    │                                                   │
    ▼                                                   ▼
┌──────────────── Retrieval Pipeline ───────────────────┐
│  [BM25] ←→ [Dense (InternVideo2)]                     │
│        └──→ [WRRF Fusion]                             │
│                 └──→ [ColBERT Reranking]              │
│                          └──→ [ITM Reranking]         │
└───────────────────────────────────────────────────────┘
    │  Top-K candidate clips
    ▼
[StoryboardMapper] ← Scene Graph attributes
    │
    ├── USE_AS_IS ──→ clip as-is
    └── TRANSFORM ──→ OpenCV color grading (18 presets)  ★ Demo
                      TokenFlow / Runway AI stylization  ★ Full system
    │
    ▼
★ PD Review (confirm / retry / skip / upload)
    │
    ▼
[VideoAssembler]
    DreamColour 3D LUT                                   ★ Full system
    DINOv2 transition scoring (CUT/CROSSFADE/MORPH)      ★ Full system
    FFmpeg CUT rendering                                 ★ Demo
    │
    ▼
Final video
    └── C2PA ES256 provenance signing                    ★ Full system
```

---

## Design Decisions

**Why 2-path routing?**
Clips that already match are used as-is. Clips with attribute mismatches or low scores go through transformation. Fully generative video creation is better handled by dedicated tools. The PD can override every routing decision — AI automation and human editorial judgment coexist.

**Why OpenCV color grading?**
The goal is attribute shifting before assembly (day→night, summer→winter). SD img2img and TokenFlow alter content unpredictably and take minutes per clip — both were evaluated and excluded due to content degradation and speed. OpenCV per-frame processing (R/G/B gain/offset, contrast, HSV saturation, sepia/Teal-Orange effects) preserves content while shifting color. Results may be subtle depending on source footage.

**Why InversePromptEngine?** *(Full system)*
Instructing a generative model to "change evening to night" tends to just darken the image. InversePromptEngine generates a cinematic prompt that encodes scene intent, improving transform quality. Example output: *"A sprawling cityscape at night, neon signs blazing in electric blue and magenta, deep indigo sky, volumetric haze catching the neon glow."*

**Why hybrid retrieval?**
BM25 captures proper nouns and numbers; InternVideo2 dense retrieval captures semantic similarity. WRRF combines both; ColBERT v2 MaxSim provides precision reranking; ITM cross-attention handles final reranking. Due to ITC collapse, the dense channel uses mean pooling instead of CLS.

**Why full ITM?**
Standard approach: ITC pre-filters to top-128, then ITM runs on those candidates. When ITC embeddings collapse (cosine ≈ 0.9997 for all pairs), that pre-filter is effectively random — 22.5% of ground truths are lost before ITM even runs (R@1 drops to 39.5%). Running ITM over all 1,000 videos directly (R@1 44.4%) outperforms the standard two-stage approach by 4.9%p.

**Why C2PA?**
In a final video that mixes archive clips and AI-transformed content, C2PA cryptographically proves which clips are original archive footage and which are AI-generated.

**Why is shot detection disabled in the demo?**
TransNetV2-based `shot_detector` accumulates per-clip frame extraction overhead — indexing takes hours. The demo uses single-frame, single-vector per clip instead.

---

## Known Limitations

### Single-frame attribute bias

Both the full system (`_compute_attribute_match`) and the demo (Canvas API) infer scene attributes from a single frame per clip.

- Full system: first frame of clip
- Demo: mid-frame of crop range — `(cropStart + cropEnd) / 2`

When a clip transitions significantly over time (e.g., day→night within the clip), the selected frame may not represent the dominant visual character. Attribute judgment accuracy degrades proportionally to within-clip visual variation.

**Planned improvement:** Sample multiple frames at equal intervals, then use majority vote or averaged attributes.

---

## Full System — `02_demo.ipynb`

Full-featured Gradio prototype on Google Colab T4. Two-tab interface:

### Tab 1: Scene Graph Workflow

Input a screenplay (JSON) → GPT-4o-mini extracts per-scene `description` (English, for retrieval) and `attributes` (time of day, season, mood, location) → system auto-proposes 2-path routing.

| Branch | Auto-routing criteria | Processing |
|--------|----------------------|-----------|
| USE_AS_IS | Retrieval score ≥ threshold AND attribute match ≥ threshold | Use clip directly |
| TRANSFORM | Attribute mismatch OR low retrieval score | Generate inverse prompt → InversePromptEngine → TokenFlow / Runway transform |

PD actions per scene: preview candidates → select clip → review inverse prompt → set crop range → confirm / retry / skip / upload → reorder scenes → final assembly (DINOv2 transitions + DreamColour + C2PA signing)

### Tab 2: PD Curation *(TBD)*

Direct text query search without Scene Graph → PD selects, excludes, reorders clips → assemble. Intended for quick B-roll extraction.

### Shared features

Real-time log panel · per-stage latency chart · TC-Score (temporal consistency) · C2PA provenance signing

---

## Project Structure

```
videorag-public/
  src/
    pipeline.py                      # Main orchestrator
    data_models.py                   # Shared data models
    input/
      query_preprocessor.py          # Papago translation           [Full system]
      script_parser.py               # GPT-4o-mini → Scene Graph    [Full system]
    phase0_indexing/
      shot_detector.py               # TransNetV2 shot detection     [Full system]
      embedder.py                    # InternVideo2-1B embedding
      vector_store.py                # FAISS IVFFlat index
      indexer.py                     # Phase 0 orchestrator
    phase12_search/
      bm25_retriever.py              # BM25 + spaCy
      dense_retriever.py             # FAISS dense retrieval
      hybrid_fusion.py               # WRRF fusion
    phase3_reranking/
      reranker.py                    # ColBERT v2 MaxSim
      itm_scorer.py                  # ITM final reranking
    phase4_assembly/
      storyboard_mapper.py           # Scene Graph → 2-path routing
      inverse_prompt_engine.py       # Inverse prompt generation     [Full system]
      tokenflow_wrapper.py           # TokenFlow wrapper             [Full system]
      assembler.py                   # Video assembly
      visual_scorer.py               # DINOv2 visual similarity      [Full system]
      transition_selector.py         # CUT/CROSSFADE/MORPH           [Full system]
      colour_normalizer.py           # DreamColour 3D LUT            [Full system]
      morph_transition.py            # Optical Flow transition       [Full system]
    phase5_c2pa/
      c2pa_tagger.py                 # C2PA ES256 signing            [Full system]
    evaluation/
      faiss_flat_eval.py             # Exact-search evaluation index
  notebooks/
    00_setup.ipynb                   # Environment setup
    01_indexing.ipynb                # Offline indexing (7,010 videos)
    01b_caption_remaining.ipynb      # Caption generation (remaining 6,010 videos)
    02_demo.ipynb                    # ★ PD Workstation (full system)
    03_evaluation.ipynb              # ★ MSR-VTT 1k-A benchmark
  scripts/
    modal_transform.py               # ★ Transform/Assemble API (Modal deployment)
  docs/
    indexing_search_flow.md          # Full indexing & retrieval flow
    tech_sources.md                  # Per-module paper & license references
    issue_report_1st–8th.md          # Per-iteration issue diagnosis & resolution logs
  data/
    msrvtt/                          # Benchmark data
    queries/                         # Demo query set
```

---

## Quick Start (Colab)

**Prerequisites:**
- Google Colab with T4 GPU
- HuggingFace token (`HF_TOKEN`) — required for InternVideo2 weights
- OpenAI API key — GPT-4o-mini (captions, Scene Graph, inverse prompts)
- *(Optional)* Papago API key — Korean query translation
- MSR-VTT videos uploaded to Google Drive as `MSR-VTT.ZIP` (see `data/msrvtt/README.md`)

```bash
notebooks/00_setup.ipynb        # environment setup
notebooks/01_indexing.ipynb     # indexing (~30 min on T4)
notebooks/02_demo.ipynb         # PD Workstation
notebooks/03_evaluation.ipynb   # quantitative retrieval evaluation
```

---

## Background

Developed for a Korean government R&D project: *"Conversational Multimodal AI-Based Media Production Technology Development"* — specifically Sub-task 3: *"High-Speed Retrieval-Based Factual Video Synthesis."* The broader project spans Sub-task 1 (vibe editing), Sub-task 2 (inverse-prompt video generation), and Sub-task 3 (retrieval-based video synthesis). This repository is the Sub-task 3 prototype.

---

## License

This project integrates multiple open-source components. See individual module headers for license and attribution details.

---

<details>
<summary>🇰🇷 한국어 설명</summary>

방송사 PD가 대본(큐시트)이나 자연어 쿼리를 입력하면 영상 아카이브에서 장면을 검색하고, 속성이 맞지 않으면 색감 변환을 적용하고, 최종 편집 영상까지 만들어주는 시스템. 검색과 생성의 경계를 PD가 직접 제어한다는 것이 핵심이다. Google Colab T4에서 개발한 프로토타입.

MSR-VTT는 벤치마크 데이터셋일 뿐이며, 파이프라인 구조는 어떤 영상 아카이브에든 붙을 수 있다. 현재 시스템이 실제로 증명한 것은 검색 정확도(R@1 44.4%)와 파이프라인 구조다. 각 단계를 독립적으로 측정·교체할 수 있는 프로토타입으로 봐야 한다.

정부 R&D 과제 "대화형 멀티모달 AI 기반 미디어 프로덕션 기술개발"의 세부3 "고속 검색 기반 사실형 영상 합성 기술개발"을 위해 개발됐다.

</details>
