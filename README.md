# VideoRAG — Hybrid Retrieval + Sourced Video Synthesis

A **Video Retrieval-Augmented Generation** system that combines sparse (BM25) and dense (InternVideo2) retrieval with late-interaction reranking (ColBERT) to find, rank, and assemble video clips from a natural language query — with C2PA provenance signing.

Built as a solo prototype in 2 weeks on Google Colab (T4 Free Tier).

## Key Results

### MSR-VTT 1k-A Zero-Shot Text-to-Video Retrieval

| Method | R@1 | R@5 | R@10 | MdR | MnR |
|---|---|---|---|---|---|
| InternVideo2-1B #F=4 (paper) | 51.9 | 74.6 | 81.7 | - | - |
| Ours: Dense-only | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

> Eval notebook: `notebooks/03_evaluation.ipynb`
> Paper baseline: InternVideo2 Table 24a (Supplementary), CVPR 2024

### Pipeline Latency Ablation (7,010 clips, Colab T4)

| Configuration | Avg Latency |
|---|---|
| BM25 only | _TBD_ |
| Dense only (InternVideo2) | _TBD_ |
| Hybrid (WRRF 0.6/0.4) | _TBD_ |
| Hybrid + ColBERT (full) | _TBD_ |

## Architecture

```
Query ─→ [QueryPreprocessor] ─→ Papago (ko→en) + spaCy lemmatize
                │
    ┌───────────┴───────────┐
    ▼                       ▼
[BM25 Retriever]    [Dense Retriever]
  spaCy lemma         InternVideo2-1B
  rank_bm25            FAISS IVFFlat
    │                       │
    └───────┬───────────────┘
            ▼
    [Hybrid Fusion (WRRF)]
     w_visual=0.6, w_text=0.4, k=60
            │
            ▼
    [ColBERT Reranker]
     Late Interaction (Phase 3)
            │
            ▼
    [Video Assembler]
     DINOv2 transition scoring
     FFmpeg concat + morph blending
            │
            ▼
    [C2PA Tagger]
     Provenance signing (ES256)
            │
            ▼
      Output Video + Metadata
```

### Phase Breakdown

| Phase | Module | Latency | Description |
|---|---|---|---|
| 0 | `indexer.py` | Offline | MSR-VTT indexing (BM25 + FAISS + metadata) |
| 1-2 | BM25 + Dense + WRRF | ~700ms | Hybrid retrieval with weighted RRF fusion |
| 3 | `reranker.py` (ColBERT) | ~400ms | Late-interaction reranking |
| 4 | `assembler.py` | ~500ms | DINOv2 transition + video assembly |
| 5 | `c2pa_tagger.py` | ~50ms | C2PA provenance signing |

## Tech Stack

| Component | Technology | Source |
|---|---|---|
| Video Embedding | InternVideo2-1B (512-dim, 4 frames) | Shanghai AI Lab, CVPR 2024 |
| Sparse Retrieval | BM25 + spaCy lemmatizer | rank_bm25 |
| Dense Index | FAISS IVFFlat (cosine/IP) | Meta AI Research |
| Fusion | WRRF (Weighted Reciprocal Rank Fusion) | Custom (based on Cormack 2009) |
| Reranking | ColBERT v2 (Late Interaction) | Stanford, NAACL 2022 |
| Transition | DINOv2 visual similarity scoring | Meta AI Research |
| Shot Detection | TransNetV2 | Soucek & Lokoc 2020 |
| Provenance | C2PA standard + ES256 signing | C2PA specification |
| Eval Index | FAISS IndexFlatIP (exact, eval only) | Custom eval helper |

## Quick Start (Colab)

```bash
# 1. Environment setup
notebooks/00_setup.ipynb

# 2. Indexing (offline, ~30 min on T4)
notebooks/01_indexing.ipynb → saves to Drive

# 3. Demo (Gradio UI)
notebooks/02_demo.ipynb

# 4. Evaluation (MSR-VTT 1k-A benchmark)
notebooks/03_evaluation.ipynb
```

### Prerequisites

- Google Colab (T4 GPU, free tier sufficient)
- HuggingFace token (`HF_TOKEN`) for InternVideo2 weights
- (Optional) Papago API credentials for Korean query translation
- MSR-VTT test videos for evaluation (see `data/msrvtt/README.md`)

## Project Structure

```
videorag_prototype/
  src/
    pipeline.py                  # Main orchestrator
    input/                       # Query preprocessing
    phase0_indexing/             # Embedder, BM25, FAISS, shot detection
    phase12_search/              # BM25, Dense, Hybrid fusion
    phase3_reranking/            # ColBERT reranker
    phase4_assembly/             # Video assembly, DINOv2 transitions
    phase5_c2pa/                 # Provenance signing
    evaluation/                  # Metrics + eval helpers
      faiss_flat_eval.py         # Exact-search eval index
      metrics.py                 # R@K, nDCG, hallucination metrics
    hallucination/               # Hallucination detection
    output/                      # Timeline export, provenance report
  notebooks/
    00_setup.ipynb               # Environment bootstrap
    01_indexing.ipynb             # Offline indexing
    02_demo.ipynb                # Gradio demo
    03_evaluation.ipynb          # MSR-VTT 1k-A benchmark
  data/
    msrvtt/                      # Benchmark data
      annotations/               # 1k-A annotation JSON
      test_1ka_videos/           # Test video files
    queries/                     # Demo query sets
  config/                        # Pipeline configuration
```

## Design Decisions

**Why Hybrid (BM25 + Dense)?** BM25 catches exact keyword matches that dense retrieval may miss (e.g., proper nouns, numbers). Dense retrieval captures semantic similarity. WRRF fusion combines both with tunable weights.

**Why ColBERT reranking?** First-stage retrieval (BM25 + Dense) trades precision for speed over a large corpus. ColBERT's token-level late interaction provides finer-grained relevance scoring on the top-K candidates.

**Why FAISS IVFFlat for production, IndexFlatIP for eval?** IVFFlat is approximate but fast for the full 7K+ corpus. For the 1k-A benchmark (1,000 videos), exact brute-force search is both faster and necessary for fair comparison with published numbers.

**Why C2PA?** Provenance tracking is increasingly important for AI-generated/assembled content. C2PA signing provides cryptographic proof of the assembly pipeline's actions.

## Evaluation Methodology

See `notebooks/03_evaluation.ipynb` for full details.

**Tier 1** validates our InternVideo2 integration by comparing Dense-only retrieval against the published paper baseline on MSR-VTT 1k-A. A match within +-3% confirms correct model setup.

**Tier 1.5** profiles end-to-end latency across 4 pipeline configurations (BM25 / Dense / Hybrid / Full) on the production corpus, showing the cost-benefit tradeoff of each component.

## License

This project integrates multiple open-source components. See individual module headers for specific licenses and attributions.
