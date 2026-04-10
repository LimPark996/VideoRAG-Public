# VideoRAG Prototype — 기술 이슈 보고서

> 작성일: 2026-04-01  
> 대상: `videorag_prototype` 전체 커밋 히스토리  
> 근거: 커밋별 실제 diff 분석 + Claude Code 대화 세션

---

## 커밋 1. `53f93ad` — 파이프라인 전면 리팩토링

**변경**: 25개 파일, +3,203 / -247 lines

### TC-Score 알고리즘 교체: DINOv2 → Optical Flow

`src/phase4_assembly/tc_scorer.py`에서 DINOv2 cosine similarity 기반 TC-Score를 **Farneback Dense Optical Flow** 기반으로 완전 교체했다.

```python
flow = cv2.calcOpticalFlowFarneback(gray_a, gray_b, None, **self.FLOW_PARAMS)
mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
consistency = max(0.0, 1.0 - avg_mag / MAX_FLOW_MAGNITUDE)
```

DINOv2는 보조 지표(`dino_tc_score`)로 강등되고, 메인 TC-Score는 optical flow가 담당하게 되었다.

### Assembler에서 scene_bonus 제거

`assembler.py`의 `_sort_clips_perception()`에서 가중치를 `w_text=0.5, w_visual=0.35, scene_bonus=0.15` → `w_text=0.6, w_visual=0.4`로 변경하여 PDF 원안을 복원했다. `same_scene` 판정 로직과 `scene_id` 비교 코드를 전부 삭제했다.

`transition_selector.py`에서도 `same_scene` 파라미터를 삭제하고, 순수 DINOv2 유사도 3단계 분기만 남겼다 (sim ≥ 0.6 → Cut, 0.3~0.6 → Crossfade, < 0.3 → Morph).

### 한국어 캡션 파이프라인 + BM25 형태소 토크나이저

- `src/phase0_indexing/caption_ko.py` (258 lines): GPT-4o-mini Vision API로 영상 8프레임 → 한국어 한 문장 캡션 생성
- `src/phase12_search/bm25_retriever.py`에 `KoreanMorphTokenizer` 추가: konlpy Okt 기반, `stem=True`로 어간 추출, Noun/Verb/Adjective/Alpha/Number만 유지

### 신규 모듈 3개

| 모듈 | 규모 | 역할 |
|------|------|------|
| `hallucination_detector.py` | 536 lines | PRISM/WCS mock 환각 탐지 (3단계 판정: pass/auto_correct/regenerate) |
| `inverse_prompt_engine.py` | 807 lines | 실내/실외 분류 + 시간대/계절 추론 + Sora/Runway 변환 프롬프트 생성 |
| `storyboard_mapper.py` | 661 lines | Scene Graph JSON 파싱 + 3경로 분기(USE_AS_IS/TRANSFORM/GENERATE) |

---

## 커밋 2. `80a22ad` — 환각 탐지기 실제 구현 (PRISM+WCS)

**변경**: 20개 파일, +944 / -420 lines

### PRISM: Mock → 실제 의미 유사도

`hallucination_detector.py`에서 단순 단어 겹침(`set(query.lower().split()) & set(caption.lower().split())`)을 `sentence-transformers`의 `paraphrase-multilingual-MiniLM-L12-v2` 모델 코사인 유사도 계산으로 교체했다. 폴백으로 `_compute_tfidf_similarity()`(자카드 0.4 + 오버랩 0.6 가중평균)를 추가하고, konlpy Okt 명사 추출로 한국어를 대응했다.

### WCS: Mock → 실제 CV 메트릭

4개 독립 메트릭을 새로 구현했다:

| 메트릭 | 방법 | 가중치 |
|--------|------|--------|
| Object Persistence | ORB 특징점 300개 + BFMatcher(NORM_HAMMING, crossCheck) | 0.3 |
| Relative Scale | Canny 엣지 밀도 변화율 | 0.2 |
| Color Consistency | H-S 2D 히스토그램 32×32 bins, HISTCMP_CORREL | 0.3 |
| Frame Physics | Farneback 옵티컬 플로우 기반 물리 법칙 위반 탐지 | 0.2 |

종합 WCS = `0.3×OP + 0.2×RS + 0.3×CC + 0.2×FP`

### 데모 노트북 마이그레이션

2-tab Gradio UI 도입. 마크다운 타이틀을 "Phase 1~5 전체 파이프라인" → "Scene Graph 기반 E2E 파이프라인"으로 변경. `all_results[scenario][query]` 중첩 구조 적용.

---

## 커밋 3. `3d6f75d` — 병목 현상 해결

**변경**: 14개 파일, +72 / -1,453 lines

### 근본 원인

Scene Graph 워크플로에서 **장면별로 전체 검색 파이프라인이 end-to-end로 반복**되는 구조적 문제. 두 가지 핵심 병목이 있었다:

1. **`cv2.VideoCapture` 3회 반복 오픈**: `_decide_branch()`, `_compute_attribute_match()`, `_generate_transform_prompt_for_review()` 각각에서 동일 영상 파일을 독립적으로 열어 프레임을 추출
2. **ColBERT 리랭킹 후보 100개**: 과다한 후보 수로 리랭킹 단계 지연

### 수정 내용

`storyboard_mapper.py`의 `_map_single_scene()` 내부에서 프레임을 최초 1회만 `_frame` 변수에 추출하고, 이후 모든 함수에 `frame=_frame` 파라미터로 전달하도록 변경:

```python
# Before: 각 함수 내부에서 cv2.VideoCapture 독립 오픈 (3회)
branch, attr_match = self._decide_branch(best_clip, req)

# After: 1회 추출 + 파라미터 전달
cap = cv2.VideoCapture(best_clip.video_path)
ret, _frame = cap.read()
cap.release()
branch, attr_match = self._decide_branch(best_clip, req, frame=_frame)
```

`_decide_branch()`, `_compute_attribute_match()` 모두 `frame: Optional[np.ndarray] = None` 파라미터를 추가하여 외부에서 전달받으면 재오픈하지 않도록 했다.

`pipeline.py`에서 ColBERT 후보 수를 `fused_results[:100]` → `fused_results[:30]`으로 축소했다.

---

## 커밋 4. `fb1b5cb` — Gradio 입력 변경 + API key 설정

**변경**: 1개 파일, +148 / -22 lines

### 수정 내용

- **Google Drive 마운트**: Step 0 셀 앞에 `drive.mount('/content/drive')` 추가
- **Sora/Runway API 키**: Colab userdata에서 `os.environ["SORA_API_KEY"] = userdata.get("SORA_API_KEY")` 설정 블록 추가
- **Scene Graph 입력**: `gr.Textbox` → `gr.Code(language="json")` 변경으로 JSON 구문 강조 지원
- **노트북 셀 포맷**: 단일 문자열 `"source": "..."` → 배열 `"source": ["...\n", "...\n"]` 형태로 변환 (Colab 호환)

---

## 커밋 5. `561fad7` — 단계별 latency 그래프 추가

**변경**: 2개 파일, +190 / -40 lines

### 수정 내용

`_make_latency_chart()` 함수를 신규 추가하여 matplotlib 수평 바 차트를 생성한다. 색상 코딩은 비율 > 40% 빨강, > 20% 주황, 나머지 초록.

핸들러 함수들(`_process()`, `on_parse()`, `on_accept()`, `on_skip()`, `on_upload()`, `on_next()`)을 `return` → `yield`/`yield from` 제너레이터로 변환하여, Gradio가 중간 진행 상태를 실시간으로 소비할 수 있게 했다.

Tab 1 출력 튜플을 13개 → 14개로 확장 (`latency_plot` 추가).

**주의**: `pipeline.py`에서 커밋 3에서 30으로 줄였던 ColBERT 후보 수를 다시 100으로 복원했다. 정확도와 속도 사이의 트레이드오프 실험 중인 것으로 보인다.

---

## 커밋 6. `3e9a8d9` — 실시간 진행 상황 + 모델 웜업

**변경**: 2개 파일, +281 / -67 lines

### 근본 원인

1. **첫 쿼리 수초 지연**: InternVideo2/ColBERT 모델이 첫 호출 시 로딩되면서 발생
2. **검색 중 UI 멈춤**: 사용자가 진행 상태를 알 수 없음

### 수정 내용

`pipeline.py`에 콜백 시스템을 추가했다:

```python
self._progress_callback = None  # (phase_name, "start"|"done", elapsed_ms)
```

각 파이프라인 단계(phase0_preprocess → phase1_bm25 → phase1_embed → phase1_faiss → phase2_fusion → phase3_reranking) 시작/완료 시 콜백을 호출한다.

Step 1 셀 끝에 모델 웜업 코드를 추가했다:

```python
pipeline.embedder.encode_query("warmup query")  # InternVideo2 웜업
pipeline.reranker._load_model()                  # ColBERT 웜업
```

Gradio 측에서는 `_run_prepare_with_progress()` 함수가 스레드 + 큐 패턴으로 콜백을 수신하여 단계별 체크박스(✅/🔄/⬜) 마크다운을 실시간 yield한다:

```python
progress_q = queue.Queue()
pipeline._progress_callback = lambda phase, status, ms: progress_q.put(...)
t = threading.Thread(target=_run, daemon=True)
t.start()
while t.is_alive():
    phase, status, ms = progress_q.get(timeout=0.25)
    yield "progress", progress_text
```

---

## 커밋 7. `5b4272a` — PD 큐레이션 통합 + 누적 검색 리셋

**변경**: 3개 파일, +1,920 / -632 lines (가장 큰 리팩토링)

### 근본 원인

Tab 2 "단순 검색"과 Tab 3 "PD 큐레이션"이 별도 UI로 분리되어 있어 코드 중복과 UX 혼란이 있었다. 누적 검색 시 이전 결과가 리셋되지 않아 검색 결과가 섞이는 버그도 있었다.

### 수정 내용

- **3탭 → 2탭 통합**: Tab 1 (Scene Graph 워크플로) + Tab 2 (PD 큐레이션 = 기존 단순 검색 + 큐레이션 합체)
- **클립 관리 헬퍼 신규**: `_extract_first_frame()`, `_get_video_duration_ms()`, `_clips_to_choices()`, `_find_clip()`, `_make_clip_result()` — 클립 목록 ↔ Gradio CheckboxGroup 변환
- **누적 검색 리셋**: 새 검색 시 `state = {"clips": [], "query": query}`로 초기화
- **디버깅 파일 생성**: `notebooks/_cell3_current_dump.py` (718 lines) — 노트북 셀 3의 전체 코드를 .py로 덤프 (추후 커밋 10에서 삭제됨)

---

## 커밋 8. `cd34177` — Scene Graph description 한국어 → 영어

**변경**: 1개 파일, +70 / -36 lines

### 근본 원인

Scene Graph의 `description` 필드가 한국어로 되어 있으면 **영어 기반 임베딩 모델(InternVideo2)과 영어 캡션(MSR-VTT) 사이의 언어 미스매치**로 검색 정확도가 급격히 떨어진다.

### 수정 내용

3개 시나리오 모두 영어 description + 한국어 description_ko 이중 구조로 변경:

```python
# Before
"description": "밤에 도시 건물들과 네온사인이 빛나는 거리"

# After
"description": "neon signs glowing on city buildings at night",
"description_ko": "밤에 도시 건물들과 네온사인이 빛나는 거리",
```

---

## 커밋 9. `71cc192` — Sora 삭제 + 검색 중복 제거 + 상위 5개 후보 + 합성 에러 반환

**변경**: 4개 파일, +186 / -275 lines

### 이슈 A: Sora API 전면 삭제

`inverse_prompt_engine.py`에서 `_apply_sora()` (~70줄)과 `_generate_sora()` (~80줄) 메서드를 전부 삭제했다. `sora_api_key`는 `openai_api_key`로 이름을 변경하여 GPT-4o-mini 프롬프트 생성에만 사용하도록 했다. `default_backend`를 `"opencv"` → `"runway"`로 변경했다.

빈 문자열 처리도 추가했다. 기존에는 Colab에서 `os.environ.get("SORA_API_KEY", "")` → 빈 문자열이 truthy로 처리되어 API 호출이 실패하고 조용히 OpenCV로 폴백되는 문제가 있었다:

```python
# Before
self.sora_api_key = sora_api_key or os.environ.get("OPENAI_API_KEY")

# After
self.openai_api_key = (openai_api_key or "").strip() or os.environ.get("OPENAI_API_KEY", "").strip() or None
```

### 이슈 B: 장면 간 검색 결과 중복

`storyboard_mapper.py`의 `prepare_scene()`에 `excluded_clip_ids: Optional[set]` 파라미터를 추가했다. 검색 결과에서 이미 다른 장면에 배정된 clip_id를 필터링한다:

```python
raw_candidates = self.search_fn(req.description, self.top_k)
candidates = [c for c in raw_candidates if c.clip_id not in excluded_clip_ids]
```

Gradio의 `_process()`에서 매핑 완료된 장면들의 clip_id를 `used_clip_ids` 집합으로 구성하여 전달한다.

### 이슈 C: PD에게 프레임(이미지)만 보여줌 → 클립(영상) 미리보기 + 상위 5개 후보

`MappedScene`과 `PDReviewRequest`에 `top_candidates: Optional[List[ClipResult]]` 필드를 추가했다. `prepare_scene()`에서 상위 5개 후보를 저장한다.

USE_AS_IS 분기에서도 이제 None 대신 `PDReviewRequest`를 반환하여 PD가 후보 중 다른 클립을 선택할 수 있다.

Gradio 측에서는 `sg_clip_radio` (Radio 컴포넌트)와 `sg_clip_preview` (Video 컴포넌트)를 추가했다. PD가 라디오 버튼으로 후보 클립을 선택하면 `on_select_clip()` 핸들러가 미리보기 영상을 변경한다. `_out()` 튜플은 15개 → 18개로 확장되었다.

### 이슈 D: 합성 오류 시 에러 메시지 미노출

`on_assemble_sg()`에 try/except 래퍼를 추가하여, 오류 발생 시 `traceback.format_exc()`를 Markdown으로 UI에 반환한다:

```python
except Exception as e:
    err_detail = traceback.format_exc()
    err_msg = f"### ❌ 합성 오류\n\n**에러**: `{str(e)}`\n\n```\n{err_detail}\n```"
    return state, err_msg, None, None
```

---

## 커밋 10. `7d93de3` — 불필요 파일 삭제

**변경**: 1개 파일, -716 lines

커밋 7에서 디버깅 용도로 생성한 `notebooks/_cell3_current_dump.py` 삭제. 어디서도 import되지 않았고, 노트북에 코드가 이미 반영되어 있으므로 정리 차원에서 제거했다.

---

## 반복 패턴 분석

### 1. cv2.VideoCapture 반복 오픈

영상 처리 파이프라인에서 동일 프레임을 여러 함수에서 독립적으로 `cv2.VideoCapture` → `cap.read()` → `cap.release()` 하는 패턴이 반복 등장했다. 커밋 3에서 파라미터 전달 방식으로 해결했지만, 이후 추가된 코드(`prepare_scene` 등)에서도 동일 패턴이 재등장할 여지가 있다.

### 2. ColBERT 후보 수 반복 조정

커밋 3에서 100 → 30으로 축소했다가, 커밋 5에서 다시 100으로 복원했다. 정확도와 속도 사이의 트레이드오프를 실험적으로 조정하고 있으며, 최적값이 아직 확정되지 않았다.

### 3. Gradio 출력 튜플 지속 확장

`_out()` 함수의 반환 튜플이 13개 → 14개 → 15개 → 18개로 계속 증가했다. UI 컴포넌트를 하나 추가할 때마다 `_out()`, `tab1_outs`, 모든 이벤트 핸들러를 동시에 수정해야 하는 구조적 부담이 누적되고 있다.

### 4. Sora API 생명주기

커밋 1에서 도입 → 커밋 4에서 API key 설정 추가 → 커밋 9에서 전면 삭제. 약 2주 만에 전체 코드가 제거되었다. 빈 문자열이 truthy로 처리되어 조용히 폴백하는 문제 등, 실제로 정상 작동한 적이 없었을 가능성이 높다.

### 5. 언어 미스매치

MSR-VTT 데이터셋(영어 캡션)과 한국어 검색 쿼리 사이의 언어 불일치가 반복적으로 문제를 일으켰다. BM25 한국어 토크나이저 추가(커밋 1), Scene Graph description 영어 전환(커밋 8) 등으로 대응했지만, 다국어 검색 파이프라인의 근본적 설계 과제로 남아 있다.

### 6. 디버깅 아티팩트 관리

노트북 셀 코드를 `.py`로 덤프하여 디버깅하는 워크플로가 보인다(커밋 7에서 생성, 커밋 10에서 삭제). txt 파일 12개(ANALYSIS.txt, cell_0~9.txt)도 커밋 3에서 일괄 삭제했다. 디버깅 파일이 커밋에 포함되는 패턴이 반복된다.

---

## 2026-04-02 세션 — 구조 변경 6건

> 근거: Claude Code 대화 세션 (PDF 설계 대조, Runway 공식 문서 확인, Gradio UI 에러 재현)

### 이슈 11. 기본 단위 변경: shot → scene (scene = clip)

**근본 원인**: PD가 5개의 장면을 요청하면 ~3초짜리 shot 조각이 5개 나와서, 이어붙여도 자연스러운 영상이 되지 않는다. PDF 설계에서도 "검색된 영상 **클립**을 자동으로 합성"이라고 명시하고 있으며, 계층 구조는 Frame → Shot(~3초) → Scene(~30초) → Video 4단계로 정의되어 있다. 검색·합성의 기본 단위가 shot이 아닌 scene이어야 PD 관점에서 의미가 있다.

**수정 파일 및 내용**:

| 파일 | 변경 |
|------|------|
| `src/phase0_indexing/indexer.py` | `_build_clips_from_videos()`: `for shot in scene.shots` 루프 제거 → scene 단위로 clip 생성. clip_id 패턴 `_shot{id}` → `_scene{id}`, 시간 범위 `shot.start_ms/end_ms` → `scene.start_ms/end_ms` |
| `src/phase0_indexing/indexer.py` | `_build_clips_from_videos_grouped()`: 동일 변경 |
| `src/data_models.py` | ClipMeta 주석 및 예시 `video0001_shot003` → `video0001_scene003` |
| `src/phase0_indexing/vector_store.py` | 주석 예시 변경 |
| `notebooks/04_phase2_output.ipynb` | 예시 데이터 `_shot001` → `_scene001` |
| `notebooks/05_phase3_advanced.ipynb` | 동일 |

**영향**: 기존 shot 단위 인덱스와 호환되지 않으므로 재인덱싱 필요. Agglomerative Clustering 자체는 `shot_detector.py`에서 동일하게 수행되며, 결과를 소비하는 방식만 변경.

### 이슈 12. 데이터 범위 확장: 1000개 샘플 → 7010개 전체

**근본 원인**: 1000개 카테고리 균형 샘플링은 프로토타입 초기 단계에서 비용·시간 절약을 위한 것이었으나, 전체 pool로 확장하여 검색 정확도와 다양성을 높여야 한다.

**수정 파일 및 내용**:

| 파일 | 변경 |
|------|------|
| `notebooks/00_setup.ipynb` Cell 2 | `VIDEO_DIR = .../videos_1000` → `VIDEO_DIR = .../videos`, 주석 "샘플링된 1000개" → "전체 MSR-VTT" |
| `notebooks/01_indexing.ipynb` Cell 3 | **전면 재작성**: `sample_list.csv` 필터링 제거, 전체 7010개 symlink, `msrvtt_full_captions.json` 우선 로드 (fallback: 1000개 + 경고) |
| `notebooks/01_indexing.ipynb` Cell 4 | `videos_1000` → `videos` |
| `notebooks/01_indexing.ipynb` Cell 8 | Drive 백업 경로 동일 변경 |
| `notebooks/02_demo.ipynb` Cell 1 | `videos_1000` → `videos` |
| 프로젝트 전체 | `videos_1000` 문자열 참조 0건 (grep 확인 완료) |

### 이슈 13. 캡션 생성 노트북 신규: `01b_caption_remaining.ipynb`

**근본 원인**: 기존 1000개에만 GPT-4o-mini 영문 캡션이 있고, 나머지 6010개는 캡션이 없어서 BM25(텍스트) 검색에서 빠진다.

**수정 내용**: `notebooks/01b_caption_remaining.ipynb` 신규 생성 (7개 셀). 기존 1000개 캡션과 동일한 depth를 보장하기 위해 프롬프트, 프레임 추출 설정, 모델 파라미터를 완전히 동일하게 맞춤:

- 모델: `gpt-4o-mini`, `max_tokens=150`, `temperature=0.3`
- 프레임: 8장, 512×288, JPEG 85%, `detail: low`
- 프롬프트: "Write one detailed English sentence... Focus on: main subjects, actions, setting, notable details. Do not start with 'The video shows' or 'In the video'."
- 50개마다 Google Drive에 체크포인트 저장 (세션 끊김 대비)
- 최종 출력: `msrvtt_full_captions.json` (기존 1000 + 신규 병합)
- 예상: ~$7.2, ~2.5시간

### 이슈 14. Runway API 작동 불가 → SDK 방식 전면 전환

**근본 원인**: `inverse_prompt_engine.py`가 `requests.post("https://api.dev.runwayml.com/v1/image_to_video")` 직접 호출 + 수동 폴링 방식을 사용하고 있었으나, Runway API가 업데이트되면서 모델명(`gen3a_turbo`), 파라미터명(`promptImage`/`promptText`), 비율 형식(`16:9`) 등이 모두 변경되어 호출이 실패했다.

**수정 파일**: `src/phase4_assembly/inverse_prompt_engine.py`

| 항목 | 이전 | 변경 |
|------|------|------|
| 호출 방식 | `requests.post()` + 수동 폴링 | `runwayml` SDK `client.image_to_video.create().wait_for_task_output()` |
| 모델 | `gen3a_turbo` | `gen4.5` |
| 파라미터 | `promptImage`, `promptText`, `ratio: "16:9"` | `prompt_image`, `prompt_text`, `ratio: "1280:720"` |
| 환경변수 | `RUNWAY_API_KEY` | `RUNWAYML_API_SECRET` (하위 호환 유지) |
| text-to-video | 별도 `text_to_video` 엔드포인트 | `prompt_image` 생략하면 자동 text-to-video |
| 의존성 | `requests` | `runwayml` (`pip install runwayml`) |

`_get_runway_client()` 메서드를 추가하여 SDK 클라이언트를 지연 초기화하고, `_apply_runway()`와 `_generate_runway()` 두 메서드를 전면 재작성.

### 이슈 15. Gradio `ClipResult` NameError + UI 4건 수정

**이슈 15-A: `ClipResult` NameError**

`02_demo.ipynb` Cell 2에서 `from src.data_models import CurationState`만 import하고 `ClipResult`가 누락되어, Cell 3의 `_assemble_sg_inner()` (line 833)에서 `NameError: name 'ClipResult' is not defined` 발생. `CurationState, ClipResult`로 수정.

**이슈 15-B: Scene Graph 탭 — Before 영상 미반영**

후보 클립 선택 시 `on_select_clip()`이 `sg_clip_preview`(미리보기)만 업데이트하고 `sg_before_img`(Before)는 업데이트하지 않았다. `on_select_clip()` 반환값을 2개 → 3개로 확장하고, outputs에 `sg_before_img` 추가. `sg_before_img`를 `gr.Image` → `gr.Video`로 변경하여 영상 재생 가능하게 함.

**이슈 15-C: PD 큐레이션 탭 — 키프레임 정지 이미지 → 영상 미리보기**

`clip_keyframe_img = gr.Image()` → `clip_preview_video = gr.Video()`로 변경. `on_clip_detail()` 함수가 키프레임 경로 대신 `video_path`를 반환하도록 수정.

**이슈 15-D: PD 큐레이션 탭 — 불필요 UI 제거**

"클립 상세 보기" 드롭다운(`clip_detail_dd`): `visible=False`로 숨김 (내부 동기화용으로만 유지). "변환" 버튼 + "제거" 버튼: UI에서 제거. Section C 변환 패널과 관련 이벤트 연결 전체 비활성화 (주석: "2차년도 구현 예정"). 제거 기능은 체크박스 해제로 대체.

---

---

## 현존 미해결 이슈

### 이슈 16. 인덱스 호환성 관리 부재

scene=clip 변경(이슈 11), 1000→7010 확장(이슈 12) 모두 기존 인덱스와 호환되지 않는다. 인덱스 버전 관리나 마이그레이션 도구가 없어서, 변경 시 항상 재인덱싱이 필요하다. 인덱싱이 수십 분 소요되므로 개발 반복 비용이 높다.

**해결 방안 제안**: 인덱스 메타데이터에 버전 해시(clip 단위 + 영상 수 + 캡션 해시)를 기록하고, `load_index()` 시 불일치하면 경고 또는 자동 재빌드 트리거.

### 이슈 17. 노트북 ↔ src 간 하드코딩 경로

`01_indexing.ipynb` Cell 3의 ZIP 경로(`/content/drive/MyDrive/videorag_prototype/data/msrvtt/videos/data/MSR-VTT.ZIP`), Drive 캡션 경로(`DRIVE_CAPTIONS`) 등이 노트북에 직접 하드코딩되어 있다. 00_setup에서 경로 CONFIG를 중앙 관리하지만, 이 셀들은 CONFIG 변수를 참조하지 않고 별도 문자열을 사용한다. src 코드를 변경해도 노트북의 경로를 별도로 수정해야 하는 동기화 부담이 존재한다.

**해결 방안 제안**: 00_setup.ipynb에서 `DRIVE_ZIP_PATH`, `DRIVE_CAPTIONS_DIR` 등을 CONFIG 변수로 정의하고, 01_indexing/01b에서 이 변수를 참조하도록 통일.

### 이슈 18. Runway 모델명 하드코딩

커밋 1에서 Sora 도입 → 커밋 9에서 Sora 삭제 + Runway Gen-3 도입 → 이번 세션에서 Runway Gen-3 삭제 + gen4.5 SDK 도입. 외부 API 종속 코드가 API 버전 변경에 취약하며, 매번 호출 방식·모델명·파라미터가 전면 교체된다. SDK 래퍼를 도입한 것은 개선이지만, 모델명(`gen4.5`)이 `_apply_runway()`와 `_generate_runway()` 두 곳에 하드코딩되어 있어 다음 모델 변경 시 동일 이슈가 재발할 수 있다.

**해결 방안 제안**: `__init__`에서 `self.runway_model = runway_model or "gen4.5"` 파라미터로 추출하여 한 곳에서 관리.
