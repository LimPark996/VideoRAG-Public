# 02_demo.ipynb 이슈 보고서 (ver3)

**작성일**: 2026-04-02  
**최종 수정**: 2026-04-03  
**대상**: `notebooks/02_demo.ipynb` — Gradio 통합 데모 (Scene Graph 워크플로 + PD 큐레이션)  
**보고자**: Claude Code 자동 분석

---

## Issue 1: Scene Graph 워크플로 — 두 번째 신부터 검색 결과가 첫 신과 동일

### 증상
- 첫 번째 장면(Scene 1)에 대한 검색 결과는 정상적으로 나옴
- 두 번째 장면(Scene 2) 이후부터 검색 결과가 **첫 번째 장면의 결과와 완전히 동일**
- 각 장면의 `description`이 서로 다름에도 불구하고 동일한 클립이 반환됨

### 원인 분석

#### 코드 추적 결과

코드 흐름을 완전히 추적함:  
`on_parse` → `_process(idx=N)` → `_run_prepare_with_progress(req)` → `mapper.prepare_scene(req)` → `search_fn(req.description)` → `pipeline.search_only(query)`

**결론: 코드 경로상 캐싱이나 상태 공유 버그는 없음.**  
각 장면마다 다른 `description`으로 검색이 실행되며, `_run_search_pipeline`은 매번 BM25 + Dense + WRRF + ColBERT를 처음부터 수행함.

#### 근본 원인 (확정 필요 — 런타임 로그 확인 필요)

**가설 1: 소규모 인덱스 + 관련 쿼리 → 동일 top-k (가장 유력)**
- 인덱스에 7010개 클립이 등록되어 있으나, 같은 시나리오 내 장면들이 유사한 주제
- CLIP/InternVideo2 임베딩 공간에서 유사 쿼리 → 유사 Dense 랭킹
- WRRF 융합에서 Dense가 60% 가중치 → Dense 랭킹이 최종 순서를 지배

**가설 2: Gradio Radio 컴포넌트 시각적 미갱신**
- `gr.update(choices=new_choices)` 호출 시, 동일 clip_id가 순위만 바뀌어 재등장하면 사용자 입장에서 "똑같다"고 인식

**가설 3 (방어적): `description_ko`만 있는 JSON 사용 시**
- `parse_scene_graph`이 `description` 키만 읽어서 빈 문자열 → 모든 장면 동일 쿼리
- 영문 `description`이 있는 예시 시나리오에서는 해당 없음, 커스텀 JSON에서만 발생 가능

**→ 정확한 원인은 Gradio 로그 패널 + pipeline 진단 로그로 런타임 확인 필요**

#### 확정 원인: 장면 간 중복 제거 범위 한정

- 이전 장면에서 **선택된 클립 1개**만 제외
- 나머지 상위 후보(top_candidates)는 모두 재등장 가능
- 따라서 두 번째 장면에서도 첫 번째와 거의 동일한 후보 목록이 표시됨

### 수정 내역 ✅

| 수정 | 파일 | 상태 |
|------|------|------|
| `description_ko` 폴백 추가 | `storyboard_mapper.py` | ✅ 완료 |
| `used_clip_ids`에 top_candidates 전체 제외 | `02_demo.ipynb` | ✅ 완료 |
| `prepare_scene`에 검색 쿼리/결과 로깅 추가 | `storyboard_mapper.py` | ✅ 완료 |
| `pipeline._last_diag`에 BM25/Dense/Rerank top3 저장 | `pipeline.py` | ✅ 완료 |
| **Gradio 실시간 로그 패널** 추가 (모든 로그 UI 표시) | `02_demo.ipynb` | ✅ 완료 |
| `_build_live_progress`에 검색 쿼리 + 제외 클립 수 표시 | `02_demo.ipynb` | ✅ 완료 |

---

## Issue 2: Runway API 결과가 나오지 않음

### 증상
- Runway 백엔드를 선택하고 실행해도 결과가 나오지 않음
- 사용자에게 아무런 에러 메시지 없이 "알 수 없는 오류" 표시
- Runway 사이트 대시보드에서 실제로는 크레딧이 사용됨

### 원인 분석

#### 근본 원인 1: 무음 OpenCV 폴백
- `_apply_runway`의 `except Exception`이 모든 에러를 삼키고 `_apply_opencv_fallback`으로 넘김
- Gradio UI에는 에러 표시 없음

#### 근본 원인 2: error 필드 미전달
- `execute_scene()`이 API 결과의 `error` 필드를 `PDExecutionResult`에 전달하지 않음
- `execution.error = None` → "알 수 없는 오류" 표시

#### 근본 원인 3: gen4.5 모델 — 과도한 크레딧 소모 + 느린 생성
- `model="gen4.5"` 하드코딩 (25 크레딧/초, 5초 클립 = **125 크레딧**)
- 생성 시간 3~5분 (가장 느린 모델)
- `gen4_turbo`는 5 크레딧/초, ~30초 생성 (5배 저렴, 5배 빠름)

#### 근본 원인 4: 실행 중 UI 블로킹
- `execute_scene()` 동기 호출 → Runway 대기 중 Gradio UI 완전 정지
- 로그 패널도 갱신되지 않아 "아직 로그 없음" 표시

#### 근본 원인 5: 생성 영상 미보존
- Runway 결과 영상이 Colab 로컬에만 저장 → 세션 끊기면 공중분해
- 크레딧을 써서 만든 영상이 사라짐

### 수정 내역 ✅

| 수정 | 파일 | 상태 |
|------|------|------|
| 무음 OpenCV 폴백 제거 → `success=False` + `error` 반환 | `inverse_prompt_engine.py` | ✅ 완료 |
| `execute_scene`에서 `error=api_result.get("error")` 전달 | `storyboard_mapper.py` | ✅ 완료 (TRANSFORM + GENERATE 양쪽) |
| OpenCV 실패 시에도 `error` 필드 추가 | `inverse_prompt_engine.py` | ✅ 완료 |
| 실패 메시지에 백엔드/output_path 등 진단 정보 포함 | `02_demo.ipynb` | ✅ 완료 |
| Step 1에 Runway 진단 출력 (키 설정 여부, 패키지 설치 여부) | `02_demo.ipynb` | ✅ 완료 |
| Tab 2 `on_exec_transform`에 `success` 체크 추가 | `02_demo.ipynb` | ✅ 완료 |
| **기본 모델 `gen4.5` → `gen4_turbo`로 변경** (5배 저렴/빠름) | `inverse_prompt_engine.py` | ✅ 완료 |
| `runway_model` 파라미터 추가 (생성자에서 설정 가능) | `inverse_prompt_engine.py` | ✅ 완료 |
| **`on_execute` 스레드 분리** — 2초마다 UI 갱신 + 경과 시간 표시 | `02_demo.ipynb` | ✅ 완료 |
| **Runway 4단계 로깅** (프레임추출→API요청→대기→다운로드) | `inverse_prompt_engine.py` | ✅ 완료 |
| **Drive 자동 백업** (`_backup_to_drive`) | `inverse_prompt_engine.py` | ✅ 완료 |
| 백업 경로: `Drive/videorag_prototype/output/runway/{transformed,generated}/` | — | ✅ 구현 |

### Runway 모델별 비용/속도 비교

| 모델 | 크레딧/초 | 5초 클립 비용 | 생성 시간 | 비고 |
|------|----------|-------------|----------|------|
| gen4.5 (이전) | 25 | 125 크레딧 ($1.25) | 3~5분 | 최고 품질, 최비쌈 |
| **gen4_turbo (현재)** | **5** | **25 크레딧 ($0.25)** | **~30초** | **권장** |
| gen3_alpha_turbo | 5 | 25 크레딧 ($0.25) | ~20초 | 빠르지만 구세대 |

---

## Issue 3: 장면 순서 재배치가 반영되지 않음

### 증상
- Scene Graph 워크플로에서 모든 장면 처리 후 "장면 재배치" 텍스트 박스 순서가 표시됨
- 사용자가 줄 순서를 변경한 후 "영상 합성" 클릭
- 합성된 영상에서 재배치된 순서가 반영되지 않음

### 원인 분석

#### 근본 원인 1: `sg_scene_order_box`의 `interactive` 속성 미설정
- output 요소로 사용되면 Gradio가 `interactive=False`로 자동 설정
- 결과: 사용자가 텍스트 박스를 편집할 수 없음

#### 근본 원인 2: 순서 파싱 실패 시 무음 폴백
- `if not ordered: ordered = mapped` — 사용자 피드백 없이 원래 순서 사용

### 수정 내역 ✅

| 수정 | 파일 | 상태 |
|------|------|------|
| `interactive=True` 명시 추가 | `02_demo.ipynb` | ✅ 완료 |
| 파싱 실패 시 `logger.warning` 추가 | `02_demo.ipynb` | ✅ 완료 |

---

## Issue 4: 합성 버튼이 표시되지 않음

### 증상
- Scene Graph 워크플로에서 장면 처리를 진행해도 "🎬 영상 합성" 버튼이 나타나지 않음

### 원인 분석
- 합성 버튼이 **모든 장면 처리 완료** 후에만 `visible=True`로 설정됨
- 중간 에러, 워크플로 미완료 시 합성 불가능

### 수정 내역 ✅

| 수정 | 파일 | 상태 |
|------|------|------|
| 장면 1개 이상 완료 시 합성 버튼 표시 | `02_demo.ipynb` | ✅ 완료 |

---

## Issue 5: TC-Score 포맷팅 에러 (float → str)

### 증상
- 합성 완료 후 TC-Score 표시에서 `ValueError` 발생
- f-string에서 float 포맷(`.4f`)이 문자열 `'N/A'`에 적용되면서 터짐

### 원인 분석

**코드**:
```python
f"- **TC-Score**: {result.tc_score:.4f if result.tc_score else 'N/A'}"
```

**문제 2가지**:
1. `tc_score=0.0`일 때 `if result.tc_score` → `False` → 정상 점수도 `N/A` 표시
2. `:.4f`가 조건식 전체에 적용 → `'N/A'`(str)에 float 포맷 → `ValueError`

### 수정 내역 ✅

| 수정 | 파일 | 상태 |
|------|------|------|
| `{f'{...:.4f}' if ... is not None else 'N/A'}` 로 변경 (2곳) | `02_demo.ipynb` | ✅ 완료 |

---

## Issue 6: Before/After가 이미지로만 표시됨

### 증상
- TRANSFORM 실행 후 Before/After 영역에 사진 1장만 표시
- 영상을 눈으로 확인할 수 없음

### 원인 분석
- Tab 1: `sg_after_img = gr.Image(...)` — 이미지 컴포넌트
- Tab 2: `before_img = gr.Image(...)`, `after_img = gr.Image(...)` — 둘 다 이미지
- `execution.after_keyframe_path` (키프레임 이미지)만 반환, 영상 경로는 미전달

### 수정 내역 ✅

| 수정 | 파일 | 상태 |
|------|------|------|
| Tab 1 After: `gr.Image` → `gr.Video` | `02_demo.ipynb` | ✅ 완료 |
| Tab 2 Before/After: `gr.Image` → `gr.Video` | `02_demo.ipynb` | ✅ 완료 |
| After에 `execution.output_path` (영상) 반환 | `02_demo.ipynb` | ✅ 완료 |
| Tab 2 Before에 `clip["video_path"]` 직접 사용 | `02_demo.ipynb` | ✅ 완료 |

---

## Issue 7: 영상 파일 1841개 누락 (인덱스 불일치)

### 증상
- 인덱스에 7010개 클립 등록, 실제 영상 파일 5169개
- 검색에서 누락 영상이 걸리면 미리보기 불가, 합성 시 스킵

### 원인 분석
- 01_indexing에서 Colab 로컬에 7010개 전부 존재 → 인덱싱 + 캡션 생성 완료
- ZIP 압축에서 symlink로 로컬에 연결했으나, **Drive 영상 폴더에는 5169개만 복사됨**
- 세션 끊기면 symlink 사라짐 → 02_demo 복원 시 Drive 기준 5169개만 가져옴

### 수정 내역 ✅

| 수정 | 파일 | 상태 |
|------|------|------|
| Step 0 영상 복원: ZIP 기반으로 변경 | `02_demo.ipynb` | ✅ 완료 |
| 인덱스 클립 수와 비교 → 부족하면 ZIP 압축 해제 → symlink | `02_demo.ipynb` | ✅ 완료 |
| 출력: `✓ 영상 복원: 7010개 (ZIP에서 1841개 추가, 인덱스: 7010개)` | — | ✅ 구현 |

---

## Issue 8: Gradio에서 에러/로그가 보이지 않음

### 증상
- `logger.error()`, `logger.warning()`, `print()` 전부 Colab 콘솔에만 표시
- Gradio UI에서는 에러를 확인할 수 없음
- `except Exception`이 에러를 삼켜서 사용자에게 전달되지 않음

### 수정 내역 ✅

| 수정 | 파일 | 상태 |
|------|------|------|
| `_GradioLogCapture` 핸들러 — Python logging을 Gradio 패널로 전달 | `02_demo.ipynb` | ✅ 완료 |
| Tab 1 하단에 `📋 실시간 로그` 패널 (`gr.Code`) 추가 | `02_demo.ipynb` | ✅ 완료 |
| `_out()` 19번째 요소로 로그 텍스트 반환 (매 갱신마다 최신 로그 표시) | `02_demo.ipynb` | ✅ 완료 |
| root logger 하나에만 핸들러 등록 (propagate 중복 제거) | `02_demo.ipynb` | ✅ 완료 |
| `on_execute` 실행 중 2초마다 yield → 로그 실시간 갱신 | `02_demo.ipynb` | ✅ 완료 |

---

## Issue 9: Before 영상이 0초에서 끝남

### 증상
- Before 영역에 영상이 표시되지만 재생하면 0:00에서 바로 끝남
- 원본 클립의 해당 구간이 아닌 전체 영상 파일이 전달됨

### 원인 분석
- `clip.video_path`는 MSR-VTT 원본 영상 전체 (10~30초)
- 클립은 `start_ms`~`end_ms` 구간만 해당하는데, 전체 영상 경로를 그대로 Gradio에 전달
- Gradio `gr.Video`가 전체 영상을 로드하지만 사용자가 클립 구간을 식별할 수 없음

### 수정 내역 ✅

| 수정 | 파일 | 상태 |
|------|------|------|
| `_clip_preview(clip)` 함수 추가 — ffmpeg로 `start_ms`~`end_ms` 구간 추출 | `02_demo.ipynb` | ✅ 완료 |
| `output/preview/{clip_id}_preview.mp4`로 캐싱 (재생성 방지) | `02_demo.ipynb` | ✅ 완료 |
| Before에 `_clip_preview(review.selected_clip)` 사용 (2곳) | `02_demo.ipynb` | ✅ 완료 |
| `on_select_clip` 반환값도 `_clip_preview(c)` 적용 | `02_demo.ipynb` | ✅ 완료 |

---

## Issue 10: TRANSFORM 프롬프트 품질 — 속성 차이만 나열, 장면 의도 없음

### 증상
- TRANSFORM 프롬프트가 `"Transform from evening to night. Adjust lighting..."` 수준
- Runway가 장면의 맥락을 모르고 단순 밝기/색온도만 변경 → 검은색으로 어두워지기만 함
- 장면이 "네온사인이 빛나는 도시 밤"인데 그냥 어둡게만 처리

### 원인 분석

#### GPT 경로 (1순위)
- system prompt가 "속성 변환 전문가" 관점으로만 지시
- user prompt에 `scene_description`이 `"장면 설명: ..."` 한 줄로만 포함
- **장면이 뭘 표현하려는지** Runway에 전달되지 않음

#### 폴백 경로 (2순위, GPT 불가 시)
- `_generate_fallback_prompt(delta)` — `scene_description` 파라미터 자체가 없었음
- 순수 속성 변환 지시만: `"from evening to night"`, `"from neutral to cool"`

### 수정 내역 ✅

| 수정 | 파일 | 상태 |
|------|------|------|
| GPT system prompt 재설계 — "SCENE INTENT 기반 시네마틱 프롬프트 생성" | `inverse_prompt_engine.py` | ✅ 완료 |
| GPT user prompt 구조화: SCENE INTENT → ATTRIBUTE CHANGES → CURRENT STATE | `inverse_prompt_engine.py` | ✅ 완료 |
| 폴백 `_generate_fallback_prompt`에 `scene_description` 파라미터 추가 | `inverse_prompt_engine.py` | ✅ 완료 |
| 폴백 프롬프트 맨 앞에 장면 의도 배치 | `inverse_prompt_engine.py` | ✅ 완료 |
| `generate_transform_prompt` → 폴백 호출 시 `scene_description` 전달 | `inverse_prompt_engine.py` | ✅ 완료 |

### 변경 전후 비교

**이전** (속성 차이만):
```
Transform the scene from evening to night. Adjust lighting, sky color, 
and ambient atmosphere accordingly. Change the overall mood from neutral 
to cool. Maintain the original composition, subjects, and camera angle.
```

**변경 후 — GPT 경로**:
```
A sprawling cityscape at night, neon signs blazing in electric blue and 
magenta across building facades. Push the exposure down to deep twilight, 
replace the warm evening sky with a cold indigo canopy. Add volumetric 
haze catching the neon glow. Cool-wash the entire palette toward cyan 
and steel blue. Preserve all building geometry, signage, and camera angle.
```

**변경 후 — 폴백 경로**:
```
This scene depicts: neon signs glowing on city buildings at night. 
Transform the visual style while preserving the scene content.
Transform the scene from evening to night. Adjust lighting...
```

---

## Issue 11: 실내/실외 분류기 오판 → TRANSFORM 프롬프트 왜곡

### 증상
- "cars driving on a road at night" 장면이 **실내(indoor)**로 판정됨
- Scene Graph에 `location: "outdoor"` 명시되어 있음에도 무시
- 실내 판정 → `target_time=night` 무시 → 실내 조명 변환 프롬프트 생성
- 결과: "chiaroscuro lighting", "room layout, furniture" 등 부적절한 프롬프트

### 원인 분석
- 야간 영상 → 하늘 어둡고(sky_ratio < 0.03 → +1), 에지 많고(+1), 색상 단조로움(+1)
- `indoor_score >= 2` → `SceneLocation.INDOOR`로 오판
- `compute_delta`가 `current.location`만 참조 → Scene Graph의 `target_location` 무시
- 실내 판정 시 `target_time`, `target_season` 변환 요청이 자동으로 무시됨

### 수정 내역 ✅

| 수정 | 파일 | 상태 |
|------|------|------|
| `compute_delta`에 `target_location` 파라미터 추가 | `inverse_prompt_engine.py` | ✅ 완료 |
| Scene Graph `location`이 명시되면 분류기 결과 오버라이드 | `inverse_prompt_engine.py` | ✅ 완료 |
| `storyboard_mapper`에서 `req.target_location` 전달 | `storyboard_mapper.py` | ✅ 완료 |

---

## Issue 12: TRANSFORM 시 "original" 선택지 없음

### 증상
- TRANSFORM 분기에서 백엔드 선택이 runway/opencv만 가능
- 검색 결과가 이미 충분히 적합한데 강제로 변환해야 하는 상황

### 수정 내역 ✅

| 수정 | 파일 | 상태 |
|------|------|------|
| `available_backends`에 `"original"` 추가 | `storyboard_mapper.py` | ✅ 완료 |
| `on_execute`에서 `backend=="original"` 처리 — 변환 없이 USE_AS_IS로 확정 | `02_demo.ipynb` | ✅ 완료 |

---

## Issue 13: 클립 후보 클릭 시 Gradio 오류 (로그 없음)

### 증상
- 후보 클립 5개 중 일부를 클릭하면 Gradio 에러 발생
- 로그 패널에 아무것도 표시되지 않음

### 원인 분석
- `on_select_clip`이 `c.video_path`를 `gr.Video`에 전달
- 해당 영상 파일이 존재하지 않으면(누락 1841개 중 하나) Gradio 프론트엔드에서 에러
- Python 예외가 아니라 Gradio JS 내부 에러 → 로그 패널에 안 잡힘

### 수정 내역 ✅

| 수정 | 파일 | 상태 |
|------|------|------|
| `os.path.exists()` 체크 추가 — 파일 없으면 `None` 반환 | `02_demo.ipynb` | ✅ 완료 |
| 파일 없는 경우 로그 패널에 경고 표시 | `02_demo.ipynb` | ✅ 완료 |

---

## Issue 14: TRANSFORM 클립 길이 vs Runway 출력 길이 불일치

### 증상
- 검색 결과 클립이 15~20초인데, Runway TRANSFORM 후 5초짜리 영상만 생성됨
- 원본 클립의 **첫 프레임만 사용**, 나머지 15초 분량은 버려짐
- 합성 시 장면 길이가 Scene Graph의 `duration_sec`과 맞지 않음

### 원인 분석
- Runway `image_to_video.create(duration=5)` — 첫 프레임 1장 → 5초 영상 생성
- 원본 클립의 전체 타임라인을 활용하지 않고 첫 프레임 하나만 추출
- Scene Graph에서 `duration_sec: 8`을 요청해도 Runway는 5초 또는 10초만 가능

### 해결 방안: PD 구간 크롭 워크플로 (채택 예정)

**현재 워크플로 (문제)**:
```
검색 → 20초 클립 통째로 선택 → Runway가 첫 프레임만 보고 5초 생성 → 15초 버림
```

**개선 워크플로 (설계)**:
```
검색 → 클립 선택 → 타임라인 표시 → PD가 원하는 구간 크롭 (예: 3초~8초)
→ 크롭된 구간으로 → runway / opencv / original 선택
```

#### UI 설계

```
┌─────────────────────────────────────────┐
│  클립 미리보기 (영상 재생)                  │
│  ▶ ━━━━━━━━●━━━━━━━━━━━━━ 20.0s        │
│                                         │
│  구간 선택:                               │
│  시작 [====3.0s====] 끝 [====8.0s====]   │
│  선택 길이: 5.0s                          │
│                                         │
│  [구간 크롭] [전체 사용]                    │
└─────────────────────────────────────────┘
          ↓ 크롭 후
┌─────────────────────────────────────────┐
│  Before (크롭 구간)  │  After (변환 결과)  │
│  ▶ 5.0s             │  ▶ 5.0s           │
│                                         │
│  백엔드: ○ runway ○ opencv ○ original    │
│  [▶ 실행]                                │
└─────────────────────────────────────────┘
```

#### 구현 포인트

1. **구간 선택 UI**: `gr.Slider` 2개 (start, end) + 크롭 버튼
2. **크롭 실행**: ffmpeg `-ss {start} -t {duration}` → `output/preview/{clip_id}_crop.mp4`
3. **Runway 입력**: 크롭된 영상의 첫 프레임 → `duration = min(crop_length, 10)`
4. **여러 구간**: 한 클립에서 여러 구간을 크롭 → 각각 변환/원본 → 순서대로 이어붙이기
5. **전체 사용**: 크롭 안 하고 원본 클립 전체를 사용 (기존 동작과 동일)

#### 장점
- PD가 **최적의 장면**을 직접 골라서 Runway에 넘김
- 크롭 길이 ≈ Runway duration → 길이 불일치 해소
- 여러 구간 크롭+이어붙이기로 **긴 장면도 처리 가능**
- `original`로 크롭만 하고 변환 안 하는 것도 가능

### 수정 내역 ✅

| 수정 | 파일 | 상태 |
|------|------|------|
| `original` 백엔드 추가 (Issue 12) | `storyboard_mapper.py` + `02_demo.ipynb` | ✅ 완료 |
| **구간 크롭 UI** — 시작/끝 슬라이더 + 크롭/전체사용 버튼 | `02_demo.ipynb` | ✅ 완료 |
| `on_select_clip`에서 슬라이더 범위를 클립 길이로 자동 설정 | `02_demo.ipynb` | ✅ 완료 |
| `on_crop` — ffmpeg로 선택 구간 추출 → Before에 표시 | `02_demo.ipynb` | ✅ 완료 |
| `on_execute`에서 `crop_video_path` 있으면 크롭 영상으로 변환 실행 | `02_demo.ipynb` | ✅ 완료 |

### 워크플로

```
검색 → 클립 선택 → 타임라인 슬라이더로 구간 선택
→ [구간 크롭] 또는 [전체 사용]
→ Before에 크롭 결과 표시
→ runway / opencv / original 선택 → 실행
```

---

## 영향도 요약

| Issue | 심각도 | 상태 |
|-------|--------|------|
| 1. 검색 결과 동일 | **Critical** | ✅ 수정 + 런타임 로그 확인 필요 |
| 2. Runway 미작동 + 에러 무음 | **Critical** | ✅ 완료 |
| 3. 순서 재배치 미반영 | **Major** | ✅ 완료 |
| 4. 합성 버튼 미표시 | **Major** | ✅ 완료 |
| 5. TC-Score 포맷 에러 | **Minor** | ✅ 완료 |
| 6. Before/After 이미지만 표시 | **Major** | ✅ 완료 |
| 7. 영상 1841개 누락 | **Critical** | ✅ 완료 |
| 8. 로그 Gradio 미표시 | **Critical** | ✅ 완료 |
| 9. Before 영상 0초 종료 | **Major** | ✅ 완료 |
| 10. TRANSFORM 프롬프트 품질 | **Major** | ✅ 완료 |
| 11. 실내/실외 분류기 오판 | **Major** | ✅ 완료 |
| 12. "original" 선택지 없음 | **Minor** | ✅ 완료 |
| 13. 클립 클릭 Gradio 오류 | **Major** | ✅ 완료 |
| 14. 클립 길이 vs Runway 출력 불일치 | **Major** | ✅ 완료 (구간 크롭 UI 구현) |

---

## 수정 파일 목록

| 파일 | 수정 내용 |
|------|----------|
| `src/pipeline.py` | `_last_en_query`, `_last_diag` 진단 저장 |
| `src/phase4_assembly/storyboard_mapper.py` | `description_ko` 폴백, 검색 로깅, error 필드 전달, `target_location` 전달, `original` 백엔드 |
| `src/phase4_assembly/inverse_prompt_engine.py` | 무음 폴백 제거, gen4_turbo 기본값, 4단계 로깅, Drive 백업, TRANSFORM 프롬프트 개선, `target_location` 오버라이드 |
| `notebooks/02_demo.ipynb` Cell 1 (Step 0) | ZIP 기반 영상 복원 (7010개) |
| `notebooks/02_demo.ipynb` Cell 2 (Step 1) | Runway 진단 출력 |
| `notebooks/02_demo.ipynb` Cell 3 (Step 2) | 로그 패널, Before/After 영상화, 클립 미리보기 구간 추출, 스레드 분리, TC-Score 수정, 중복 제거 확대, interactive, 합성 버튼 조기 표시, `original` 백엔드, 클립 클릭 파일 체크, **구간 크롭 UI** |
