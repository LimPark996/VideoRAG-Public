# VideoRAG 프로토타입 — 이슈 보고서 (4차)

작성일: 2026-03-20

이 문서는 03_evaluation.ipynb에서 5개 쿼리 통합 테스트를 실행하면서 발견된 이슈 2건을 정리한 것입니다. 두 이슈 모두 Phase 4 영상 조립 단계에서 발생했으며, 하나는 Morph 전환이 100% 실패하는 문제, 다른 하나는 FFmpeg concat이 불필요하게 느린 문제입니다.

---

## 목차

1. [Morph 전환 100% 실패 — `_get_last_frame_bgr` seek 범위 초과](#1-morph-전환-100-실패--_get_last_frame_bgr-seek-범위-초과)
2. [FFmpeg concat 재인코딩 병목 — `-c copy` 미사용](#2-ffmpeg-concat-재인코딩-병목---c-copy-미사용)

---

## 1. Morph 전환 100% 실패 — `_get_last_frame_bgr` seek 범위 초과

### 상태: 해결 완료

### 이게 뭔 문제야?

Phase 4에서 영상 클립들을 이어붙일 때, 클립 간 유사도가 낮으면(DINOv2 cosine similarity < 0.3) "Morph 전환"을 사용합니다. Morph 전환은 Farneback Dense Optical Flow로 두 프레임 사이의 움직임 벡터를 추정해서, 한 장면이 다른 장면으로 자연스럽게 변형되는 중간 프레임들을 생성하는 기법입니다. 단순히 투명도를 섞는 Crossfade보다 물체의 움직임 방향을 고려하기 때문에 시각적으로 더 자연스럽습니다.

문제는 5개 쿼리 테스트에서 Morph 전환이 **한 건도 성공하지 못했다**는 것입니다. 로그에서 `[Render] ⚠️ Morph 전환 X 실패 (Yms)`라는 메시지가 모든 Morph 시도에서 출력되었습니다. 실패 시간이 18~175ms로 매우 짧았는데, 이는 optical flow 계산(수백ms 이상 소요)까지 도달하기도 전에 초기 단계에서 실패했다는 의미입니다.

### 원인을 어떻게 찾았는지

처음에는 `_render_morph_segment` 함수가 `None`을 반환하는 것까지는 알 수 있었지만, 이 함수 안에서 `None`을 반환하는 지점이 3곳이나 있어서 정확히 어디서 실패하는지 알 수 없었습니다:

```python
# 지점 1: 다음 클립이 없는 경우
if clip_to is None:
    return None

# 지점 2: 프레임 추출 실패
if frame_a is None or frame_b is None:
    return None

# 지점 3: morph 프레임 생성 실패
if not morph_frames:
    return None
```

그래서 각 `return None` 지점에 디버깅 로그를 추가했습니다. `_get_last_frame_bgr()`과 `_get_first_frame_bgr()` 함수에도 상세한 로그를 넣었습니다. 로그를 추가한 뒤 다시 실행하니 원인이 즉시 드러났습니다:

```
[FrameExtract] _get_last_frame: read 실패 clip=video253 seek_ms=29900.0, total_frames=480, fps=29.97
[Morph 0] ❌ 프레임 추출 실패: frame_a=None (...), frame_b=OK (...)
```

**100% `_get_last_frame_bgr`의 `frame_a`가 None**이었고, `_get_first_frame_bgr`의 `frame_b`는 항상 정상이었습니다.

### 근본 원인

MSR-VTT 데이터셋의 메타데이터에서 대부분의 클립이 `start_ms=0, end_ms=30000`으로 기록되어 있습니다. 이건 "최대 30초"라는 데이터셋 규격의 기본값이지, 실제 영상 길이가 정확히 30초라는 뜻이 아닙니다.

실제 영상 길이를 확인하면:

| 클립 | total_frames | fps | 실제 길이 | 메타데이터 end_ms |
|------|-------------|-----|----------|-----------------|
| video5698 | 275 | 25.0 | **11.0초** | 30,000 |
| video253 | 480 | 29.97 | **16.0초** | 30,000 |
| video3089 | 302 | 27.42 | **11.0초** | 30,000 |
| video4663 | 264 | 23.98 | **11.0초** | 30,000 |
| video6674 | 325 | 25.0 | **13.0초** | 30,000 |

기존 코드는 이렇게 동작했습니다:

```python
end_ms = clip.end_ms - 100  # 30000 - 100 = 29900
seek_ms = max(end_ms, clip.start_ms)  # 29900
cap.set(cv2.CAP_PROP_POS_MSEC, seek_ms)  # 29.9초 지점으로 이동 시도
ret, frame = cap.read()  # 영상이 11초인데 29.9초? → 실패!
```

11초짜리 영상에서 29.9초 지점을 찾으려고 하니, OpenCV의 `VideoCapture`가 유효한 프레임을 찾지 못하고 `ret=False`를 반환한 것입니다. `CAP_PROP_POS_MSEC` seek은 영상 끝을 넘어서는 위치에 대해 에러를 던지지 않고, 그냥 조용히 실패합니다.

한편 `_get_first_frame_bgr`은 `start_ms=0`으로 seek하기 때문에 항상 성공했습니다.

### 어떻게 해결했는지

`_get_last_frame_bgr`에 두 가지 보호 장치를 추가했습니다.

**보호 1: 실제 영상 길이 기반 seek 위치 계산**

메타데이터의 `end_ms`를 그대로 믿지 않고, OpenCV에서 읽은 `total_frames`와 `fps`로 실제 영상 길이를 계산합니다:

```python
actual_duration_ms = (total_frames / video_fps * 1000) if video_fps > 0 else clip.end_ms
effective_end_ms = min(clip.end_ms, actual_duration_ms)
seek_ms = max(effective_end_ms - 200, clip.start_ms)
```

예를 들어 `video5698`의 경우:
- `actual_duration_ms = 275 / 25.0 * 1000 = 11000ms`
- `effective_end_ms = min(30000, 11000) = 11000ms`
- `seek_ms = max(11000 - 200, 0) = 10800ms` ← 실제 영상 범위 안의 안전한 위치

**보호 2: 프레임 번호 기반 폴백**

시간 기반 seek이 실패할 경우(컨테이너에 따라 타임스탬프 seek이 정확하지 않을 수 있음), 프레임 번호로 직접 이동하는 폴백을 추가했습니다:

```python
if not ret and total_frames > 0:
    fallback_frame = max(total_frames - 2, 0)
    cap.set(cv2.CAP_PROP_POS_FRAMES, fallback_frame)
    ret, frame = cap.read()
```

`total_frames - 2`를 사용하는 이유는, 일부 코덱에서 마지막 프레임(`total_frames - 1`)이 불완전할 수 있기 때문입니다. 끝에서 두 번째 프레임을 읽으면 안전합니다.

### 관련 코드 위치

- `src/phase4_assembly/assembler.py` — `_get_last_frame_bgr()` 메서드
- `src/phase4_assembly/assembler.py` — `_render_morph_segment()` 메서드 (디버깅 로그 추가 부분)

---

## 2. FFmpeg concat 재인코딩 병목 — `-c copy` 미사용

### 상태: 해결 완료

### 이게 뭔 문제야?

Phase 4 영상 렌더링 시간의 **90% 이상**이 FFmpeg concat 단계에서 소비되고 있었습니다. 5개 쿼리 테스트의 수정 전 로그를 보면:

```
[Assembly]   4f_ffmpeg_render: 46517.2ms (93.9%)
```

전체 렌더링에서 FFmpeg 단계가 93.9%를 차지합니다. 그중에서도 개별 세그먼트 추출(3~16초)이 아닌 **최종 concat**(16~30초)이 대부분이었습니다.

수정 전 쿼리별 concat 소요 시간:

| 쿼리 | 세그먼트 추출 | concat | 비율 |
|------|------------|--------|------|
| guitar | 6,516ms | **21,287ms** | 76.6% |
| cooking | 11,941ms | **29,354ms** | 71.1% |
| dog | 7,266ms | **30,515ms** | 80.8% |
| dancing | 3,051ms | **16,680ms** | 84.6% |
| cars | 15,844ms | **30,670ms** | 65.9% |

120초 영상을 합치는 데 16~30초가 걸리는 건 합리적이지 않습니다.

### 근본 원인

기존 코드에서 FFmpeg concat 명령어를 보면:

```python
cmd_concat = [
    'ffmpeg', '-y',
    '-f', 'concat', '-safe', '0',
    '-i', concat_list,
    '-c:v', 'libx264', '-preset', 'fast',
    '-pix_fmt', 'yuv420p',
    '-movflags', '+faststart',
    output_path
]
```

핵심은 `-c:v libx264 -preset fast`입니다. 이 옵션은 입력 파일의 모든 프레임을 디코딩한 뒤, H.264로 다시 처음부터 인코딩하라는 의미입니다. 120초 × 25fps = 3,000프레임을 전부 디코딩하고 다시 인코딩하는 것입니다.

그런데 concat에 사용되는 세그먼트들은 바로 직전 단계에서 이미 동일한 파라미터로 인코딩된 파일들입니다:

```python
# 세그먼트 추출 단계
cmd = [
    'ffmpeg', '-y',
    '-ss', ..., '-i', ..., '-t', ...,
    '-vf', 'scale=640:480:...',
    '-c:v', 'libx264', '-preset', 'ultrafast',
    '-pix_fmt', 'yuv420p',
    '-an', '-r', '25',
    seg_path
]
```

모든 세그먼트가 H.264 / yuv420p / 640×480 / 25fps로 동일하게 인코딩되어 있으므로, concat 시 재인코딩은 완전히 불필요합니다. FFmpeg의 `-c copy` 옵션을 사용하면 비트스트림을 디코딩/재인코딩 없이 그대로 복사해서 합칩니다.

### 어떻게 해결했는지

concat 명령어를 `-c copy`로 변경했습니다:

```python
cmd_concat = [
    'ffmpeg', '-y',
    '-f', 'concat', '-safe', '0',
    '-i', concat_list,
    '-c', 'copy',
    '-movflags', '+faststart',
    output_path
]
```

`-c copy`가 실패할 수 있는 경우도 있습니다. Morph 전환 세그먼트는 OpenCV의 `cv2.VideoWriter`로 mp4v 코덱으로 먼저 생성한 뒤 H.264로 재인코딩하는데, 이 과정에서 코덱 파라미터가 미묘하게 다를 수 있습니다. 예를 들어 profile이나 level이 다르면 `-c copy` concat이 실패합니다.

이를 대비해서 `-c copy` 실패 시 재인코딩 폴백을 추가했습니다:

```python
except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
    # -c copy 실패 → 재인코딩 폴백 (ultrafast로 빠르게)
    cmd_concat_reencode = [
        'ffmpeg', '-y',
        '-f', 'concat', '-safe', '0',
        '-i', concat_list,
        '-c:v', 'libx264', '-preset', 'ultrafast',
        '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart',
        output_path
    ]
```

폴백에서도 기존의 `-preset fast` 대신 `-preset ultrafast`를 사용합니다. ultrafast는 fast보다 압축률은 약간 떨어지지만(파일 크기가 약 20% 더 클 수 있음) 인코딩 속도가 약 2~3배 빠릅니다. 프로토타입에서는 파일 크기보다 속도가 중요합니다.

### 개선 결과

수정 후 concat 소요 시간:

| 쿼리 | 수정 전 | 수정 후 | 개선 배율 |
|------|---------|---------|----------|
| guitar | 21,287ms | **126ms** | **169×** |
| cooking | 29,354ms | **131ms** | **224×** |
| dog | 30,515ms | **162ms** | **188×** |
| dancing | 16,680ms | **120ms** | **139×** |
| cars | 30,670ms | **139ms** | **221×** |

평균 **188배** 빨라졌습니다. 전체 Phase 4 assembly 시간도 크게 줄었습니다:

| 쿼리 | 수정 전 assembly | 수정 후 assembly | 개선 |
|------|-----------------|-----------------|------|
| guitar | 39,197ms | **12,726ms** | 3.1× |
| cooking | 46,408ms | **15,539ms** | 3.0× |
| dog | 40,356ms | **8,536ms** | 4.7× |
| dancing | 20,595ms | **3,845ms** | 5.4× |
| cars | 49,520ms | **17,895ms** | 2.8× |

### 관련 코드 위치

- `src/phase4_assembly/assembler.py` — `_render_video()` 메서드의 FFmpeg concat 부분

---

## 전체 파이프라인 구조 요약 (이슈의 맥락)

두 이슈 모두 Phase 4 영상 조립 단계의 렌더링 파이프라인에서 발생했습니다.

```
Phase 4: 영상 조립 (온라인)
  ├── Step 1: Perception Sort (DINOv2)
  ├── Step 2: Keyframe Extract
  ├── Step 2.5: Visual Similarity (DINOv2)
  ├── Step 3: Transition Select
  ├── Step 3.5: DreamColour LUT
  └── Step 4: FFmpeg Render              ← 두 이슈 모두 여기
       ├── 세그먼트 추출 (FFmpeg)
       ├── Morph/Crossfade 전환 생성      ← 이슈 #1
       └── FFmpeg concat 최종 합성        ← 이슈 #2
```

---

## 이슈 현황 요약표

| # | 이슈 | 위치 | 상태 | 심각도 |
|---|------|------|------|--------|
| 1 | Morph 전환 100% 실패 (seek 범위 초과) | `assembler.py` `_get_last_frame_bgr()` | 해결 완료 | 높음 (Morph 전환 전면 불능) |
| 2 | FFmpeg concat 재인코딩 병목 | `assembler.py` `_render_video()` | 해결 완료 | 중간 (성능 저하, concat 평균 188× 개선) |

---

## 아직 검증이 필요한 것

이슈 #1의 수정사항(실제 영상 길이 기반 seek + 프레임 번호 폴백)이 코드에 반영되었지만, Morph 전환이 실제로 성공하여 optical flow 기반 중간 프레임이 정상 생성되는지는 아직 확인되지 않았습니다. 다음 Colab 실행 시 로그에서 `[Morph X] 프레임 추출 성공: frame_a=..., frame_b=...` 메시지가 출력되고, `[Render] Morph 전환 X: Yms` (성공 로그)가 나오는지 검증이 필요합니다.
