# TokenFlow 디버깅 전체 이슈 보고서

**날짜**: 2026-04-20  
**파일**: `src/phase4_assembly/tokenflow_wrapper.py`  
**상태**: 수정 완료 (일부 항목 검증 진행 중)

---

## 전체 수정 목록 요약

| # | 증상 | 원인 | 수정 방법 | 상태 |
|---|---|---|---|---|
| 1 | `preprocess.py` 항상 실패 (returncode=2) | `--config yaml` 방식으로 호출했는데 preprocess는 argparse 개별 인자 방식 | argparse 개별 인자 방식으로 변경 | ✅ 완료 |
| 2 | SD 모델 로드 시 404 에러 | `revision="fp16"` 브랜치가 HuggingFace에서 삭제됨 | `_patch_tokenflow()` 로 정규식 제거 | ✅ 완료 |
| 3 | `preprocess.py` basename 경로 오류 | 절대경로를 넘기면 basename이 꼬여서 프레임 못 찾음 | `data/tf_{uid}` 상대경로로 통일 | ✅ 완료 |
| 4 | `preprocess.py` 파라미터 누락 | 필수 config 키값 누락 | 필수 인자 추가 | ✅ 완료 |
| 5 | latent 경로 불일치 | `save_dir` 와 `latents_path` 경로가 달랐음 | `rel_data` 로 통일 | ✅ 완료 |
| 6 | 출력 프레임 중복 수집으로 영상 반복 | `**` 재귀 glob이 `fps_10/`, `fps_20/`, `fps_30/` 까지 수집 | `img_ode/` 폴더만 탐색하도록 변경 | ✅ 완료 |
| 7 | `Missing latents at t 925` | `--save_steps` 미전달로 preprocess(50)와 run_tokenflow_pnp(30) timestep 목록 불일치 | `--save_steps` 를 `N_TIMESTEPS` 와 동일하게 전달 | ✅ 완료 |
| 8 | finally 블록이 출력 폴더를 프레임 수집 전에 삭제 | `_cleanup_tokenflow_dirs()` 가 finally에서 출력 폴더까지 삭제 | 출력 폴더는 프레임 수집 완료 직후 별도 삭제 | ✅ 완료 |
| 9 | 7초 영상 → 4초로 짧아짐 + 앞 구간 반복 | `EXTRACT_FPS=8`, `KEYFRAME_FREQ=10` 이 짧은 클립에서 keyframe 부족 | `EXTRACT_FPS=4`, `KEYFRAME_FREQ=4`, `N_TIMESTEPS=30` 으로 변경 | ✅ 완료 |
| 10 | 3fps 영상에서 프레임 수 부족 | 원본 fps가 EXTRACT_FPS보다 낮아 추출 프레임이 너무 적음 | RIFE 광학 흐름 보간 추가 (3fps → 24fps) | ✅ 완료 (검증 진행 중) |
| 11 | n_frames가 batch_size 배수 아닐 때 프레임 손실 | TokenFlow 내부에서 `n_frames % batch_size != 0` 이면 자동 잘림 | batch_size를 n_frames 약수 중 최댓값으로 동적 조정 | ✅ 완료 |
| 12 | `img_ode` 폴더 탐색 깊이 부족 | run_tokenflow_pnp가 output 경로에 여러 단계 하위 폴더를 자동 생성 | `**` 재귀 탐색으로 `img_ode` 폴더 위치 동적 탐색 | ✅ 완료 |
| 13 | RIFE 가중치 다운로드 실패 | 존재하지 않는 GitHub Releases URL 사용 | HuggingFace `jbilcke-hf/varnish` 미러 URL로 변경 | ✅ 완료 |

---

## 이슈 1: `preprocess.py` 항상 실패 (returncode=2)

### 증상
```
[TokenFlow] preprocess.py 실패 (returncode=2)
```

### 원인
`preprocess.py` 는 argparse 개별 인자 방식으로 실행되는 스크립트인데, `--config yaml파일` 형식으로 호출했다. argparse가 `--config` 인자를 모르니까 즉시 returncode=2 로 종료.

### 수정
```python
# 수정 전 (잘못됨)
subprocess.run([sys.executable, preprocess_script, "--config", cfg_path])

# 수정 후
subprocess.run([
    sys.executable, preprocess_script,
    "--data_path", rel_data,
    "--save_dir",  rel_data,
    "--steps",     str(N_TIMESTEPS),
    ...
])
```

---

## 이슈 2: SD 모델 로드 시 HuggingFace 404 에러

### 증상
```
OSError: runwayml/stable-diffusion-v1-5 does not appear to have a file named revision=fp16
```

### 원인
TokenFlow 코드가 SD 모델 로드 시 `revision="fp16"` 을 하드코딩하는데, HuggingFace에서 해당 브랜치가 삭제됨.

### 수정
`_patch_tokenflow()` 함수로 TokenFlow 스크립트 3개에서 `revision="fp16"` 패턴을 정규식으로 제거. `_ensure_ready()` 에서 매번 호출해서 TokenFlow가 업데이트돼도 자동 적용.

---

## 이슈 3~5: 경로 관련 버그 (basename, save_dir, latents_path)

### 증상
```
FileNotFoundError: data/frames/00000.jpg not found
```

### 원인
- `preprocess.py` 내부에서 `os.path.basename(data_path)` 를 사용해 경로를 재조합
- 절대경로를 넘기면 `basename("/content/TokenFlow/data/tf_abc")` = `"tf_abc"` 가 되어 경로가 꼬임
- `save_dir` 와 `latents_path` 가 서로 다른 경로를 가리키던 문제

### 수정
모든 경로를 `cwd=TOKENFLOW_DIR` 기준 상대경로로 통일.
```python
rel_data = f"data/tf_{uid}"
# preprocess: --data_path data/tf_{uid} --save_dir data/tf_{uid}
# yaml: latents_path: data/tf_{uid}
```

---

## 이슈 6: 출력 프레임 중복 수집으로 영상 반복

### 증상
- 7초 영상을 넣었는데 영상이 두 번 반복되어 나옴
- 로그: `편집 완료: 48프레임` (실제 편집 프레임은 24개인데 2배)

### 원인
`run_tokenflow_pnp.py` 가 편집 결과를 `img_ode/`, `fps_10/`, `fps_20/`, `fps_30/` 여러 폴더에 저장하는데, `**` 재귀 glob으로 전체 탐색하면 모든 폴더의 이미지가 중복 수집됨.

### 수정
프레임이 저장되는 폴더는 `img_ode/` 뿐이므로 해당 폴더만 탐색하도록 변경.
```python
# 수정 전: 전체 재귀 탐색 → 중복 수집
_glob.glob(os.path.join(TOKENFLOW_DIR, f"{rel_output}*", "**", "*"), recursive=True)

# 수정 후: img_ode/ 폴더만 탐색
img_ode_dirs = [d for d in _glob.glob(
    os.path.join(TOKENFLOW_DIR, f"{rel_output}*", "**", "img_ode"),
    recursive=True,
) if os.path.isdir(d)]
```

---

## 이슈 7: `Missing latents at t 925`

### 증상
```
AssertionError: Missing latents at t 925 path
data/tf_7f01cf01/sd_1.5/tf_7f01cf01/steps_30/nframes_27/latents/noisy_latents_925.pt
```

### 원인
`preprocess.py` 의 `--steps` 와 `--save_steps` 는 서로 다른 인자다.

| 인자 | 역할 |
|---|---|
| `--steps` | DDIM inversion 단계 수. latent 저장 경로의 `steps_{N}` 에 반영 |
| `--save_steps` | 실제로 latent 파일을 저장할 timestep 목록 결정. `DDIMScheduler.set_timesteps(save_steps)` 로 생성 |

`--save_steps` 를 넘기지 않으면 기본값 50으로 실행된다. `run_tokenflow_pnp.py` 는 `n_timesteps=30` 기준으로 `set_timesteps(30)` → `[958, 925, 892, ...]` 목록을 생성하고 `t=925` 를 요청하는데, preprocess는 `save_steps=50` 기준 목록으로 저장했기 때문에 `t=925` 파일이 없어서 에러 발생.

실제로 `set_timesteps(30)` 결과를 확인:
```python
scheduler.set_timesteps(30)
# → [958, 925, 892, 859, 826, 793, 760, 727, ...]
```
`t=925` 는 `set_timesteps(30)` 목록에 포함되어 있다.

### 수정
```python
preprocess_cmd = [
    ...
    "--steps",      str(N_TIMESTEPS),  # 30
    "--save_steps", str(N_TIMESTEPS),  # 30 ← 추가
    ...
]
```

---

## 이슈 8: finally 블록이 출력 폴더를 프레임 수집 전에 삭제

### 증상
```
[ERROR] [TokenFlow] 출력 프레임 없음 — img_ode 탐색 경로: .../img_ode/
```
Sampling 30/30 완료됐는데 프레임을 못 찾는 상황.

### 원인
`_cleanup_tokenflow_dirs()` 가 `finally` 블록에서 출력 폴더(`rel_output*`)를 삭제했다. Python에서 `finally` 는 `try` 블록이 끝나는 순간 실행되기 때문에, 프레임 수집 코드 실행 전에 폴더가 사라짐.

실행 순서:
```
run_tokenflow_pnp 완료 → img_ode/ 생성
finally 실행 → img_ode/ 포함 출력 폴더 삭제  ← 문제
프레임 수집 시도 → 폴더 없음 → 에러
```

### 수정
출력 폴더는 `finally` 에서 삭제하지 않고, 프레임 수집 및 `edited_dir` 복사 완료 직후에 별도로 삭제.
```python
# 프레임 수집 완료 후 출력 폴더 삭제
for out_dir in _glob2.glob(os.path.join(TOKENFLOW_DIR, f"{rel_output}*")):
    shutil.rmtree(out_dir, ignore_errors=True)

return edited

# finally에서는 입력 폴더(data/tf_*)와 yaml config만 정리
finally:
    _cleanup_tokenflow_dirs(uid, rel_output)
```

---

## 이슈 9: 7초 영상 → 4초 출력, 앞 구간 반복, 픽셀 깨짐

### 증상
- 7초 영상을 넣으면 4초짜리 영상이 나옴
- 앞 2~3초가 반복되는 느낌
- 1~2초 구간에서 픽셀 깨짐

### 원인
`EXTRACT_FPS=8`, `KEYFRAME_FREQ=10` 조합이 짧은 클립에서 keyframe 부족을 일으킴.

- 7초 × 8fps = 56프레임 추출
- 56 ÷ 10 = keyframe 5~6개
- keyframe이 너무 적으면 보간이 뭉개지고 앞 구간이 반복됨
- `N_TIMESTEPS=50` 은 DDIM 초반 구간 불안정으로 픽셀 깨짐 발생

### 수정
```python
# 수정 전
EXTRACT_FPS   = 8
KEYFRAME_FREQ = 10
N_TIMESTEPS   = 50

# 수정 후
EXTRACT_FPS   = 4
KEYFRAME_FREQ = 4
N_TIMESTEPS   = 30
```

7초 클립 기준:
- 추출 프레임 = 7 × 4 = 28개
- keyframe = 28 ÷ 4 = 7개 (충분)
- DDIM steps 감소로 초반 픽셀 깨짐 완화

---

## 이슈 10: 3fps 영상에서 프레임 수 부족으로 품질 저하

### 증상
MSR-VTT 데이터셋 영상이 원본 fps=3.0 이라서 21개 프레임밖에 추출되지 않음. TokenFlow 품질 저하.

### 원인
MSR-VTT 영상이 원래 3fps로 인코딩되어 있어서, `EXTRACT_FPS=4` 로 추출하려 해도 원본에 프레임이 부족함.

`effective_fps = min(EXTRACT_FPS, orig_fps) = min(4, 3) = 3` → step=1 → 21개 추출

### 수정
RIFE(Real-Time Intermediate Flow Estimation) 광학 흐름 보간 추가.

ffmpeg 단순 보간(픽셀 평균)과 달리 RIFE는 프레임 사이의 움직임 벡터를 추정해서 자연스러운 중간 프레임을 생성한다.

```
3fps 영상
  → RIFE 8배 보간 → 24fps
  → EXTRACT_FPS=4 로 추출 (step=6)
  → 7초 × 4 = 28프레임
```

RIFE 레포: `hzwer/Practical-RIFE`  
가중치: HuggingFace `jbilcke-hf/varnish/rife/flownet.pkl`  
(SHA256: `fe854fc8996547c953f732aaa3b78cae76cc0a12833ae856ea0749c4c570d7d8` — 여러 레포에서 동일 해시 확인됨)

실패 시 원본 fps로 폴백하여 파이프라인이 중단되지 않음.

---

## 이슈 11: n_frames가 batch_size 배수 아닐 때 프레임 손실

### 증상
- 21프레임을 넣으면 16프레임짜리 영상이 나옴 (5프레임 손실)
- 로그: `편집 완료: 16프레임`

### 원인
`run_tokenflow_pnp.py` 내부 코드:
```python
if self.config["n_frames"] % self.config["batch_size"] != 0:
    self.config["n_frames"] = self.config["n_frames"] - (self.config["n_frames"] % self.config["batch_size"])
```
TokenFlow가 `n_frames % batch_size != 0` 이면 자동으로 잘라버림.
- 21프레임, batch_size=8 → 21 % 8 = 5 → 21 - 5 = 16프레임으로 잘림

### 수정
`BATCH_SIZE` 를 고정으로 넘기는 대신, `n_frames` 의 약수 중 `BATCH_SIZE` 이하 최댓값을 동적으로 계산해서 넘김.

```python
def _best_batch_size(n: int, max_bs: int) -> int:
    best = 1
    for d in range(1, max_bs + 1):
        if n % d == 0:
            best = d
    return best

effective_batch_size = _best_batch_size(n_frames, BATCH_SIZE)
# n_frames=21 → effective_batch_size=7 (21÷7=3배치, 손실 없음)
# n_frames=27 → effective_batch_size=3 (27÷3=9배치, 손실 없음)
# n_frames=24 → effective_batch_size=8 (24÷8=3배치, 변화 없음)
```

---

## 이슈 12: `img_ode` 폴더 탐색 깊이 부족

### 증상
```
[ERROR] [TokenFlow] 출력 프레임 없음 — img_ode 탐색 경로: .../img_ode/
```
Sampling이 정상 완료됐는데도 프레임을 못 찾음.

### 원인
`run_tokenflow_pnp.py` 가 output 경로를 아래처럼 여러 단계로 만들어서 저장함:
```
tf_{uid}_out_.../
  _pnp_SD_1.5/
    tf_{uid}/
      {프롬프트}/
        attn_0.5_f_0.8/
          batch_size_8/
            30/
              img_ode/  ← 실제 프레임 저장 위치
```

기존 코드는 한 단계만 들어가서 `img_ode` 를 찾지 못함.

### 수정
`**` 재귀 glob으로 `img_ode` 폴더를 동적으로 탐색.
```python
img_ode_dirs = [
    d for d in _glob.glob(
        os.path.join(TOKENFLOW_DIR, f"{rel_output}*", "**", "img_ode"),
        recursive=True,
    )
    if os.path.isdir(d)
]
```

---

## 이슈 13: RIFE 가중치 다운로드 실패

### 증상
```
[RIFE] 가중치 다운로드 실패 → 원본 fps로 진행
```

### 원인
처음에 넣은 URL이 존재하지 않는 GitHub Releases 경로였음.  
`Practical-RIFE` 공식 배포는 Google Drive와 바이두 네트판만 제공하는데, 두 서비스 모두 wget 자동 다운로드가 막혀 있음.

### 수정
HuggingFace `jbilcke-hf/varnish` 에 미러된 파일 사용.
- 여러 독립적인 레포(`LeonJoe13/Sonic`, `DeepBeepMeep/Wan2.1` 등)에서 SHA256 해시가 동일하게 확인됨
- `jbilcke-hf` 는 HuggingFace staff 계정

```python
weight_url = "https://huggingface.co/jbilcke-hf/varnish/resolve/main/rife/flownet.pkl"
```

다운로드 후 파일 크기 검증 추가 (`< 1000 bytes` 이면 실패 처리).

---

## TokenFlow end-to-end 정상 흐름 (전체 수정 후)

```
Gradio에서 변환 버튼 클릭
  ↓
_ensure_ready()
  ├── TokenFlow 설치 확인
  └── revision="fp16" 패치 적용
  ↓
_extract_frames()
  ├── 원본 fps 확인 (cv2)
  ├── orig_fps < EXTRACT_FPS 이면 RIFE 보간 시도
  │     3fps → 24fps (8배 보간)
  │     실패 시 원본 fps로 폴백
  ├── effective_fps = min(EXTRACT_FPS, orig_fps)
  ├── step = round(orig_fps / effective_fps)
  └── 매 step번째 프레임 → PNG 저장
  ↓
_run_tokenflow()
  ├── uid 생성, rel_data/rel_output 경로 설정
  ├── PNG → JPG 변환 후 /content/TokenFlow/data/tf_{uid}/ 에 복사
  ├── n_frames의 약수 중 BATCH_SIZE 이하 최댓값으로 effective_batch_size 계산
  │
  ├── [Step 1] preprocess.py 실행
  │     --data_path data/tf_{uid}  (상대경로)
  │     --save_dir  data/tf_{uid}
  │     --steps 30 --save_steps 30  (동일한 timestep 목록 보장)
  │     --n_frames {n_frames}
  │     --batch_size {effective_batch_size}
  │     → latents 저장: data/tf_{uid}/sd_1.5/tf_{uid}/steps_30/nframes_{N}/latents/
  │
  ├── [Step 2] run_tokenflow_pnp.py 실행
  │     n_timesteps=30, n_inversion_steps=30  (preprocess와 동일)
  │     batch_size={effective_batch_size}
  │     → 편집 프레임 저장: tf_{uid}_out_.../_pnp_SD_1.5/.../30/img_ode/
  │
  ├── img_ode/ 폴더를 ** 재귀 탐색으로 찾음
  ├── 프레임을 edited_dir 로 복사
  ├── 출력 폴더 삭제 (수집 완료 후)
  └── finally: data/tf_{uid}/ + yaml config 삭제
  ↓
_reconstruct_video()
  ├── ffmpeg로 mp4 재조립 (fps=4)
  └── 실패 시 OpenCV 폴백
  ↓
최종 영상 경로 반환
```

---

## 관련 파일

| 파일 | 역할 |
|---|---|
| `src/phase4_assembly/tokenflow_wrapper.py` | 수정 대상 전체 |
| `/content/TokenFlow/preprocess.py` | DDIM inversion, latent 저장 |
| `/content/TokenFlow/run_tokenflow_pnp.py` | PnP 편집, latent 요청, img_ode 저장 |
| `/content/TokenFlow/tokenflow_utils.py` | `load_source_latents_t()` — latent 로드 |
| `/content/Practical-RIFE/inference_video.py` | RIFE 보간 실행 |