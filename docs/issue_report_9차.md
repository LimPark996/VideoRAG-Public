# 이슈 보고서: TokenFlow `Missing latents` 오류

**날짜**: 2026-04-20  
**파일**: `src/phase4_assembly/tokenflow_wrapper.py`  
**상태**: 원인 확인 완료, 수정 필요

---

## 증상

Gradio에서 변환 버튼을 누르면 아래 에러가 발생하며 원본 영상이 반환됨.

```
AssertionError: Missing latents at t 925 path
data/tf_7f01cf01/sd_1.5/tf_7f01cf01/steps_30/nframes_27/latents/noisy_latents_925.pt
```

```
[ERROR] [TokenFlow] run_tokenflow_pnp.py 실패 (returncode=1)
[WARNING] [TokenFlow] 편집 실패 — 원본 반환
```

---

## 원인 분석

### TokenFlow 실행 흐름

TokenFlow는 두 단계로 실행된다.

**1단계: preprocess.py — DDIM inversion**

입력 프레임들을 점점 노이즈를 추가해가면서 중간 상태를 저장하는 과정이다. 예를 들어 `t=925` 시점에 27개 프레임 전체가 얼마나 노이즈화됐는지를 `noisy_latents_925.pt` 파일로 저장한다.

어떤 timestep에 저장할지는 `--save_steps` 인자로 결정된다. `save_steps=N` 이면 DDIM scheduler가 1000개의 timestep 중 N개를 균등하게 선택해 그 시점의 latent만 저장한다.

```python
toy_scheduler.set_timesteps(opt.save_steps)
timesteps_to_save, _ = get_timesteps(...)

# ddim_inversion 내부
if save_latents and t in timesteps_to_save:
    torch.save(latent_frames, f'noisy_latents_{t}.pt')
```

**2단계: run_tokenflow_pnp.py — PnP 편집**

저장된 latent 파일들을 읽어 역방향으로 denoising하며 스타일을 변환한다. 어떤 timestep을 요청할지는 `n_timesteps` 인자로 결정된다. `n_timesteps=N` 이면 scheduler가 동일하게 N개의 timestep 목록을 생성하고, 그 목록에 있는 t값들에 해당하는 latent 파일을 하나씩 읽어온다.

```python
self.scheduler.set_timesteps(config["n_timesteps"], device=self.device)
# ...
source_latents = load_source_latents_t(t, self.latents_path)[indices]
# → noisy_latents_{t}.pt 파일을 읽음
```

---

### 핵심 원인: `save_steps`와 `n_timesteps` 불일치

두 스크립트가 **같은 N값**으로 scheduler를 설정해야 같은 timestep 목록을 공유한다.

실제로 `set_timesteps(30)` 으로 생성되는 timestep 목록을 확인한 결과:

```python
from diffusers import DDIMScheduler
scheduler = DDIMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")
scheduler.set_timesteps(30)
print(scheduler.timesteps.tolist())
# → [958, 925, 892, 859, 826, 793, 760, 727, 694, 661, 628, 595,
#     562, 529, 496, 463, 430, 397, 364, 331, 298, 265, 232, 199,
#     166, 133, 100, 67, 34, 1]
```

`t=925` 는 `set_timesteps(30)` 기준 목록에 포함되어 있다.

현재 `tokenflow_wrapper.py` 에서 preprocess에 넘기는 인자:

```python
"--steps",  str(N_TIMESTEPS),   # 30 전달
# --save_steps 미전달 → 기본값 50으로 실행
```

결과적으로:

| 항목 | 값 | timestep 목록 기준 |
|---|---|---|
| preprocess `save_steps` | **50** (기본값) | `set_timesteps(50)` 기준으로 저장 |
| run_tokenflow_pnp `n_timesteps` | **30** | `set_timesteps(30)` 기준으로 요청 |

`set_timesteps(50)` 기준 목록과 `set_timesteps(30)` 기준 목록은 서로 다르다. preprocess가 `save_steps=50` 기준으로 저장하면 `t=925` 가 그 목록에 없어서 파일이 생성되지 않는다. run_tokenflow_pnp는 `n_timesteps=30` 기준으로 `t=925` 를 요청하는데 파일이 없으니 에러가 발생한다.

---

### 부가 원인: 입력 영상 fps 문제 (선행 이슈, 이미 수정됨)

MSR-VTT 데이터셋 영상의 원본 fps가 3.0으로 낮아, `EXTRACT_FPS=4` 로 추출하면 step=1이 되어 21개밖에 뽑히지 않았다. 이는 ffmpeg 30fps 보간으로 수정 완료.

| | 수정 전 | 수정 후 |
|---|---|---|
| 원본 fps | 3.0 | 3.0 |
| 보간 후 fps | 없음 | 30.0 |
| 추출 프레임 수 | 21개 | 27개 |

---

## 해결 방법

`preprocess_cmd` 에 `--save_steps` 를 `N_TIMESTEPS` 와 동일한 값으로 추가한다.

```python
# 수정 전
preprocess_cmd = [
    sys.executable, preprocess_script,
    "--steps",  str(N_TIMESTEPS),   # 30
    ...
]

# 수정 후
preprocess_cmd = [
    sys.executable, preprocess_script,
    "--steps",      str(N_TIMESTEPS),   # 30
    "--save_steps", str(N_TIMESTEPS),   # 30 ← 추가
    ...
]
```

이렇게 하면:

- preprocess: `set_timesteps(30)` 기준으로 `t=958, 925, 892 ...` 에 latent 저장
- run_tokenflow_pnp: `set_timesteps(30)` 기준으로 동일한 `t=958, 925, 892 ...` 요청
- `noisy_latents_925.pt` 파일 존재 → 정상 로드 → 에러 없음

---

## TokenFlow end-to-end 정상 흐름 (수정 후)

```
1. _extract_frames
   - 입력 영상 3fps → ffmpeg 30fps 보간
   - 30fps 기준 EXTRACT_FPS=4 로 추출 (step=8)
   - 결과: 27개 프레임

2. 프레임 복사 (PNG → JPG)
   - /content/TokenFlow/data/tf_xxxx/00000.jpg ~ 00026.jpg

3. preprocess.py 실행
   - --steps 30 --save_steps 30 --n_frames 27
   - set_timesteps(30) → [958, 925, 892, ... 1] 목록 생성
   - 27개 프레임을 각 timestep마다 DDIM inversion
   - 30개 파일 저장:
     noisy_latents_958.pt, noisy_latents_925.pt, ... noisy_latents_1.pt

4. run_tokenflow_pnp.py 실행
   - n_timesteps=30, n_inversion_steps=30
   - set_timesteps(30) → 동일하게 [958, 925, 892, ... 1] 목록 생성
   - 각 timestep마다 noisy_latents_{t}.pt 읽어서 PnP 편집 수행
   - t=925 요청 → noisy_latents_925.pt 존재 ✅
   - 편집된 프레임 27개 출력

5. _reconstruct_video
   - 27개 프레임 → ffmpeg로 mp4 재조립
   - 최종 영상 반환
```

---

## 영향 범위

- **직접 영향**: TokenFlow 변환 기능 전체 (`on_exec_transform`, `on_execute`)
- **간접 영향**: Scene Graph 워크플로의 TRANSFORM 분기 전체
- **미영향**: USE_AS_IS, GENERATE(Runway) 분기, 검색 기능

---

## 관련 파일

| 파일 | 역할 |
|---|---|
| `src/phase4_assembly/tokenflow_wrapper.py` | 수정 대상 |
| `/content/TokenFlow/preprocess.py` | `--steps`, `--save_steps` 인자 정의, latent 저장 |
| `/content/TokenFlow/run_tokenflow_pnp.py` | `n_timesteps` 기준으로 latent 요청 |
| `/content/TokenFlow/tokenflow_utils.py` | `load_source_latents_t()` — 실제 에러 발생 지점 |