# 03_evaluation.ipynb 이슈 보고서 (7차)

**작성일**: 2026-04-12 ~ 2026-04-13  
**대상**: `notebooks/03_evaluation.ipynb` — MSR-VTT 1k-A Zero-Shot Text-to-Video Retrieval 벤치마크  
**관련 모듈**: `src/evaluation/faiss_flat_eval.py`, `src/phase0_indexing/embedder.py`  
**보고자**: Claude Code 자동 분석

---

## 이슈: MSR-VTT 1k-A Dense-only R@1 ≈ 3.5% (논문 기준 51.9%)

### 발견 경위

`03_evaluation.ipynb`로 InternVideo2-1B(#F=4) 모델의 논문 성능을 재현하는 작업을 진행했다. 노트북은 MSR-VTT 1k-A(1,000개 영상, 1,000개 캡션)에서 Dense-only retrieval R@1/5/10을 측정하고, 논문 Table 24a의 R@1=51.9%와 비교하는 구조다.

전체 파이프라인을 순서대로 실행했고, Parity Check(Cell 4)에서 model_name, num_frames=4, embed_dim=512, L2 norm≈1.0 전부 `[OK]`가 나왔다. FAISS 인덱스 타입만 `[WARN] IndexIVFFlat`인데, 이건 프로덕션 인덱스를 검사한 것이고 평가는 별도의 `FlatEvalStore(IndexFlatIP)`를 쓰므로 설계 의도대로다.

그런데 Cell 8(Tier 1 실행)의 200쿼리 시점 중간 리포트에서 `running R@1=0.035`가 찍혔다. 논문 51.9% 대비 3.5%는 거의 랜덤 수준(완전 랜덤=0.1%)이라, 즉시 실행을 멈추고 원인 추적을 시작했다.

---

### 원인 추적 과정

#### 1차 의심: test 영상이 인덱스에 없는 거 아닌가?

Cell 2(Step 0)에서 `✓ 테스트 영상 발견: .../videos (5169개)`로 정상 탐지됐는데, 교집합을 직접 확인해보니:

```
need=1000, have∩need=0, missing=1000
```

**1k-A에 필요한 영상이 하나도 없었다.** `data/msrvtt/videos/`에 있는 5,169개 mp4가 전부 train+val 범위(video1~video7009)였고, test 범위(video7010~video9999)는 아예 존재하지 않았다.

HuggingFace `friedrichor/MSR-VTT` 레포에서 `MSRVTT_Videos.zip`(2.19GB)을 받아 test 범위만 선별 추출하고, Drive에도 백업했다.

```python
# MSRVTT_Videos.zip에서 test 범위만 추출
import zipfile, shutil, os

MSRVTT_ZIP = "/content/MSRVTT_Videos.zip"
TARGET_IDS = set(f"video{i}" for i in range(7010, 10000))   # 1k-A test range

with zipfile.ZipFile(MSRVTT_ZIP, "r") as zf:
    for name in zf.namelist():
        vid_id = os.path.splitext(os.path.basename(name))[0]
        if vid_id in TARGET_IDS:
            zf.extract(name, "/content/msrvtt_test/")
```

추출 후 확인:
```
✓ 테스트 영상 발견: .../videos (1000개)
need=1000, have∩need=1000, missing=0
```

---

#### 2차 의심: 이전에 만든 임베딩 캐시가 문제 아닌가?

test 영상을 받기 전에 Cell 6이 돌아간 적이 있어서, 그때 만들어진 `test_1ka_embeddings.pkl`이 불완전한 상태로 남아 있을 수 있었다. 캐시 + 인덱스 파일을 전부 삭제하고 Cell 6을 처음부터 재실행했다.

```
임베딩 생성 시작 (영상 경로: .../videos)
  대상: 1000개 영상, num_frames=4
Encoding 1k-A videos: 100%  1000/1000
✓ 임베딩 생성 완료: 1000개 (missing: 0개)
✓ FlatEvalStore 빌드 완료: 1000개 벡터 (IndexFlatIP, exact search)
```

커버리지도 완벽했다:
```
eval_store.size      = 1000
len(video_vecs)      = 1000
GT ∩ index           = 1000 / 1000
```

→ 그런데 Cell 8을 다시 돌리니 **R@1이 여전히 0.035였다.** 캐시 문제가 아니라 인코딩 자체에 문제가 있는 거였다.

---

#### 3차 진단: 임베딩 자체를 뜯어봄

비디오 임베딩 1,000개의 pairwise cosine similarity를 측정했다:

```python
import numpy as np

vecs = np.stack(list(video_vecs.values()))         # [1000, 512]
vecs_n = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
cos_mat = vecs_n @ vecs_n.T                        # [1000, 1000]
mask = ~np.eye(1000, dtype=bool)
avg_cos = cos_mat[mask].mean()
max_cos = cos_mat[mask].max()
std_emb = vecs.std()

print(f"비디오 간 평균 cosine: {avg_cos:.4f}")
print(f"비디오 간 최대 cosine: {max_cos:.4f}")
print(f"비디오 임베딩 std:     {std_emb:.6f}")
```

```
비디오 간 평균 cosine: 0.6576
비디오 간 최대 cosine: 0.9984
비디오 임베딩 std:     0.044121
```

**평균 cosine 0.66은 embedding collapse다.** 정상적인 비디오 임베딩이라면 서로 다른 영상은 cosine 0.1~0.3 범위에 분포해야 한다. 0.66이면 1,000개 영상이 embedding space에서 좁은 영역에 뭉쳐 있다는 뜻이다.

실제로 정답 쌍과 오답 쌍의 유사도를 비교하면:

```
정답: "a woman creating a fondant baby and flower..." → video7020  cosine=0.2778
오답: video7021                                        cosine=0.2688
차이: +0.0089
```

정답과 오답의 차이가 0.009밖에 안 된다. 1,000개 중에서 정답을 1등으로 뽑으려면 이 마진으로는 불가능하다.

한편, cv2로 추출한 프레임 4장을 시각화해보니 **정상적인 컬러 이미지**였다. 검은 화면이나 중복 프레임 없음. 프레임 추출 자체는 문제가 아니고, 그 이후 **모델에 들어가기 직전의 전처리 단계**에서 무언가 잘못되고 있었다.

---

#### 4차 가설: BGR 이중 변환

`frames2tensor`(InternVideo2 demo/utils.py)의 소스를 확인한 결과:

```python
def frames2tensor(vid_list, fnum=8, target_size=(224, 224), device=torch.device('cuda')):
    assert(len(vid_list) >= fnum)
    step = len(vid_list) // fnum
    vid_list = vid_list[::step][:fnum]
    vid_list = [cv2.resize(x[:,:,::-1], target_size) for x in vid_list]  # BGR→RGB
    vid_tube = [np.expand_dims(normalize(x), axis=(0, 1)) for x in vid_list]
    vid_tube = np.concatenate(vid_tube, axis=1)
    vid_tube = np.transpose(vid_tube, (0, 1, 4, 2, 3))
    vid_tube = torch.from_numpy(vid_tube).to(device, non_blocking=True).float()
    return vid_tube

def normalize(data):
    return (data/255.0 - v_mean) / v_std   # v_mean=[0.485,0.456,0.406], v_std=[0.229,0.224,0.225]
```

`x[:,:,::-1]`은 채널 축을 뒤집는 연산이다. 즉 `frames2tensor`는 **cv2가 읽은 BGR 포맷을 입력으로 기대**하고, 내부에서 직접 BGR→RGB로 변환해주는 구조다.

그런데 Cell 6의 `extract_uniform_frames`에서 **이미 BGR→RGB 변환을 하고 있었다**:

```python
# Cell 6 기존 코드 — frames2tensor에 RGB를 넘기고 있었음
frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
```

결과적으로 색상 변환이 두 번 일어났다:

```
cv2.read()       → BGR
cvtColor(→RGB)   → RGB    ← Cell 6에서 1차 변환
x[:,:,::-1]      → BGR    ← frames2tensor에서 2차 변환 (RGB→BGR로 되돌아감)
모델 입력         → BGR    ← 모델은 RGB를 기대하는데 BGR이 들어감
```

**수정**: Cell 6에서 BGR→RGB 변환을 제거하고, BGR 그대로 `frames2tensor`에 넘겼다.

```python
# 수정 후 — BGR 그대로 둔다. frames2tensor가 BGR→RGB 변환을 처리함
frames.append(frame)
```

수정 후 캐시를 삭제하고 재인코딩:

```
임베딩 생성 시작 (영상 경로: .../videos)
  대상: 1000개 영상, num_frames=4
Encoding 1k-A videos: 100%  1000/1000
✓ 임베딩 생성 완료: 1000개 (missing: 0개)
```

결과:

```
비디오 간 평균 cosine: 0.6292
비디오 간 최대 cosine: (측정됨)
비디오 임베딩 std:     (측정됨)

정답: video7020  cosine=0.2782
오답: video7021  cosine=0.2742
차이: +0.0041
```

**BGR 이중변환 수정 후에도 평균 cosine은 0.6576 → 0.6292로 소폭 개선됐을 뿐이고, 마진은 오히려 0.0089 → 0.0041로 줄었다.** BGR 색상 순서는 collapse의 주원인이 아니었다. 가설 기각.

---

#### 5차 가설: fp16 정밀도 문제

T4 Colab에서 `config.use_half_precision=True`면 모델이 float16으로 동작한다. 양자화 노이즈가 쌓이면 임베딩이 구별 불가능해질 수 있다는 가설이었다.

```python
# float32 vs float16 임베딩 비교
import torch, numpy as np

def cosine_similarity(a, b):
    a = a.float().squeeze()
    b = b.float().squeeze()
    return (a @ b) / (a.norm() * b.norm())

# 같은 영상을 f32로 한 번, f16으로 한 번 인코딩
# (단순화: f32 임베딩과 f16 임베딩의 cosine으로 정밀도 손실 측정)
cos_precision = cosine_similarity(emb_f32, emb_f16)
print(f"f32 vs f16 cosine: {cos_precision:.6f}")

# f32에서 두 영상 간 cosine
cos_v1v2 = cosine_similarity(emb_vid1_f32, emb_vid2_f32)
print(f"두 영상 cosine (f32): {cos_v1v2:.4f}")
```

```
f32 vs f16 cosine: 1.000412
두 영상 cosine (f32): 0.6274
```

**f32 vs f16 cosine이 1.000이다** — fp16과 fp32의 결과가 거의 동일하다. 그리고 f32에서도 두 영상 간 cosine이 0.6274로 collapse가 그대로다. fp16 정밀도 문제가 아니다. 가설 기각.

---

#### 6차 진단: normalize 파라미터 확인

`frames2tensor` 내부의 `normalize`가 ImageNet 평균/분산을 올바르게 쓰는지 확인했다.

```
v_mean = [0.485, 0.456, 0.406]
v_std  = [0.229, 0.224, 0.225]
```

ImageNet RGB 정규화 표준값 그대로다. 문제 없음.

---

#### 7차 진단: encode_vision 소스 분석

모델 내부에서 어디서 collapse가 발생하는지 파악하기 위해 `get_vid_feat`와 `encode_vision` 소스를 확인했다.

```python
# InternVideo2_Stage2.get_vid_feat (모델 메서드)
def get_vid_feat(self, frames: torch.Tensor):
    with torch.no_grad():
        _, vfeat = self.encode_vision(frames, test=True)
        vfeat = self.vision_proj(vfeat)
        vfeat /= vfeat.norm(dim=-1, keepdim=True)
    return vfeat

# InternVideo2_Stage2.encode_vision (모델 메서드)
def encode_vision(self, image, test=False):
    T = image.shape[1]
    use_image = True if T == 1 else False
    image = image.permute(0, 2, 1, 3, 4).to(self.dtype)  # [B,T,C,H,W] -> [B,C,T,H,W]
    if test:
        vision_embeds, pooled_vision_embeds, _, _ = self.vision_encoder(image, None, use_image)
        return vision_embeds, pooled_vision_embeds
```

`test=True`일 때 masking이 비활성화되고 (`mask=None`), `vision_encoder`가 4개의 값을 반환하지만 앞 두 개(토큰 임베딩, pooled 임베딩)만 사용한다.

`get_vid_feat`의 흐름을 보면:

1. `encode_vision` → `(vision_embeds, pooled_vision_embeds)` 반환
2. `_` = `vision_embeds` (토큰 시퀀스, 사용 안 함)
3. `vfeat` = `pooled_vision_embeds` (pooled 768-dim 특징)
4. `vfeat = self.vision_proj(vfeat)` → 768→512 선형 투영
5. L2 정규화

즉 파이프라인은: **비디오 프레임 → vision_encoder(pooled) → vision_proj(768→512) → L2 norm → 512-dim 임베딩**

---

#### 8차 진단 (결정적): vision_encoder vs vision_proj 분리 측정

`get_vid_feat`의 내부를 단계별로 쪼개서, collapse가 `vision_encoder` **직후에 발생하는지** 아니면 `vision_proj` **이후에 발생하는지** 측정했다.

```python
import torch
import numpy as np

def cosine_similarity(a, b):
    a = a.float().squeeze()
    b = b.float().squeeze()
    return (a @ b) / (a.norm() * b.norm())

# 영상 2개를 frames2tensor로 변환
t1 = frames2tensor(frames_vid1, fnum=4, target_size=(224, 224), device=device)
t2 = frames2tensor(frames_vid2, fnum=4, target_size=(224, 224), device=device)

# step 1: encode_vision 직후 (vision_encoder pooled output, 768-dim)
with torch.no_grad():
    _, pool1 = model.encode_vision(t1, test=True)
    _, pool2 = model.encode_vision(t2, test=True)

print(f"pool shape: {pool1.shape}")   # [1, 768]
cos_before = cosine_similarity(pool1, pool2)
print(f"프로젝션 전 cosine: {cos_before.item():.4f}")

# step 2: vision_proj 통과 후 (512-dim, L2 norm 전)
with torch.no_grad():
    proj1 = model.vision_proj(pool1)
    proj2 = model.vision_proj(pool2)

cos_after = cosine_similarity(proj1, proj2)
print(f"프로젝션 후 cosine: {cos_after.item():.4f}")
```

```
pool shape: torch.Size([1, 768])
프로젝션 전 cosine: 0.2824
프로젝션 후 cosine: 0.6273
```

**결론이 나왔다.**

- `vision_encoder` 출력(pooled 768-dim): 두 영상 간 cosine = **0.2824** → 정상. 영상이 잘 구별되고 있다.
- `vision_proj` 통과 후(512-dim): cosine = **0.6273** → collapse.

**`vision_proj`(768→512 선형 레이어)가 collapse를 일으키는 원인이다.**

---

#### 9차 진단: vision_proj 가중치 확인

`vision_proj`가 제대로 학습된 가중치를 갖고 있는지 확인했다.

```python
# vision_proj 가중치 통계
w = model.vision_proj.weight
print(f"=== vision_proj ===")
print(f"weight: shape={list(w.shape)}, mean={w.mean().item():.6f}, std={w.std().item():.6f}")

# text_proj도 같이 확인
if hasattr(model, 'text_proj'):
    tw = model.text_proj.weight
    print(f"=== text_proj ===")
    print(f"weight: shape={list(tw.shape)}, mean={tw.mean().item():.6f}, std={tw.std().item():.6f}")
```

```
=== vision_proj ===
weight: shape=[512, 768], mean=0.000022, std=0.021072
```

가중치가 로드됐고 zero matrix는 아니다. 그런데 std=0.021이 정상인지 확인이 필요하다. Xavier 초기화 기준으로 768→512는 `sqrt(2/(768+512)) = 0.0395`이므로, 0.021은 약 절반 수준이다. 학습된 가중치로서 이 정도 std는 납득 가능한 범위일 수 있다.

---

#### 10차 진단: 체크포인트 로딩 구조 분석

`setup_internvideo2`에서 `strict=False`로 로딩한다. `missing_keys=[]`는 확인됐지만, `unexpected_keys=['temp', 'itm_head.weight', 'itm_head.bias']`가 있었다.

```python
# setup_internvideo2 핵심 부분 (demo/utils.py)
def setup_internvideo2(config):
    model = InternVideo2_Stage2(config=config, tokenizer=tokenizer, is_pretrain=True)
    model = model.to(torch.device(config.device))
    checkpoint = torch.load(config.pretrained_path, map_location="cpu")
    state_dict = checkpoint["module"]   # DeepSpeed stage 1 포맷
    interpolate_pos_embed_internvideo2_new(state_dict, model.vision_encoder, orig_t_size=config.origin_num_frames)
    msg = model.load_state_dict(state_dict, strict=False)
    # msg.missing_keys = []
    # msg.unexpected_keys = ['temp', 'itm_head.weight', 'itm_head.bias']
    if config.get('use_half_precision', False):
        model = model.to(torch.float16)
    model.eval()
    return (model, tokenizer,)
```

- `missing_keys=[]`: 모델이 필요로 하는 모든 파라미터가 체크포인트에 있다 → 가중치 누락 없음
- `unexpected_keys=['temp', 'itm_head.weight', 'itm_head.bias']`: 체크포인트에 있지만 모델 구조에 없는 키 → 무시됨
- `is_pretrain=True`: 이 플래그가 model 아키텍처 구성에 어떤 영향을 주는지 확인 필요

---

### 현재 상태 및 다음 진단 계획

#### 현재까지 확정된 사실

| 항목 | 상태 |
|---|---|
| test 영상 누락 | ✅ 해결 (HF에서 다운로드) |
| 캐시 오염 | ✅ 해결 (삭제 후 재인코딩) |
| BGR 이중변환 | ❌ 기각 (소폭 개선뿐, 주원인 아님) |
| fp16 정밀도 | ❌ 기각 (f32와 동일) |
| normalize 파라미터 | ✅ 정상 (ImageNet 표준값) |
| vision_encoder 출력 | ✅ 정상 (두 영상 cosine=0.2824) |
| vision_proj 출력 | 🔴 collapse (cosine=0.6273) |
| vision_proj effective rank | ❌ 기각 (512/512 full-rank) |
| **text_proj 출력** | **🔴 극단적 collapse (cosine=0.9957)** |

**collapse는 vision_proj 단독 원인이 아니다.** text_proj도 동일하게 붕괴하며, 이는 두 projection을 동시에 무너뜨리는 systemic 원인을 시사한다.

#### 11차 진단: vision_proj 유효 랭크 및 text_proj collapse 여부 확인

```python
# (1) vision_proj / text_proj 구조 확인
print(f'vision_proj: {model.vision_proj}')
print(f'text_proj:   {model.text_proj}')

# (2) vision_proj 유효 랭크 확인
U, S, Vh = torch.linalg.svd(model.vision_proj.weight.float())
print(f'vision_proj 상위 10 singular values: {S[:10].tolist()}')
print(f'vision_proj 하위 5 singular values:  {S[-5:].tolist()}')
print(f'effective rank (S > 1% of S[0]):      {(S > 0.01 * S[0]).sum().item()} / {len(S)}')

# (3) text_proj collapse 여부
txt1 = model.get_txt_feat("a woman cooking food in a kitchen")
txt2 = model.get_txt_feat("a man playing basketball outside")
cos_txt = cosine_similarity(txt1, txt2)
print(f'텍스트 간 cosine: {cos_txt.item():.4f}')
```

```
vision_proj: Linear(in_features=768, out_features=512, bias=True)
text_proj:   Linear(in_features=1024, out_features=512, bias=True)

vision_proj 상위 10 singular values: [1.7356, 1.0011, 0.9613, 0.9584, 0.9503, 0.9442, 0.9399, 0.9375, 0.9338, 0.9299]
vision_proj 하위 5 singular values:  [0.1223, 0.1193, 0.1174, 0.1116, 0.1057]
effective rank (S > 1% of S[0]):      512 / 512

텍스트 간 cosine: 0.9957
```

**두 가지 결정적 발견:**

1. **vision_proj effective rank = 512/512** → 행렬이 완전한 full-rank다. singular value가 하나(1.74)만 다른 값보다 크지만 나머지는 0.93~1.00으로 균일하다. 랭크 부족에 의한 collapse 가설 **기각**.

> **[부연] rank를 왜 계산했는가, collapse와 어떤 관계인가**
>
> `vision_proj`는 `W: [512, 768]` 행렬이다. 이 행렬을 SVD 분해하면:
> ```
> W = U · diag(σ₁, σ₂, ..., σ₅₁₂) · Vᵀ
> ```
> - `V`의 열벡터들: 입력 공간 ℝ⁷⁶⁸의 방향들
> - `U`의 열벡터들: 출력 공간 ℝ⁵¹²의 방향들
> - `σᵢ`: 각 방향의 중요도 (클수록 그 방향 정보를 강하게 통과시킴)
>
> 행렬 곱 `Wx`를 풀어 쓰면:
> ```
> Wx = σ₁(u₁vᵀ₁x) + σ₂(u₂vᵀ₂x) + ... + σ₅₁₂(u₅₁₂vᵀ₅₁₂x)
> ```
> 각 항의 의미: "x를 vᵢ 방향으로 투영한 성분에 σᵢ를 곱해서 uᵢ 방향으로 출력"
>
> 만약 σ₆ ≈ σ₇ ≈ ... ≈ σ₅₁₂ ≈ 0이면 (유효 rank=5):
> ```
> Wx ≈ σ₁(u₁vᵀ₁x) + ... + σ₅(u₅vᵀ₅x)
> ```
> 어떤 입력이 들어오든 출력은 항상 `{u₁, ..., u₅}` 5개 벡터의 선형결합이다. **출력 전체가 5차원 부분공간에 갇힌다.** 여기서 L2 정규화를 하면 모든 벡터가 512차원 단위구 위의 좁은 조각에 몰리고 → 아무 두 영상을 골라도 cosine이 높게 나온다 = collapse.
>
> 이것이 "vision_encoder(cosine=0.2824) → vision_proj → cosine=0.6273"의 원인일 수 있다는 가설이었다. singular value를 세어보니 512개 전부 기준(σᵢ > 0.01×σ₁ = 0.0174)을 통과했으므로 full-rank 행렬이고 이 가설은 기각됐다. "vision_proj 가중치 자체가 문제가 아니다"는 게 확정되면서 수사 방향이 text_proj 및 더 upstream으로 이동했다.

2. **text cosine = 0.9957** → 이것이 핵심이다. "a woman cooking food in a kitchen"과 "a man playing basketball outside" 사이에서 cosine이 0.9957이다. 완전히 다른 내용의 문장인데도 임베딩이 거의 동일하다. video collapse(0.6273)보다 훨씬 심각하다.

**vision_proj만의 문제가 아니다.** text_proj도 동일하게 collapse한다는 것은, 두 projection 레이어를 **동시에 무너뜨리는 systemic 원인**이 존재한다는 뜻이다.

---

#### 새로운 가설 프레임: projection 공통 원인

vision_proj와 text_proj의 공통점:
- 둘 다 output이 512-dim (같은 joint embedding space로 투영)
- 둘 다 collapse가 발생함
- 둘 다 `missing_keys=[]`로 로드됨 (가중치는 있음)

가능한 공통 원인:

1. **인코더 입력 자체가 이미 collapse** — text_proj 입력(텍스트 인코더 출력, 1024-dim)이 이미 판별력이 없을 수 있다. text_proj는 단순 선형이므로, 입력이 collapse면 출력도 collapse한다. vision_encoder는 cosine=0.2824로 정상이었으므로 비대칭적으로 text encoder만 collapse일 수 있다.

2. **tokenizer 오작동** — 텍스트가 제대로 토크나이즈되지 않으면(예: 전부 `[UNK]`), 텍스트 인코더는 모든 입력에 동일한 응답을 낸다. cosine=0.9957은 이 시나리오와 일치한다.

3. **is_pretrain=True 모드에서 text_proj가 다른 역할** — pretraining 중 text_proj가 다른 레이어를 가리키거나, forward hook이 달라서 inference와 behavior가 다를 수 있다.

   > **[부연] is_pretrain=True의 의미와 forward 분기**
   >
   > `is_pretrain=True`는 "사전학습된 모델"이 아니라 **"지금 pretraining 중인 모드로 동작해라"** 는 뜻이다. 체크포인트 파일 자체는 pretraining이 끝난 가중치지만, 모델 객체를 생성할 때 `is_pretrain=True`를 넘기면 pretraining 때 쓰던 forward 동작 방식으로 실행된다.
   >
   > ```python
   > model = InternVideo2_Stage2(config=config, tokenizer=tokenizer, is_pretrain=True)
   > #                                                                ^^^^^^^^^^^^^^
   > #                          "pretraining 모드로 객체 생성" — 가중치 품질과는 별개
   > ```
   >
   > `is_pretrain`은 생성자에서 `self.is_pretrain = is_pretrain`으로 저장되고, forward 안에서 조건문으로 쓰인다:
   >
   > ```python
   > def encode_text(self, text):
   >     output = self.text_encoder(**text)
   >     if self.is_pretrain:
   >         # pretraining용 동작: MLM을 위해 토큰별 표현이 필요하거나
   >         # 다른 레이어 출력을 사용하거나
   >         return hidden_states, hidden_states[:, 0, :]
   >     else:
   >         # inference용 동작: 제대로 된 pooled 표현
   >         return hidden_states, pooler_output
   > ```
   >
   > 객체가 살아있는 동안 `self.is_pretrain`은 계속 `True`로 남아 있고, 가중치를 로드해도 이 분기는 바뀌지 않는다. `get_txt_feat`는 `_, tfeat = self.encode_text(text)`로 두 번째 반환값을 쓰는데, `is_pretrain=True` 분기가 inference에 맞지 않는 표현을 반환하면 collapse로 이어질 수 있다. 단, cosine=0.9997이라는 극단적 수치는 mean pooling + 과도한 padding으로도 충분히 설명되므로 이 가설은 보조 후보다. `encode_text` 소스를 보면 동시에 판별된다.

4. **vision_proj가 실제로는 멀쩡하고 collapse는 비디오 임베딩 공간의 특성** — vision_encoder 출력(768-dim) cosine=0.2824가 정상으로 보이지만, 이것도 절대적 기준에서는 낮지 않을 수 있다. 서로 다른 영상인데도 0.28이라면 이미 어느 정도 집중된 상태일 수 있고, vision_proj가 이를 512-dim으로 압축하면서 더 집중되는 것일 수 있다.

---

#### 12차 진단: get_txt_feat 소스 + text_proj hook + tokenizer 확인

```python
# A. get_txt_feat 소스
import inspect
print(inspect.getsource(model.get_txt_feat))
```

```python
def get_txt_feat(self, text: str):
    """get the text features for the given text."""
    with torch.no_grad():
        text = self.tokenizer(
            text, 
            padding="max_length", 
            truncation=True, 
            max_length=self.config.max_txt_l, 
            return_tensors="pt",).to(self.config.device)
        _, tfeat = self.encode_text(text)
        tfeat = self.text_proj(tfeat)
        tfeat /= tfeat.norm(dim=-1, keepdim=True)
    return tfeat
```

소스를 보면 흐름이 `encode_text → text_proj → L2 norm`이다. text_proj 입력이 `encode_text`의 출력이고, text_proj 출력이 최종 임베딩의 직전 단계다. collapse가 `encode_text`에서 이미 일어나는지, 아니면 `text_proj`를 통과하면서 생기는지 알려면 **text_proj 입구와 출구의 cosine을 동시에 잡아야** 한다. 소스를 수정하지 않고 이를 할 수 있는 방법이 forward hook이다. hook을 text_proj에 걸면, `get_txt_feat`를 그냥 호출하기만 해도 text_proj에 들어오는 값(encode_text 출력)과 나가는 값(proj 후)을 모두 캡처할 수 있다.

```python
# B. forward hook으로 text_proj 입력/출력 cosine 측정
with torch.no_grad():
    captured = {}
    def hook(module, input, output):
        captured['input']  = input[0].detach().clone()
        captured['output'] = output.detach().clone()
    handle = model.text_proj.register_forward_hook(hook)
    _ = model.get_txt_feat("a woman cooking food in a kitchen")
    feat_in_1  = captured['input'].clone()
    feat_out_1 = captured['output'].clone()   # 두 번째 호출이 덮어쓰기 전에 꺼냄
    _ = model.get_txt_feat("a man playing basketball outside")
    feat_in_2  = captured['input'].clone()
    feat_out_2 = captured['output'].clone()
    handle.remove()

print(f"text_proj 입력 cosine  (텍스트 인코더 출력): {cos(feat_in_1, feat_in_2).item():.4f}")
print(f"text_proj 출력 cosine  (proj 후, norm 전):   {cos(feat_out_1, feat_out_2).item():.4f}")
```

```python
# C. tokenizer 출력 확인
tok1 = model.tokenizer("a woman cooking food in a kitchen", return_tensors="pt")
tok2 = model.tokenizer("a man playing basketball outside",  return_tensors="pt")
print(f"text1 token ids: {tok1['input_ids']}")
print(f"text2 token ids: {tok2['input_ids']}")
print(f"text1 tokens:    {model.tokenizer.convert_ids_to_tokens(tok1['input_ids'][0])}")
print(f"text2 tokens:    {model.tokenizer.convert_ids_to_tokens(tok2['input_ids'][0])}")
```

결과:

```
text_proj 입력 cosine  (텍스트 인코더 출력): 0.9997
text_proj 출력 cosine  (proj 후, norm 전):   0.9957

text1 token ids: tensor([[ 101, 1037, 2450, 8434, 2833, 1999, 1037, 3829,  102]])
text2 token ids: tensor([[ 101, 1037, 2158, 2652, 3455, 2648,  102]])
text1 tokens: ['[CLS]', 'a', 'woman', 'cooking', 'food', 'in', 'a', 'kitchen', '[SEP]']
text2 tokens: ['[CLS]', 'a', 'man', 'playing', 'basketball', 'outside', '[SEP]']
```

**두 가지가 확정됐다:**

1. **tokenizer는 정상이다.** 의미있는 WordPiece 토큰이 제대로 나왔다. UNK 없음.

2. **text_proj 입력 cosine이 이미 0.9997이다.** text_proj 통과 후 0.9957로 미세하게 감소할 뿐, **collapse는 `encode_text` 내부에서 이미 발생한 상태로 text_proj에 들어온다.**

---

#### 13차 진단: 토큰별 cosine 측정 — CLS만 collapsed인가, 전체가 collapsed인가

```python
device = next(model.parameters()).device

with torch.no_grad():
    tok1 = model.tokenizer("a woman cooking food in a kitchen",
                           padding="max_length", truncation=True,
                           max_length=model.config.max_txt_l,
                           return_tensors="pt").to(device)
    tok2 = model.tokenizer("a man playing basketball outside",
                           padding="max_length", truncation=True,
                           max_length=model.config.max_txt_l,
                           return_tensors="pt").to(device)

    enc = model.get_text_encoder()
    out1 = enc(tok1.input_ids, attention_mask=tok1.attention_mask,
               return_dict=True, mode="text")
    out2 = enc(tok2.input_ids, attention_mask=tok2.attention_mask,
               return_dict=True, mode="text")

    hs1 = out1.last_hidden_state
    hs2 = out2.last_hidden_state

    print(f"CLS cosine:                            {cos(hs1[:,0,:], hs2[:,0,:]).item():.4f}")
    print(f"token[1] cosine ('a'  vs 'a'):         {cos(hs1[:,1,:], hs2[:,1,:]).item():.4f}")
    print(f"token[2] cosine ('woman' vs 'man'):    {cos(hs1[:,2,:], hs2[:,2,:]).item():.4f}")
    print(f"token[3] cosine ('cooking'/'playing'): {cos(hs1[:,3,:], hs2[:,3,:]).item():.4f}")

    # attention_mask를 이용한 mean pooling
    mask1 = tok1.attention_mask.unsqueeze(-1).float()
    mask2 = tok2.attention_mask.unsqueeze(-1).float()
    mean1 = (hs1 * mask1).sum(1) / mask1.sum(1)
    mean2 = (hs2 * mask2).sum(1) / mask2.sum(1)
    print(f"mean pooling cosine:                   {cos(mean1, mean2).item():.4f}")

    # CLS/SEP 제외 content 토큰 mean pooling
    content_mask1 = tok1.attention_mask[:, 1:-1].unsqueeze(-1).float()
    content_mask2 = tok2.attention_mask[:, 1:-1].unsqueeze(-1).float()
    cmean1 = (hs1[:, 1:-1, :] * content_mask1).sum(1) / content_mask1.sum(1)
    cmean2 = (hs2[:, 1:-1, :] * content_mask2).sum(1) / content_mask2.sum(1)
    print(f"content mean pooling cosine (CLS/SEP 제외): {cos(cmean1, cmean2).item():.4f}")
```

```
CLS cosine:                            0.9997
token[1] cosine ('a'  vs 'a'):         0.9374
token[2] cosine ('woman' vs 'man'):    0.7388
token[3] cosine ('cooking'/'playing'): 0.5657
mean pooling cosine:                   0.7811
content mean pooling cosine (CLS/SEP 제외): 0.7751
```

**결론:**

- **비CLS 토큰들은 정상이다.** 'woman' vs 'man' = 0.7388, 'cooking' vs 'playing' = 0.5657 — xbert 인코더가 텍스트 내용을 처리하고 있다.
- **CLS만 0.9997로 collapsed다.** 인코더 전체 문제가 아니라 CLS 토큰이 시퀀스 정보를 aggregation하지 못하고 있다.
- **mean pooling = 0.7811** — 개선됐지만, 이는 BERT 계열의 고질적인 anisotropy 문제(fine-tuning 없이 raw mean pooling하면 서로 다른 문장도 cosine이 0.7~0.9로 높게 나오는 현상)이지 근본 해결이 아니다. text_proj는 CLS 특징으로 학습됐으므로 mean pooling으로 교체해도 joint embedding space 정렬이 틀어진다.

---

#### 14차 진단: text encoder 타입 확인 및 max_txt_l 확인

```python
print(type(model.text_encoder))
print(type(model.get_text_encoder()))
print(f"max_txt_l = {model.config.max_txt_l}")
```

```
<class 'models.backbones.bert.xbert.BertForMaskedLM'>
<class 'models.backbones.bert.xbert.BertModel'>
max_txt_l = 40
```

- **xbert**: BLIP에서 가져온 커스텀 BERT. `mode` 파라미터로 텍스트 전용(`mode="text"`)과 cross-modal fusion(`mode="fusion"`)을 분기한다.
- `get_text_encoder()`는 `BertForMaskedLM`에서 MLM head를 제거하고 내부 `BertModel`만 반환한다.
- `max_txt_l = 40`: text1(9토큰) → 31개 PAD. 비율 22.5% 실제 토큰. **padding 비율 과다 가설 기각** (mean pooling이 0.7811인 것은 padding 때문이 아니라 BERT anisotropy 때문).

[부연] xbert는 우리 코드가 쓰는 게 아니라 InternVideo2 원본에 포함된 것이다
xbert는 InternVideo2 논문 저자들이 직접 가져다 쓴 텍스트 인코더다. BLIP에서 fork한 커스텀 BERT인데, InternVideo2 repo 안에 models/backbones/bert/xbert.py로 포함되어 있다. model.text_encoder가 xbert인 것은 우리 코드 선택이 아니라, InternVideo2를 쓰면 원래부터 xbert가 텍스트 인코더로 들어와 있는 것이다.


[부연] padding 40개 중 31개가 PAD인데 왜 문제가 아닌가
핵심은 어떤 pooling 방식을 쓰느냐다.

mean pooling을 쓰면 → PAD 토큰 31개가 평균에 희석되므로 padding이 문제가 맞다.
CLS pooling을 쓰면 → PAD가 아무리 많아도 CLS 위치 벡터 하나만 꺼내므로 padding 비율은 무관하다.

encode_text는 last_hidden_state[:, 0]으로 CLS만 꺼낸다. 따라서 padding이 40개든 400개든 CLS 결과에 직접 영향이 없다. attention 메커니즘 안에서 CLS가 PAD 토큰에 attend할 수 있지만, attention mask로 PAD는 -inf 처리되므로 그것도 아니다.


[부연] 왜 갑자기 encoder 타입과 max_txt_l을 확인했는가
13차에서 결론이 "CLS만 collapsed, 비CLS 토큰은 정상"으로 났다. 다음 질문은 자연스럽게 "CLS가 왜 이러지?"인데, 본격적으로 xbert 내부를 파고들기 전에 3줄짜리 코드로 소거할 수 있는 후보들을 먼저 처리한 것이다.

혹시 encoder가 이상한 타입이라서? → type() 확인
혹시 padding이 너무 많아서 mean pooling이 희석됐나? → max_txt_l 확인

둘 다 기각한 다음, 15차에서 본격적으로 xbert BertModel.forward 소스를 분석했다. "쉬운 가설 먼저 소거" 전략이다.

---

#### 15차 진단: xbert BertModel.forward 소스 분석

```python
import inspect
print(inspect.getsource(model.get_text_encoder().forward))
```

소스의 핵심 부분:

```python
def forward(self, ..., mode="multi_modal", normalize_attention=True):
    # ... attention mask 처리 ...
    embedding_output = self.embeddings(input_ids=input_ids, ...)

    encoder_outputs = self.encoder(
        embedding_output,
        attention_mask=extended_attention_mask,
        ...
        mode=mode,                    # ← mode가 encoder 레이어까지 전달됨
        normalize_attention=normalize_attention,
    )
    sequence_output = encoder_outputs[0]
    pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

    return BaseModelOutputWithPoolingAndCrossAttentions(
        last_hidden_state=sequence_output,
        pooler_output=pooled_output,  # ← pooler_output이 존재함
        ...
    )
```

**두 가지 새로운 발견:**

1. **`mode`가 `self.encoder`(transformer 레이어들)까지 전달된다.** xbert의 각 레이어가 `mode="text"`와 `mode="multi_modal"` 사이에서 다른 동작을 한다. mode="text"에서 CLS의 self-attention 동작이 무엇인지가 핵심이다.

2. **`pooler_output`이 존재하는데 `encode_text`가 쓰지 않는다.** 표준 BERT에서 `pooler_output = tanh(Dense(last_hidden_state[:, 0]))` 이다. 단순 raw CLS가 아니라 학습된 Dense+tanh를 거친 버전이다. 그런데 `encode_text`는:

```python
text_embeds = text_output.last_hidden_state
pooled_text_embeds = text_embeds[:, 0]   # raw CLS. pooler_output 무시
```

`text_output.pooler_output`을 버리고 `last_hidden_state[:, 0]`(raw CLS)를 직접 쓰고 있다.

---

#### 16차 진단: raw CLS vs pooler_output 비교

`encode_text`가 `last_hidden_state[:, 0]`(raw CLS)을 쓰는데, 실제로 discriminative한 표현은 `pooler_output`(tanh(Dense(CLS)))일 수 있다는 가설을 검증했다.

```python
device = next(model.parameters()).device

with torch.no_grad():
    tok1 = model.tokenizer("a woman cooking food in a kitchen",
                           padding="max_length", truncation=True,
                           max_length=model.config.max_txt_l,
                           return_tensors="pt").to(device)
    tok2 = model.tokenizer("a man playing basketball outside",
                           padding="max_length", truncation=True,
                           max_length=model.config.max_txt_l,
                           return_tensors="pt").to(device)

    enc = model.get_text_encoder()
    out1 = enc(tok1.input_ids, attention_mask=tok1.attention_mask,
               return_dict=True, mode="text")
    out2 = enc(tok2.input_ids, attention_mask=tok2.attention_mask,
               return_dict=True, mode="text")

    print(f"raw CLS cosine (encode_text 현재): {cos(out1.last_hidden_state[:,0,:], out2.last_hidden_state[:,0,:]).item():.4f}")

    if out1.pooler_output is not None:
        print(f"pooler_output cosine:              {cos(out1.pooler_output, out2.pooler_output).item():.4f}")
    else:
        print("pooler_output is None")
```

```
raw CLS cosine (encode_text 현재): 0.9997
pooler_output is None
```

`pooler_output`이 존재하지 않는다. xbert BertModel이 pooler 없이 생성됐다. 가설 기각.

---

[부연]pooler_output이 None이라는 게 무슨 의미인가

표준 BERT의 구조

일반 BERT는 이렇게 생겼어:

```
입력 토큰들
    ↓
Transformer 레이어 × N
    ↓
last_hidden_state   ← 모든 토큰의 마지막 hidden state [seq_len, 1024]
    ↓
Pooler (Dense + tanh)   ← CLS 토큰만 받아서 변환
    ↓
pooler_output       ← [1, 1024], 문장 전체를 대표하는 벡터
```

`pooler_output = tanh(W · last_hidden_state[:, 0] + b)`

즉 **pooler는 raw CLS를 한 번 더 학습된 선형변환으로 가공**해주는 레이어야. NSP(Next Sentence Prediction) 같은 문장 단위 태스크를 위해 BERT 사전학습 때 같이 학습됨.

---

[부연]xbert에서 pooler가 없는 이유

`build_bert`에서 분기를 보면:

```python
if pretrain:
    text_encoder = BertForMaskedLM.from_pretrained(...)   # MLM용
else:
    text_encoder = BertModel.from_pretrained(
        ..., add_pooling_layer=False)                      # pooler 제거
```

`is_pretrain=True`면 `BertForMaskedLM`이 생성돼. 이 클래스는 내부에 `BertModel`을 가지고 있는데, **`add_pooling_layer`를 명시하지 않으므로 기본값(True)으로 pooler가 있어야 할 것 같지만**, `get_text_encoder()`로 꺼낸 내부 BertModel에는 실제로 pooler가 없는 거야.

왜냐면 **xbert의 BertModel은 처음부터 multimodal fusion용으로 설계**됐기 때문이야. pooler가 필요 없어 — CLS pooling 대신 cross-modal attention으로 텍스트와 영상을 융합하는 게 목적이거든. 그래서 pooler 자체가 아예 안 만들어진 거야.

---

[부연]그래서 이 진단이 의미하는 것

15차에서 소스를 보니까 `pooler_output`을 반환하는 코드가 있었어:

```python
pooled_output = self.pooler(sequence_output) if self.pooler is not None else None
```

"pooler가 있으면 쓰고, 없으면 None" — 즉 **pooler가 없으면 pooler_output은 그냥 None이 돼.**

그리고 실제로 확인해보니 `pooler_output is None`이었던 거야.

결론적으로 이 진단이 말하는 건:

1. **"pooler_output을 쓰면 더 discriminative하지 않을까?"** 라는 15차의 가설 자체가 실현 불가능함 — pooler가 없으니까
2. **encode_text가 raw CLS를 쓰는 건 버그가 아니라 xbert의 정상 동작** — pooler_output이 없으니 raw CLS 외에 선택지가 없음
3. **따라서 CLS collapse의 원인을 pooler 유무에서 찾는 방향은 막힘** → 18차(oversmoothing) 방향으로 수사가 이동

---

#### 17차 진단: BertEncoder.forward 소스 분석 — mode별 실행 레이어

```python
import inspect
print(inspect.getsource(model.get_text_encoder().encoder.forward))
```

핵심 분기:

```python
if mode == "text" or mode == "temporal":
    start_layer = 0
    output_layer = self.config.fusion_layer

elif mode == "fusion":
    start_layer = self.config.fusion_layer
    output_layer = self.config.num_hidden_layers

elif mode == "multi_modal":
    start_layer = 0
    output_layer = self.config.num_hidden_layers
```

```python
print(f"fusion_layer      = {model.get_text_encoder().config.fusion_layer}")
print(f"num_hidden_layers = {model.get_text_encoder().config.num_hidden_layers}")
```

```
fusion_layer      = 19
num_hidden_layers = 24
```

`mode="text"`는 레이어 0~18(19개)를 실행한다. `mode="multi_modal"`은 0~23(24개) 전체를 실행하지만 cross-attention 레이어에서 visual feature가 없으면 AssertionError가 난다.

```
AssertionError: encoder_hidden_states must be given for cross-attention layers
```

`mode="text"`가 올바른 선택이고, 19개 레이어가 실행되는 것도 맞다. 레이어 부족 가설 기각.

---

[부연]배경: 왜 이걸 확인했나

13차에서 **CLS만 collapsed** 라는 걸 확인했어. 그러면 "왜 CLS가 collapsed인가?" 를 파고들어야 하는데, 후보 중 하나가 **"레이어를 충분히 안 돌린 거 아닌가?"** 였어.

BERT 계열은 레이어가 깊을수록 CLS에 더 많은 문맥 정보가 축적돼. 만약 19개 레이어 중 5개만 돌리고 있었다면, CLS가 제대로 문장을 표현 못 하는 게 당연하겠지. **"혹시 레이어 부족이 원인 아닌가?"** 가 17차의 가설이야.

---

[부연]xbert의 특이한 구조: 레이어가 두 종류야

일반 BERT는 레이어가 전부 동일한 self-attention이야. 그런데 xbert는 달라:

```
레이어 0~18:   self-attention만 있음   ← 텍스트 전용
레이어 19~23:  self-attention + cross-attention  ← 영상과 융합하는 레이어
```

cross-attention 레이어는 **"텍스트 토큰이 영상 토큰을 참조"** 하는 구조야. 텍스트와 영상을 같이 넣어서 멀티모달 이해를 하는 용도지.

---

[부연]mode 파라미터가 하는 일

xbert는 이 두 종류 레이어를 상황에 따라 골라 실행해. 그게 `mode` 파라미터야:

| mode | 실행 레이어 | 언제 쓰나 |
|---|---|---|
| `"text"` | 0~18 (19개) | 텍스트만 인코딩할 때 |
| `"fusion"` | 19~23 (5개) | 텍스트+영상 융합할 때 |
| `"multi_modal"` | 0~23 (24개) | 전체 |

우리는 텍스트만 인코딩하니까 `mode="text"` → 레이어 0~18, 19개 실행.

---

[부연]AssertionError가 왜 나오나

`mode="multi_modal"`을 쓰면 레이어 19~23도 실행되는데, 이 레이어들은 **영상 feature를 필수로 요구**해:

```python
# cross-attention 레이어 내부
assert encoder_hidden_states is not None, \
    "encoder_hidden_states must be given for cross-attention layers"
```

영상 없이 텍스트만 넣으면 이 assert에서 터져버려. 그러니까 `mode="text"`가 맞는 선택이야.

---

[부연]결론: 레이어 부족 가설 기각

- `mode="text"` → 레이어 19개 실행 ✅
- 19개면 fusion_layer(19) 직전까지 전부 돌리는 거라 충분함 ✅
- 레이어 부족이 CLS collapse의 원인이 아님 → **기각**

그래서 다음 18차에서 "레이어는 충분히 돌리는데, 그럼에도 CLS가 왜 collapsed인가?" 를 직접 파고든 거야 — 그게 oversmoothing 발견으로 이어지지.

---

#### 18차 진단: 레이어별 CLS attention weight — oversmoothing 확인

19개 레이어가 실행됨에도 CLS가 0.9997인 원인을 파악하기 위해 각 레이어에서 CLS(position 0)가 자기 자신에게 얼마나 attention하는지 측정했다.

```python
with torch.no_grad():
    out1_attn = enc(tok1.input_ids, attention_mask=tok1.attention_mask,
                    return_dict=True, mode="text", output_attentions=True)

    # layer 0, 9, 18 — CLS의 pos[0..9] attention 분포
    for layer_idx in [0, 9, 18]:
        attn = out1_attn.attentions[layer_idx]
        print(f"Layer {layer_idx:2d}, head 0, CLS→pos[0..9]: {attn[0, 0, 0, :10].tolist()}")
```

```
Layer  0, head 0, CLS→pos[0..9]: [0.0003, 0.4504, 0.0008, 0.0004, 0.0007, 0.0008, 0.5454, 0.0008, 0.0005, 0.0]
Layer  9, head 0, CLS→pos[0..9]: [0.1558, 0.0974, 0.0646, 0.1208, 0.1071, 0.1209, 0.1131, 0.1177, 0.1027, 0.0]
Layer 18, head 0, CLS→pos[0..9]: [0.7925, 0.0214, 0.0160, 0.0473, 0.0359, 0.0146, 0.0312, 0.0183, 0.0231, 0.0]
```

**패턴이 확인됐다.**

- **Layer 0**: CLS가 'a'(pos 1: 0.45)와 'a'(pos 6: 0.55)에 집중. 초기 레이어에서 흔한 단어에 편향.
- **Layer 9**: CLS attention이 pos[0]~pos[8]에 0.06~0.16으로 고르게 분산. **정상 작동**.
- **Layer 18**: CLS가 자기 자신(pos[0])에 **0.79**. 실제 내용 토큰들은 0.01~0.05로 무시됨.

**oversmoothing 확정.** 레이어가 깊어질수록 CLS의 self-attention이 자기 자신으로 수렴하고 있다. Layer 18에서 CLS는 입력 문장 내용과 무관하게 이전 레이어의 자기 자신만 보고 있다. 이 상태로 19레이어를 거치면 CLS 출력은 입력 문장에 무관한 상수에 수렴한다 → cosine=0.9997.

---

#### 미해결 질문: 이것이 버그인가, 의도된 설계인가

oversmoothing이 **우리 설정에서만 발생하는 버그**인지, 아니면 **InternVideo2 xbert의 의도된 동작**인지 불분명하다.

InternVideo2가 T2V retrieval R@1=51.9%를 보고했다면 어떤 방식으로든 텍스트를 discriminative하게 인코딩하고 있어야 한다. 가능성:

1. **우리 설정에서만 oversmoothing이 생긴다** — 원래 코드에서는 CLS가 정상적으로 discriminative하다. 설정(config, flash_attn stub, is_pretrain 플래그 등)이 attention 동작에 영향을 준다.
2. **InternVideo2는 retrieval에 xbert CLS를 쓰지 않는다** — `get_txt_feat`가 아닌 다른 경로로 텍스트를 인코딩한다.

---

#### 다음에 실행할 진단 코드

```python
# 모델에 어떤 텍스트 관련 메서드들이 있는지 확인
text_methods = [m for m in dir(model) if 'text' in m.lower() or 'txt' in m.lower()]
print(text_methods)
```

```python
# 레이어별 CLS self-attention 전체 추이 — oversmoothing 진행 양상 확인
with torch.no_grad():
    out_attn = enc(tok1.input_ids, attention_mask=tok1.attention_mask,
                   return_dict=True, mode="text", output_attentions=True)
    for i, attn in enumerate(out_attn.attentions):
        cls_self = attn[0, :, 0, 0].mean().item()   # 모든 head에서 CLS→CLS 평균
        print(f"Layer {i:2d}: CLS→self (avg over heads) = {cls_self:.4f}")
```

---

#### 19차 진단: 텍스트 메서드 목록 + 레이어별 CLS self-attention 전체 추이

```python
text_methods = [m for m in dir(model) if 'text' in m.lower() or 'txt' in m.lower()]
print(text_methods)
```

```
['encode_text', 'get_text_encoder', 'get_txt_feat', 'text_encoder', 'text_proj']
```

`model`이 노출하는 텍스트 관련 인터페이스는 `get_txt_feat`(공개 API), `encode_text`(내부 로직), `get_text_encoder`(xbert 추출기), `text_encoder`(BertForMaskedLM), `text_proj`(1024→512 선형) 다섯 개다. 별도의 hidden retrieval path는 없다. InternVideo2가 retrieval에서 다른 경로를 쓴다면 메서드 이름이 여기 나타났을 것인데, 나타나지 않았다. `get_txt_feat` 하나만 있다는 게 확인됐다.

---

레이어별 CLS self-attention 전체 추이:

```python
with torch.no_grad():
    out_attn = enc(tok1.input_ids, attention_mask=tok1.attention_mask,
                   return_dict=True, mode="text", output_attentions=True)
    for i, attn in enumerate(out_attn.attentions):
        cls_self = attn[0, :, 0, 0].mean().item()
        print(f"Layer {i:2d}: CLS→self (avg over heads) = {cls_self:.4f}")
```

```
Layer  0: CLS→self (avg over heads) = 0.0347
Layer  1: CLS→self (avg over heads) = 0.0563
Layer  2: CLS→self (avg over heads) = 0.0622
Layer  3: CLS→self (avg over heads) = 0.0791
Layer  4: CLS→self (avg over heads) = 0.0903
Layer  5: CLS→self (avg over heads) = 0.1147
Layer  6: CLS→self (avg over heads) = 0.1489
Layer  7: CLS→self (avg over heads) = 0.2043
Layer  8: CLS→self (avg over heads) = 0.2784
Layer  9: CLS→self (avg over heads) = 0.3517
Layer 10: CLS→self (avg over heads) = 0.4822
Layer 11: CLS→self (avg over heads) = 0.8631
Layer 12: CLS→self (avg over heads) = 0.9044
Layer 13: CLS→self (avg over heads) = 0.9218
Layer 14: CLS→self (avg over heads) = 0.9317
Layer 15: CLS→self (avg over heads) = 0.9389
Layer 16: CLS→self (avg over heads) = 0.9441
Layer 17: CLS→self (avg over heads) = 0.9482
Layer 18: CLS→self (avg over heads) = 0.9511
```

**layer 10→11 사이에 CLS self-attention이 0.48 → 0.86으로 폭등한다.** layer 11 이후로는 계속 상승해 layer 18에서 0.95에 달한다. 구조적 경계(fusion_layer=19)와는 무관하고, layer 11이라는 특정 깊이에서 갑자기 attention sink가 형성된다.

이 패턴은 **attention sink**(일부 토큰이 다른 토큰의 attention을 흡수하는 현상) 중 CLS가 자기 자신에게 sink를 형성하는 변종이다. 일반적인 attention sink는 CLS나 `[SEP]` 같은 특수 토큰이 다른 토큰들의 attention을 흡수하는 방향이지만, 여기서는 CLS가 content 토큰을 무시하고 자기 자신만 본다는 점에서 방향이 반대다. 이로 인해 layer 11 이후 CLS hidden state가 입력 시퀀스 내용과 무관한 상수에 수렴한다.

**핵심 미해결 질문**: layer 10→11에서 왜 이 전환이 발생하는가? 이것이 우리 설정에서만 발생하는 버그인가, 아니면 InternVideo2 xbert의 의도된 동작인가?

---

#### 20차 진단: has_cross_attention 레이어별 확인

cross-attention의 경계가 실제로 fusion_layer=19에서 시작하는지 확인했다.

```python
for i, layer in enumerate(model.get_text_encoder().encoder.layer):
    hca = getattr(layer, 'has_cross_attention', False)
    print(f"Layer {i:2d}: has_cross_attention = {hca}")
```

```
Layer  0: has_cross_attention = False
Layer  1: has_cross_attention = False
Layer  2: has_cross_attention = False
Layer  3: has_cross_attention = False
Layer  4: has_cross_attention = False
Layer  5: has_cross_attention = False
Layer  6: has_cross_attention = False
Layer  7: has_cross_attention = False
Layer  8: has_cross_attention = False
Layer  9: has_cross_attention = False
Layer 10: has_cross_attention = False
Layer 11: has_cross_attention = False
Layer 12: has_cross_attention = False
Layer 13: has_cross_attention = False
Layer 14: has_cross_attention = False
Layer 15: has_cross_attention = False
Layer 16: has_cross_attention = False
Layer 17: has_cross_attention = False
Layer 18: has_cross_attention = False
Layer 19: has_cross_attention = True
Layer 20: has_cross_attention = True
Layer 21: has_cross_attention = True
Layer 22: has_cross_attention = True
Layer 23: has_cross_attention = True
```

레이어 0~18은 self-attention만 가지고, 레이어 19~23은 cross-attention을 추가로 가진다. fusion_layer=19 설정과 정확히 일치한다. 아키텍처 자체는 올바르게 구성되어 있다.

**중요한 관찰**: CLS oversmoothing의 급등 지점(layer 10→11)은 구조적 경계(layer 18/19)와 전혀 무관하다. 아키텍처 설계가 아니라 self-attention 레이어들 내부의 weight가 이 동작을 만든다는 뜻이다.

---

#### 21차 진단: build_bert 소스 및 공식 BertConfig 확인

`build_bert`의 소스와 공식 BertConfig JSON을 확인해서, 모델이 어떤 config로 초기화됐는지를 검증했다.

```python
import inspect
from models.backbones.bert.builder import build_bert
print(inspect.getsource(build_bert))
```

```python
def build_bert(model_config, pretrain, checkpoint, encoder_width=None):
    bert_config = BertConfig.from_json_file(model_config.text_encoder.config)
    bert_config.fusion_layer = model_config.text_encoder.fusion_layer
    if not model_config.multimodal.enable:
        bert_config.fusion_layer = bert_config.num_hidden_layers
    if pretrain:
        text_encoder = BertForMaskedLM.from_pretrained(
            model_config.text_encoder.pretrained, config=bert_config, ...)
    else:
        text_encoder = BertModel.from_pretrained(
            model_config.text_encoder.pretrained, config=bert_config,
            add_pooling_layer=False, ...)
    return text_encoder
```

`pretrain=True`(우리 설정의 `is_pretrain=True`)이면 `BertForMaskedLM`이 반환된다. `BertForMaskedLM`은 MLM head가 달린 버전이고, `get_text_encoder()`는 `encoder.bert if hasattr(encoder, "bert") else encoder`로 내부 BertModel만 추출해 쓴다. 가중치 로딩은 정상(`missing_keys=[]`).

공식 BertConfig JSON:

```json
{
  "architectures": ["BertForMaskedLM"],
  "encoder_width": 1408,
  "fusion_layer": 19,
  "hidden_size": 1024,
  "num_attention_heads": 16,
  "num_hidden_layers": 24
}
```

`src/phase0_indexing/embedder.py`에서도:

```python
cfg.model.text_encoder.fusion_layer = 19
```

로 수동 설정하고 있어서, JSON의 `fusion_layer=19`와 완전히 일치한다. **설정 불일치 가설 기각.**

---

#### 현재까지 확인된 사실 정리

| 진단 | 결과 |
|---|---|
| 아키텍처 (fusion_layer=19, has_cross_attention 경계) | ✅ 정상 — 공식 config와 일치 |
| 가중치 로딩 (missing_keys) | ✅ 정상 — missing_keys=[] |
| 텍스트 인코딩 경로 | ✅ `get_txt_feat` 단일 경로, hidden path 없음 |
| 레이어별 CLS self-attention | 🔴 layer 11 이후 0.86~0.95, attention sink 형성 |
| CLS collapse 원인 | ❓ 미해결 — layer 10→11 전환 이유 불명 |
| oversmoothing이 버그인지 의도인지 | ❓ 미해결 — InternVideo2 원래 코드와 비교 필요 |

**남은 핵심 질문**: InternVideo2가 R@1=51.9%를 달성했다면, 우리와 다른 점이 무엇인가? 가능성:

1. **flash_attn stub 영향** — T4 Colab 호환성을 위해 주입된 flash attention stub이 attention 계산 방식을 바꿨을 수 있다. 원래 구현은 CUDA flash attention을 쓰고, stub는 `F.scaled_dot_product_attention` 등으로 fallback한다. 이 차이가 attention sink 형성에 영향줄 수 있다.

2. **is_pretrain=True 모드** — inference에는 `is_pretrain=False`로 초기화해야 하는지 확인이 필요하다. `BertModel(add_pooling_layer=False)` 경로가 pooler를 제거하는 것 외에 다른 차이가 있는지도 확인 필요.

3. **normalize_attention 파라미터** — xbert BertModel.forward의 시그니처에 `normalize_attention=True` 기본값이 있다. encode_text 호출 경로에서 이 값이 어떻게 전달되는지 확인 필요.

---

### 배제한 원인들

| 후보 | 왜 배제했는가 |
|---|---|
| 프레임 수 (4 vs 8) | 논문 Table 24a에서 #F=4 R@1=51.9, #F=8 R@1=51.9로 동일. 모델 체크포인트도 `-f4` |
| 캡션 품질 | HuggingFace 공식 annotation 사용. 논문 벤치마크와 동일 소스 |
| 모델 가중치 누락 | `load_state_dict: missing_keys=[]`. parity check 전항목 OK |
| FAISS 인덱스 타입 | 평가는 `IndexFlatIP`(exact). 프로덕션 `IVFFlat`과 독립 |
| L2 정규화 | query/video 모두 `norm≈1.0000` 확인 |
| cv2 프레임 추출 | 육안 확인 결과 정상 컬러 프레임 4장 |
| GT 커버리지 | `eval_store.size=1000`, `GT ∩ index = 1000/1000` |
| BGR 이중변환 | 수정 후 cosine 0.6576→0.6292로 소폭 개선만 됨, 주원인 아님 |
| fp16 정밀도 | f32 vs f16 cosine=1.000412, f32에서도 동일한 collapse 존재 |
| normalize 파라미터 | ImageNet 표준값(v_mean/v_std) 그대로 사용 확인 |
| vision_encoder 출력 | 두 영상 간 cosine=0.2824, 정상 판별력 확인 |
| vision_proj rank 부족 | 기각: effective rank=512/512, full-rank 행렬 |
| mean pooling + 과도한 padding | 기각: max_txt_l=40이고 encode_text는 CLS pooling 사용 확인 |
| xbert 인코더 전체 collapse | 기각: 비CLS 토큰들은 정상 (woman/man=0.7388, cooking/playing=0.5657) |
| pooler_output 오용 | 기각: pooler_output is None, pooler 자체가 없음 |
| mode="text" 레이어 부족 | 기각: fusion_layer=19, 19개 레이어 실행 확인 |
| mode 설정 오류 | 기각: mode="multi_modal"은 visual feature 없이 AssertionError 발생 → "text"가 맞음 |
| **CLS oversmoothing** | **🔴 확정: Layer 18에서 CLS self-attention=0.79, 레이어가 쌓일수록 자기 자신으로 수렴** |
| 아키텍처/config 불일치 | 기각: has_cross_attention 경계 layer 19 확인, fusion_layer=19 공식 JSON·embedder.py 일치 |
| build_bert pretrain 분기 오용 | 보류: is_pretrain=True → BertForMaskedLM이나 get_text_encoder()로 내부 BertModel 정상 추출됨. 동작 차이 추가 확인 필요 |
| flash_attn stub | ✅ 기각: BertSelfAttention.forward 소스 확인 결과 flash_attn 미사용, 순수 PyTorch matmul·softmax |
| ITC 단독 평가 방식 | **🔴 근본 원인 확정**: 공식 평가는 ITC+ITM 2단계, 우리 노트북은 ITC만 사용 |

---

#### 22차 진단: BertSelfAttention.forward 소스 분석 — flash_attn 미사용 확인

flash_attn이 built-in module로 설치돼 있어 소스를 열람할 수 없었다. 실제 attention 계산이 어디서 일어나는지 BertSelfAttention.forward를 확인했다.

```python
print(inspect.getsource(model.get_text_encoder().encoder.layer[0].attention.self.forward))
```

핵심 부분:

```python
attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
attention_scores = attention_scores / math.sqrt(self.attention_head_size)
if attention_mask is not None:
    attention_scores = attention_scores + attention_mask
attention_probs = nn.Softmax(dim=-1)(attention_scores)
```

**flash_attn이 전혀 쓰이지 않는다.** `torch.matmul → / sqrt(d) → + mask → Softmax` 순수 PyTorch다. flash_attn stub은 이 코드 경로에 관여하지 않는다. **flash_attn stub 가설 기각.**

코드 자체에도 버그가 없다. attention 계산 공식은 표준 BERT 그대로다. 따라서 attention sink는 **코드 버그가 아니라 weight 수준의 동작**이다.

---

#### 23차 진단: Softmax 이전 raw attention score 측정 — PAD sink 발견

layer 10→11에서 CLS self-attention이 0.48→0.86으로 폭등하는 원인을 파악하기 위해 softmax 이전 raw score를 직접 캡처했다.

```python
captured_scores = {}

def make_hook(i):
    def hook(mod, inp, out):
        if len(out) >= 3:
            captured_scores[i] = out[2].detach()  # attention_scores (마스크 포함, softmax 이전)
    return hook

handles = [
    model.get_text_encoder().encoder.layer[i].attention.self.register_forward_hook(make_hook(i))
    for i in [10, 11]
]

enc = model.get_text_encoder()
with torch.no_grad():
    enc(tok1.input_ids, attention_mask=tok1.attention_mask,
        return_dict=True, mode="text", output_attentions=True)

for h in handles:
    h.remove()

for i in [10, 11]:
    s = captured_scores[i][0, 0, 0, :12]  # batch 0, head 0, CLS row, pos 0..11
    print(f"Layer {i:2d} raw scores (pre-softmax, CLS→pos[0..11]): {s.tolist()}")
```

```
Layer 10 raw scores (pre-softmax, CLS→pos[0..11]):
  [6.19, 2.28, 1.12, 0.98, 0.83, 1.75, 1.98, 1.01, 5.33, -9992.0, -9992.0, -9992.0]

Layer 11 raw scores (pre-softmax, CLS→pos[0..11]):
  [2.94, 0.06, -1.13, -1.20, -1.81, -0.57, -0.28, -0.70, 0.13, -10000.0, -10000.0, -10000.0]
```

**두 가지 결정적 발견:**

**1. PAD raw score 역산** — 마스크(-10000)가 더해진 후의 값이 `-9992`이므로, 마스크 이전 원본 raw score는 `−9992 − (−10000) = +8`이다. Layer 10에서 CLS query가 `[PAD]` key와의 dot product가 **+8**이다. CLS 자기 자신(6.19)보다도 높다. 즉 **CLS가 PAD 토큰에 가장 강하게 끌리고 있는데 마스크가 이를 차단**하고 있다.

**2. Layer 11 전환** — content 토큰들의 raw score가 Layer 10의 +0.83~+2.28 범위에서 Layer 11에서 -1.81~+0.06으로 전부 음수가 됐다. PAD raw score도 ≈0으로 떨어졌다. CLS→self(2.94)만 독보적으로 높아 softmax 후 0.86이 된다.

**PAD attention sink 메커니즘 확정:**

InternVideo2 text encoder는 `[PAD]`(token_id=0) key 방향으로 CLS query를 강하게 학습했다. `[PAD]` 임베딩은 항상 동일하고, 학습 내내 loss에서 무시되는 "중립 토큰"이다. 마스크가 이를 차단하면 attention이 자기 자신(CLS)으로 수렴한다. 이것이 layer 11부터 CLS self-attention이 0.86 이상으로 치솟는 이유다.

이 패턴은 cross-modal 사전학습의 부산물이다. layer 0~18 text encoder는 layer 19~23 cross-attention에서 영상 feature를 받아 enriching되는 것을 전제로 학습됐다. mode="text"에서 영상 없이 CLS만 쓰는 단독 retrieval은 이 모델의 설계 의도가 아니다.

---

[부연]1단계: attention이 뭘 하는 건지부터

BERT 같은 트랜스포머에서 각 토큰은 문장의 다른 토큰들을 "얼마나 참고할지"를 학습해. 이게 attention이야.

예를 들어 `"a woman cooking food in a kitchen"` 이 문장에서:

```
토큰들: [CLS] a woman cooking food in a kitchen [SEP] [PAD] [PAD] ...
위치:     0   1   2      3      4    5  6    7      8     9    10
```

CLS 토큰(위치 0)은 문장 전체를 대표하는 특수 토큰이야. 나머지 토큰들을 참고해서 "이 문장이 뭔지"를 담아야 해.

attention이 하는 일은 **"CLS가 각 위치를 얼마나 볼 건지" 비율을 결정하는 것**이야. 합이 1이 되도록. 예를 들면:

```
CLS가 보는 비율:
woman:   0.30
cooking: 0.25
kitchen: 0.20
food:    0.15
...
```

이러면 CLS가 의미 있는 단어들을 골고루 참고해서 문장 표현을 만드는 거야.

---

[부연]2단계: raw score가 뭔지

저 비율(0.30, 0.25 ...)은 어떻게 결정되냐면, 먼저 각 위치마다 **점수**를 계산해. 이게 raw score야.

```
CLS → woman:   점수 2.5
CLS → cooking: 점수 2.1
CLS → kitchen: 점수 1.8
CLS → food:    점수 1.5
...
```

그다음 이 점수들을 softmax에 넣으면 합이 1인 비율로 바뀌어. 점수가 높을수록 비율이 높아지는데, **softmax는 차이를 증폭시켜.** 점수 차이가 조금만 나도 비율 차이는 크게 벌어져.

---

[부연]3단계: PAD 토큰이 뭔지

문장 길이가 제각각이니까 짧은 문장은 뒤를 `[PAD]`로 채워. max_txt_l=40이니까:

```
[CLS] a woman cooking food in a kitchen [SEP] [PAD] [PAD] ... [PAD]
  0   1   2      3      4    5  6    7     8     9    10  ...   39
```

실제 내용은 9개 토큰인데 나머지 31개가 전부 PAD야.

PAD는 "아무 의미 없는 빈칸"이야. CLS가 PAD를 참고하면 안 되니까, **마스크**를 써서 PAD 위치의 점수에 -10000을 강제로 더해. softmax에 -10000이 들어가면 확률이 사실상 0이 돼서 CLS가 PAD를 못 보게 막는 거야.

```
PAD 위치 raw score: +8
마스크 적용 후:     +8 + (-10000) = -9992
softmax 후 확률:    ≈ 0  (완전히 무시됨)
```

---

[부연]4단계: 23차에서 뭘 발견했냐

23차에서 layer 10의 raw score를 직접 꺼내보니:

```
CLS → 자기자신(pos 0): 6.19
CLS → woman(pos 2):   1.12
CLS → cooking(pos 3): 0.98
CLS → kitchen(pos 7): 1.01
CLS → PAD(pos 9):    -9992  ← 마스크 적용된 값
```

여기서 역산을 하면:

```
마스크 적용 후 = raw score + (-10000)
-9992         = raw score + (-10000)
raw score     = +8
```

**PAD 방향 raw score가 +8이야. CLS 자기 자신(6.19)보다도 높아.**

이게 무슨 의미냐면, 마스크가 없었다면 CLS는 content 토큰보다 PAD를 제일 많이 봤을 거야. 근데 마스크가 강제로 막아버리니까, 그 다음으로 높은 게 뭐냐면...

layer 11을 보면:

```
CLS → 자기자신: 2.94  (제일 높음)
CLS → woman:   -1.13
CLS → cooking: -1.20
CLS → kitchen: -0.70
```

content 토큰들이 전부 음수야. CLS 자기 자신만 2.94로 혼자 높아. softmax 넣으면 CLS→self 비율이 0.86이 돼.

**즉 CLS는 PAD를 제일 보고 싶은데 막혀있으니까, 그 다음으로 자기 자신을 보는 거야.** 레이어가 깊어질수록 이게 굳어져서 layer 18에서는 자기 자신 비율이 0.95까지 올라가.

---

[부연]5단계: 그래서 문제가 뭔데

CLS가 자기 자신만 보게 되면, CLS가 담는 정보가 **문장 내용과 무관해져.**

"a woman cooking food in a kitchen"이든 "a man playing basketball outside"든, layer 11 이후 CLS는 content 토큰을 거의 안 보고 자기 자신만 봐. 그러면 두 문장의 CLS 출력이 거의 똑같아지고, cosine similarity가 0.9997이 나오는 거야.

이게 바로 text collapse의 원인이었고, 24차에서 "애초에 이 모델은 CLS 하나로 retrieval 하도록 설계된 게 아니라 ITM(영상과 cross-attention)까지 써야 했다"는 근본 원인으로 이어지는 거야.

---

요약하면:

> CLS가 PAD한테 제일 끌리는데 마스크가 막음 → 어쩔 수 없이 자기 자신만 봄 → 레이어가 깊어질수록 심해짐 → 문장 내용을 못 담게 됨 → 어떤 문장이든 CLS가 똑같아짐 → cosine=0.9997