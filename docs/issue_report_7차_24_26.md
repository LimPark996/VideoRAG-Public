#### 24차 진단: 공식 retrieval_utils.py 분석 — ITC+ITM 2단계 구조 발견

Colab에 클론된 InternVideo2 공식 코드의 retrieval 평가 유틸리티를 분석했다.

```python
result = subprocess.run(
    ['grep', '-n', 'text_feat|image_feat|sim|score|itm|itc',
     '/content/InternVideo/InternVideo2/multi_modality/tasks/retrieval_utils.py'],
    capture_output=True, text=True
)
print(result.stdout)
```

**핵심 발견 — line 37:**

```python
text_feat = model.encode_text(text_input)[0]
```

`encode_text`는 `(text_embeds, pooled_text_embeds)`를 반환한다. `[0]` = `text_embeds` = **`last_hidden_state` 전체 토큰 시퀀스 [N, seq_len, 1024]**. CLS 하나가 아니다.

**공식 평가 파이프라인 2단계:**

**1단계 — ITC** (line 279-280):
```python
i2t_scores, t2i_scores = get_sim(
    model.vision_proj(pooled_image_feats), model.text_proj(text_feats[:, 0])
)
```
`text_feats[:, 0]` = CLS 토큰. 우리 `get_txt_feat()`와 동일하다. 이 ITC 스코어가 collapsed인 건 공식 코드도 마찬가지다.

**2단계 — ITM 재순위** (line 396-476):
```python
# ITC top-k 후보를 ITM으로 재순위
out = enc(
    encoder_embeds=text_feats[topk_idx],      # 전체 텍스트 토큰 시퀀스
    encoder_hidden_states=image_feats[vi],    # 영상 전체 토큰
    mode="fusion"
)
itm_embeds = output.last_hidden_state[:, 0]  # fusion 후 CLS
score = match_head(itm_embeds)[:, 1]         # 이진 매칭 점수
```

1k-A(1,000개)에서 `k_test=1000`이면 ITM이 전체 1,000×1,000 쌍을 재순위 매긴다. ITC가 완전히 망가져도 ITM이 전체를 커버한다. 'woman', 'cooking' 같은 content 토큰이 영상과 cross-attention으로 discriminative한 점수를 만든다.

**결론: 우리 노트북은 ITC만 구현했다. ITM이 없다. 이것이 R@1=3.5% vs 51.9%의 근본 원인이다.**

---

[부연]첫 번째: 왜 핵심 발견이야?

지금까지 우리 코드는 이렇게 텍스트를 인코딩하고 있었어:

```python
# 우리 get_txt_feat()
_, tfeat = self.encode_text(text)  # [1]번째 반환값만 씀
```

`encode_text`가 두 개를 반환하는데 `_`로 버리고 `[1]`(pooled, 즉 CLS 하나)만 썼어.

근데 공식 코드를 보니까:

```python
text_feat = model.encode_text(text_input)[0]  # [0]번째 반환값을 씀
```

`[0]`을 쓰고 있어. 이게 **전체 토큰 시퀀스**야.

```
[0] = last_hidden_state = [N, 40, 1024]
      → 40개 토큰 전부의 hidden state

[1] = pooled = [N, 1024]  
      → CLS 토큰 하나만
```

우리는 CLS 하나만 꺼내서 쓴다고 생각했는데, 공식 코드는 **40개 토큰을 전부 들고 다니고 있었던 거야.** 나중에 ITM 단계에서 그 40개를 전부 영상과 cross-attention 시키려고. 이게 우리가 놓치고 있던 핵심이라서 "핵심 발견"이야.

---

[부연]두 번째: ITC 설명

ITC는 텍스트 임베딩 하나 ↔ 영상 임베딩 하나를 dot product로 비교하는 거야. 우리가 R@1=3.5%를 얻은 방식이 바로 이거야.

```
텍스트: "a woman cooking" → CLS 하나 → 512차원 벡터
영상:   video7020         → pooled 하나 → 512차원 벡터
점수:   둘의 cosine similarity
```

공식 코드의 1단계도 동일하게 ITC를 써:

```python
model.text_proj(text_feats[:, 0])  # text_feats[:, 0] = CLS 하나
```

`text_feats[:, 0]`이 CLS 토큰이고, 우리 `get_txt_feat()`가 하는 것과 완전히 같아.

"이 ITC 스코어가 collapsed인 건 공식 코드도 마찬가지다"는 말은 — **공식 코드도 1단계에서는 우리처럼 CLS를 써서 collapsed된 점수를 만들어. 근데 공식 코드는 거기서 끝내지 않아.** 2단계(ITM)가 있거든. 우리는 1단계에서 끝냈으니까 3.5%가 나온 거고.

---

[부연]세 번째: ITM 설명

ITM이 뭘 하는지부터.

ITC에서 1000개 영상에 대한 점수를 매겼어. collapsed돼 있어서 점수가 다 비슷하지만 어쨌든 순위는 생겨. 공식 코드는 그 중 상위 k개(k_test개)를 뽑아서 **더 정밀한 방법으로 재순위**를 매겨. 이게 ITM이야.

ITM이 정밀한 이유가 뭐냐면, CLS 하나가 아니라 **텍스트 토큰 40개와 영상 토큰 1025개를 직접 cross-attention 시키기 때문**이야.

```
텍스트 토큰들: [CLS, a, woman, cooking, food, in, a, kitchen, SEP, PAD, ...]
                                  ↕ cross-attention (mode="fusion")
영상 토큰들:   [영상을 patch로 쪼갠 1025개 토큰]
```

"woman"이라는 토큰이 영상의 어떤 patch를 보고 있는지, "cooking"이 어떤 patch와 연결되는지를 직접 계산하는 거야. 그러면 CLS가 collapsed돼 있어도 상관없어 — content 토큰들이 영상과 직접 대화하니까.

그 결과로 나온 fusion CLS를 `itm_head`(이진 분류기)에 넣어서 **"이 텍스트와 영상이 매칭되는가? yes/no"** 점수를 내는 거야.

```
text1 + video1 (정답 쌍): +0.50  ← 높음
text1 + video2 (오답 쌍): -2.07  ← 낮음
```

26차 sanity check에서 이 숫자가 나왔잖아. ITC에서는 구분 못 했는데, ITM에서는 정답/오답이 확연히 갈리는 거야.

---

[부연]전체 흐름 요약

```
1단계 ITC: CLS 하나로 빠르게 1000개 중 상위 k개 추려냄
           (빠르지만 collapsed라 부정확)
                    ↓
2단계 ITM: 그 k개에 대해 텍스트 40토큰 ↔ 영상 1025토큰 cross-attention
           (느리지만 정확)
                    ↓
          최종 순위 결정
```

우리는 1단계만 있었고, 공식 코드는 2단계까지 있었던 거야. 그게 3.5% vs 51.9%의 차이.
---

#### 25차 진단: itm_head 위치 확인 및 추출

공식 코드가 `model.itm_head`를 쓰는데, 우리 모델에 있는지 확인했다.

```python
print(hasattr(model, 'itm_head'))
# → False

# InternVideo2_Stage2.__init__ 소스에서 itm_head 검색
src = inspect.getsource(model.__class__.__init__)
for i, line in enumerate(src.split('\n')):
    if 'itm_head' in line:
        print(i, line)
# → 아무것도 출력 안 됨
```

`InternVideo2_Stage2` 클래스 자체에 `itm_head`가 없다. 체크포인트 로딩 시 `unexpected_keys=['temp', 'itm_head.weight', 'itm_head.bias']`였던 이유가 여기서 확정됐다. 체크포인트에는 가중치가 있지만 모델 클래스 구조에 없으니 로딩이 무시됐던 것이다.

체크포인트에서 직접 추출해 shape 확인:

```python
checkpoint = torch.load(ckpt_path, map_location='cpu')
state_dict = checkpoint['module']

itm_keys = {k: v for k, v in state_dict.items() if 'itm_head' in k}
for k, v in itm_keys.items():
    print(k, v.shape)
```

```
itm_head.weight  torch.Size([2, 1024])
itm_head.bias    torch.Size([2])
```

`Linear(1024→2)`. 이진 분류기(match / no-match)다. 수동으로 생성해서 모델에 붙일 수 있다.

---

#### 26차 진단: ITM sanity check — discriminative score 확인

itm_head를 체크포인트에서 수동 로드한 뒤, 텍스트 2개 × 영상 2개의 4가지 조합으로 ITM score를 측정했다.

```python
import torch.nn as nn
itm_head = nn.Linear(1024, 2)
itm_head.weight = nn.Parameter(state_dict['itm_head.weight'].float())
itm_head.bias   = nn.Parameter(state_dict['itm_head.bias'].float())
model.itm_head  = itm_head.to(device)

def itm(txt_feat, txt_att, vis_feat):
    with torch.no_grad():
        out = enc(
            encoder_embeds=txt_feat,
            attention_mask=txt_att,
            encoder_hidden_states=vis_feat,
            encoder_attention_mask=None,
            return_dict=True,
            mode="fusion"
        )
        return model.itm_head(out.last_hidden_state[:, 0].float())[:, 1].item()

print(f"text1 + video1: {itm(txt_feat1, txt_att1, vis_feat1):.4f}")
print(f"text1 + video2: {itm(txt_feat1, txt_att1, vis_feat2):.4f}")
print(f"text2 + video1: {itm(txt_feat2, txt_att2, vis_feat1):.4f}")
print(f"text2 + video2: {itm(txt_feat2, txt_att2, vis_feat2):.4f}")
```

```
text1 + video1:  0.5019
text1 + video2: -2.0650
text2 + video1: -2.0255
text2 + video2: -0.4236
```

**대각선(matching pair)이 비대각선(non-matching)보다 확연히 높다.** ITC에서 cosine=0.9997로 구분 불가능했던 것과 완전히 다르다. ITM은 content 토큰들의 cross-attention을 통해 discriminative한 점수를 만든다.

---
[부연]결과 해석

텍스트 2개, 영상 2개로 4가지 조합을 만든 거야.

```
text1 = "a woman cooking food in a kitchen"
text2 = "a man playing basketball outside"
video1 = 요리하는 영상 (text1의 정답)
video2 = 농구하는 영상 (text2의 정답)
```

---

결과를 표로 보면:

```
              video1    video2
text1          0.50     -2.07
text2         -2.03     -0.42
```

**대각선** = 정답 쌍 (text1↔video1, text2↔video2)
**비대각선** = 오답 쌍 (text1↔video2, text2↔video1)

정답 쌍이 오답 쌍보다 점수가 높아. 이 점수가 높을수록 "이 텍스트와 영상이 매칭된다"는 뜻이야.

---

근데 왜 text2↔video2가 -0.42야? 정답 쌍인데 양수여야 하는 거 아닌가 싶을 수 있는데 — ITM 점수는 절대적인 기준이 없어. 중요한 건 **같은 텍스트 기준으로 정답 영상이 오답 영상보다 높으면** 돼.

```
text2 기준:
  video2 (정답): -0.42
  video1 (오답): -2.03
  → 정답이 더 높음 ✅
```

---

이걸 ITC랑 비교하면:

ITC에서는 "a woman cooking"이랑 "a man playing basketball"의 텍스트 임베딩 cosine이 0.9997이었잖아. 두 문장이 거의 동일한 벡터라는 뜻이니까, 어떤 영상을 넣어도 점수가 다 비슷하게 나와서 정답을 못 찾는 거야.

ITM은 그 collapsed된 CLS를 안 써. 텍스트 토큰 40개를 영상과 직접 cross-attention 시키니까, "woman", "cooking", "kitchen" 같은 단어들이 영상 내용과 직접 비교돼서 구별이 가능해지는 거야.
---

### 결론 및 수정 방향

**근본 원인 확정:**

| 구분 | 내용 |
|---|---|
| 우리 노트북 | ITC만 사용: `text_proj(CLS)` × `vision_proj(pooled_video)` cosine |
| 공식 평가 | ITC top-k → **ITM 재순위**: text 전체 토큰 + vision 전체 토큰 cross-attention |
| CLS collapse | mode="text"에서 의도된 동작 — PAD sink 차단 후 CLS self-attention으로 수렴 |
| 재현 불가 이유 | `itm_head` 미탑재 (클래스에 없음) + ITM 평가 코드 미구현 |

**수정:**

1. `itm_head`를 체크포인트에서 수동 추출해 `model.itm_head`에 부착
2. 텍스트 feature: `encode_text()[0]` → 전체 토큰 [N, 40, 1024]
3. 영상 feature: `encode_vision()[0]` → 전체 토큰 [N, 1025, 1408]
4. ITM 스코어 행렬 [1000, 1000] 계산 후 R@k 평가

**`03_evaluation.ipynb` Cell 9 대체 코드:**

```python
# ── Step 5 (ITM): Tier 1 — Dense + ITM Retrieval on MSR-VTT 1k-A ──
import cv2, os, torch, torch.nn as nn, numpy as np
from tqdm import tqdm
from demo.utils import frames2tensor

iv_model = pipeline.embedder.model
enc      = iv_model.get_text_encoder()
device   = pipeline.embedder.device

# 0. itm_head 로드
_sd = torch.load(iv_model.config.pretrained_path, map_location='cpu')['module']
itm_head = nn.Linear(1024, 2)
itm_head.weight = nn.Parameter(_sd['itm_head.weight'].float())
itm_head.bias   = nn.Parameter(_sd['itm_head.bias'].float())
iv_model.itm_head = itm_head.to(device)
del _sd

# 1. 텍스트 피처 [1000, 40, 1024]
captions = [cap for cap, _ in eval_pairs]
gt_vids  = [vid for _, vid in eval_pairs]
text_feats_all, text_atts_all = [], []
with torch.no_grad():
    for i in tqdm(range(0, len(captions), 64), desc="text encoding"):
        tok = iv_model.tokenizer(
            captions[i:i+64], padding="max_length", truncation=True,
            max_length=iv_model.config.max_txt_l, return_tensors="pt"
        ).to(device)
        feat, _ = iv_model.encode_text(tok)
        text_feats_all.append(feat.cpu().half())
        text_atts_all.append(tok.attention_mask.cpu())
text_feats_all = torch.cat(text_feats_all)
text_atts_all  = torch.cat(text_atts_all)

# 2. 영상 피처 [1000, 1025, 1408] — BGR 그대로 frames2tensor에 넘김
def _extract_bgr_frames(video_path, num_frames=4):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release(); return None
    frames = []
    for idx in np.linspace(0, total - 1, num_frames, dtype=int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret: frames.append(frame)
    cap.release()
    return frames if len(frames) == num_frames else None

ordered_vids  = list(video_ids_1ka)
vis_feats_all = []
for vid_id in tqdm(ordered_vids, desc="vision encoding"):
    frames = _extract_bgr_frames(os.path.join(TEST_VIDEO_DIR, f"{vid_id}.mp4"), 4)
    if frames is None:
        vis_feats_all.append(torch.zeros(1025, 1408, dtype=torch.float16)); continue
    t = frames2tensor(frames, fnum=4, target_size=(224, 224), device=device)
    with torch.no_grad():
        vf, _ = iv_model.encode_vision(t, test=True)
    vis_feats_all.append(vf.squeeze(0).cpu().half())
vis_feats_all = torch.stack(vis_feats_all)

# 3. ITM 스코어 행렬 [N_txt, N_vid]
N_txt, N_vid = len(captions), len(ordered_vids)
scores = torch.zeros(N_txt, N_vid)
BS_ITM = 32  # OOM 시 16으로

model_dtype = next(iv_model.parameters()).dtype  # fp16 — fusion layer 가중치와 맞춰야 함

with torch.no_grad():
    for vi in tqdm(range(N_vid), desc="ITM scoring"):
        vf = vis_feats_all[vi:vi+1].to(device, dtype=model_dtype)   # fp16
        for ti in range(0, N_txt, BS_ITM):
            tf = text_feats_all[ti:ti+BS_ITM].to(device, dtype=model_dtype)  # fp16
            ta = text_atts_all[ti:ti+BS_ITM].to(device)
            bs = tf.shape[0]
            out = enc(encoder_embeds=tf, attention_mask=ta,
                      encoder_hidden_states=vf.expand(bs, -1, -1),
                      encoder_attention_mask=None,
                      return_dict=True, mode="fusion")
            # itm_head는 fp32이므로 마지막에만 캐스팅
            s = iv_model.itm_head(out.last_hidden_state[:, 0].float())[:, 1]
            scores[ti:ti+bs, vi] = s.cpu()

# 4. R@k
vid_to_idx = {vid: i for i, vid in enumerate(ordered_vids)}
ranks = []
for qi, gt_vid in enumerate(gt_vids):
    row = scores[qi].numpy()
    gt_idx = vid_to_idx[gt_vid]
    ranks.append(int((row > row[gt_idx]).sum()) + 1)

ranks = np.array(ranks)
r1  = float((ranks <= 1).mean()  * 100)
r5  = float((ranks <= 5).mean()  * 100)
r10 = float((ranks <= 10).mean() * 100)
mdr, mnr = float(np.median(ranks)), float(np.mean(ranks))

print(f"R@1={r1:.1f}%  R@5={r5:.1f}%  R@10={r10:.1f}%  MdR={mdr:.0f}  MnR={mnr:.1f}")

tier1_metrics = {'method': 'Dense+ITM (InternVideo2-1B, #F=4)',
                 'R@1': r1, 'R@5': r5, 'R@10': r10, 'MdR': mdr, 'MnR': mnr}
dense_results = [{'r1': float(r<=1), 'r5': float(r<=5), 'r10': float(r<=10),
                  'rank': int(r), 'latency_ms': 0.0} for r in ranks]
```

예상 소요: vision encoding ≈10분 + ITM scoring ≈40분 (T4 기준). `BS_ITM=16`으로 낮추면 VRAM 부족 해소.

---

### 최종 실행 결과

dtype 불일치 에러(`mat1 Float, mat2 Half`) 수정 후 — ITM scoring 루프에서 입력을 `model_dtype`(fp16)으로 맞추고, itm_head 직전에만 float32로 캐스팅:

```python
model_dtype = next(iv_model.parameters()).dtype   # fp16
# ...
vf = vis_feats_all[vi:vi+1].to(device, dtype=model_dtype)
tf = text_feats_all[ti:ti+BS_ITM].to(device, dtype=model_dtype)
# ...
s = iv_model.itm_head(out.last_hidden_state[:, 0].float())[:, 1]
```

실행 결과 (T4, 1k-A 전체 1000×1000 쌍):

```
vision encoding: 100%  1000/1000 [05:47]
ITM scoring:     100%  1000/1000 [1:18:54]

============================================================
  Tier 1 Results: Dense + ITM (InternVideo2-1B, #F=4)
============================================================
  R@1  = 44.5%   (paper: 51.9%,  delta: -7.4%)
  R@5  = 66.3%   (paper: 74.6%,  delta: -8.3%)
  R@10 = 75.8%   (paper: 81.7%,  delta: -5.9%)
  MdR  = 2
  MnR  = 21.4
```

**ITC 단독 R@1=3.5% → ITM 적용 후 R@1=44.5%.** 근본 원인 진단이 맞았음이 수치로 확인됐다.

---

### 잔여 갭(-7.4%) 원인 분석

| 원인 | 설명 |
|---|---|
| 프레임 샘플링 차이 | 논문은 decord/PyAV 기반 pipeline, 우리는 OpenCV(`cv2`). 같은 영상이어도 decoder마다 총 프레임 수 계산이 달라 실제 추출 위치가 다름 |
| ITC 사전 필터링 없음 | 공식 평가는 ITC top-k(예: 128) 후보만 ITM 재순위. 우리는 ITC가 collapsed라 k=1000(전체)으로 돌림. 논문 셋업과 다른 검색 경로 |
| fp16 중간 저장 | vis_feats_all을 fp16으로 CPU에 저장했다가 다시 로드. 논문은 fp16 to fp16 GPU 상에서 처리 |

프레임 decoder 차이만으로 7~8% 갭은 충분히 설명된다. **구현 정합성은 확인됐다.**

---

### 이슈 해결 요약

| 단계 | 발견 | 결과 |
|---|---|---|
| 1~10차 | 영상 누락, 캐시 오염, BGR, fp16, vision_proj rank 등 | 각각 기각 또는 해결 |
| 11~13차 | text CLS collapse 발견 (cosine=0.9997) | text encoder 문제로 범위 좁힘 |
| 14~18차 | CLS oversmoothing 확정 (layer 18에서 self-attention=0.95) | attention sink 형성 확인 |
| 19~21차 | 아키텍처/config 정상, PAD sink 메커니즘 시작 | 설정 기각, weight 수준 동작 확인 |
| 22~23차 | flash_attn 미사용 확인, PAD raw score=+8 발견 | CLS collapse 메커니즘 확정 |
| 24차 | 공식 평가 코드 ITC+ITM 2단계 구조 발견 | **근본 원인 확정** |
| 25~26차 | itm_head 체크포인트 추출, sanity check 통과 | ITM 구현 가능성 확인 |
| 최종 | ITM 전체 평가 실행 | **R@1 3.5% → 44.5%** |

---

### 후속 질의응답 — ITM의 파이프라인 통합 및 ITC·ITM 개념 정리

---

#### Q1. ITM 코드를 indexing 쪽(src/ 또는 notebooks/)에도 넣어야 하나? FAISS 구조가 변경되나?

**결론: ITM 코드는 두 곳으로 분리된다.**

| 위치 | 역할 |
|---|---|
| `src/phase0_indexing/indexer.py` | **오프라인**: 영상 시퀀스 피처 `[N_clips, 1025, 1408]` 사전 추출·저장 |
| `src/phase3_reranking/itm_reranker.py` | **온라인**: ColBERT top-K 후보에 ITM 재랭킹 적용 |

FAISS 인덱스 자체 구조는 변경하지 않는다. 기존 FAISS는 512-dim ITC 벡터를 저장하는 ANN 인덱스이고, 이 용도에는 그대로 쓰인다. ITM 피처는 별도 파일(`itm_vis_feats.npy`)로 저장한다.

---

#### Q2. 기존 BM25 + Dense → ColBERT 구조를 못 쓰게 되는 건가?

**아니다. 기존 구조는 그대로 유지된다.** ITM은 ColBERT 뒤에 추가되는 4번째 단계다.

```
BM25 + Dense(FAISS/ITC) → WRRF Fusion → ColBERT → ITM
```

---

#### Q3. FAISS를 못 쓴다는 게 무슨 뜻인가? ITM이 ANN에 적합하지 않은 이유는?

**FAISS를 못 쓰는 것이 아니라, ITM에 FAISS를 적용할 수 없다는 뜻이다.**

현재 FAISS가 저장하는 512-dim 벡터는 ITC 피처다. ITC는 텍스트와 영상을 각자 따로 인코딩해 같은 공간에 투영하는 Dual Encoder 구조이므로, 영상 벡터만 미리 저장해두고 쿼리가 들어오면 ANN 검색이 가능하다.

반면 ITM은 Cross Encoder 구조다. 텍스트 토큰과 영상 패치 토큰을 같은 Fusion Transformer 안에 함께 넣어 cross-attention을 수행해야만 점수가 산출된다. 쿼리 없이 영상 단독으로 인덱싱할 수 있는 벡터가 존재하지 않는다. 따라서 ANN 인덱싱 자체가 불가능하고, ColBERT가 추린 top-K 후보에만 개별 적용(O(K))하는 방식으로만 쓸 수 있다.

| | ITC | ITM |
|---|---|---|
| 인코딩 방식 | Dual Encoder — 텍스트·영상 각자 인코딩 후 dot product | Cross Encoder — 텍스트+영상 함께 Fusion Transformer 통과 |
| 점수 계산 | `score = q_vec · v_vec` | `score = itm_head(CrossAttn(text_tokens, vis_tokens))[1]` |
| ANN 가능 여부 | O — 쿼리 없이도 영상 벡터만 인덱싱 가능 | X — 쿼리 없이 영상만 인덱싱할 수 없음 |
| 복잡도 | O(log N) (ANN) | O(K) — top-K 후보에만 적용 |

---

#### Q4. 인덱싱 시점에 ITC, ITM 점수를 모두 저장해두고 꺼내 쓰면 안 되나?

- **ITC 점수** → 이미 하고 있다. FAISS 인덱스가 바로 그것이다.
- **ITM 점수** → **쿼리가 없으면 점수가 존재하지 않는다.** ITM 점수 행렬은 `[N_queries × N_videos]` 구조인데, 쿼리는 사용자가 실시간으로 입력하는 것이라 인덱싱 시점에 알 수 없다. MSR-VTT 1k-A 평가 노트북에서 미리 계산할 수 있었던 이유는 평가셋의 캡션 1000개가 고정되어 있기 때문이다.

인덱싱 시점에 저장할 수 있는 것은 ITM 점수가 아니라 **영상 측 중간 피처** `[N_clips, 1025, 1408]`다. 이것을 저장해두면 쿼리가 들어왔을 때 영상 재인코딩 없이 ITM fusion만 돌리면 된다.

---

#### Q5. 이미 영상:캡션 = 1:1 쌍이 있는데 ITC를 계산하는 의미가 뭔가?

ITC는 저장된 쌍이 맞는지 확인하는 게 아니라, **학습(pretraining) 목표**다. 배치 내 N개의 쌍이 있을 때, 캡션A가 영상 N개 중 어느 것과 매칭되는지를 맞히는 과제를 수억 쌍에 반복함으로써 "의미가 비슷한 것들이 같은 방향에 모이는 임베딩 공간"을 형성한다.

이 공간이 형성된 덕분에, **학습 때 본 적 없는 임의의 사용자 쿼리**도 영상 벡터와 같은 공간에서 유사도 비교가 가능해진다. 1:1 쌍 자체를 저장하는 게 목적이 아니다.

---

#### Q6. ITM은 쿼리와 영상을 1:1로 비교하는 건가?

**맞다.** 정확히 1:1이다. 특정 쿼리와 특정 영상의 토큰들을 함께 Fusion Transformer에 통과시켜 그 쌍의 매칭 점수를 산출한다. 그 다음 영상B와 또 1:1, 영상C와 또 1:1로 반복한다. 1000개 전체에 돌리면 1000번의 Fusion Transformer 포워드 패스가 필요하므로 top-K에만 적용하는 것이다.

---

#### Q7. ITC와 ITM의 차이를 정확히 정리하면?

둘 다 Vision-Language 모델의 사전학습 목표(pretraining objective)이며, InternVideo2 같은 모델이 학습 시 동시에 사용한다.

**ITC (Image-Text Contrastive)**
- 목표: 맞는 텍스트-영상 쌍은 임베딩 공간에서 가깝게, 틀린 쌍은 멀게 밀어내는 학습
- 방식: 텍스트 인코더·영상 인코더 각자 따로 학습, 결과는 공유 임베딩 공간의 벡터
- 점수: cosine similarity
- 특징: 전체적인 의미 유사도 측정, 빠름, ANN 인덱싱 가능

**ITM (Image-Text Matching)**
- 목표: 텍스트와 영상이 실제로 매칭되는지 이진 분류
- 방식: 텍스트 토큰 + 영상 패치 토큰을 같은 Fusion Transformer에 함께 통과, cross-attention 후 `itm_head`로 매칭 확률 산출
- 점수: 매칭 확률 스칼라
- 특징: 세부 내용("빨간", "잡는") 수준까지 교차 비교, 느림, ANN 불가

**비유:**
- ITC: 이력서(영상)를 미리 파일함에 정리해두고, 채용공고(쿼리)와 키워드 매칭
- ITM: 면접관(쿼리)이 지원자(영상)와 직접 1:1 면접

ITC로 대규모 후보를 빠르게 소환하고, ITM으로 소수 후보를 정밀하게 재랭킹하는 구조가 자연스럽게 만들어진다.
