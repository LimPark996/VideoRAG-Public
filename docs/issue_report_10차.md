# VideoRAG ITC Collapse 원인 진단 — 이슈 보고서 (10차)

**작성일**: 2026-04-27  
**대상**: ITC text embedding collapse 원인 추적  
**핵심 발견**: CLS token embedding이 모든 문장에서 동일하게 시작하며, BERT가 이를 충분히 분화시키지 못함

---

## 배경

8차 이슈 보고서에서 ITC cosine collapse(≈0.9997)의 원인이 미확정 상태로 남아있었다.  
당시 소거된 가설:
- BGR 이중 변환 → 기각
- fp16 양자화 노이즈 → 기각
- vision_proj 가중치 문제 (SVD full-rank) → 기각
- is_pretrain=True → 기각
- PAD attention sink → 기각 (BERT 내부에서 이미 mask 적용)
- 배치 크기 차이 → 기각 (BERT self-attention은 per-sequence)

**미확정으로 남은 것**: collapse가 어느 단계에서 발생하는가.

10차 진단에서는 단계별 측정 실험으로 이를 추적했다.

---

## 실험 환경

- 모델: InternVideo2-Stage2_1B-224p-f4
- 샘플: MSR-VTT 1k-A eval_pairs에서 캡션 200개 추출
- 환경: Google Colab T4

---

## 실험 1 — 영상 ITC 벡터 PCA 시각화

### 목적

영상 벡터가 collapse됐는지 시각적으로 확인.

### 방법

```python
import faiss
index = faiss.read_index(f'{INDEX_DIR}/faiss_ivfflat.index')
vecs = index.reconstruct_n(0, index.ntotal)  # production 7,010개 영상 ITC 벡터

# 별도로 1k-A 1,000개 영상 ITC 벡터
video_ids = list(video_vecs.keys())
vecs = np.stack(list(video_vecs.values()))  # [1000, 512]

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
reduced = pca.fit_transform(vecs)
plt.scatter(reduced[:, 0], reduced[:, 1], alpha=0.3, s=5)
```

### 결과

2D PCA 시각화에서 벡터들이 고르게 퍼져있음. 한 점으로 뭉치는 패턴 없음.

### 해석

```
영상 ITC 벡터 → 정상 분포 ✓
텍스트 ITC 벡터 → collapse ✗ (이전 실험에서 cosine ≈ 0.9997 확인)
```

collapse는 영상 쪽이 아닌 **텍스트 쪽에 국한**됨.

---

## 실험 2 — 텍스트 단계별 cosine 측정

### 목적

텍스트 ITC 벡터의 collapse가 어느 단계에서 발생하는지 특정.

### 방법

```python
# 200개 캡션 인코딩
tok = iv_model.tokenizer(
    sample_captions,
    padding="max_length",
    truncation=True,
    max_length=iv_model.config.max_txt_l,
    return_tensors="pt"
).to(device)

with torch.no_grad():
    feat, _ = iv_model.encode_text(tok)  # [200, 40, 1024]

n = 200

# 1. BERT mean pooling cosine
mask = tok.attention_mask.unsqueeze(-1).float()
mean_pooled = (feat * mask).sum(1) / mask.sum(1)
mean_pooled = F.normalize(mean_pooled, dim=-1)
cos_bert_mean = (mean_pooled @ mean_pooled.T)
print(cos_bert_mean.fill_diagonal_(0).sum() / (n*(n-1)))

# 2. CLS 추출 후 cosine
cls = feat[:, 0].float()
cls_norm = F.normalize(cls, dim=-1)
cos_cls = (cls_norm @ cls_norm.T)
print(cos_cls.fill_diagonal_(0).sum() / (n*(n-1)))

# 3. text_proj 통과 후 cosine
projected = iv_model.text_proj(cls)
proj_norm = F.normalize(projected, dim=-1)
cos_proj = (proj_norm @ proj_norm.T)
print(cos_proj.fill_diagonal_(0).sum() / (n*(n-1)))
```

### 결과

| 단계 | cosine 평균 | 변화량 |
|------|------------|--------|
| BERT mean pooling | 0.6738 | — |
| CLS 추출 후 | 0.9326 | **+0.2588 ← 급등** |
| text_proj 통과 후 | 0.9360 | +0.0034 (미미) |

### 해석

- mean pooling(0.6738) → CLS(0.9326): **+0.2588 급등** → collapse의 주원인
- CLS(0.9326) → text_proj(0.9360): **+0.0034** → text_proj는 거의 무관

**text_proj가 범인이 아니라 CLS 추출 단계에서 이미 collapse가 발생함.**

---

## 실험 3 — BERT 레이어별 CLS cosine 측정

### 목적

CLS가 어느 레이어부터 collapse되는지 추적.

### BertConfig 확인

```
num_hidden_layers: 24
fusion_layer: 19
```

`fusion_layer=19`에 의해 `mode="text"`에서는 layer 0~18만 실행됨 (총 19레이어).  
hidden_states는 embedding(layer 0) + 19레이어 = 20개.

### 방법

```python
bert_model = iv_model.get_text_encoder()

with torch.no_grad():
    output = bert_model(
        input_ids=tok.input_ids,
        attention_mask=tok.attention_mask,
        output_hidden_states=True,
        return_dict=True,
        mode="text"
    )

hidden_states = output.hidden_states  # 튜플 20개

for i, hs in enumerate(hidden_states):
    cls = hs[:, 0].float()
    cls_norm = F.normalize(cls, dim=-1)
    cos = (cls_norm @ cls_norm.T)
    cos.fill_diagonal_(0)
    print(f"layer {i:2d}: {cos.sum() / (n*(n-1)):.4f}")
```

### 결과

```
layer  0: 1.0000  ← token embedding 단계, 완전 동일
layer  1: 0.9957
layer  2: 0.9898
layer  3: 0.9902
layer  4: 0.9824
layer  5: 0.9681
layer  6: 0.9254  ← 최저점 (그나마 가장 구별 가능)
layer  7: 0.9545
layer  8: 0.9799
layer  9: 0.9899
layer 10: 0.9952
layer 11: 0.9969
layer 12: 0.9972
layer 13: 0.9978
layer 14: 0.9988
layer 15: 0.9987
layer 16: 0.9986
layer 17: 0.9964
layer 18: 0.9719
layer 19: 0.9330  ← 최종 출력
```

### 해석

**layer 0에서 이미 1.0000.**

CLS 토큰(id=101)은 모든 문장에서 항상 동일한 위치(position 0)에 있다.  
token embedding table에서 101번 벡터는 항상 동일하게 꺼내진다.  
따라서 layer 0에서 모든 문장의 CLS가 완전히 동일한 벡터 → cosine = 1.0000.

BERT 레이어를 거치면서 주변 토큰들을 self-attention으로 보고 조금씩 달라지는데:
- layer 6에서 0.9254까지 내려가며 잠깐 분화
- 이후 다시 0.99대로 수렴하며 분화 실패
- 최종 layer 19에서 0.9330

---

## 종합 결론

### collapse 원인 특정

```
CLS token(id=101)
  → token embedding table에서 항상 동일한 벡터로 시작 (layer 0: cosine=1.0000)
  → BERT self-attention이 주변 문맥을 보고 분화를 시도 (layer 6: cosine=0.9254)
  → 이후 다시 수렴, 분화 실패 (layer 10~: cosine=0.99대)
  → 최종 CLS: cosine=0.9326
  → text_proj: cosine=0.9360 (거의 변화 없음)
```

**collapse의 근본 원인은 CLS token embedding이 모든 문장에서 동일하게 시작하고, 이 체크포인트의 BERT가 이를 충분히 분화시키지 못하는 것이다. CLS 토큰(id=101)은 모든 문장에서 항상 같은 번호라서 token embedding부터 동일한 벡터로 시작한다. (layer 0: cosine=1.0000) BERT가 레이어를 거치면서 분화를 시도하지만 layer 6(0.9254)에서 잠깐 내려갔다가 다시 0.99대로 수렴한다. 이 체크포인트에서 ITC 파인튜닝이 충분하지 않았던 것으로 추정된다.
text_proj는 무관(+0.0034). 해결책은 CLS 대신 실제 내용 토큰들의 평균(mean pooling)을 쓰는 것이고, 이로써 cosine이 0.9997에서 0.6738로 내려간다.**

### 왜 논문은 51.9%를 냈는가

논문의 체크포인트에서는 ITC 파인튜닝이 제대로 돼서 BERT가 CLS를 문장 내용에 맞게 충분히 분화시켰을 것으로 추정된다. HuggingFace에 공개된 체크포인트가 논문 평가에 쓰인 것과 동일한지 확인할 방법이 없어 이 부분은 미확정으로 남는다.

### 왜 mean pooling이 해결책인가

mean pooling은 CLS를 쓰지 않는다.  
"a", "woman", "cooking" 같은 내용 토큰들은 문장마다 다른 token id를 가지므로 embedding부터 달라진다.  
따라서 layer 0에서부터 구별력이 있고, BERT 통과 후에도 0.6738 수준을 유지한다.

### 영상 벡터는 왜 정상인가

영상은 CLS 문제가 없다. vision encoder의 CLS는 4프레임 패치들을 self-attention으로 처리한 결과물이라 영상마다 다른 입력을 받는다. 따라서 PCA에서 정상 분포를 보였다.

---

## 수치 요약

| 측정 | cosine 평균 |
|------|------------|
| 영상 ITC 벡터 (PCA) | 정상 분포 |
| 텍스트 BERT mean pooling | 0.6738 |
| 텍스트 BERT CLS (layer 19) | 0.9326 |
| 텍스트 text_proj 통과 후 | 0.9360 |
| 텍스트 CLS token embedding (layer 0) | 1.0000 |
| 텍스트 CLS 최저점 (layer 6) | 0.9254 |

---

## 8차와의 비교

| 항목 | 8차 | 10차 |
|------|-----|------|
| collapse 위치 | 미확정 | CLS token embedding 단계로 특정 |
| text_proj 역할 | 의심 | 무관 확인 (+0.0034) |
| BERT 자체 | 의심 | layer 6까지 분화 시도하나 실패 확인 |
| 영상 벡터 | 미확인 | PCA로 정상 분포 확인 |

---

## 향후 과제

- 논문 저자의 정확한 체크포인트와 비교 실험 (현재 불가)
- layer 6에서 분화가 일어나는 이유 분석
- mean pooling ITC + ITM 결합으로 R@1 51.9% 재현 시도
