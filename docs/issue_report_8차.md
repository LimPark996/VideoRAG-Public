# VideoRAG ITC Collapse 원인 분석 및 해결 방향

**작성일**: 2026-04-14  
**대상**: MSR-VTT 1k-A Zero-Shot Text-to-Video Retrieval 벤치마크  
**핵심 이슈**: ITC cosine similarity collapse (≈0.9997) → pre-filter 불가 → 논문 파이프라인 재현 실패

---

## 1. 현상 요약

| 항목 | 수치 |
|------|------|
| ITC only R@1 | 3.5% |
| ITM 전체 적용 R@1 | 41.1% |
| 논문 ITC+ITM R@1 | 51.9% |
| 논문 대비 갭 | -10.8%p |

### 갭의 구조적 원인

논문 파이프라인:
```
ITC 인코딩 → top-128 후보 → ITM 재순위 → 51.9%
```

우리 파이프라인:
```
ITM 전체 1000개 직접 적용 → 41.1%
```

ITC cosine이 전부 ≈0.9997로 collapse되어 top-128 필터를 쓸 수 없는 상태.  
랜덤 순위에서 top-128 안에 정답이 들어올 확률 = 128/1000 = 12.8%이므로,  
ITC pre-filter를 강제로 쓰면 오히려 41.1% → 최대 12.8%로 하락.

---

## 2. 원인 추적 과정

### 2-1. 초기 가설 (3개)

1. `encode_text()` vs `get_txt_feat()` 호출 경로 차이
2. attention mask 미적용 가능성
3. Stage 2 학습 목표 가중치 문제

### 2-2. embedder.py 분석 → 가설 1 제거

`encode_query()` 코드 확인:

```python
# embedder.py — encode_query()
emb = self.model.get_txt_feat(text).cpu().float().numpy()
```

우리도 이미 `get_txt_feat()`을 쓰고 있었음.  
→ **"호출 경로 차이" 가설 제거.**

### 2-3. `get_txt_feat()` 내부 확인

```python
def get_txt_feat(self, text: str):
    with torch.no_grad():
        text = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.config.max_txt_l,
            return_tensors="pt",
        ).to(self.config.device)
        _, tfeat = self.encode_text(text)   # ← 두 번째 반환값
        tfeat = self.text_proj(tfeat)
        tfeat /= tfeat.norm(dim=-1, keepdim=True)
    return tfeat
```

`encode_text()`의 **두 번째 반환값**을 사용.  
→ `encode_text()` 내부 확인 필요.

### 2-4. `encode_text()` 내부 확인 → CLS 추출 구조 파악

```python
def encode_text(self, text: dict):
    text_output = self.get_text_encoder()(
        text.input_ids,
        attention_mask=text.attention_mask,
        return_dict=True,
        mode="text",
    )
    text_embeds = text_output.last_hidden_state      # [B, 40, 1024]
    pooled_text_embeds = text_embeds[:, 0]           # ← CLS 토큰만 추출
    return text_embeds, pooled_text_embeds
```

**주의 — 아래 해석은 틀림**:  
~~"attention mask는 BERT에 전달되지만, CLS를 뽑을 때는 mask를 전혀 사용하지 않음 → PAD attention sink로 오염된 CLS"~~

**올바른 해석**:  
`attention_mask`는 BERT의 각 self-attention 레이어 내부에서 이미 사용된다.  
BERT는 PAD 위치에 `-10000`을 더해 softmax 이후 가중치를 0에 수렴시키므로,  
**CLS 토큰이 PAD를 직접 attend하는 경로 자체가 차단된다**.  
따라서 `text_embeds[:, 0]`으로 CLS만 꺼낼 때 mask를 별도로 사용하지 않더라도,  
"PAD에 오염된 CLS"라는 해석은 기술적으로 성립하지 않는다.

### 2-5. CLS 벡터 완성 흐름 (토큰 → 최종 ITC 벡터)

텍스트 1개가 ITC 벡터로 변환되는 전체 경로:

```
입력 문자열 ("a man playing guitar")
    ↓  tokenizer
input_ids [1, 40]  +  attention_mask [1, 40]
    ↓  BERT 24개 레이어 (각 레이어에서 attention_mask 적용)
last_hidden_state [1, 40, 1024]   ← 모든 토큰의 컨텍스트 표현
    ↓  [:, 0]  (CLS 토큰 위치)
pooled_text_embeds [1, 1024]
    ↓  model.text_proj  (Linear 1024→512)
tfeat [1, 512]
    ↓  L2 정규화
ITC 벡터 [1, 512]
```

`text_proj`는 CLS 표현을 512차원 공간으로 선형 투영하고,  
L2 정규화로 단위 벡터를 만들어 cosine similarity 계산을 내적으로 대체한다.

### 2-6. 논문 `retrieval_utils.py` 분석 → 코드 경로 비교

```python
# retrieval_utils.py — extract_text_feats()
def extract_text_feats(texts, max_txt_l, tokenizer, model, device, return_ids=False):
    num_text = len(texts)
    text_bs = 256                          # ← 256개씩 배치 처리
    text_feats = []
    text_atts = []

    for i in range(0, num_text, text_bs):
        text = texts[i : min(num_text, i + text_bs)]
        text_input = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=max_txt_l,
            return_tensors="pt",
        ).to(device)

        text_feat = model.encode_text(text_input)[0]   # ← 첫 번째 반환값 (전체 토큰)
        text_feats.append(text_feat)
        text_atts.append(text_input.attention_mask)    # ← attention_mask도 보관

    text_feats = torch.cat(text_feats, dim=0)  # [1000, 40, 1024]
    text_atts = torch.cat(text_atts, dim=0)    # [1000, 40]
    return text_feats, text_atts
```

**`torch.cat(text_feats, dim=0)` 의미**:  
256개씩 처리한 결과 텐서 4개 ([256,40,1024] × 3 + [232,40,1024] × 1)를  
배치 차원(dim=0)으로 이어 붙여 [1000,40,1024]로 만든다.

그리고 ITC 점수 계산 시:

```python
# retrieval_utils.py — evaluation()
i2t_scores, t2i_scores = get_sim(
    model.vision_proj(pooled_image_feats),
    model.text_proj(text_feats[:, 0])      # ← 여기서 CLS 추출
)
```

**`model.text_proj(text_feats[:, 0])` 의미**:  
`text_feats` = [1000, 40, 1024]에서 `[:, 0]`은 모든 1000개 텍스트의 0번 위치(CLS)를 뽑아  
[1000, 1024]로 만든 뒤, `text_proj`(Linear 1024→512)를 통과시켜 [1000, 512]를 얻는다.

---

## 3. 현재 상태의 정확한 이해

### 논문과 우리의 구조 차이

| 항목 | 논문 (`retrieval_utils.py`) | 우리 (`embedder.py`) |
|------|---------------------------|----------------------|
| 호출 경로 | `encode_text()[0]` → 전체 토큰 저장 → `text_feats[:, 0]`으로 CLS 추출 | `get_txt_feat()` → `encode_text()[1]` → 즉시 CLS 반환 |
| 처리 단위 | 256개 배치 | 1개씩 |
| 저장 방식 | 전체 시퀀스 [N, 40, 1024] 보관 | CLS [1, 512]만 반환 |
| 최종 벡터 | 동일 (`text_proj(CLS)` → normalize) | 동일 |

**핵심**: CLS를 추출하는 수식 자체는 동일하다. 차이는 **배치 크기**와 **전체 시퀀스 보관 여부**다.

### 배치 크기가 CLS collapse의 원인인가?

기술적으로는 **아니다**. BERT self-attention은 시퀀스별로 독립 계산된다.  
`padding="max_length"` 설정 하에서는 배치 크기가 1이든 256이든  
각 시퀀스에 적용되는 패딩 개수와 attention mask 패턴은 동일하다.  
따라서 배치 크기가 달라져도 개별 시퀀스의 CLS 벡터는 이론상 변하지 않는다.

**collapse의 실제 원인은 아직 확정되지 않았다.**  
가능한 원인:
- ITC projection head의 fine-tuning이 이 배포 시나리오와 맞지 않을 가능성
- 평가 시 사용하는 체크포인트와 논문 평가 시 체크포인트의 미묘한 차이
- 영상 feature 추출 경로와 텍스트 feature 추출 경로 간 mismatch

### 그럼에도 배치 처리 전환이 권장되는 이유

논문의 `retrieval_utils.py`와 동일한 코드 경로로 전환하면:
- 혹시 있을 수 있는 미묘한 구현 차이를 제거한다
- `text_feats` 전체 시퀀스를 보관해 ITM reranking에 그대로 재사용할 수 있다
- 논문 재현 검증의 기준점이 명확해진다

---

## 4. 왜 InternVideo2를 처음부터 그대로 쓰지 못했나

### 4-1. 목표 자체가 달랐다

InternVideo2 원본은 **오프라인 배치 평가** 도구다.

```
[1000 쿼리] × [1000 영상] → 전체 유사도 행렬 → R@1 출력
```

VideoRAG가 하려는 건 **온라인 단일 쿼리 검색**이다.

```
[쿼리 1개] → FAISS 인덱스 → 관련 클립 반환 → LLM 답변 생성
```

InternVideo2의 `retrieval_utils.py`는 "1000개를 한꺼번에 평가"하도록 설계됐고,  
VideoRAG는 "쿼리 하나씩 실시간 처리"가 필요하다. 애초에 풀려는 문제가 다르다.

### 4-2. 그래서 `get_txt_feat()`이 만들어졌다

모델 클래스에 단일 쿼리용 메서드가 따로 있다.

```python
# get_txt_feat(): 쿼리 1개 → ITC 벡터 반환
emb = model.get_txt_feat(text)  # [1, 512]
```

VideoRAG의 `embedder.py`는 이걸 쓰는 게 맞다.  
**그런데 이 경로가 평가 시에도 그대로 쓰인 게 문제였다.**

### 4-3. 충돌 지점: 평가

평가(`03_evaluation.ipynb`)에서는 1000개 쿼리를 처리해야 한다.  
이때 선택지가 두 개였다.

| 선택지 | 방법 | 결과 |
|--------|------|------|
| A (우리가 한 것) | `get_txt_feat()` 1000번 반복 | 단일 문장 처리 → collapse |
| B (논문 방식) | `extract_text_feats()` 배치 처리 | 논문 재현 가능 |

`get_txt_feat()`은 RAG 파이프라인용 API인데, 평가에서도 그걸 그대로 돌려버린 것이다.

### 4-4. 본질적인 문제 -> """틀렸다."""

InternVideo2를 "그대로" 쓰려면 InternVideo2의 전체 코드베이스 구조(config, dataloader, trainer 등)가 따라와야 한다. VideoRAG는 모델만 가져다가 자체 파이프라인에 통합한 것이기 때문에, 평가 코드만 골라서 이식하는 과정에서 **"배치 평가용 API"와 "단일 쿼리용 API"를 혼용**하는 실수가 생겼다.

### 4-5. 결론 및 해결책

**핵심 실수**: RAG 파이프라인 통합 과정에서 단일 쿼리용 API(`get_txt_feat`)를 평가에도 그대로 쓴 것.  
**해결책**: 평가만큼은 논문의 `extract_text_feats()` 방식(배치 처리)을 그대로 가져오면 된다.

`03_evaluation.ipynb`에서:
- 지금: `get_txt_feat(query)` 1000번 반복
- 수정: `extract_text_feats_batch(queries, ...)` 한 번에 처리

---

## 5. 해결 방향

### 방향 A: 논문 방식 그대로 구현 (권장)

텍스트 1000개를 배치로 한꺼번에 인코딩 → `text_feats[:, 0]`으로 ITC 점수 산출.

```python
import torch
import torch.nn.functional as F

# Step 1: 텍스트 1000개 배치 인코딩 (논문 방식)
def extract_text_feats_batch(texts, max_txt_l, tokenizer, model, device, text_bs=256):
    text_feats = []
    text_atts = []
    
    for i in range(0, len(texts), text_bs):
        batch = texts[i : i + text_bs]
        text_input = tokenizer(
            batch,
            padding="max_length",
            truncation=True,
            max_length=max_txt_l,
            return_tensors="pt",
        ).to(device)
        
        with torch.no_grad():
            feat = model.encode_text(text_input)[0]  # [B, 40, 1024]
        
        text_feats.append(feat.cpu())
        text_atts.append(text_input.attention_mask.cpu())
    
    text_feats = torch.cat(text_feats, dim=0)  # [1000, 40, 1024]
    text_atts  = torch.cat(text_atts,  dim=0)  # [1000, 40]
    return text_feats, text_atts


# Step 2: ITC 점수로 top-128 후보 추출
def get_itc_topk(text_feats, image_feats_pooled, model, device, k=128):
    """
    text_feats: [1000, 40, 1024]
    image_feats_pooled: [1000, 512]  ← test_1ka_embeddings.pkl의 영상 벡터
    """
    # CLS → text_proj → 정규화
    text_cls = text_feats[:, 0].to(device)              # [1000, 1024]
    text_itc = model.text_proj(text_cls)                 # [1000, 512]
    text_itc = F.normalize(text_itc, dim=-1)             # [1000, 512]
    
    vid_itc = F.normalize(image_feats_pooled.to(device), dim=-1)  # [1000, 512]
    
    # cosine similarity matrix [1000, 1000]
    sim_matrix = text_itc @ vid_itc.T
    
    # 각 텍스트 쿼리에 대해 top-k 영상 인덱스
    topk_indices = sim_matrix.topk(k, dim=-1).indices   # [1000, 128]
    return topk_indices, sim_matrix


# Step 3: top-128 후보에 대해 ITM 재순위
def itm_rerank(text_feats, text_atts, image_feats_full, topk_indices, model, device):
    """
    text_feats: [1000, 40, 1024]
    text_atts:  [1000, 40]
    image_feats_full: [1000, 1025, 1408]  ← encode_clips_itm() 결과
    topk_indices: [1000, 128]
    """
    scores = torch.zeros(len(text_feats), len(text_feats))
    
    for i, (t_feat, t_att, topk) in enumerate(
        zip(text_feats, text_atts, topk_indices)
    ):
        t_feat = t_feat.unsqueeze(0).to(device)   # [1, 40, 1024]
        t_att  = t_att.unsqueeze(0).to(device)    # [1, 40]
        
        for j in topk.tolist():
            v_feat = image_feats_full[j].unsqueeze(0).to(device)  # [1, 1025, 1408]
            
            with torch.no_grad():
                itm_score = model.itm_head(
                    model.get_multimodal_embeds(t_feat, t_att, v_feat)
                )[:, 1]  # positive score
            
            scores[i, j] = itm_score.item()
    
    return scores
```

### 방향 B: Mean Pooling으로 CLS 우회 (대안)

CLS 대신 PAD를 제외한 토큰들의 평균을 ITC에 사용.

```python
# encode_text() 결과에서 mean pooling 적용
def mean_pool_text(text_feats, attention_mask):
    """
    text_feats: [B, 40, 1024]
    attention_mask: [B, 40]
    """
    mask = attention_mask.unsqueeze(-1).float()          # [B, 40, 1]
    mean = (text_feats * mask).sum(1) / mask.sum(1)      # [B, 1024]
    return mean
```

**방향 A 권장 이유**: 논문과 동일한 체크포인트이므로 `text_proj`가 CLS 표현으로 학습됨. 논문의 `retrieval_utils.py` 코드 경로와 완전히 일치시켜 재현 가능성을 높인다.

---

## 6. 현재 상태 및 다음 단계

| 단계 | 상태 |
|------|------|
| CLS collapse 현상 확인 | ✅ 완료 (cosine ≈ 0.9997) |
| collapse 발생 구조 파악 | ✅ 완료 (CLS 추출 흐름 및 논문과의 코드 경로 비교) |
| "PAD attention sink" 가설 재검토 | ✅ 완료 (BERT 내부 mask 적용으로 PAD→CLS 오염 경로 차단됨 — 기존 설명 수정) |
| 해결 방향 설계 | ✅ 완료 (논문 방식 배치 처리) |
| `compute_itm_topk_matrix()` 구현 | ✅ 완료 (`itm_scorer.py`에 추가) |
| `03_evaluation.ipynb` Cell 9 ITC pre-filter 적용 | ✅ 완료 (Step 1.5 추가, recall@128 진단 포함) |
| 논문 수치 (R@1 51.9%) 재현 검증 | ⬜ 진행 예정 (Colab 실행 필요) |

---

## 7. 구현 변경 사항

### `get_txt_feat()` vs 배치 인코딩 — 차이는 배치 크기만이 아니다

| | `get_txt_feat()` | `03_evaluation.ipynb` 배치 인코딩 |
|--|--|--|
| 처리 단위 | 1개 | 64개 |
| `encode_text()` 반환값 | `[1]` = CLS만 `[B, 1024]` | `[0]` = 전체 시퀀스 `[B, 40, 1024]` |
| `text_proj` + 정규화 | 함수 내부에서 즉시 적용 → `[1, 512]` 반환 | 적용 안 함 → raw BERT 출력 반환 |
| attention_mask | 버림 | 보관 → ITM fusion cross-attention에 필수 |

`get_txt_feat()`은 RAG 파이프라인에서 쿼리 하나를 실시간으로 인코딩하는 용도다.  
평가에서도 이걸 1000번 반복한 것이 문제의 출발점이었다.  
배치 인코딩은 전체 시퀀스를 들고 다니기 때문에 ITM에 바로 재사용할 수 있다.

### `itm_scorer.py` — `compute_itm_topk_matrix()` 추가

기존 `compute_itm_matrix()`는 1000×1000 전체 쌍을 계산한다.  
새로 추가한 `compute_itm_topk_matrix()`는 `topk_indices [N_txt, K]`를 받아서 처리한다.

**핵심 구조 — vid_query_map 역방향 인덱스**:

```python
# 영상 인덱스 → 이 영상이 top-K에 포함된 쿼리 목록
vid_query_map: Dict[int, List[int]] = {}
for qi in range(N_txt):
    for vi in topk_indices[qi].tolist():
        vid_query_map.setdefault(vi, []).append(qi)
```

영상 단위로 순회하되, top-K에 등장한 영상만 로드하고 해당 쿼리들에 대해서만 ITM을 계산한다.  
top-K 밖의 `(쿼리, 영상)` 쌍은 `-inf`로 초기화되어 R@k 계산 시 자동으로 최하위가 된다.

### `03_evaluation.ipynb` Cell 9 — ITC pre-filter 추가 (Step 1.5)

텍스트 배치 인코딩 이후, ITM 호출 이전에 다음 블록이 추가됐다.

```python
# video_vecs: {vid_id: ndarray[512]} — L2 정규화된 ITC 벡터 (Cell 7에서 생성)
vid_vecs = torch.tensor(np.stack([video_vecs[v] for v in ordered_vids]), dtype=torch.float32)

with torch.no_grad():
    text_cls = text_feats_all[:, 0].float().to(device)   # [N_txt, 1024]
    text_itc = iv_model.text_proj(text_cls)               # [N_txt, 512]
    text_itc = F.normalize(text_itc, dim=-1).cpu()        # [N_txt, 512]

sim_matrix   = text_itc @ vid_vecs.T                      # [N_txt, N_vid]
topk_indices = sim_matrix.topk(128, dim=-1).indices       # [N_txt, 128]

recall_128 = ...  # 정답이 top-128 안에 포함된 쿼리 비율
USE_ITC_FILTER = recall_128 >= 80.0
```

- `recall_128 >= 80%`: `compute_itm_topk_matrix()` 호출 → 전체 대비 약 8× 빠름
- `recall_128 < 80%`: ITC가 아직 collapse 상태로 판단 → `compute_itm_matrix()` 폴백

실행 시 `recall_128` 수치가 먼저 출력되므로, 배치 처리 전환 후 ITC collapse 해소 여부를 즉시 확인할 수 있다.

---

## 9. 포트폴리오 서술 요약

> InternVideo2 공식 `retrieval_utils.py` 소스 분석을 통해 논문의 ITC+ITM 2단계 평가 파이프라인 구조를 확인. 자체 구현의 ITC cosine collapse(≈0.9997) 원인을 분석하는 과정에서, BERT의 attention_mask 처리 구조(PAD 위치에 -10000 적용 → softmax 후 0에 수렴)를 추적해 기존 "PAD attention sink → CLS 오염" 가설이 기술적으로 성립하지 않음을 확인. collapse의 정확한 원인은 미확정이나, 논문과 동일한 256개 배치 처리 코드 경로로 전환해 ITC+ITM 파이프라인 재현을 진행 중.
>
> 현재: ITM 전체 적용으로 R@1 3.5% → 41.1% 개선 완료. 배치 처리 전환 후 ITC+ITM 파이프라인 재현 진행 중.

---

---

## 10. 7차~8차 전체 삽질 기록

### 출발점

> R@1 = **3.5%** (논문: 51.9%)  
> 원인을 모름. 어디가 문제인지도 모름.

---

### 1구간 — 색상 변환 버그를 고쳤는데 소용없었다 (7차 4차 진단)

**삽질 내용**: BGR 이중 변환 발견.

```
cv2.read()       → BGR
cvtColor(→RGB)   → RGB   ← 우리가 1차 변환
x[:,:,::-1]      → BGR   ← frames2tensor가 2차 변환
모델 입력: BGR        (모델은 RGB를 기대함)
```

분명히 버그였고 수정했다.

**결과**: cosine 0.6576 → 0.6292. 소폭 개선.  
마진은 오히려 0.0089 → 0.0041로 더 줄었다.  
→ BGR 이중변환은 주원인이 아니었다. 기각.

---

### 2구간 — fp16, 정규화 파라미터도 다 정상이었다 (7차 5~6차 진단)

fp16 양자화 노이즈 의심 → f32로 측정해도 동일 → 기각.  
ImageNet normalize 파라미터 확인 → 표준값 그대로 → 기각.

---

### 3구간 — 범인을 vision_proj로 잘못 특정했다 (7차 7~10차 진단)

**삽질 내용**: 단계별 쪼개기로 collapse 위치를 측정했다.

```
vision_encoder 출력 (768-dim): cosine = 0.2824  ← 정상
vision_proj 통과 후 (512-dim): cosine = 0.6273  ← collapse
```

"범인은 vision_proj다"라고 특정하고 SVD까지 했다.

```
effective rank = 512 / 512  (full-rank)
```

가중치 랭크가 멀쩡하다는 게 나와서 혼란스러워졌는데, 그때 text_proj도 측정해봤다.

```
텍스트 간 cosine: 0.9957
```

visual collapse(0.6273)보다 text collapse(0.9997)가 훨씬 심각했다.  
→ vision_proj 단독 원인이 아님. 두 projection을 동시에 무너뜨리는 systemic 원인 존재.

---

### 4구간 — is_pretrain=True 가설 (7차 11~23차 진단)

`setup_internvideo2`가 `is_pretrain=True`로 모델을 생성하는 것 발견.  
inference 동작을 바꾸는지 의심 → 소스 추적 → 실제로는 영향 없음.  
tokenizer 오작동 의심 → 정상 확인.  
다양한 가설을 하나씩 소거하면서 12~23차까지 이어졌다.

---

### 5구간 — 진짜 원인 발견: ITM이 없었다 (7차 24차 진단, 결정적)

InternVideo2 공식 `retrieval_utils.py` 소스 분석으로 2단계 파이프라인 발견.

```
1단계 ITC: text_proj(CLS) × vision_proj(pooled) → top-128 후보 선별
2단계 ITM: 텍스트 40토큰 ↔ 영상 1025토큰 cross-attention → 재순위
```

우리는 **1단계(ITC)만** 구현하고 끝냈던 것이다.  
ITC가 collapse돼도 ITM이 전체 1000개를 cross-attention으로 처리하면 51.9%가 나왔던 것.

---

### 6구간 — ITM 구현 (7차 25~26차 진단)

`itm_head`가 모델 클래스에 없다는 것 발견.  
체크포인트 로딩 시 `unexpected_keys`로 조용히 무시됐던 것이었다.  
`Linear(1024→2)` 가중치를 체크포인트에서 직접 꺼내 수동으로 탑재.

sanity check:
```
text1 + video1 (정답):  +0.50
text1 + video2 (오답):  -2.07
```

ITM 전체 1000개 적용 → **R@1 = 41.1%** (3.5%에서 +37.6%p)

---

### 7구간 — 남은 갭 -10.8%p 분석 (8차)

논문 51.9% vs 우리 41.1%. 갭의 원인: 논문은 `ITC → top-128 → ITM`인데 우리는 ITC가 collapse돼서 top-128 필터를 못 쓰고 ITM을 전체 1000개에 돌렸다.

ITC collapse 원인 추적:
- "PAD attention sink로 CLS 오염" 가설 → 틀렸다 (BERT가 내부에서 이미 mask 적용)
- "배치 크기 차이 때문" 가설 → 기술적으로 틀렸다 (BERT self-attention은 per-sequence)
- **진짜 원인**: 미확정

**해결책**: 평가 시 `get_txt_feat()` 1000번 반복 대신 논문의 배치 처리로 전환.  
ITC pre-filter가 작동하면 top-128 → ITM → 논문 수치 51.9% 재현 목표.

---

### 한 줄로

> test 영상 없음 → BGR 버그 → fp16 의심 → vision_proj SVD → is_pretrain 가설 → **ITM 없었음** → ITM 구현 → R@1 41.1% → ITC collapse 원인 추적 → 배치 처리 전환으로 51.9% 재현 시도 중
