# VideoRAG 02/03 노트북 이슈 보고서

**작성일**: 2026-03-15
**영향 범위**: `02_demo.ipynb`, `03_evaluation.ipynb`, `src/pipeline.py`, `src/phase0_indexing/embedder.py`
**상태**: 수정 완료 (GitHub push 대기)

---

## 이슈 #1 — 02/03 노트북 환경 설정 누락

**증상**: 02_demo.ipynb, 03_evaluation.ipynb 실행 시 모듈 import 실패, 인덱스 로드 실패 등 다수 에러.

**원인**: 01_indexing.ipynb를 수정하면서 InternVideo2 sparse checkout, TransNetV2 설치, open_clip_torch, GitHub src clone, HF_TOKEN 설정, sys.path 등록 등이 추가됐는데 02/03에는 반영되지 않음.

**조치**: 02/03 노트북의 Step 0을 01과 동일한 구조로 전면 재작성.

- Google Drive 마운트
- GitHub에서 최신 `src/` clone
- `open_clip_torch`, `easydict`, `TransNetV2` 등 의존성 설치
- `sys.path` 등록 (프로젝트, TransNetV2)
- `HF_TOKEN` 환경변수 설정
- 인덱스 존재 확인 및 Drive 복원

---

## 이슈 #2 — embed_dim 1024 → 512

**증상**: FAISS 인덱스 로드 시 차원 불일치, 검색 결과 비정상.

**원인**: InternVideo2-Stage2_1B-224p-f4 모델의 실제 임베딩 차원은 **512**인데, 02/03 노트북과 `pipeline.py`에서 기본값이 **1024**로 설정되어 있었음.

```
01_indexing에서 생성된 인덱스: 512차원
02/03에서 로드 시 기대 차원:  1024차원  ← 불일치
```

**조치**:

- 02/03 노트북: `config={'embed_dim': 512}`로 변경
- `pipeline.py` 55행: `FAISSVectorStore` 기본 dim `1024` → `512`
- 02/03 노트북에 남아있던 중복 Step 1 셀 (embed_dim: 1024) 삭제

---

## 이슈 #3 — BertPreTrainedModel 패치 WARNING

**증상**: `WARNING: BertPreTrainedModel 패치 실패: No module named 'models'`

**원인**: `embedder.py`의 `_patch_transformers_tied_weights()`가 InternVideo 미설치 상태에서 `from models.backbones.bert.xbert import BertPreTrainedModel`을 시도. 이 런타임 패치는 **보조 안전망**이고, `_patch_internvideo2_source()`가 이미 xbert.py 소스 파일에 `_tied_weights_keys = []`를 직접 삽입하기 때문에 런타임 패치 실패는 무해함.

**조치**:

- `os.path.isdir(models_dir)` 선확인 추가: 경로 없으면 import 시도 자체를 스킵
- 로그 레벨 `WARNING` → `DEBUG`로 변경
- 안내 메시지: "패치 실패" → "소스 패치로 대체 (생략)"

---

## 이슈 #4 — transformers 4.48+ `get_head_mask` 호환

**증상**: `AttributeError: 'BertModel' object has no attribute 'get_head_mask'`

**원인**: Colab의 `transformers` 패키지가 4.48+로 업데이트되면서 `PreTrainedModel.get_head_mask()` 메서드가 제거됨. InternVideo2의 xbert.py는 구버전 기준으로 `self.get_head_mask()`를 호출.

**배경**: embedder.py에는 이미 transformers 호환 패치 블록이 있었음 (`apply_chunking_to_forward`, `find_pruneable_heads_and_indices`, `prune_linear_layer`). `get_head_mask`만 누락.

**조치**: xbert.py 호환 패치 블록에 `get_head_mask` 구현 추가.

```python
if not hasattr(_tf.PreTrainedModel, 'get_head_mask'):
    def _compat_get_head_mask(self, head_mask, num_hidden_layers, ...):
        if head_mask is not None:
            # head_mask 차원 변환 (1D→5D, 2D→5D)
            ...
        else:
            head_mask = [None] * num_hidden_layers
        return head_mask
    _tf.PreTrainedModel.get_head_mask = _compat_get_head_mask
```

`hasattr` 체크를 하기 때문에 transformers가 이 메서드를 다시 포함하는 버전에서는 패치가 자동 스킵됨.

---

## 이슈 #5 — `No module named 'demo'` (Python import 캐시 오염)

**증상**: InternVideo Step 0.5을 실행했는데도 `ModuleNotFoundError: No module named 'demo'` 발생. 2.8GB 체크포인트를 다 다운로드한 후에야 에러 발생.

**원인 — Python의 import 캐시 메커니즘**:

Python은 `import`를 할 때, 매번 디렉토리를 새로 스캔하면 느리니까 "이 경로에 뭐가 있더라"를 내부적으로 캐싱한다.

```
Step 0:   sys.path에 '/content/InternVideo/.../multi_modality' 등록
          → Python이 이 경로를 스캔: "demo 폴더 없음" 캐싱 ❌

Step 0.5: InternVideo clone → demo 폴더 생김 ✓

Step 1:   from src.pipeline import ... → 내부적으로 경로 재탐색
          → 캐시에 "demo 없음" 남아있음

Step 2:   from demo.utils import ... → 캐시 보고 "없음" → 에러 💥
```

실제로는 파일이 있는데, 캐시가 "없다"고 기억하고 있어서 못 찾는 것.

01_indexing에서는 이 문제가 안 일어났던 이유: clone(Step 0.5) 후 첫 import(Step 3 build_index)까지 사이에 해당 경로로의 import 시도가 없었기 때문에 캐시 오염이 발생하지 않음.

**조치 3가지**:

① **노트북 Step 0** — InternVideo 관련 `sys.path.insert` 줄 삭제. clone 전에 등록하는 게 문제의 시작이니까 아예 제거.

② **노트북 Step 0.5** — clone 완료 **후에** `sys.path` 등록하고, 바로 뒤에 캐시 강제 초기화.

```python
# clone 완료 후
sys.path.insert(0, INTERNVIDEO_PATH)
importlib.invalidate_caches()  # "캐시 다 지워, 다시 스캔해"
```

③ **embedder.py** — 코드 내부에서 `sys.path` 등록할 때도 똑같이 `invalidate_caches()` 추가. 노트북 셀 순서가 꼬여도 방어.

**한 줄 요약**: "파일 생기기 전에 경로 등록해서 '없음'이 캐싱됐으니, clone 후로 등록 시점을 옮기고 캐시도 강제로 초기화."

---

## 이슈 #6 — Colab에 수정 코드가 반영 안 됨

**증상**: 위 수정을 전부 적용했는데 Colab에서 여전히 `No module named 'demo'` 발생.

**원인**: Colab의 Step 0이 매번 GitHub에서 `src/`를 clone해서 로컬 코드를 덮어씀. 로컬에서 embedder.py를 아무리 수정해도 **GitHub에 push 안 하면 Colab에는 옛날 코드**가 들어감.

```
Colab Step 0:
  git clone https://github.com/.../VideoRAG-Prototype.git
  cp -r /tmp/VideoRAG-Prototype/src → /content/videorag_prototype/src
  ↑ GitHub의 옛 버전이 복사됨
```

**추가 조치**: `_load_internvideo2()` 맨 앞에 사전 검증 추가. InternVideo의 `demo/`와 `models/` 디렉토리 존재 여부를 먼저 확인하고, 없으면 2.8GB 다운로드 없이 즉시 에러 + 해결 방법 안내.

```python
demo_dir = os.path.join(INTERNVIDEO2_CODE_PATH, "demo")
models_dir = os.path.join(INTERNVIDEO2_CODE_PATH, "models")
if not os.path.isdir(demo_dir) or not os.path.isdir(models_dir):
    raise ModuleNotFoundError(
        "InternVideo2 코드가 없습니다.\n"
        "→ 노트북의 Step 0.5 (InternVideo sparse checkout) 셀을 먼저 실행하세요."
    )
```

---

## 수정 파일 목록

| 파일 | 수정 내용 |
|------|-----------|
| `src/pipeline.py` | FAISSVectorStore 기본 dim `1024` → `512` |
| `src/phase0_indexing/embedder.py` | InternVideo 사전 검증, `get_head_mask` 호환 패치, `invalidate_caches()`, BertPreTrainedModel 패치 안정화 |
| `02_demo.ipynb` | Step 0 재작성, Step 0.5 추가, 중복 셀 삭제, embed_dim 512 |
| `03_evaluation.ipynb` | Step 0 재작성, Step 0.5 추가, 중복 셀 삭제, embed_dim 512 |

---

## 실행 순서 (수정 후)

```
Step 0   : 환경 부트스트랩 (Drive, GitHub src, 의존성, sys.path, HF_TOKEN)
Step 0.5 : InternVideo sparse checkout → sys.path 등록 → invalidate_caches()
Step 1   : 파이프라인 초기화 (embed_dim=512)
Step 2+  : 검색 / 평가 실행
```
