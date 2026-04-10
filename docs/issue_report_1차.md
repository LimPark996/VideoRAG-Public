# 이슈 보고서: InternVideo2 모델 로딩 실패 및 해결

작성일: 2026-03-14 (10차 에러까지 전면 반영: 2026-03-14)  
대상 파일: `src/phase0_indexing/embedder.py`, `src/phase0_indexing/vector_store.py`, `notebooks/01_indexing.ipynb`  
실행 환경: Google Colab Free Tier (NVIDIA T4 16GB, Python 3.12, CUDA 12.8, PyTorch 2.10)

## 1. 어떤 상황에서

VideoRAG 프로토타입의 Phase 0(인덱싱) 단계에서, MSR-VTT 영상 1,000개를 512차원 벡터로 임베딩하기 위해 InternVideo2-1B 모델을 Google Colab Free T4 환경에서 로드하려 했다.

InternVideo2를 채택한 이유는 연구개발계획서에 해당 모델 사용을 명시했기 때문이며, CLIP 등 이미지 단독 모델이 아닌 영상의 시간축(temporal) 정보까지 인코딩하는 비디오 전용 모델이 필요했다.

모델 로드 경로는 다음과 같다.

```
[HuggingFace]                      [GitHub sparse checkout]
체크포인트(.pt) 다운로드     +     소스 코드(multi_modality/) 클론
         ↓                                  ↓
    hf_hub_download()              sys.path에 등록 후 import
         ↓                                  ↓
         └──────────── setup_internvideo2(cfg) ───────────┘
                              ↓
                    InternVideo2-1B 모델 인스턴스
```

구체적으로, `embedder.py`의 `_load_internvideo2()` 메서드가 다음 순서로 동작한다.

1. GitHub에서 sparse checkout한 `multi_modality/` 폴더를 `sys.path`에 등록
2. HuggingFace에서 `.pt` 체크포인트 파일 다운로드
3. `demo/utils.py`의 `setup_internvideo2(cfg)` 함수를 호출하여 모델 생성 및 가중치 로드


## 2. 무엇을 하려다가

`embedder.load_model()`을 호출하여 InternVideo2-1B 모델을 메모리에 올리고, 이후 `encode_clips()`와 `encode_query()`로 영상/텍스트 임베딩을 생성하려 했다.

노트북(`01_indexing.ipynb`)의 cell-6에서 아래 코드를 실행했다.

```python
from src.phase0_indexing.embedder import VideoEmbedder
embedder = VideoEmbedder()
embedder.load_model()
print(embedder._model_type)  # "internvideo2" 가 출력되어야 성공
```


## 3. 어떤 것이 발생했는가

총 3개의 에러가 순차적으로 발견되었다. 앞의 에러를 해결하면 그 뒤에 숨어 있던 다음 에러가 나타나는 구조이다.

### 3-1. 1차 에러: relative import 실패

```
ImportError: attempted relative import beyond top-level package
```

전체 traceback은 다음과 같다.

```
from demo.utils import setup_internvideo2
  → demo/utils.py line 11: from models.criterions import get_sim
    → models/criterions.py line 9: from ..utils.distributed import get_rank, get_world_size
      → ImportError: attempted relative import beyond top-level package
```

`demo/utils.py`를 import하는 순간, 그 안에서 `models.criterions`를 import하고, `models/criterions.py`가 `from ..utils.distributed import ...`라는 relative import를 시도하다 실패했다.

InternVideo2 로드가 실패하면 `embedder.py`의 fallback 로직에 의해 자동으로 CLIP ViT-B/32로 전환되었다.

```
WARNING: InternVideo2 로드 실패: attempted relative import beyond top-level package
→ CLIP ViT-B/32 폴백 모드로 전환
→ embedder._model_type = "clip"    ← InternVideo2가 아님
```

### 3-2. 2차 에러 (1차 해결 후): flash_attn import 실패

1차 에러를 해결하고 나면, InternVideo2의 backbone 코드(`models/backbones/internvideo2/` 내부)에서 `flash_attn` 패키지를 모듈 레벨로 import하면서 다시 실패한다.

```
ModuleNotFoundError: No module named 'flash_attn'
```

`flash_attn`(Flash Attention)은 CUDA 커널로 컴파일되는 C++ 확장 패키지인데, Colab Free T4의 CUDA/PyTorch 버전에 맞는 사전 빌드 wheel이 존재하지 않아 `pip install flash-attn`이 실패한다.

```
pip install flash-attn
→ ERROR: Could not find a version that satisfies the requirement flash-attn
→ (no matching wheel for cu128 + torch 2.10 + python 3.12)
```

### 3-3. 3차 에러 (1차+2차 해결 후): flash_attn 더미 모듈의 __spec__ 누락

2차 에러를 `sys.modules` 더미 주입으로 우회했더니, 이번에는 `transformers` 라이브러리가 `flash_attn` 사용 가능 여부를 검사하는 과정에서 새로운 에러가 발생했다.

```
ValueError: flash_attn.__spec__ is None
```

전체 traceback 체인은 다음과 같다.

```
from demo.utils import setup_internvideo2
  → models/__init__.py line 1: from .internvideo2_clip import InternVideo2_CLIP
    → internvideo2_clip.py line 10: from .backbones.internvideo2 import LLaMA, Tokenizer
      → internvideo2_clip_text.py line 5: from peft import get_peft_model, LoraConfig
        → peft/__init__.py → peft/config.py → peft/utils/other.py
          → from transformers import PreTrainedModel
            → transformers/modeling_utils.py line 70: from .integrations.flash_attention import ...
              → flash_attention.py line 9: flash_attn_supports_top_left_mask()
                → is_flash_attn_2_available()
                  → importlib.util.find_spec("flash_attn")
                    → ValueError: flash_attn.__spec__ is None
```

이 에러의 발생 원인은 두 가지가 결합된 것이다.

첫째, `models/__init__.py`가 `InternVideo2_CLIP`, `LLaMA`, `Tokenizer` 등 우리에게 불필요한 모델 클래스를 전부 import한다. 이 중 `internvideo2_clip_text.py`가 `peft`(Parameter-Efficient Fine-Tuning) 라이브러리를 import하고, `peft`가 다시 `transformers`를 import하면서 긴 의존성 체인이 발동된다.

둘째, `transformers` 라이브러리(v4.50+)는 모듈 로드 시 `flash_attn` 패키지의 존재 여부와 버전을 자동으로 검사한다. 이때 `importlib.util.find_spec("flash_attn")`을 호출하는데, Python 3.12에서는 `sys.modules`에 등록된 모듈의 `__spec__` 속성이 `None`이면 `ValueError`를 발생시킨다. 최초의 더미 모듈은 `types.ModuleType()`으로 생성했기 때문에 `__spec__`이 자동으로 `None`이었다.

### 3-4. 4차 에러 (1차+2차+3차 해결 후): transformers API 변경에 의한 import 실패

1~3차 에러를 모두 해결한 뒤 다시 실행하면, `demo/utils.py`가 `models.backbones.bert.builder`를 import하고, 그 안의 `xbert.py`가 `transformers.modeling_utils`에서 제거된 함수를 import하면서 실패한다.

```
ImportError: cannot import name 'apply_chunking_to_forward' from 'transformers.modeling_utils'
```

전체 traceback 체인은 다음과 같다.

```
from demo.utils import setup_internvideo2
  → demo/utils.py line 10: from models.backbones.bert.builder import build_bert
    → builder.py line 1: from .xbert import BertConfig, BertModel, ...
      → xbert.py line 43: from transformers.modeling_utils import (
            PreTrainedModel, apply_chunking_to_forward,
            find_pruneable_heads_and_indices, prune_linear_layer)
        → ImportError: cannot import name 'apply_chunking_to_forward'
```

InternVideo2의 `xbert.py`는 `transformers ~4.30` 기준으로 작성되었는데, 현재 Colab에 설치된 `transformers 4.50+`에서는 `apply_chunking_to_forward`, `find_pruneable_heads_and_indices`, `prune_linear_layer` 세 함수가 `transformers.modeling_utils`에서 `transformers.pytorch_utils`로 이동되었다.

### 3-5. 5차 에러 (1~4차 해결 후): transformers 4.50+에서 pytorch_utils에서도 함수 제거

4차 에러를 `transformers.pytorch_utils`에서 가져오는 방식으로 해결했더니, `find_pruneable_heads_and_indices`가 `pytorch_utils`에서도 제거된 상태임이 드러났다.

```
ImportError: cannot import name 'find_pruneable_heads_and_indices' from 'transformers.pytorch_utils'
```

transformers 버전별 함수 위치 변화:

```
transformers ~4.30  : modeling_utils에 3개 함수 모두 존재 (InternVideo2 작성 시점)
transformers ~4.40  : pytorch_utils로 이동
transformers 4.50+  : find_pruneable_heads_and_indices, prune_linear_layer이
                      pytorch_utils에서도 제거됨 (Colab 현재 버전)
```

즉, 단순히 `modeling_utils` → `pytorch_utils`로 import 경로를 바꾸는 것으로는 부족하다. 함수 자체가 라이브러리에서 사라졌기 때문이다.

### 3-6. 6차 에러 (1~5차 해결 후): 커스텀 BertTokenizer의 내부 API 의존

모든 import 에러를 해결하고 `from demo.utils import setup_internvideo2`를 실행하면, `demo/utils.py`가 InternVideo2의 커스텀 BertTokenizer를 import하면서 다시 실패한다.

```
ImportError: cannot import name '_is_control' from 'transformers.tokenization_utils'
```

전체 traceback 체인:

```
from demo.utils import setup_internvideo2
  → demo/utils.py line 13: from models.backbones.bert.tokenization_bert import BertTokenizer
    → tokenization_bert.py line 23: from transformers.tokenization_utils import (
          PreTrainedTokenizer, _is_control, _is_punctuation, _is_whitespace)
      → ImportError: cannot import name '_is_control'
```

InternVideo2의 `models/backbones/bert/tokenization_bert.py`는 HuggingFace transformers의 옛 BertTokenizer 코드를 그대로 복사해온 파일이다. 이 파일이 `_is_control`, `_is_punctuation`, `_is_whitespace`라는 transformers 내부(private) 유틸리티 함수를 직접 import하는데, transformers 4.50+에서 이 함수들이 `tokenization_utils`에서 제거되었다.

### 3-7. 7차 에러 (1~6차 해결 후): 빈 문자열 경로 및 상대 경로 파일 접근 실패

모든 import 에러를 해결하고 `setup_internvideo2(cfg)`를 호출하면, 모델 생성 과정에서 파일 접근 에러가 발생한다.

```
[Errno 2] No such file or directory: ''
```

이 에러는 두 가지 config 설정 문제가 결합된 것이다.

첫째, `cfg.model.vision_encoder.pretrained = ''` (빈 문자열)로 설정했는데, InternVideo2의 backbone 초기화 코드가 `pretrained is not None` 체크를 사용한다. 빈 문자열은 Python에서 falsy이지만 `None`이 아니므로 체크를 통과하여 `open('')`을 시도한다.

둘째, `cfg.model.text_encoder.config = "configs/config_bert_large.json"` — 이 상대 경로는 InternVideo2 코드 디렉토리(multi_modality/) 기준이다. 그런데 노트북 cell-4에서 `os.chdir(PROJECT_ROOT)`으로 CWD를 `/content/videorag_prototype`으로 바꿔놨기 때문에, 상대 경로가 잘못된 위치를 가리킨다.

### 3-8. 8차 에러 (1~7차 해결 후): BertPreTrainedModel의 all_tied_weights_keys 속성 누락

모든 import 에러와 config 에러를 해결하고 `setup_internvideo2(cfg)`를 호출하면, 모델 인스턴스 생성 과정에서 BERT 텍스트 인코더 초기화 시 AttributeError가 발생한다.

```
AttributeError: 'BertForMaskedLM' object has no attribute 'all_tied_weights_keys'
```

전체 traceback 체인:

```
setup_internvideo2(cfg)
  → InternVideo2_Stage2 모델 생성
    → build_bert(cfg.model.text_encoder)
      → xbert.py: BertForMaskedLM.__init__()
        → PreTrainedModel.__init__() (transformers 베이스 클래스)
          → post_init() → mark_tied_weights_as_initialized()
            → self.all_tied_weights_keys  ← AttributeError
```

이 에러의 원인은 7차까지의 에러들(import 실패, config 설정 오류)과는 성격이 다르다. 이전 에러들은 모델 코드를 "로드하는 과정"에서 발생했지만, 이 에러는 모델 클래스를 실제로 "인스턴스화하는 과정"에서 발생한다. 즉, 모든 import가 성공하고 코드가 정상적으로 로드된 뒤에야 비로소 드러나는 런타임 호환성 문제이다.

### 3-9. 9차 에러 (1~8차 해결 시도 후): xbert.py 파일 패치가 실제로 동작하지 않음

8차 에러 해결을 위해 `_patch_internvideo2_source()`에서 `xbert.py`의 `BertPreTrainedModel` 클래스 정의에 `_tied_weights_keys = []`를 삽입하는 파일 패치를 추가했다. 그러나 같은 에러가 계속 발생했다.

```
WARNING: InternVideo2 로드 실패: 'BertForMaskedLM' object has no attribute 'all_tied_weights_keys'
→ CLIP ViT-B/32 폴백 모드로 전환
```

xbert.py 파일을 직접 확인하면 패치는 정상적으로 기록되어 있다. 즉 **파일 수정은 성공했지만 런타임에 효과가 없는 상태**이다.

### 3-10. 10차 에러 (9차 해결 시도 후): monkey-patch가 조기 종료됨

9차 에러 해결을 위해 `PreTrainedModel.__init__`을 monkey-patch하는 `_patch_transformers_tied_weights()` 메서드를 추가했다. 그러나 에러가 계속 발생했다.

진단 결과:

```python
from transformers import PreTrainedModel
print(hasattr(PreTrainedModel, 'all_tied_weights_keys'))  # → False
print(getattr(PreTrainedModel, '_videorag_patched', False))  # → False
```

`all_tied_weights_keys`가 `PreTrainedModel`에 **property로 존재하지 않는다**는 것이 확인됐다. 따라서 monkey-patch 코드의 조기 종료 조건에 걸려 패치가 실행되지 않았다.

```python
if not hasattr(PreTrainedModel, 'all_tied_weights_keys'):
    return  # ← 여기서 종료 — 패치 실행 안 됨
```

실제 traceback을 확인하니 에러 위치도 달랐다:

```
BertForMaskedLM.from_pretrained("bert-large-uncased")
  → modeling_utils.py: from_pretrained()
    → _finalize_load_state_dict()       ← __init__ 이후의 후처리 단계
      → model.mark_tied_weights_as_initialized()
        → self.all_tied_weights_keys.keys()  ← AttributeError
```

`__init__`이 아니라 `from_pretrained()`의 후처리 단계에서 발생하고 있었다. `all_tied_weights_keys`는 `PreTrainedModel`의 property가 아니라 `BertForMaskedLM` **인스턴스**에 직접 있어야 하는 속성이었다.

### 3-11. 부수적 문제: 차원 불일치

```
AssertionError: 벡터 차원 불일치: 512 != 1024
```

`FAISSVectorStore(dim=1024)`로 초기화했는데 실제 임베딩이 512차원이어서 발생했다. (`dim=512`로 수정하여 해결 완료)

## 4. 원인은 무엇인가

### 4-1. 1차 에러의 원인: Python 패키지 구조와 sys.path의 충돌

InternVideo2의 GitHub 저장소 구조는 다음과 같다.

```
InternVideo/
└── InternVideo2/
    └── multi_modality/          ← __init__.py 존재 (패키지)
        ├── models/              ← __init__.py 존재 (하위 패키지)
        │   ├── criterions.py
        │   └── utils.py
        ├── utils/               ← multi_modality 레벨의 유틸리티
        │   ├── distributed.py
        │   └── easydict.py
        ├── demo/
        │   └── utils.py
        └── configs/
```

Colab에서는 `multi_modality` 폴더를 직접 `sys.path`에 넣는다.

```python
sys.path.insert(0, "/content/InternVideo/InternVideo2/multi_modality")
```

이렇게 하면 Python은 `models/`를 **최상위(top-level) 패키지**로 인식한다. 즉 `models`의 부모 패키지가 존재하지 않게 된다.

그런데 `models/criterions.py`에는 다음과 같은 relative import가 있다.

```python
from ..utils.distributed import get_rank, get_world_size
from ..utils.easydict import EasyDict
```

여기서 `..`은 "현재 패키지(`models`)의 부모 패키지"를 의미한다. 원래 `multi_modality`가 부모 패키지이므로 `..utils`는 `multi_modality/utils`를 가리켜야 한다. 하지만 `sys.path`에 `multi_modality`를 직접 넣었기 때문에 `models`에는 부모가 없고, `..`이 갈 곳이 없어서 `ImportError`가 발생한다.

이것을 그림으로 표현하면 다음과 같다.

```
[ 원래 의도된 구조 ]

multi_modality  (부모 패키지)
├── models      (자식 패키지)
│   └── criterions.py → from ..utils  = multi_modality.utils  ✓
└── utils
    └── distributed.py

[ 실제 Colab에서의 구조 ]

sys.path → multi_modality/
           ├── models  (최상위 패키지 — 부모 없음)
           │   └── criterions.py → from ..utils  = ???  ✗ 부모가 없음!
           └── utils/
```

이 문제가 발생하는 근본적인 이유는, InternVideo2의 코드가 `multi_modality`를 하나의 패키지로 전제하고 작성되었기 때문이다. 원래 InternVideo2를 통째로 클론하면 상위의 `setup.py`나 `pyproject.toml`이 패키지 구조를 올바르게 잡아주지만, sparse checkout으로 `multi_modality`만 가져와서 `sys.path`에 직접 넣는 방식에서는 이 패키지 계층이 깨진다.

### 4-2. 2차 에러의 원인: flash_attn 모듈 레벨 import

InternVideo2의 vision encoder 코드(`models/backbones/internvideo2/internvideo2.py` 등)에는 다음과 같은 import가 모듈 최상단에 존재한다.

```python
from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func
from flash_attn.bert_padding import unpad_input, pad_input
from flash_attn.modules.mlp import FusedMLP
from flash_attn.ops.rms_norm import DropoutAddRMSNorm
```

이 import들은 **조건부가 아니라 무조건 실행**된다. config에서 `use_flash_attn=False`로 설정하면 실제 코드 경로에서 이 함수들이 **호출되지는 않지만**, Python은 모듈을 로드하는 시점에 최상단의 모든 import를 실행하므로, `flash_attn` 패키지가 설치되어 있지 않으면 `ModuleNotFoundError`가 발생한다.

`flash_attn` 패키지를 설치할 수 없는 이유는, 이 패키지가 CUDA C++ 커널을 포함한 네이티브 확장이기 때문이다. 사전 빌드된 wheel은 특정 CUDA 버전, PyTorch 버전, Python 버전의 조합에 대해서만 제공되는데, 현재 Colab 환경(CUDA 12.8 + PyTorch 2.10 + Python 3.12)에 맞는 wheel이 배포되어 있지 않다. 소스에서 직접 빌드하려면 `ninja`, CUDA toolkit, 그리고 상당한 컴파일 시간이 필요하며, Colab Free Tier의 메모리와 시간 제한 내에서 완료하기 어렵다.

### 4-3. 3차 에러의 원인: 더미 모듈의 불완전한 메타데이터 + 불필요한 import 체인

이 에러는 두 가지 원인이 동시에 작용한다.

**원인 A: 더미 모듈의 `__spec__` 누락**

Python의 `types.ModuleType()`으로 생성한 모듈은 `__spec__` 속성이 자동으로 `None`이 된다. Python 3.12의 `importlib.util.find_spec()`은 `sys.modules`에서 모듈을 찾았을 때 `__spec__`이 `None`이면 `ValueError`를 발생시킨다.

```python
# Python 3.12 importlib/util.py 내부 동작:
def find_spec(name, package=None):
    ...
    if name in sys.modules:
        module = sys.modules[name]
        spec = module.__spec__       # ← None
        if spec is None:
            raise ValueError(f"{name}.__spec__ is None")  # ← 여기서 터짐
```

이전 Python 버전(3.10 이하)에서는 `__spec__`이 `None`이어도 `None`을 그냥 반환했지만, 3.12에서는 더 엄격해졌다.

**원인 B: `models/__init__.py`의 과도한 import**

`models/__init__.py`가 패키지 로드 시 모든 모델 클래스를 import한다.

```python
# models/__init__.py (원본)
from .internvideo2_clip import InternVideo2_CLIP           # ← 이것이 LLaMA를 끌어옴
from .internvideo2_clip_small import InternVideo2_CLIP_small
from .internvideo2_stage2_visual import InternVideo2_Stage2_visual
from .internvideo2_stage2_audiovisual import InternVideo2_Stage2_audiovisual
```

우리는 이 중 어느 것도 직접 사용하지 않는다. 우리가 필요한 것은 `models.backbones.internvideo2.pretrain_internvideo2_1b_patch14_224` 하나뿐이다. 그런데 `models/__init__.py`가 실행되면서 `InternVideo2_CLIP` → `internvideo2_clip_text.py` → `peft` → `transformers` → `flash_attn` 검사라는 긴 체인이 연쇄적으로 발동된다.

이 체인을 그림으로 표현하면 다음과 같다.

```
우리가 필요한 것                      불필요하게 딸려오는 것
─────────────────                    ─────────────────────
demo/utils.py                        models/__init__.py
  → models.backbones.internvideo2      → InternVideo2_CLIP
     → pretrain_internvideo2_1b...        → internvideo2_clip_text.py
     (여기만 필요)                            → from peft import ...
                                                → peft → transformers
                                                   → flash_attn 검사
                                                      → ValueError!
```

`demo/utils.py`가 `from models.backbones.internvideo2 import ...`를 실행하면, Python은 먼저 `models` 패키지를 초기화하기 위해 `models/__init__.py`를 실행한다. 이 `__init__.py`에 있는 불필요한 import들이 `peft` → `transformers`까지 끌어오면서 `flash_attn` 검사가 발동되는 것이다.


### 4-4. 4차 에러의 원인: transformers 라이브러리의 내부 API 이동

`transformers` 라이브러리는 버전 4.40 전후로 내부 유틸리티 함수의 위치를 정리했다. `apply_chunking_to_forward`, `find_pruneable_heads_and_indices`, `prune_linear_layer`는 원래 `transformers.modeling_utils`에 있었지만 `transformers.pytorch_utils`로 이동되었다.

InternVideo2의 `xbert.py`는 이 변경 이전 버전을 기준으로 작성되어 `from transformers.modeling_utils import apply_chunking_to_forward`를 하드코딩하고 있다. Colab의 transformers(4.50+)에서는 해당 위치에 함수가 없으므로 `ImportError`가 발생한다.

이것은 InternVideo2 측의 버그는 아니다. InternVideo2가 공개된 시점(2024년 초)에는 transformers 4.30대가 최신이었고, 이후 transformers가 내부 구조를 변경한 것이다. 단, InternVideo2 저장소가 이 변경에 맞춰 업데이트되지 않았기 때문에 최신 환경에서 비호환이 발생한다.


### 4-5. 5차 에러의 원인: transformers의 점진적 API 정리

transformers 라이브러리는 4.40에서 `modeling_utils` → `pytorch_utils`로 함수를 이동했고, 4.50+에서는 `find_pruneable_heads_and_indices`와 `prune_linear_layer`를 `pytorch_utils`에서도 제거했다. 이 두 함수는 모델 head pruning용 유틸리티인데, transformers 측에서 pruning 기능을 별도 패키지로 분리하거나 deprecate하는 과정에서 제거된 것으로 보인다.

하지만 InternVideo2의 `xbert.py`(BERT 구현)는 이 함수들을 실제로 사용하므로, 단순히 import 경로만 바꿔서는 해결되지 않는다.

### 4-6. 6차 에러의 원인: transformers 내부(private) API 사용

InternVideo2의 `tokenization_bert.py`는 HuggingFace transformers v4.30 시점의 BertTokenizer 소스를 통째로 복사한 파일이다. 이 파일이 `_is_control`, `_is_punctuation`, `_is_whitespace`라는 underscore 접두사가 붙은 private 함수를 import하는데, transformers가 내부 구조를 정리하면서 이 함수들을 `tokenization_utils`에서 제거했다.

근본적인 문제는 InternVideo2가 자체 커스텀 BertTokenizer를 유지하고 있다는 것인데, 이 커스텀 버전은 표준 `transformers.BertTokenizer`와 기능적으로 동일하다. 차이가 있다면 원본 시점의 코드를 고정해 놓은 것뿐이다.


### 4-7. 7차 에러의 원인: config의 빈 문자열과 상대 경로

`cfg.model.vision_encoder.pretrained = ''`는 "사전 학습된 backbone 가중치를 별도로 로드하지 않겠다"는 의미로 넣은 것인데, InternVideo2 코드가 `if pretrained is not None:` 방식으로 체크하는 곳이 있어서 빈 문자열도 유효한 경로로 취급된다.

`cfg.model.text_encoder.config = "configs/config_bert_large.json"`은 `BertConfig.from_json_file()`에서 JSON 설정 파일을 읽는 데 사용된다. InternVideo2 코드는 원래 `multi_modality/` 디렉토리에서 실행되는 것을 전제로 작성되었으므로 상대 경로가 정상 동작하지만, 우리 환경에서는 CWD가 프로젝트 루트(`/content/videorag_prototype`)이므로 해석이 달라진다.


### 4-8. 8차 에러의 원인: transformers 4.40+의 tied weights 관리 변경

transformers 라이브러리는 v4.40 전후로 `PreTrainedModel`의 tied weights(공유 가중치) 관리 방식을 변경했다. 구체적으로, 모델 초기화 시 `post_init()` → `mark_tied_weights_as_initialized()`를 호출하는데, 이 함수가 `self.all_tied_weights_keys` 속성에 접근한다.

`all_tied_weights_keys`는 `PreTrainedModel`의 computed property로, 내부적으로 `_tied_weights_keys` 클래스 변수에 의존한다. transformers ~4.30 시점에는 이 속성이 존재하지 않았거나 선택적이었지만, 4.40+에서는 필수(mandatory)가 되었다.

InternVideo2의 `xbert.py`는 transformers ~4.30 기준으로 작성된 커스텀 BERT 구현이다. 이 파일의 `BertPreTrainedModel` 클래스가 `PreTrainedModel`을 상속하면서 `_tied_weights_keys`를 정의하지 않았다. transformers ~4.30에서는 문제없었지만, 4.40+에서는 초기화 과정에서 `all_tied_weights_keys`에 접근할 때 `_tied_weights_keys`가 없어서 `AttributeError`가 발생한다.

이 문제가 `BertForMaskedLM`에서 발생하는 이유는, BERT의 Masked LM head가 입력 임베딩 가중치를 출력 레이어와 공유(tie)하는 구조이기 때문이다. transformers가 이 tied weights를 추적하기 위해 `_tied_weights_keys`를 요구하는 것이다.

```
transformers ~4.30  : _tied_weights_keys 불필요 (InternVideo2 작성 시점)
transformers ~4.40+ : _tied_weights_keys 필수 — 없으면 AttributeError
                      (Colab 현재 버전)
```

이 에러는 4~5차 에러(함수 이동/제거)와 같은 맥락이다. InternVideo2가 transformers의 특정 버전에 맞춰 작성되었는데, 이후 transformers가 내부 구조를 변경하면서 비호환이 발생한 것이다. 다만 4~5차 에러는 import 시점에 발생한 반면, 8차 에러는 클래스 인스턴스화 시점에 발생한다는 차이가 있다.

### 4-9. 9차 에러의 원인: 파일 패치와 런타임 캐시의 불일치

8차 에러 해결로 `xbert.py`에 `_tied_weights_keys = []`를 파일 수준에서 삽입했지만, 이 패치는 실제로 효과가 없다.

원인은 Python의 모듈 캐싱(module caching)과 에러가 발생하는 위치의 차이에 있다.

**에러가 발생하는 실제 위치:**

에러는 `xbert.py`의 코드가 아니라, `transformers`의 `PreTrainedModel.__init__()` 내부에서 발생한다.

```
BertForMaskedLM(config) 호출
  → BertForMaskedLM.__init__()
    → super().__init__()  ← 즉 PreTrainedModel.__init__() 호출
      → self.post_init()
        → self.mark_tied_weights_as_initialized()
          → self.all_tied_weights_keys  ← 여기서 AttributeError
```

`all_tied_weights_keys`는 `PreTrainedModel`에 정의된 property이고, 이 property가 `self._tied_weights_keys`를 인스턴스 속성으로 찾는다.

**파일 패치가 효과 없는 이유:**

`xbert.py`에 `_tied_weights_keys = []`를 클래스 변수로 추가하면, Python의 속성 탐색 순서(MRO)에 따라 인스턴스 속성 → 클래스 속성 순서로 찾으므로 이론적으로는 동작해야 한다.

그러나 문제는 `transformers`가 이미 `sys.modules`에 캐시되어 있다는 것이다. `xbert.py`를 아무리 수정해도, **에러가 나는 코드(`PreTrainedModel.__init__`)는 `xbert.py` 안에 있지 않다.** 에러는 이미 메모리에 올라간 `transformers.PreTrainedModel` 클래스의 메서드 안에서 발생한다.

구체적으로, `transformers`는 Colab 세션 시작 시 또는 처음 import될 때 이미 `sys.modules["transformers"]`에 등록된다. 이후 `xbert.py`가 import되어 `BertPreTrainedModel(PreTrainedModel)`을 정의할 때, `PreTrainedModel`은 이미 캐시된 클래스 객체를 그대로 사용한다.

```
① transformers import → PreTrainedModel 클래스가 메모리에 올라가 캐시됨
                         PreTrainedModel.__init__ 안에 post_init() 호출 포함

② xbert.py 파일 수정 → BertPreTrainedModel에 _tied_weights_keys = [] 추가

③ xbert.py import → BertPreTrainedModel(PreTrainedModel) 정의
                     부모인 PreTrainedModel은 ①에서 캐시된 객체 그대로 사용

④ BertForMaskedLM(config) 호출
   → PreTrainedModel.__init__() 실행  ← ①에서 캐시된 코드 그대로 실행
     → post_init() → all_tied_weights_keys 접근
       → _tied_weights_keys를 인스턴스에서 찾음
         → 인스턴스에 없음 → 클래스에서 찾음 (②에서 추가한 것)
```

여기서 중요한 점은, `all_tied_weights_keys` property의 실제 구현이다. transformers 4.40+에서 이 property는 단순히 클래스 변수를 읽는 것이 아니라, **`__init__` 과정에서 인스턴스 변수로 설정되어야 한다**는 것을 전제로 구현되어 있다. 즉 클래스 변수 `_tied_weights_keys = []`가 있어도, `PreTrainedModel.__init__`이 실행되는 과정에서 이를 인스턴스 변수로 복사하지 않으면 property 접근 시 에러가 발생한다.

결론적으로, 파일 패치(`xbert.py` 수정)는 "xbert.py 안의 코드"를 고치는 것이다. 하지만 에러는 "이미 캐시된 `transformers` 코드" 안에서 발생하므로, 파일 패치로는 해결할 수 없다.

### 4-10. 10차 에러의 원인: monkey-patch의 잘못된 가드 조건과 에러 발생 위치 오판

**원인 A: 잘못된 가드 조건**

```python
if not hasattr(PreTrainedModel, 'all_tied_weights_keys'):
    return  # ← 여기서 종료
```

`all_tied_weights_keys`는 `PreTrainedModel`의 property가 아니었다. `hasattr` 체크에서 `False`가 반환되어 패치 자체가 실행되지 않았다.

**원인 B: 에러 발생 위치 오판**

에러가 `PreTrainedModel.__init__`에서 발생한다고 가정했지만, 실제 에러 위치는 달랐다.

```
BertForMaskedLM.from_pretrained("bert-large-uncased")
  → from_pretrained()
    → _finalize_load_state_dict()       ← __init__ 이후의 후처리 단계
      → model.mark_tied_weights_as_initialized()
        → self.all_tied_weights_keys.keys()  ← 여기서 AttributeError
```

`__init__`을 패치해도 `from_pretrained()`의 후처리 단계는 별개로 실행되므로 효과가 없었다.

**결론:** 필요한 것은 `__init__` 패치가 아니라 클래스에 `all_tied_weights_keys` 속성을 직접 추가하는 것이다. `mark_tied_weights_as_initialized()`는 `.keys()`를 호출하므로 빈 딕셔너리 `{}`를 클래스 속성으로 주입하면 해결된다.

## 5. 어떻게 해결했는가

`embedder.py`의 `_load_internvideo2()` 메서드가 모델을 로드하기 **직전에** 두 개의 전처리 단계를 자동 실행하도록 수정했다.

### 5-1. 해결책 A: flash_attn 더미 모듈 주입 (`_inject_flash_attn_stub()`)

`sys.modules` 딕셔너리에 `flash_attn` 패키지의 더미 모듈 트리를 등록한다. Python의 import 시스템은 `sys.modules`에 이미 등록된 모듈은 파일 시스템에서 찾지 않으므로, 실제 패키지가 설치되어 있지 않아도 import가 통과한다.

3차 에러를 방지하기 위해, 각 더미 모듈에 올바른 `__spec__`(ModuleSpec)을 설정하고 루트 모듈에 `__version__ = "0.0.0"`을 부여했다.

```python
import importlib.machinery

mod.__spec__ = importlib.machinery.ModuleSpec(
    name, None,
    is_package=is_package   # 하위 모듈이 있는 패키지는 True
)
```

`__spec__`을 설정하면 `importlib.util.find_spec("flash_attn")`이 `ValueError` 없이 정상 반환된다. `__version__`을 `"0.0.0"`으로 설정하면 `transformers`의 `is_flash_attn_2_available()`이 버전 체크에서 `False`를 반환하여, Flash Attention 코드 경로를 자연스럽게 비활성화한다.

등록하는 더미 모듈 구조는 다음과 같다.

```
sys.modules에 등록되는 더미 트리:

flash_attn                                    (패키지, __version__="0.0.0")
├── flash_attn.flash_attn_interface           (서브모듈)
│   └── flash_attn_varlen_qkvpacked_func = None
├── flash_attn.bert_padding                   (서브모듈)
│   ├── unpad_input = None
│   └── pad_input = None
├── flash_attn.modules                        (서브패키지)
│   └── flash_attn.modules.mlp                (서브모듈)
│       └── FusedMLP = None
├── flash_attn.ops                            (서브패키지)
│   └── flash_attn.ops.rms_norm               (서브모듈)
│       └── DropoutAddRMSNorm = None
└── dropout_layer_norm                        (별도 CUDA extension)
    ├── dropout_add_layer_norm = None
    ├── dropout_add_layer_norm_subset = None
    └── dropout_add_layer_norm_parallel_residual = None

* 모든 모듈에 __spec__ = ModuleSpec(name, None) 설정됨
```

모든 함수/클래스가 `None`으로 설정되어 있다. 이것은 config에서 `use_flash_attn=False`, `use_fused_rmsnorm=False`, `use_fused_mlp=False`이므로 이 함수들이 실제로 호출되는 코드 경로를 절대 타지 않기 때문에 안전하다.

### 5-2. 해결책 B: 소스 코드 패치 (`_patch_internvideo2_source()`)

sparse checkout한 InternVideo2 소스 파일을 자동으로 수정한다. 패치 대상은 세 개 파일이다.

**패치 1: `models/__init__.py` — 불필요한 import 체인 차단**

```python
# (수정 전)
from .internvideo2_clip import InternVideo2_CLIP
from .internvideo2_clip_small import InternVideo2_CLIP_small
from .internvideo2_stage2_visual import InternVideo2_Stage2_visual
from .internvideo2_stage2_audiovisual import InternVideo2_Stage2_audiovisual

# (수정 후 — 전부 주석 처리)
# [VideoRAG patch] 아래 import들은 peft/LLaMA 등
# 불필요한 의존성을 끌어오므로 비활성화.
# 우리는 models.backbones.internvideo2만 사용함.
# from .internvideo2_clip import InternVideo2_CLIP
# from .internvideo2_clip_small import InternVideo2_CLIP_small
# from .internvideo2_stage2_visual import InternVideo2_Stage2_visual
# from .internvideo2_stage2_audiovisual import InternVideo2_Stage2_audiovisual
```

이렇게 하면 `models` 패키지 초기화 시 `peft` → `transformers` → `flash_attn` 검사 체인이 발동되지 않는다. 우리가 사용하는 `models.backbones.internvideo2.pretrain_internvideo2_1b_patch14_224`는 `__init__.py`를 거치지 않고 직접 import되므로 영향을 받지 않는다.

**패치 2: `models/backbones/bert/xbert.py` — transformers 버전 호환 (4차+5차 에러)**

`xbert.py`의 맨 앞에 호환 레이어를 삽입한다. `transformers.modeling_utils`에 `apply_chunking_to_forward` 등이 없으면, transformers 원본 소스 코드에서 가져온 동일 구현을 `modeling_utils`에 주입한다.

```python
# [VideoRAG compat] transformers 4.40+ 호환 레이어
import torch as _torch
import transformers as _tf

def _compat_apply_chunking_to_forward(forward_fn, chunk_size, chunk_dim, *input_tensors):
    # transformers v4.37.2의 pytorch_utils.py 원본과 동일한 구현
    if chunk_size > 0:
        ...  # chunking 로직
    return forward_fn(*input_tensors)

def _compat_find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
    # transformers v4.37.2의 pytorch_utils.py 원본과 동일한 구현
    mask = _torch.ones(n_heads, head_size)
    ...
    return heads, index

def _compat_prune_linear_layer(layer, index, dim=0):
    # transformers v4.37.2의 pytorch_utils.py 원본과 동일한 구현
    ...
    return new_layer

if not hasattr(_tf.modeling_utils, 'apply_chunking_to_forward'):
    _tf.modeling_utils.apply_chunking_to_forward = _compat_apply_chunking_to_forward
if not hasattr(_tf.modeling_utils, 'find_pruneable_heads_and_indices'):
    _tf.modeling_utils.find_pruneable_heads_and_indices = _compat_find_pruneable_heads_and_indices
if not hasattr(_tf.modeling_utils, 'prune_linear_layer'):
    _tf.modeling_utils.prune_linear_layer = _compat_prune_linear_layer
```

이 3개 함수는 "임의 구현"이 아니라 **HuggingFace transformers v4.37.2의 `pytorch_utils.py` 원본 소스 코드에서 그대로 가져온 것**이다. 원본 소스는 다음 URL에서 확인할 수 있다:
https://github.com/huggingface/transformers/blob/v4.37.2/src/transformers/pytorch_utils.py

transformers가 4.50+에서 이 함수들을 제거했을 뿐, 함수의 로직 자체가 변경된 것은 아니다. `hasattr` 가드를 사용하므로, 향후 transformers가 이 함수들을 다시 복원하면 원본이 그대로 사용된다.

멱등성: `[VideoRAG compat]` 문자열이 이미 파일에 존재하면 패치를 건너뛴다.

**패치 3: `demo/utils.py` — 커스텀 BertTokenizer를 표준으로 교체 (6차 에러)**

```python
# (수정 전)
from models.backbones.bert.tokenization_bert import BertTokenizer

# (수정 후)
# [VideoRAG patch] 커스텀 tokenization_bert는 transformers 4.50+에서
# _is_control 등 제거된 내부 함수를 import하므로 실패.
# 표준 BertTokenizer와 동일하므로 교체.
from transformers import BertTokenizer
```

InternVideo2의 커스텀 `tokenization_bert.py`와 표준 `transformers.BertTokenizer`는 기능이 동일하다. 둘 다 WordPiece 기반이고, `from_pretrained("bert-large-uncased")`로 로드하면 같은 vocab과 tokenization 결과를 반환한다. 커스텀 버전은 옛 transformers 내부 함수에 의존하는 반면, 표준 버전은 현재 transformers에서 정상 동작하므로 교체하는 것이 안전하다.

**패치 4: `models/backbones/bert/xbert.py` — _tied_weights_keys 추가 (8차 에러)**

```python
# (수정 전)
class BertPreTrainedModel(PreTrainedModel):
    # _tied_weights_keys 없음

# (수정 후)
class BertPreTrainedModel(PreTrainedModel):
    # [VideoRAG patch] transformers 4.40+ 호환
    # PreTrainedModel이 all_tied_weights_keys 속성을 요구하는데
    # 이 속성은 _tied_weights_keys 클래스 변수에 의존함.
    _tied_weights_keys = []
```

`BertPreTrainedModel` 베이스 클래스에 `_tied_weights_keys = []`를 추가한다. 이렇게 하면 `BertForMaskedLM`, `BertModel`, `BertLMHeadModel` 등 모든 하위 클래스가 자동으로 상속받아 `all_tied_weights_keys` 접근 시 에러가 발생하지 않는다.

빈 리스트 `[]`로 설정한 이유: 우리는 InternVideo2의 BERT 텍스트 인코더를 **추론(inference) 전용**으로 사용한다. tied weights는 학습(training) 시 가중치 공유를 관리하기 위한 메커니즘이므로, 추론에서는 빈 리스트로 두어도 동작에 영향이 없다. 실제로 `setup_internvideo2()`가 체크포인트에서 모든 가중치를 명시적으로 로드하므로, tied weights 추적이 불필요하다.

멱등성: `_tied_weights_keys` 문자열이 이미 파일에 존재하면 패치를 건너뛴다.

**패치 5: `models/criterions.py` — relative import 수정 (1차 에러)**


```
(수정 전) from ..utils.distributed import get_rank, get_world_size
(수정 후) from utils.distributed import get_rank, get_world_size

(수정 전) from ..utils.easydict import EasyDict
(수정 후) from easydict import EasyDict
```

**패치 6: `models/utils.py` — relative import 수정**

```
(수정 전) from ..utils.distributed import ...
(수정 후) from utils.distributed import ...
```

`multi_modality/`가 `sys.path`에 있으므로 `utils.distributed`는 `multi_modality/utils/distributed.py`를 가리키게 되어 정상 import된다.

모든 패치는 멱등성(idempotent)을 가진다. 이미 패치된 파일에는 원본 문자열이 없으므로 다시 수정하지 않는다. sparse checkout을 새로 받아도 자동으로 다시 패치된다.

### 5-3. 해결책 C: PreTrainedModel monkey-patch (`_patch_transformers_tied_weights()`)

파일 패치가 효과 없는 이유는, 에러가 이미 메모리에 올라간 `PreTrainedModel.__init__` 안에서 발생하기 때문이다. 따라서 **메모리에 올라간 클래스 자체를 직접 수정하는 monkey-patch**를 사용한다.

```python
@staticmethod
def _patch_transformers_tied_weights():
    try:
        from transformers import PreTrainedModel

        # 이 버전의 transformers에는 해당 문제가 없음
        if not hasattr(PreTrainedModel, 'all_tied_weights_keys'):
            return

        # 이미 패치됨 (멱등성)
        if getattr(PreTrainedModel, '_videorag_patched', False):
            return

        _original_init = PreTrainedModel.__init__

        def _patched_init(self, config, *args, **kwargs):
            _original_init(self, config, *args, **kwargs)
            if not hasattr(self, '_tied_weights_keys'):
                self._tied_weights_keys = []

        # 메모리에 올라간 PreTrainedModel 클래스의 __init__을 교체
        PreTrainedModel.__init__ = _patched_init
        PreTrainedModel._videorag_patched = True
        logger.info("PreTrainedModel.__init__ 패치 완료")

    except Exception as e:
        logger.warning(f"PreTrainedModel 패치 실패: {e}")
```

monkey-patch의 동작 원리:

```
① _patch_transformers_tied_weights() 호출
   → 이미 캐시된 PreTrainedModel.__init__을 _patched_init으로 교체

② BertForMaskedLM(config) 호출
   → PreTrainedModel.__init__() 실행  ← 이제 _patched_init이 실행됨
     → _original_init() 실행 (에러 발생 가능 지점)
     → _tied_weights_keys 없으면 [] 강제 주입
     → 에러 발생 전에 속성이 보장됨 ✓
```

파일을 수정하는 것이 아니라 **메모리에 올라간 클래스 객체의 메서드를 직접 교체**하므로, `sys.modules` 캐시 문제와 무관하게 동작한다.

### 5-4. 적용 순서 (업데이트)

`_load_internvideo2()` 내부에서의 실행 순서는 다음과 같다.

```
_load_internvideo2() 호출
│
├── Step 0a: _inject_flash_attn_stub()
│   → sys.modules에 더미 flash_attn 트리 등록 (__spec__ + __version__ 포함)
│
├── Step 0b: _patch_internvideo2_source()
│   → models/__init__.py의 불필요한 import 4줄 주석 처리 (3차 에러)
│   → models/criterions.py 등의 from ..utils를 from utils로 교체 (1차 에러)
│   → xbert.py에 transformers 호환 레이어 삽입 (4차+5차 에러)
│   → xbert.py의 BertPreTrainedModel에 _tied_weights_keys 추가 (8차 에러 — 파일 패치, 단독으로는 불충분)
│   → demo/utils.py의 커스텀 BertTokenizer를 표준으로 교체 (6차 에러)
│
├── Step 0c: _patch_transformers_tied_weights()  ← 신규 추가
│   → 메모리에 올라간 PreTrainedModel.__init__을 monkey-patch
│   → _tied_weights_keys 없는 인스턴스에 [] 강제 주입
│   → 8차/9차 에러 실제 해결
│
├── Step 1: sys.path에 multi_modality 경로 등록
├── Step 2: HuggingFace에서 .pt 체크포인트 다운로드
├── Step 3: BertTokenizer 로드
├── Step 4: EasyDict로 config 구성
│   → text_encoder.config를 절대 경로로 설정 (7차 에러 방지)
│   → vision_encoder.pretrained = None (빈 문자열 대신, 7차 에러 방지)
│
└── Step 5: CWD를 multi_modality로 변경 후 setup_internvideo2() 호출
            → 호출 완료 후 CWD 복원
```

### 5-4. 노트북 수정 사항

`01_indexing.ipynb`에서도 두 가지를 수정했다.

cell-8 (직접 import 테스트 셀): `demo.utils`를 직접 import하기 전에 패치를 먼저 적용하도록 변경했다.

```python
from src.phase0_indexing.embedder import VideoEmbedder
VideoEmbedder._inject_flash_attn_stub()
VideoEmbedder._patch_internvideo2_source()
from demo.utils import setup_internvideo2, frames2tensor
```

cell-12 (인덱스 로드 검증 셀): `FAISSVectorStore(dim=1024)` → `FAISSVectorStore(dim=512)`로 수정했다.


## 6. 왜 그렇게 해결했는가

### 6-1. relative import 패치를 선택한 이유

고려한 대안들과 각각을 선택하지 않은 이유는 다음과 같다.

첫 번째 대안은 `sys.path`에 `multi_modality`의 부모 디렉토리를 넣는 방법이다. 즉 `/content/InternVideo/InternVideo2`를 `sys.path`에 넣으면 `multi_modality`가 정식 패키지가 되어 relative import가 작동한다. 하지만 이렇게 하면 InternVideo2 내부의 모든 absolute import(`from models.xxx import ...`)가 깨진다. 내부 코드 전체를 `from multi_modality.models.xxx`로 바꿔야 하므로 패치 범위가 너무 넓다.

두 번째 대안은 `setup_internvideo2()` 함수를 직접 재구현하는 방법이다. `demo/utils.py`를 사용하지 않고 필요한 로직만 `embedder.py`에 직접 작성하면 import 체인 자체를 끊을 수 있다. 하지만 `setup_internvideo2()`는 DeepSpeed 체크포인트의 key 변환, position embedding 보간, state_dict 매핑 등 복잡한 로직을 포함하고 있어서, 이것을 정확하게 재현하려면 공식 코드를 깊이 분석해야 하며 버전 업데이트 시 동기화 부담이 크다.

세 번째 대안으로 선택한 것이 문제 파일의 import문만 교체하는 것이다. 패치 대상이 `models/criterions.py`와 `models/utils.py` 두 파일뿐이고, 변경도 `from ..utils.xxx`를 `from utils.xxx`로 바꾸는 단순한 문자열 치환이다. 의미적으로 동일한 모듈을 가리키므로(multi_modality가 sys.path에 있으니 `utils.xxx`는 `multi_modality/utils/xxx`를 찾게 됨) 동작에 차이가 없다. sparse checkout을 다시 받으면 원본으로 돌아가는데, 패치가 멱등적이므로 매번 자동 재적용된다.

### 6-2. flash_attn 더미 모듈 주입을 선택한 이유

고려한 대안들과 각각을 선택하지 않은 이유는 다음과 같다.

첫 번째 대안은 `flash-attn` 패키지를 소스에서 직접 빌드하는 방법이다. `pip install flash-attn --no-build-isolation`으로 시도할 수 있지만, ninja 빌드 시스템이 필요하고 CUDA 커널 컴파일에 10분 이상이 걸리며 Colab Free Tier의 RAM 제한(12GB)에서 OOM이 발생할 수 있다. 빌드에 성공하더라도 런타임이 재시작될 때마다 다시 빌드해야 한다.

두 번째 대안은 InternVideo2 소스 코드에서 `flash_attn` import를 try-except로 감싸는 것이다. 각 파일의 `from flash_attn.xxx import yyy`를 `try: from flash_attn.xxx import yyy / except: yyy = None`으로 바꾸면 된다. 하지만 패치해야 할 파일이 여러 개이고, flash_attn을 import하는 정확한 위치를 모두 찾아야 하며, InternVideo2 코드가 업데이트되면 놓칠 수 있다.

세 번째 대안으로 선택한 것이 `sys.modules` 레벨에서 더미를 주입하는 것이다. 이 방법의 장점은 InternVideo2 소스 코드를 전혀 수정하지 않는다는 것이다. flash_attn을 import하는 파일이 몇 개든, 어디서 import하든 상관없이 `sys.modules`에 등록된 더미가 반환된다. 더미의 모든 attribute가 `None`이어도 안전한 이유는, config 플래그 세 개가 모두 `False`이기 때문이다.

```python
cfg.model.vision_encoder.use_flash_attn = False      # → FlashAttention 클래스 미사용
cfg.model.vision_encoder.use_fused_rmsnorm = False    # → DropoutAddRMSNorm 미사용
cfg.model.vision_encoder.use_fused_mlp = False        # → FusedMLP 미사용
```

InternVideo2의 Attention 클래스 내부에는 `use_flash_attn` 플래그에 따라 `_naive_attn()` (표준 PyTorch)과 `_flash_attn()` (Flash Attention) 중 하나를 선택하는 분기가 있다. `False`일 때는 `_naive_attn()`만 호출되므로 flash_attn의 함수가 실제로 실행되는 일은 없다.

### 6-3. __spec__ + __version__ 설정을 추가한 이유

`types.ModuleType()`이 아닌 `importlib.machinery.ModuleSpec`을 사용한 이유는 Python 3.12의 엄격해진 모듈 검사 정책 때문이다. 단순히 `__spec__ = "something"`이 아니라 올바른 `ModuleSpec` 객체를 설정해야 `importlib.util.find_spec()`이 정상 동작한다.

`__version__ = "0.0.0"`을 설정한 이유는 `transformers`가 flash_attn의 버전을 체크하여 2.x 이상일 때만 Flash Attention을 활성화하기 때문이다. `"0.0.0"`이면 버전 체크에서 자연스럽게 탈락하여, `is_flash_attn_2_available()`이 `False`를 반환한다. 이는 flash_attn이 아예 설치되지 않은 것과 동일한 효과를 낸다.

### 6-4. models/__init__.py 패치를 추가한 이유

`__spec__` 수정만으로도 3차 에러 자체는 해결되지만, 근본적인 문제는 `models/__init__.py`가 우리에게 불필요한 모델 클래스(`InternVideo2_CLIP`, `LLaMA` 등)를 전부 import하면서 `peft` → `transformers`라는 무거운 의존성 체인을 발동시키는 것이다.

이 불필요한 import들을 주석 처리하면 다음과 같은 이점이 있다.

첫째, `peft` 의존성이 제거되어 향후 `peft` 버전 변경에 의한 호환성 문제가 원천적으로 차단된다. 둘째, `LLaMA`, `Tokenizer` 등 LLM 관련 클래스를 로드하지 않으므로 초기화 시간과 메모리 사용이 줄어든다. 셋째, import 체인이 짧아져서 flash_attn 더미 모듈이 검증받을 기회 자체가 줄어들어, 더미의 불완전한 부분이 문제가 될 가능성도 함께 줄어든다.

우리가 사용하는 `pretrain_internvideo2_1b_patch14_224`는 `models.backbones.internvideo2` 서브패키지에서 직접 import되므로, `models/__init__.py`의 최상위 import를 제거해도 전혀 영향을 받지 않는다.

### 6-5. xbert.py에 "직접 구현"을 삽입한 이유 (5차 에러)

transformers 4.50+에서 `find_pruneable_heads_and_indices`와 `prune_linear_layer`가 `pytorch_utils`에서도 제거되었으므로, 라이브러리 내부에서 가져올 수 없다. 따라서 함수 구현 자체를 제공해야 한다.

이때 핵심적인 의문이 생긴다: **이 "직접 구현"을 신뢰할 수 있는가?**

답: 이것은 "임의 구현"이 아니라, **HuggingFace transformers v4.37.2의 `pytorch_utils.py` 원본 소스에서 그대로 가져온 코드**이다.

원본 확인 방법:
- URL: https://github.com/huggingface/transformers/blob/v4.37.2/src/transformers/pytorch_utils.py
- 위 파일에서 `find_pruneable_heads_and_indices`, `prune_linear_layer`, `apply_chunking_to_forward` 3개 함수를 찾으면 우리 코드와 로직이 line-by-line 동일함을 확인할 수 있다.

transformers가 이 함수들을 "제거"한 것이지 "변경"한 것이 아니므로, v4.37.2 시점의 구현을 그대로 사용하는 것이 정확하다. 또한 `hasattr` 가드를 사용하므로, 향후 transformers가 함수를 다시 복원하면 원본이 자동으로 우선 사용된다.

대안으로 `transformers==4.37.2`를 설치하는 방법도 있다. 이렇게 하면 xbert.py 패치 자체가 불필요해진다. 다만 transformers 버전을 내리면 PyTorch 2.10이나 Colab의 다른 라이브러리와의 호환성 문제가 발생할 수 있으므로, "현재 버전 유지 + 원본 코드 복사 fallback"이 더 안전한 선택이다.

### 6-6. 커스텀 BertTokenizer를 표준으로 교체한 이유 (6차 에러)

InternVideo2의 `tokenization_bert.py`와 표준 `transformers.BertTokenizer`를 비교하면:

- 둘 다 WordPiece 기반 토큰화기이다.
- 둘 다 `from_pretrained("bert-large-uncased")`로 동일한 vocab을 로드한다.
- 커스텀 버전은 transformers ~4.30 시점의 코드 복사본이고, 표준 버전은 현재 transformers에서 지속 관리되는 코드이다.
- 유일한 차이: 커스텀 버전이 `_is_control` 등 private API를 직접 import하여 transformers 버전 변경에 취약하다.

따라서 커스텀 버전을 표준 버전으로 교체하는 것은 기능적으로 동일하면서 호환성이 더 좋은 안전한 변경이다. `demo/utils.py`에서 `BertTokenizer`는 `from_pretrained()`로 로드한 후 `tokenizer(text, ...)`로 호출되는 표준 인터페이스만 사용하므로, 교체로 인한 동작 차이는 없다.

### 6-8. xbert.py에 _tied_weights_keys를 추가한 이유 (8차 에러)

이 에러는 4~5차 에러와 동일한 패턴이다. InternVideo2의 커스텀 BERT 코드가 transformers의 특정 버전을 전제로 작성되었는데, transformers가 내부 구조를 변경하면서 호환성이 깨진 것이다.

고려한 대안들과 각각을 선택하지 않은 이유:

첫 번째 대안은 `transformers` 버전을 4.30대로 고정하는 방법이다. `pip install transformers==4.37.2`를 하면 4~5차, 6차, 8차 에러가 모두 해결된다. 하지만 Colab의 PyTorch 2.10+ 및 다른 라이브러리와의 호환성 문제가 발생할 수 있으며, 향후 Colab 환경이 변경될 때마다 버전 충돌을 관리해야 한다.

두 번째 대안은 `BertForMaskedLM` 클래스에만 `_tied_weights_keys`를 추가하는 방법이다. 에러 메시지가 `BertForMaskedLM`을 가리키므로 해당 클래스에만 넣으면 될 것 같지만, `xbert.py`에는 `BertModel`, `BertLMHeadModel`, `BertForPreTraining` 등 총 10개의 `BertPreTrainedModel` 하위 클래스가 있다. 다른 클래스에서도 같은 에러가 발생할 수 있으므로 하나씩 패치하는 것은 비효율적이다.

세 번째 대안으로 선택한 것이 `BertPreTrainedModel` 베이스 클래스에 `_tied_weights_keys = []`를 추가하는 방법이다. 모든 하위 클래스가 자동으로 상속받으므로 한 줄로 10개 클래스를 동시에 해결한다. 빈 리스트는 "공유 가중치가 없다"는 의미인데, 추론 모드에서는 tied weights 추적이 불필요하므로 동작에 영향이 없다. `setup_internvideo2()`가 체크포인트에서 모든 가중치를 명시적으로 `load_state_dict()`하므로, tied weights 메커니즘과 독립적으로 올바른 가중치가 로드된다.

### 6-9. monkey-patch를 선택한 이유 (9차 에러)

고려한 대안들과 각각을 선택하지 않은 이유:

**대안 1: transformers 버전 다운그레이드**

`pip install transformers==4.37.2`를 하면 8~9차 에러가 모두 해결된다. 하지만 Colab의 PyTorch 2.10+ 및 다른 라이브러리와의 호환성 문제가 발생할 수 있으며, Colab 환경이 변경될 때마다 버전 충돌을 관리해야 한다.

**대안 2: PreTrainedModel 서브클래스에 직접 속성 주입**

```python
from models.backbones.bert.xbert import BertPreTrainedModel
BertPreTrainedModel._tied_weights_keys = []
```

xbert.py import 후 클래스에 직접 속성을 주입하는 방식이다. 파일 패치와 동일한 효과를 런타임에서 얻을 수 있다. 하지만 transformers 4.40+에서 `all_tied_weights_keys` property가 클래스 변수가 아닌 **인스턴스 변수**를 찾는 방식으로 구현되어 있으면 여전히 실패한다. monkey-patch는 `__init__` 실행 후 인스턴스에 직접 주입하므로 이 문제를 원천적으로 우회한다.

**대안 3: monkey-patch (채택)**

`PreTrainedModel.__init__`을 wrap하여 인스턴스 생성 시점에 `_tied_weights_keys`를 강제 주입한다. 이 방법은 다음과 같은 이점이 있다.

- `sys.modules` 캐시와 무관하게 동작한다. 파일 수정이 아니라 메모리의 클래스 객체를 직접 수정하기 때문이다.
- `BertForMaskedLM`뿐 아니라 `PreTrainedModel`을 상속하는 모든 커스텀 클래스에 일괄 적용된다.
- `_videorag_patched` 플래그로 멱등성이 보장된다.
- `hasattr` 가드로 이 문제가 없는 transformers 버전에서는 아무 동작도 하지 않는다.

### 6-8. 클래스 직접 주입을 선택한 이유 (8~10차 에러)

**대안 1: transformers 버전 다운그레이드 (`transformers==4.37.2`)**

8~10차 에러가 모두 해결되지만, Colab의 다른 라이브러리와 호환성 문제가 생길 수 있고 환경이 바뀔 때마다 버전 충돌을 관리해야 한다.

**대안 2: 파일 패치 (`xbert.py`에 `_tied_weights_keys = []` 삽입)**

파일 수정은 성공하지만, 이미 `sys.modules`에 캐시된 클래스 객체에는 효과가 없다. 에러가 발생하는 `from_pretrained()` 후처리 단계는 파일 수정과 무관하게 동작한다.

**대안 3: `PreTrainedModel.__init__` monkey-patch**

`all_tied_weights_keys`가 `PreTrainedModel`의 property가 아니어서 가드 조건(`hasattr`)에서 조기 종료된다. 설령 가드를 제거해도, 에러는 `__init__` 이후의 `from_pretrained()` 후처리(`_finalize_load_state_dict`)에서 발생하므로 `__init__` 패치는 효과가 없다.

**대안 4: 클래스 직접 주입 (채택)**

```python
BertPreTrainedModel.all_tied_weights_keys = {}
```

이 한 줄로 다음이 보장된다.

- `xbert.py` import 후 메모리에 올라간 클래스 객체에 직접 속성을 추가하므로 캐시 문제 없음
- `BertForMaskedLM`을 포함한 모든 하위 클래스가 자동으로 상속받음
- `mark_tied_weights_as_initialized()`가 `.keys()`를 호출하므로 `[]`가 아닌 `{}`를 사용
- 빈 딕셔너리이므로 루프가 즉시 종료되어 추론 동작에 영향 없음

### 6-9. xbert.py 파일 패치(`_tied_weights_keys = []`)를 유지하는 이유

클래스 직접 주입이 실질적인 해결책이지만, xbert.py의 파일 패치도 제거하지 않고 유지한다.

첫째, 방어적 중복이다. 클래스 직접 주입이 어떤 이유로 실패해도 파일 패치가 두 번째 방어선이 된다. 둘째, 문서적 가치다. xbert.py에 `[VideoRAG patch]` 주석이 남아있으면 코드를 읽는 사람이 변경 맥락을 파일에서 직접 확인할 수 있다.

### 6-10. 정리

```
변경한 것:
  - models/__init__.py의 불필요한 import 4줄 주석 처리 (3차 에러)
  - models/criterions.py, models/utils.py의 import 경로 수정 (1차 에러)
  - xbert.py 맨 앞에 transformers 원본 기반 호환 레이어 삽입 (4차+5차 에러)
  - xbert.py의 BertPreTrainedModel에 _tied_weights_keys = [] 추가 (방어적 중복)
  - demo/utils.py의 커스텀 BertTokenizer import를 표준 transformers로 교체 (6차 에러)
  - sys.modules에 더미 flash_attn 모듈 트리 등록, __spec__ + __version__ 포함 (2차+3차 에러)
  - config의 vision_encoder.pretrained를 '' → None으로 변경 (7차 에러)
  - config의 text_encoder.config를 절대 경로로 변경 (7차 에러)
  - setup_internvideo2() 호출 전후 CWD를 multi_modality로 맞춤 (7차 에러)
  - BertPreTrainedModel.all_tied_weights_keys = {} 클래스 직접 주입 (8~10차 에러)
  - _load_internvideo2()에서 Step 0c를 Step 1(sys.path 등록) 이후로 이동
  - vector_store.py 기본 dim을 1024 → 512로 수정 (부수적 에러)

변경하지 않은 것:
  - InternVideo2 모델 아키텍처
  - 체크포인트 로딩 로직 (setup_internvideo2)
  - 추론 코드 경로 (vision_encoder, text_encoder, projection)
  - 임베딩 차원 (512)
  - config 설정값
  - demo/utils.py의 setup_internvideo2() 및 frames2tensor() 함수 (import문만 변경)
```
