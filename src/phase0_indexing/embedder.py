"""
② embedder.py — InternVideo2-1B 멀티모달 임베딩

출처:
- 논문: "InternVideo2: Scaling Foundation Models for Multimodal Video Understanding"
        (Wang et al., 2024) — CVPR 2024 / Shanghai AI Lab
        https://arxiv.org/abs/2403.15377
- GitHub: https://github.com/OpenGVLab/InternVideo/tree/main/InternVideo2
- 라이선스: Apache 2.0
- MSR-VTT R@10 = 85.1% (6B), 프로토타입은 1B 사용 (T4 16GB 호환)

역할:
  - 영상 클립을 512차원 Joint Latent Space 벡터로 인코딩
  - 텍스트 쿼리도 같은 공간으로 인코딩 (Cross-modal Alignment)

코랩 VRAM 판단:
  - 1B float16 → ~2GB 모델 + ~2-4GB activations = 총 4-6GB (T4 안전)
  - 6B float16 → ~12GB 모델 + ~4-8GB activations = 16-20GB (T4 불가)
"""

import os
import sys
import logging
from typing import List, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

# InternVideo2 GitHub 코드 경로 (Colab sparse checkout 위치)
INTERNVIDEO2_CODE_PATH = "/content/InternVideo/InternVideo2/multi_modality"

# 최종 임베딩 차원
# InternVideo2 config 기준: model.embed_dim = 512
# (vision_encoder 출력 768이 vision_proj를 거쳐 512로 projection됨)
EMBED_DIM = 512


class VideoEmbedder:
    """InternVideo2-1B 기반 멀티모달 임베더

    encode_clips(clip_list, batch_size=16) → np.ndarray [N, 512]
    encode_query(text) → np.ndarray [1, 512]
    """

    def __init__(
        self,
        model_name: str = "OpenGVLab/InternVideo2-Stage2_1B-224p-f4",
        device: Optional[str] = None,
        use_fp16: bool = True,
        num_frames: int = 4,
    ):
        """
        Args:
            model_name: HuggingFace 저장소 이름 (체크포인트 다운로드용)
            device: 'cuda' or 'cpu' (None이면 자동 감지)
            use_fp16: float16 사용 여부 (VRAM 절약)
            num_frames: 영상 인코딩 시 사용할 프레임 수 (config의 num_frames_test)
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_fp16 = use_fp16 and self.device == "cuda"
        self.num_frames = num_frames
        self.model = None
        self.tokenizer = None
        self._loaded = False
        self._model_type = "internvideo2"

        print(f'\n🤖 [VideoEmbedder 초기화]')
        print(f'   model_name : {model_name}')
        print(f'   device     : {self.device}')
        print(f'   use_fp16   : {self.use_fp16}  (VRAM 절약용, cuda일 때만 활성)')
        print(f'   num_frames : {self.num_frames}  (클립당 인코딩 프레임 수)')

    def load_model(self):
        """모델 로드 (Lazy Loading — 필요 시에만 메모리에 올림)"""
        if self._loaded:
            return

        print(f'\n📦 [모델 로드] Lazy Loading 시작 → {self.model_name}')
        try:
            self._load_internvideo2()
        except Exception as e:
            logger.warning(f"InternVideo2 로드 실패: {e}")
            print(f'   ⚠️  InternVideo2 로드 실패: {e}')
            print(f'   → CLIP ViT-B/32 폴백 모드로 전환')
            self._load_clip_fallback()

        self._loaded = True
        print(f'✅ [모델 로드 완료] model_type={self._model_type}  embed_dim={self.embed_dim}')

    def _build_internvideo2_config(self, pretrained_path: str):
        """
        InternVideo2 config 객체를 직접 구성.

        원본 config 파일(internvideo2_stage2_config.py)은 ${num_frames} 같은
        커스텀 문자열 치환 문법을 사용하므로 Python dict로 바로 쓸 수 없음.
        EasyDict로 동일한 구조를 직접 구성하고 ${} 값은 Python에서 대입.

        config 출처:
          InternVideo/InternVideo2/multi_modality/demo/internvideo2_stage2_config.py
          InternVideo/InternVideo2/multi_modality/configs/model.py
        """
        from easydict import EasyDict

        cfg = EasyDict()

        # ── 입력 설정 ──────────────────────────────────────────────
        cfg.num_frames = self.num_frames
        cfg.num_frames_test = self.num_frames
        cfg.batch_size = 8
        cfg.batch_size_test = 4
        cfg.size_t = 224
        cfg.max_txt_l = 40
        cfg.origin_num_frames = self.num_frames

        # ── 모델 설정 ──────────────────────────────────────────────
        cfg.model = EasyDict()
        cfg.model.model_cls = "InternVideo2_Stage2"
        cfg.model.embed_dim = 512   # vision_proj / text_proj 출력 차원
        cfg.model.temp = 0.07
        cfg.model.find_unused_parameters = False

        # vision encoder (pretrain_internvideo2_1b_patch14_224)
        cfg.model.vision_encoder = EasyDict()
        cfg.model.vision_encoder.name = "pretrain_internvideo2_1b_patch14_224"
        cfg.model.vision_encoder.img_size = 224
        cfg.model.vision_encoder.num_frames = self.num_frames
        cfg.model.vision_encoder.tubelet_size = 1
        cfg.model.vision_encoder.patch_size = 14
        cfg.model.vision_encoder.d_model = 1408
        cfg.model.vision_encoder.clip_embed_dim = 768
        cfg.model.vision_encoder.clip_teacher_embed_dim = 3200
        cfg.model.vision_encoder.clip_teacher_final_dim = 768
        cfg.model.vision_encoder.clip_norm_type = 'l2'
        cfg.model.vision_encoder.clip_return_layer = 6
        cfg.model.vision_encoder.clip_student_return_interval = 1
        cfg.model.vision_encoder.pretrained = None
        cfg.model.vision_encoder.use_checkpoint = True
        cfg.model.vision_encoder.checkpoint_num = 40
        cfg.model.vision_encoder.use_flash_attn = False
        cfg.model.vision_encoder.use_fused_rmsnorm = False
        cfg.model.vision_encoder.use_fused_mlp = False
        cfg.model.vision_encoder.clip_teacher = None
        cfg.model.vision_encoder.clip_input_resolution = 224
        cfg.model.vision_encoder.clip_teacher_return_interval = 1
        cfg.model.vision_encoder.video_mask_type = "random"
        cfg.model.vision_encoder.video_mask_ratio = 0.8
        cfg.model.vision_encoder.image_mask_type = "random"
        cfg.model.vision_encoder.image_mask_ratio = 0.5
        cfg.model.vision_encoder.sep_image_video_pos_embed = True
        cfg.model.vision_encoder.keep_temporal = False
        cfg.model.vision_encoder.only_mask = True

        # text encoder (BERT-large)
        cfg.model.text_encoder = EasyDict()
        cfg.model.text_encoder.name = "bert_large"
        cfg.model.text_encoder.pretrained = "bert-large-uncased"
        cfg.model.text_encoder.config = os.path.join(INTERNVIDEO2_CODE_PATH, "configs", "config_bert_large.json")
        cfg.model.text_encoder.d_model = 1024
        cfg.model.text_encoder.fusion_layer = 19

        # multimodal
        cfg.model.multimodal = EasyDict()
        cfg.model.multimodal.enable = True

        # ── 로드/실행 설정 ──────────────────────────────────────────
        cfg.pretrained_path = pretrained_path
        cfg.device = self.device
        cfg.use_half_precision = self.use_fp16
        cfg.use_bf16 = False
        cfg.gradient_checkpointing = True
        cfg.compile_model = False
        cfg.evaluate = True
        cfg.deep_fusion = False

        return cfg

    @staticmethod
    def _inject_flash_attn_stub():
        """
        flash_attn 더미 모듈을 sys.modules에 주입.

        왜 필요한가:
          InternVideo2 소스 코드가 모듈 레벨에서 flash_attn 하위 모듈을 import함.
          config에서 use_flash_attn=False로 설정하면 실제 호출은 없지만
          import 자체가 실패하면 모델 클래스 정의가 안 됨.
          Colab free T4의 CUDA/PyTorch 버전에 맞는 wheel이 없어 설치 불가.
        """
        import types
        import importlib.machinery

        if "flash_attn" in sys.modules:
            print(f'   flash_attn: 이미 등록됨 (주입 생략)')
            return

        def _make_module(name, is_package=False):
            mod = types.ModuleType(name)
            mod.__spec__ = importlib.machinery.ModuleSpec(
                name, None, is_package=is_package
            )
            sys.modules[name] = mod
            return mod

        flash_attn = _make_module("flash_attn", is_package=True)
        flash_attn.__version__ = "0.0.0"

        interface = _make_module("flash_attn.flash_attn_interface")
        interface.flash_attn_varlen_qkvpacked_func = None
        flash_attn.flash_attn_interface = interface

        bert_padding = _make_module("flash_attn.bert_padding")
        bert_padding.unpad_input = None
        bert_padding.pad_input = None
        flash_attn.bert_padding = bert_padding

        modules = _make_module("flash_attn.modules", is_package=True)
        mlp = _make_module("flash_attn.modules.mlp")
        mlp.FusedMLP = None
        modules.mlp = mlp
        flash_attn.modules = modules

        ops = _make_module("flash_attn.ops", is_package=True)
        rms_norm = _make_module("flash_attn.ops.rms_norm")
        rms_norm.DropoutAddRMSNorm = None
        ops.rms_norm = rms_norm
        flash_attn.ops = ops

        if "dropout_layer_norm" not in sys.modules:
            dln = _make_module("dropout_layer_norm")
            dln.dropout_add_layer_norm = None
            dln.dropout_add_layer_norm_subset = None
            dln.dropout_add_layer_norm_parallel_residual = None

        print(f'   flash_attn 더미 모듈 주입 완료 (use_flash_attn=False → 실제 호출 없음)')
        logger.info("flash_attn 더미 모듈 주입 완료 (use_flash_attn=False이므로 실제 호출 없음)")

    @staticmethod
    def _patch_internvideo2_source():
        """
        InternVideo2 소스 코드의 relative import 문제를 수정.
        멱등성 보장 — 이미 패치된 파일은 다시 수정하지 않음.
        """
        patches = [
            {
                "file": os.path.join(INTERNVIDEO2_CODE_PATH, "models", "__init__.py"),
                "replacements": [
                    (
                        "from .internvideo2_clip import InternVideo2_CLIP",
                        "# [VideoRAG patch] 아래 import들은 peft/LLaMA 등\n"
                        "# 불필요한 의존성을 끌어오므로 비활성화.\n"
                        "# 우리는 models.backbones.internvideo2만 사용함.\n"
                        "# from .internvideo2_clip import InternVideo2_CLIP",
                    ),
                    ("from .internvideo2_clip_small import InternVideo2_CLIP_small",
                     "# from .internvideo2_clip_small import InternVideo2_CLIP_small"),
                    ("from .internvideo2_stage2_visual import InternVideo2_Stage2_visual",
                     "# from .internvideo2_stage2_visual import InternVideo2_Stage2_visual"),
                    ("from .internvideo2_stage2_audiovisual import InternVideo2_Stage2_audiovisual",
                     "# from .internvideo2_stage2_audiovisual import InternVideo2_Stage2_audiovisual"),
                ]
            },
            {
                "file": os.path.join(INTERNVIDEO2_CODE_PATH, "models", "criterions.py"),
                "replacements": [
                    ("from ..utils.distributed import", "from utils.distributed import"),
                    ("from ..utils.easydict import", "from easydict import"),
                ]
            },
            {
                "file": os.path.join(INTERNVIDEO2_CODE_PATH, "models", "utils.py"),
                "replacements": [
                    ("from ..utils.distributed import", "from utils.distributed import"),
                ]
            },
            {
                "file": os.path.join(INTERNVIDEO2_CODE_PATH, "demo", "utils.py"),
                "replacements": [
                    (
                        "from models.backbones.bert.tokenization_bert import BertTokenizer",
                        "# [VideoRAG patch] 커스텀 tokenization_bert는 transformers 4.50+에서\n"
                        "# _is_control 등 제거된 내부 함수를 import하므로 실패.\n"
                        "# 표준 BertTokenizer와 동일하므로 교체.\n"
                        "from transformers import BertTokenizer",
                    ),
                ]
            },
        ]

        patched_files = []
        for patch in patches:
            fpath = patch["file"]
            if not os.path.exists(fpath):
                continue

            with open(fpath, "r") as f:
                content = f.read()

            modified = False
            for old, new in patch["replacements"]:
                if old in content:
                    content = content.replace(old, new)
                    modified = True

            if modified:
                with open(fpath, "w") as f:
                    f.write(content)
                patched_files.append(os.path.basename(fpath))
                logger.info(f"소스 패치 완료: {os.path.basename(fpath)}")

        if patched_files:
            print(f'   소스 패치 적용: {patched_files}')
        else:
            print(f'   소스 패치: 이미 적용됨 (멱등성 — 재패치 생략)')

        # ── xbert.py 별도 처리 ──────────────────────────────────────
        xbert_path = os.path.join(
            INTERNVIDEO2_CODE_PATH, "models", "backbones", "bert", "xbert.py"
        )
        if os.path.exists(xbert_path):
            with open(xbert_path, "r") as f:
                xbert_content = f.read()

            if ("apply_chunking_to_forward" in xbert_content
                    and "[VideoRAG compat]" not in xbert_content):
                compat_block = (
                    "# [VideoRAG compat] transformers 4.40+ 호환 레이어\n"
                    "# apply_chunking_to_forward 등이 버전에 따라 위치가 다름.\n"
                    "# 어디서든 찾을 수 없으면 직접 구현을 제공.\n"
                    "import torch as _torch\n"
                    "import transformers as _tf\n"
                    "\n"
                    "def _compat_apply_chunking_to_forward(forward_fn, chunk_size, chunk_dim, *input_tensors):\n"
                    "    if chunk_size > 0:\n"
                    "        tensor_shape = input_tensors[0].shape[chunk_dim]\n"
                    "        for input_tensor in input_tensors:\n"
                    "            if input_tensor.shape[chunk_dim] != tensor_shape:\n"
                    "                raise ValueError('All input tensors must have the same shape')\n"
                    "        if tensor_shape % chunk_size != 0:\n"
                    "            raise ValueError(f'dimension {tensor_shape} not divisible by {chunk_size}')\n"
                    "        num_chunks = tensor_shape // chunk_size\n"
                    "        input_chunks = tuple(t.chunk(num_chunks, dim=chunk_dim) for t in input_tensors)\n"
                    "        output_chunks = [forward_fn(*ic) for ic in zip(*input_chunks)]\n"
                    "        return _torch.cat(output_chunks, dim=chunk_dim)\n"
                    "    return forward_fn(*input_tensors)\n"
                    "\n"
                    "def _compat_find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):\n"
                    "    mask = _torch.ones(n_heads, head_size)\n"
                    "    heads = set(heads) - already_pruned_heads\n"
                    "    for head in heads:\n"
                    "        head = head - sum(1 if h < head else 0 for h in already_pruned_heads)\n"
                    "        mask[head] = 0\n"
                    "    mask = mask.view(-1).contiguous().eq(1)\n"
                    "    index = _torch.arange(len(mask))[mask].long()\n"
                    "    return heads, index\n"
                    "\n"
                    "def _compat_prune_linear_layer(layer, index, dim=0):\n"
                    "    index = index.to(layer.weight.device)\n"
                    "    W = layer.weight.index_select(dim, index).clone().detach()\n"
                    "    if layer.bias is not None:\n"
                    "        if dim == 1:\n"
                    "            b = layer.bias.clone().detach()\n"
                    "        else:\n"
                    "            b = layer.bias[index].clone().detach()\n"
                    "    new_size = list(layer.weight.size())\n"
                    "    new_size[dim] = len(index)\n"
                    "    new_layer = _torch.nn.Linear(new_size[1], new_size[0], bias=layer.bias is not None)\n"
                    "    new_layer = new_layer.to(layer.weight.device)\n"
                    "    new_layer.weight.requires_grad = False\n"
                    "    new_layer.weight.copy_(W.contiguous())\n"
                    "    new_layer.weight.requires_grad = True\n"
                    "    if layer.bias is not None:\n"
                    "        new_layer.bias.requires_grad = False\n"
                    "        new_layer.bias.copy_(b.contiguous())\n"
                    "        new_layer.bias.requires_grad = True\n"
                    "    return new_layer\n"
                    "\n"
                    "if not hasattr(_tf.modeling_utils, 'apply_chunking_to_forward'):\n"
                    "    _tf.modeling_utils.apply_chunking_to_forward = _compat_apply_chunking_to_forward\n"
                    "if not hasattr(_tf.modeling_utils, 'find_pruneable_heads_and_indices'):\n"
                    "    _tf.modeling_utils.find_pruneable_heads_and_indices = _compat_find_pruneable_heads_and_indices\n"
                    "if not hasattr(_tf.modeling_utils, 'prune_linear_layer'):\n"
                    "    _tf.modeling_utils.prune_linear_layer = _compat_prune_linear_layer\n"
                    "\n"
                    "# [VideoRAG compat] get_head_mask — transformers 4.48+ 에서 제거됨\n"
                    "if not hasattr(_tf.PreTrainedModel, 'get_head_mask'):\n"
                    "    def _compat_get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):\n"
                    "        if head_mask is not None:\n"
                    "            if head_mask.dim() == 1:\n"
                    "                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)\n"
                    "                head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)\n"
                    "            elif head_mask.dim() == 2:\n"
                    "                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)\n"
                    "            if is_attention_chunked:\n"
                    "                head_mask = head_mask.unsqueeze(-1)\n"
                    "        else:\n"
                    "            head_mask = [None] * num_hidden_layers\n"
                    "        return head_mask\n"
                    "    _tf.PreTrainedModel.get_head_mask = _compat_get_head_mask\n"
                    "\n"
                    "del _tf, _torch\n\n"
                )
                xbert_content = compat_block + xbert_content
                with open(xbert_path, "w") as f:
                    f.write(xbert_content)
                print(f'   소스 패치 적용: xbert.py (transformers 호환 레이어 삽입)')
                logger.info("소스 패치 완료: xbert.py (transformers 호환)")

        if os.path.exists(xbert_path):
            with open(xbert_path, "r") as f:
                xbert_content = f.read()

            if ("_tied_weights_keys" not in xbert_content
                    and "class BertPreTrainedModel" in xbert_content):
                xbert_content = xbert_content.replace(
                    "class BertPreTrainedModel(PreTrainedModel):",
                    "class BertPreTrainedModel(PreTrainedModel):\n"
                    "    # [VideoRAG patch] transformers 4.40+ 호환\n"
                    "    _tied_weights_keys = []\n"
                )
                with open(xbert_path, "w") as f:
                    f.write(xbert_content)
                print(f'   소스 패치 적용: xbert.py (_tied_weights_keys 추가)')
                logger.info("소스 패치 완료: xbert.py (_tied_weights_keys 추가)")

    @staticmethod
    def _patch_transformers_tied_weights():
        """BertPreTrainedModel에 _tied_weights_keys 속성을 런타임 주입.

        이 패치는 보조 안전망입니다 — _patch_internvideo2_source()가 이미
        xbert.py 소스 파일에 _tied_weights_keys = [] 를 직접 삽입하므로,
        이 런타임 패치가 실패해도 모델 로드에는 영향이 없습니다.
        """
        # InternVideo 코드가 아직 clone 안 된 경우 조용히 스킵
        models_dir = os.path.join(INTERNVIDEO2_CODE_PATH, "models")
        if not os.path.isdir(models_dir):
            logger.debug(
                f"InternVideo2 models 디렉토리 없음 ({models_dir}) "
                "→ 런타임 패치 생략 (소스 패치로 대체)"
            )
            print(f'   BertPreTrainedModel 런타임 패치: 소스 패치로 대체 (생략)')
            return

        try:
            if INTERNVIDEO2_CODE_PATH not in sys.path:
                sys.path.insert(0, INTERNVIDEO2_CODE_PATH)

            from models.backbones.bert.xbert import BertPreTrainedModel

            if getattr(BertPreTrainedModel, '_videorag_patched', False):
                print(f'   BertPreTrainedModel 패치: 이미 적용됨 (생략)')
                return

            BertPreTrainedModel.all_tied_weights_keys = {}
            BertPreTrainedModel._videorag_patched = True
            print(f'   BertPreTrainedModel.all_tied_weights_keys 패치 완료')
            logger.info("BertPreTrainedModel.all_tied_weights_keys 패치 완료")

        except Exception as e:
            # 소스 패치가 이미 적용됐으므로 런타임 패치 실패는 치명적이지 않음
            logger.debug(f"BertPreTrainedModel 런타임 패치 생략: {e}")
            print(f'   BertPreTrainedModel 런타임 패치 생략 (소스 패치 사용)')

    def _load_internvideo2(self):
        """
        InternVideo2-1B 모델 로드 (GitHub 코드 기반)

        HuggingFace에 표준 포맷 파일 없음 → DeepSpeed .pt 체크포인트 직접 로드
        """
        # ── 사전 검증: InternVideo 코드가 clone 되어 있는지 확인 ──
        demo_dir = os.path.join(INTERNVIDEO2_CODE_PATH, "demo")
        models_dir = os.path.join(INTERNVIDEO2_CODE_PATH, "models")
        if not os.path.isdir(demo_dir) or not os.path.isdir(models_dir):
            raise ModuleNotFoundError(
                f"InternVideo2 코드가 없습니다.\n"
                f"  확인 경로: {INTERNVIDEO2_CODE_PATH}\n"
                f"  demo/ 존재: {os.path.isdir(demo_dir)}\n"
                f"  models/ 존재: {os.path.isdir(models_dir)}\n\n"
                f"  → 노트북의 Step 0.5 (InternVideo sparse checkout) 셀을 먼저 실행하세요.\n"
                f"  → 셀 내용:\n"
                f"     !git clone --no-checkout --depth=1 "
                f"https://github.com/OpenGVLab/InternVideo.git /content/InternVideo\n"
                f"     %cd /content/InternVideo\n"
                f"     !git sparse-checkout init --cone\n"
                f"     !git sparse-checkout set InternVideo2/multi_modality\n"
                f"     !git checkout main"
            )

        print(f'\n   [InternVideo2 로드 Step 0a] flash_attn 더미 모듈 주입')
        self._inject_flash_attn_stub()

        print(f'   [InternVideo2 로드 Step 0b] 소스 relative import 패치')
        self._patch_internvideo2_source()

        print(f'   [InternVideo2 로드 Step 1] sys.path에 InternVideo2 코드 등록')
        if INTERNVIDEO2_CODE_PATH not in sys.path:
            sys.path.insert(0, INTERNVIDEO2_CODE_PATH)
            print(f'     추가됨: {INTERNVIDEO2_CODE_PATH}')
        else:
            print(f'     이미 등록됨: {INTERNVIDEO2_CODE_PATH}')

        # clone 이전에 sys.path 등록 → import 캐시 오염 방지
        import importlib
        importlib.invalidate_caches()

        print(f'   [InternVideo2 로드 Step 0c] BertPreTrainedModel monkey-patch')
        self._patch_transformers_tied_weights()

        print(f'   [InternVideo2 로드 Step 2] HuggingFace에서 .pt 체크포인트 다운로드')
        from huggingface_hub import hf_hub_download

        hf_token = os.environ.get("HF_TOKEN")
        pt_filename = "InternVideo2-stage2_1b-224p-f4.pt"
        print(f'     저장소: {self.model_name}')
        print(f'     파일명: {pt_filename}')
        print(f'     HF_TOKEN: {"설정됨" if hf_token else "미설정 (공개 저장소면 무관)"}')

        pretrained_path = hf_hub_download(
            repo_id=self.model_name,
            filename=pt_filename,
            token=hf_token
        )
        print(f'     다운로드 완료: {pretrained_path}')
        logger.info(f"체크포인트 경로: {pretrained_path}")

        print(f'   [InternVideo2 로드 Step 3] BertTokenizer 로드 (bert-large-uncased)')
        from transformers import BertTokenizer
        self.tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")
        print(f'     BertTokenizer 로드 완료  (vocab_size={self.tokenizer.vocab_size})')
        logger.info("BertTokenizer(bert-large-uncased) 로드 완료")

        print(f'   [InternVideo2 로드 Step 4] EasyDict config 구성')
        cfg = self._build_internvideo2_config(pretrained_path)
        print(f'     embed_dim={cfg.model.embed_dim}  num_frames={cfg.num_frames}')
        print(f'     vision_encoder: {cfg.model.vision_encoder.name}')
        print(f'     text_encoder  : {cfg.model.text_encoder.name} ({cfg.model.text_encoder.pretrained})')
        print(f'     use_fp16      : {cfg.use_half_precision}')

        print(f'   [InternVideo2 로드 Step 5] setup_internvideo2() 호출 → 가중치 로드 중...')
        print(f'     (DeepSpeed stage1 포맷 .pt → state_dict 추출 → load_state_dict)')
        from demo.utils import setup_internvideo2
        prev_cwd = os.getcwd()
        try:
            os.chdir(INTERNVIDEO2_CODE_PATH)
            self.model, _ = setup_internvideo2(cfg)
        finally:
            os.chdir(prev_cwd)
        self.model.eval()

        self._model_type = "internvideo2"
        print(f'     모델 eval 모드 전환 완료')
        logger.info("InternVideo2-1B 로드 완료 (embed_dim=512)")

    def _load_clip_fallback(self):
        """
        폴백: CLIP ViT-B/32

        출처: "Learning Transferable Visual Models From Natural Language Supervision"
              (Radford et al., 2021) — OpenAI
              https://arxiv.org/abs/2103.00020
        """
        try:
            from transformers import CLIPModel, CLIPProcessor

            fallback_name = "openai/clip-vit-base-patch32"
            print(f'   [CLIP 폴백] {fallback_name} 로드 중...')
            logger.info(f"CLIP 폴백 모델 로딩: {fallback_name}")

            self.model = CLIPModel.from_pretrained(fallback_name).to(self.device).eval()
            self.tokenizer = CLIPProcessor.from_pretrained(fallback_name)
            self._model_type = "clip"
            print(f'   [CLIP 폴백] 로드 완료  embed_dim=512  device={self.device}')
            logger.info("CLIP ViT-B/32 폴백 로드 완료 (embed_dim=512)")
        except Exception as e2:
            logger.error(f"CLIP 폴백도 실패: {e2}")
            print(f'   ⚠️  CLIP 폴백도 실패: {e2}')
            print(f'   → 랜덤 임베딩 모드로 전환 (구조 테스트용, 실제 검색 불가)')
            self._model_type = "random"

    @torch.no_grad()
    def encode_clips(
        self,
        clip_frames: List[np.ndarray],
        batch_size: int = 8
    ) -> np.ndarray:
        """영상 클립 프레임들을 임베딩 벡터로 인코딩

        Args:
            clip_frames: 클립별 대표 프레임 리스트 [N, H, W, 3] (uint8)
            batch_size: 배치 크기 (T4 기준 8 권장 — InternVideo2가 CLIP보다 무거움)

        Returns:
            np.ndarray [N, 512] float32
        """
        self.load_model()

        n_clips = len(clip_frames)
        print(f'\n🎞️  [encode_clips] 클립 임베딩 시작')
        print(f'   클립 수   : {n_clips}개')
        print(f'   batch_size: {batch_size}')
        print(f'   model_type: {self._model_type}')
        n_batches = (n_clips + batch_size - 1) // batch_size
        print(f'   배치 수   : {n_batches}개')

        if self._model_type == "random":
            print(f'   ⚠️  랜덤 임베딩 모드 → shape=({n_clips}, {EMBED_DIM})')
            return self._random_embed(n_clips)

        all_embeddings = []

        for i in range(0, n_clips, batch_size):
            batch = clip_frames[i:i + batch_size]
            batch_embs = self._encode_image_batch(batch)
            all_embeddings.append(batch_embs)

            batch_idx = i // batch_size + 1
            if batch_idx % 10 == 0 or batch_idx == n_batches:
                print(f'   배치 진행: {batch_idx}/{n_batches}  '
                      f'({min(i + batch_size, n_clips)}/{n_clips} clips)')
            logger.info(
                f"임베딩 진행: {min(i + batch_size, n_clips)}/{n_clips}"
            )

        embeddings = np.vstack(all_embeddings)

        # L2 정규화 (cosine similarity 사전 준비)
        # InternVideo2의 get_vid_feat()가 이미 정규화하지만
        # CLIP 폴백 경로를 위해 여기서도 한 번 더 수행
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        embeddings = embeddings / norms

        embeddings = embeddings.astype(np.float32)
        print(f'   L2 정규화 완료')
        print(f'✅ [encode_clips 완료] shape={embeddings.shape}  dtype={embeddings.dtype}')
        return embeddings

    @torch.no_grad()
    def encode_query(self, text: str) -> np.ndarray:
        """텍스트 쿼리를 임베딩 벡터로 인코딩

        Args:
            text: 자연어 쿼리 (e.g., "a man playing guitar")

        Returns:
            np.ndarray [1, 512] float32
        """
        self.load_model()

        print(f'\n🔤 [encode_query] 텍스트 임베딩')
        print(f'   query      : "{text}"')
        print(f'   model_type : {self._model_type}')

        if self._model_type == "random":
            print(f'   ⚠️  랜덤 임베딩 모드')
            return self._random_embed(1)

        if self._model_type == "clip":
            inputs = self.tokenizer(
                text=[text], return_tensors="pt", padding=True
            ).to(self.device)
            text_features = self.model.get_text_features(**inputs)
            if isinstance(text_features, torch.Tensor):
                emb = text_features.cpu().float().numpy()
            else:
                emb = text_features.pooler_output.cpu().float().numpy()
        else:
            # InternVideo2: get_txt_feat() — BertTokenizer 토큰화 → text_proj → L2 정규화
            emb = self.model.get_txt_feat(text).cpu().float().numpy()

        # L2 정규화
        norm = np.linalg.norm(emb, axis=1, keepdims=True)
        emb = emb / np.maximum(norm, 1e-8)
        emb = emb.astype(np.float32)

        print(f'   임베딩 완료: shape={emb.shape}  norm≈{np.linalg.norm(emb):.4f} (L2 정규화 후 ≈1.0)')
        return emb

    def _encode_image_batch(
        self,
        clip_frame_lists: List[List[np.ndarray]]
    ) -> np.ndarray:
        """클립 배치 인코딩

        Args:
            clip_frame_lists: 클립당 프레임 리스트
                              List[List[np.ndarray]] — [B, num_frames, H, W, 3]
        """
        if self._model_type == "clip":
            # CLIP은 단일 프레임 기반 → 각 클립의 중간 프레임만 사용
            from PIL import Image
            mid_idx = self.num_frames // 2
            images = [Image.fromarray(fl[mid_idx]) for fl in clip_frame_lists]
            inputs = self.tokenizer(
                images=images, return_tensors="pt", padding=True
            ).to(self.device)
            features = self.model.get_image_features(**inputs)
            if isinstance(features, torch.Tensor):
                return features.cpu().float().numpy()
            else:
                return features.pooler_output.cpu().float().numpy()

        else:
            # InternVideo2: 균등 샘플링된 num_frames장 프레임을 그대로 사용
            # frames2tensor(): List[frame] → [1, T, C, H, W]
            if INTERNVIDEO2_CODE_PATH not in sys.path:
                sys.path.insert(0, INTERNVIDEO2_CODE_PATH)
            from demo.utils import frames2tensor

            batch_tensors = []
            for frame_list in clip_frame_lists:
                tensor = frames2tensor(
                    frame_list,
                    fnum=self.num_frames,
                    target_size=(224, 224),
                    device=torch.device(self.device)
                )
                batch_tensors.append(tensor)

            # [B, T, C, H, W]로 합치기
            batch = torch.cat(batch_tensors, dim=0)

            if self.use_fp16:
                batch = batch.half()

            # get_vid_feat(): vision_encoder → vision_proj → L2 정규화
            # 반환 shape: [B, embed_dim=512]
            feat = self.model.get_vid_feat(batch)
            return feat.cpu().float().numpy()

    @torch.no_grad()
    def encode_clips_itm(
        self,
        clip_frames: List[List[np.ndarray]],
        batch_size: int = 4,
    ) -> "torch.Tensor":
        """ITM용 vision full token 시퀀스 추출 — InternVideo2 전용

        ITC용 encode_clips()는 vision_proj 후 CLS 512-dim 벡터만 반환하지만,
        ITM은 text 40토큰과 cross-attention하려면 vision encoder의 전체 토큰이 필요.

        encode_vision()[0] = last_hidden_state [B, 1025, 1408]
          - 1025 = 1(cls) + 1024(patch tokens, 4프레임 × 14×14×4 = 1024... T4 224px 기준)
          - 1408 = vision encoder hidden dim (d_model)

        Args:
            clip_frames: 클립별 프레임 리스트 List[List[ndarray]] — [N, num_frames, H, W, 3]
            batch_size: 배치 크기 (full token은 무거우므로 4 권장, OOM 시 2로 낮춤)

        Returns:
            torch.Tensor [N, 1025, 1408] float16, CPU
            (메모리 절약: fp16 + CPU 보관, 사용 시 .to(device, dtype=model_dtype))
        """
        self.load_model()

        if self._model_type != "internvideo2":
            raise RuntimeError(
                "encode_clips_itm()는 InternVideo2 모델에서만 지원됩니다. "
                f"현재 model_type='{self._model_type}'"
            )

        if INTERNVIDEO2_CODE_PATH not in sys.path:
            sys.path.insert(0, INTERNVIDEO2_CODE_PATH)
        from demo.utils import frames2tensor

        n_clips = len(clip_frames)
        n_batches = (n_clips + batch_size - 1) // batch_size
        print(f'\n🎞️  [encode_clips_itm] ITM용 vision full token 추출')
        print(f'   클립 수   : {n_clips}개')
        print(f'   batch_size: {batch_size}  (ITC보다 작음 — full token 메모리 부담)')
        print(f'   예상 출력 : [{n_clips}, 1025, 1408] fp16')

        all_vis_feats = []

        for i in range(0, n_clips, batch_size):
            batch_frame_lists = clip_frames[i:i + batch_size]
            batch_tensors = []

            for frame_list in batch_frame_lists:
                tensor = frames2tensor(
                    frame_list,
                    fnum=self.num_frames,
                    target_size=(224, 224),
                    device=torch.device(self.device)
                )
                batch_tensors.append(tensor)

            # [B, T, C, H, W]
            batch = torch.cat(batch_tensors, dim=0)
            if self.use_fp16:
                batch = batch.half()

            # encode_vision()[0] = last_hidden_state [B, 1025, 1408]
            # (ITC의 get_vid_feat()는 vision_proj(cls_token) → 512-dim만 반환)
            vis_feat, _ = self.model.encode_vision(batch, test=True)

            # fp16으로 CPU 저장 (VRAM 해제)
            all_vis_feats.append(vis_feat.cpu().half())

            batch_idx = i // batch_size + 1
            if batch_idx % 5 == 0 or batch_idx == n_batches:
                print(f'   배치 진행: {batch_idx}/{n_batches}  '
                      f'({min(i + batch_size, n_clips)}/{n_clips} clips)')

        result = torch.cat(all_vis_feats, dim=0)  # [N, 1025, 1408]
        mem_gb = result.element_size() * result.nelement() / 1e9
        print(f'✅ [encode_clips_itm 완료] shape={tuple(result.shape)}  '
              f'dtype={result.dtype}  메모리≈{mem_gb:.2f}GB')
        return result

    def _random_embed(self, n: int) -> np.ndarray:
        """랜덤 임베딩 (모델 없을 때 구조 테스트용)"""
        emb = np.random.randn(n, EMBED_DIM).astype(np.float32)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        return emb / norms

    @property
    def embed_dim(self) -> int:
        if self._model_type == "clip":
            return 512
        return EMBED_DIM  # InternVideo2도 512