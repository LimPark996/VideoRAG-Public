"""
⑦b itm_scorer.py — ITM (Image-Text Matching) 재순위화

역할:
  ColBERT 재순위 이후, 상위 k개 후보에 대해 InternVideo2의
  ITM cross-attention 점수로 최종 재순위를 수행한다.

배경 (이슈 리포트 24~26차 진단):
  ITC (Image-Text Contrastive):
    - text_proj(CLS) × vision_proj(pooled) cosine → 512-dim 단일 벡터 비교
    - CLS 토큰이 attention sink 현상으로 collapse → 서로 다른 텍스트도 cosine≈0.9997
    - 1k-A R@1 = 3.5% (공식 51.9%와 크게 괴리)

  ITM (Image-Text Matching):
    - text 전체 40토큰 × vision 전체 1025토큰 cross-attention (mode="fusion")
    - "woman", "cooking" 등 content 토큰이 영상 patch와 직접 대화
    - itm_head(Linear 1024→2)가 매칭 점수 산출
    - 1k-A R@1 = 44.5% (ITC 대비 +41%p)

  itm_head는 InternVideo2_Stage2 클래스에 정의되어 있지 않아
  체크포인트 로딩 시 unexpected_keys로 무시되므로, 수동으로 추출·탑재해야 함.

사용 패턴:
  # 파이프라인 초기화 시
  itm_scorer = ITMScorer(
      iv_model=pipeline.embedder.model,
      itm_features_path='index/itm_vision_features.pt'
  )
  # 쿼리 타임 (ColBERT 이후)
  reranked = itm_scorer.rerank(en_query, colbert_top_k, top_k=20)
"""

import logging
from typing import List, Dict, Optional, Tuple

import torch
import torch.nn as nn

from ..data_models import ClipResult

logger = logging.getLogger(__name__)

# ITM text encoder hidden dim (BERT-large: 1024)
ITM_HIDDEN_DIM = 1024
# itm_head 출력: 이진 분류 (0=no-match, 1=match) → logit[:, 1] 사용
ITM_HEAD_OUT = 2


class ITMScorer:
    """InternVideo2 ITM cross-attention 기반 재순위화

    ColBERT 이후 상위 k개 후보에 대해 ITM 점수로 최종 재순위.

    핵심 메커니즘:
      1. query text → encode_text()[0] → [1, 40, 1024] (전체 토큰 시퀀스)
      2. candidate clip → 사전 계산된 vis_feat [1025, 1408] 로드
      3. enc(encoder_embeds=text_feat, encoder_hidden_states=vis_feat, mode='fusion')
         → fusion CLS [1, 1024]
      4. itm_head(fusion_cls)[:, 1] → 매칭 점수 (높을수록 텍스트-영상 매칭)
    """

    def __init__(
        self,
        iv_model,
        itm_features_path: str,
        device: Optional[str] = None,
        bs_itm: int = 32,
    ):
        """
        Args:
            iv_model: VideoEmbedder.model (InternVideo2_Stage2 인스턴스)
            itm_features_path: 사전 계산된 vision feature 경로
                               (indexer.build_itm_vision_features() 출력)
            device: 'cuda' or 'cpu' (None이면 iv_model에서 자동 감지)
            bs_itm: ITM 배치 크기 (OOM 시 16으로 낮춤)
        """
        self.iv_model = iv_model
        self.itm_features_path = itm_features_path
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.bs_itm = bs_itm

        # 지연 초기화 (load() 호출 전까지 None)
        self.itm_head: Optional[nn.Linear] = None
        self.vis_feat_dict: Optional[Dict[str, torch.Tensor]] = None
        self._loaded = False

        print(f'\n🔗 [ITMScorer 초기화]')
        print(f'   itm_features_path: {itm_features_path}')
        print(f'   device           : {self.device}')
        print(f'   bs_itm           : {bs_itm}')

    def load(self):
        """itm_head + vision feature 사전 데이터 로드"""
        if self._loaded:
            return

        self._load_itm_head()
        self._load_vision_features()
        self._loaded = True
        print(f'✅ [ITMScorer 로드 완료] '
              f'clips={len(self.vis_feat_dict)}개  device={self.device}')

    def _load_itm_head(self):
        """체크포인트에서 itm_head 수동 추출 및 탑재

        InternVideo2_Stage2 클래스에 itm_head가 정의되어 있지 않아
        체크포인트 로딩 시 unexpected_keys로 무시됨.
        → state_dict에서 직접 꺼내 nn.Linear로 재구성.

        itm_head.weight: [2, 1024]
        itm_head.bias:   [2]
        → Linear(in_features=1024, out_features=2)
        """
        print(f'   [ITMScorer] itm_head 체크포인트에서 추출 중...')

        ckpt_path = self.iv_model.config.pretrained_path
        state_dict = torch.load(ckpt_path, map_location='cpu')['module']

        itm_keys = {k: v for k, v in state_dict.items() if 'itm_head' in k}
        if not itm_keys:
            raise KeyError(
                f"체크포인트에서 itm_head 가중치를 찾을 수 없습니다.\n"
                f"  경로: {ckpt_path}\n"
                f"  'itm_head'를 포함한 키가 없습니다."
            )

        for k, v in itm_keys.items():
            print(f'     {k}: {tuple(v.shape)}')

        self.itm_head = nn.Linear(ITM_HIDDEN_DIM, ITM_HEAD_OUT)
        self.itm_head.weight = nn.Parameter(
            state_dict['itm_head.weight'].float()
        )
        self.itm_head.bias = nn.Parameter(
            state_dict['itm_head.bias'].float()
        )
        self.itm_head = self.itm_head.to(self.device)
        self.itm_head.eval()

        # iv_model에도 탑재 (외부에서 model.itm_head로 접근 가능하도록)
        self.iv_model.itm_head = self.itm_head

        del state_dict
        print(f'   [ITMScorer] itm_head 탑재 완료  Linear({ITM_HIDDEN_DIM}→{ITM_HEAD_OUT})  device={self.device}')
        logger.info(f'ITMScorer: itm_head 로드 완료 (Linear {ITM_HIDDEN_DIM}→{ITM_HEAD_OUT})')

    def _load_vision_features(self):
        """사전 계산된 vision full token 로드

        indexer.build_itm_vision_features() 결과:
          {'clip_ids': List[str], 'features': Tensor[N, 1025, 1408] fp16}
        → self.vis_feat_dict: {clip_id → Tensor[1025, 1408] fp16 CPU}
        """
        print(f'   [ITMScorer] vision feature 로드 중: {self.itm_features_path}')

        data = torch.load(self.itm_features_path, map_location='cpu')
        clip_ids = data['clip_ids']
        features = data['features']  # [N, 1025, 1408] fp16

        print(f'     clip 수   : {len(clip_ids)}개')
        print(f'     shape     : {tuple(features.shape)}')
        print(f'     dtype     : {features.dtype}')
        mem_gb = features.element_size() * features.nelement() / 1e9
        print(f'     메모리    : {mem_gb:.2f}GB (CPU)')

        self.vis_feat_dict = {
            cid: features[i]  # [1025, 1408] fp16 — view (메모리 공유)
            for i, cid in enumerate(clip_ids)
        }
        logger.info(
            f'ITMScorer: vision feature 로드 완료 '
            f'({len(clip_ids)} clips, {tuple(features.shape)}, {mem_gb:.2f}GB)'
        )

    @torch.no_grad()
    def score_batch(
        self,
        text_feats: torch.Tensor,   # [B, 40, 1024] fp16
        text_atts: torch.Tensor,    # [B, 40]
        vis_feat: torch.Tensor,     # [1, 1025, 1408] fp16
    ) -> torch.Tensor:
        """텍스트 배치 × 단일 영상 ITM 점수 계산

        Args:
            text_feats: 텍스트 full token [B, 40, 1024]
            text_atts: 어텐션 마스크 [B, 40]
            vis_feat: vision full token [1, 1025, 1408]

        Returns:
            torch.Tensor [B] — 각 텍스트의 ITM 매칭 점수 (logit[:, 1])
        """
        enc = self.iv_model.get_text_encoder()
        model_dtype = next(self.iv_model.parameters()).dtype  # fp16

        bs = text_feats.shape[0]
        tf = text_feats.to(self.device, dtype=model_dtype)
        ta = text_atts.to(self.device)
        vf = vis_feat.to(self.device, dtype=model_dtype).expand(bs, -1, -1)

        out = enc(
            encoder_embeds=tf,
            attention_mask=ta,
            encoder_hidden_states=vf,
            encoder_attention_mask=None,
            return_dict=True,
            mode="fusion",
        )
        # itm_head는 fp32 가중치 → fusion CLS를 float()으로 캐스팅
        scores = self.itm_head(out.last_hidden_state[:, 0].float())[:, 1]
        return scores.cpu()

    @torch.no_grad()
    def rerank(
        self,
        query: str,
        candidates: List[ClipResult],
        top_k: int = 20,
    ) -> List[ClipResult]:
        """ColBERT 후보에 ITM 점수 적용 후 재순위

        Args:
            query: 영어 자연어 쿼리 (e.g. "a woman cooking food")
            candidates: ColBERT 재순위 결과 (ClipResult 리스트)
            top_k: 최종 반환 클립 수

        Returns:
            List[ClipResult] — ITM 점수로 정렬된 상위 top_k개
                               (ClipResult.score = ITM logit[:, 1])
        """
        self.load()

        if not candidates:
            return []

        # ── 1. 텍스트 인코딩: encode_text()[0] = full token [1, 40, 1024] ──
        tokenizer = self.iv_model.tokenizer
        max_txt_l = self.iv_model.config.max_txt_l  # 40

        tok = tokenizer(
            [query],
            padding="max_length",
            truncation=True,
            max_length=max_txt_l,
            return_tensors="pt",
        )
        tok = tok.to(self.device)  # BatchEncoding.to() — 구조 유지

        # encode_text()[0] = last_hidden_state [1, 40, 1024]  (full 토큰 시퀀스)
        # encode_text()[1] = pooled_text_embeds [1, 1024]     (ITC용 CLS — 여기선 불필요)
        text_feat_full, _ = self.iv_model.encode_text(tok)
        text_feat_cpu = text_feat_full.cpu().half()    # [1, 40, 1024] fp16 CPU
        text_att_cpu = tok['attention_mask'].cpu()     # [1, 40]

        # 후보 수만큼 repeat (배치 처리용)
        n_cands = len(candidates)
        text_feats_rep = text_feat_cpu.expand(n_cands, -1, -1)  # [N, 40, 1024]
        text_atts_rep = text_att_cpu.expand(n_cands, -1)        # [N, 40]

        # ── 2. 영상별 ITM 점수 계산 (배치로 처리) ─────────────────────
        itm_scores_all = torch.zeros(n_cands)
        missing_count = 0

        for vi, clip in enumerate(candidates):
            cid = clip.clip_id
            if cid not in self.vis_feat_dict:
                missing_count += 1
                itm_scores_all[vi] = -999.0  # feature 없는 clip은 최하위
                continue

            vis_feat = self.vis_feat_dict[cid].unsqueeze(0)  # [1, 1025, 1408]

            # 단일 영상에 대해 텍스트 1개 scoring
            # (여러 쿼리를 batch로 처리할 수 있지만 여기선 쿼리=1개)
            score = self.score_batch(
                text_feats=text_feat_cpu,   # [1, 40, 1024]
                text_atts=text_att_cpu,     # [1, 40]
                vis_feat=vis_feat,          # [1, 1025, 1408]
            )
            itm_scores_all[vi] = score.item()

        if missing_count > 0:
            logger.warning(
                f'ITM feature 누락: {missing_count}/{n_cands}개 clip '
                f'(itm_vision_features.pt에 없는 clip_id)'
            )

        # ── 3. ITM 점수로 재순위 ──────────────────────────────────────
        sorted_indices = itm_scores_all.argsort(descending=True)[:top_k]

        result = []
        for rank, idx in enumerate(sorted_indices):
            clip = candidates[idx.item()]
            # score 필드를 ITM logit으로 덮어씀
            clip.score = float(itm_scores_all[idx].item())
            result.append(clip)

        logger.info(
            f'ITMScorer.rerank: {n_cands}개 → top-{top_k} | '
            f'상위 ITM 점수: {itm_scores_all.max():.3f}'
        )
        return result

    @torch.no_grad()
    def compute_itm_topk_matrix(
        self,
        text_feats_all: torch.Tensor,   # [N_txt, 40, 1024] fp16 CPU
        text_atts_all: torch.Tensor,    # [N_txt, 40] CPU
        ordered_vids: List[str],        # N_vid개 video_id 순서
        topk_indices: torch.Tensor,     # [N_txt, K] — ITC top-K 영상 인덱스
    ) -> torch.Tensor:
        """평가용: ITC top-K 후보에 대해서만 ITM 스코어 계산

        compute_itm_matrix()의 top-k 필터 버전.
        top-K에 없는 (쿼리, 영상) 쌍의 점수는 -inf로 초기화되므로,
        R@k 계산 시 정답이 top-K 밖이면 자동으로 낮은 순위가 된다.

        Args:
            text_feats_all: [N_txt, 40, 1024]
            text_atts_all:  [N_txt, 40]
            ordered_vids:   N_vid개 video_id 순서
            topk_indices:   [N_txt, K] — 각 쿼리의 ITC top-K 영상 인덱스

        Returns:
            torch.Tensor [N_txt, N_vid] — ITM 스코어 행렬
            (top-K 외 항목은 float('-inf'))
        """
        self.load()

        N_txt = text_feats_all.shape[0]
        N_vid = len(ordered_vids)
        scores = torch.full((N_txt, N_vid), float('-inf'))
        model_dtype = next(self.iv_model.parameters()).dtype
        enc = self.iv_model.get_text_encoder()

        # 영상 인덱스 → 해당 영상이 top-K에 포함된 쿼리 목록
        # (불필요한 영상 로드를 완전히 스킵하기 위한 역방향 인덱스)
        vid_query_map: Dict[int, List[int]] = {}
        for qi in range(N_txt):
            for vi in topk_indices[qi].tolist():
                vid_query_map.setdefault(vi, []).append(qi)

        from tqdm import tqdm
        for vi, query_list in tqdm(
            vid_query_map.items(), desc="ITM scoring (top-k)", total=len(vid_query_map)
        ):
            vid_id = ordered_vids[vi]
            if vid_id not in self.vis_feat_dict:
                for qi in query_list:
                    scores[qi, vi] = -999.0
                continue

            vf = self.vis_feat_dict[vid_id].unsqueeze(0).to(
                self.device, dtype=model_dtype
            )  # [1, 1025, 1408]

            # query_list를 bs_itm 단위로 배치 처리
            for batch_start in range(0, len(query_list), self.bs_itm):
                batch_qi = query_list[batch_start : batch_start + self.bs_itm]
                idx = torch.tensor(batch_qi)
                tf = text_feats_all[idx].to(self.device, dtype=model_dtype)
                ta = text_atts_all[idx].to(self.device)
                bs = tf.shape[0]

                out = enc(
                    encoder_embeds=tf,
                    attention_mask=ta,
                    encoder_hidden_states=vf.expand(bs, -1, -1),
                    encoder_attention_mask=None,
                    return_dict=True,
                    mode="fusion",
                )
                s = self.itm_head(out.last_hidden_state[:, 0].float())[:, 1]
                for k_idx, qi in enumerate(batch_qi):
                    scores[qi, vi] = s[k_idx].item()

        return scores

    @torch.no_grad()
    def compute_itm_matrix(
        self,
        text_feats_all: torch.Tensor,   # [N_txt, 40, 1024] fp16 CPU
        text_atts_all: torch.Tensor,    # [N_txt, 40] CPU
        ordered_vids: List[str],        # N_vid개 video_id 순서
    ) -> torch.Tensor:
        """평가용: N_txt × N_vid 전체 ITM 스코어 행렬 계산

        03_evaluation.ipynb Cell 9의 로직을 함수로 캡슐화.
        1k-A 전체(1000×1000) ITM 행렬 계산 시 사용.

        Args:
            text_feats_all: 전체 텍스트 full token [N_txt, 40, 1024]
            text_atts_all: 어텐션 마스크 [N_txt, 40]
            ordered_vids: 영상 ID 순서 (컬럼 순서)

        Returns:
            torch.Tensor [N_txt, N_vid] — ITM 스코어 행렬
        """
        self.load()

        N_txt = text_feats_all.shape[0]
        N_vid = len(ordered_vids)
        scores = torch.zeros(N_txt, N_vid)
        model_dtype = next(self.iv_model.parameters()).dtype
        enc = self.iv_model.get_text_encoder()

        from tqdm import tqdm
        for vi, vid_id in enumerate(tqdm(ordered_vids, desc="ITM scoring")):
            if vid_id not in self.vis_feat_dict:
                continue
            vf = self.vis_feat_dict[vid_id].unsqueeze(0).to(
                self.device, dtype=model_dtype
            )  # [1, 1025, 1408]

            for ti in range(0, N_txt, self.bs_itm):
                tf = text_feats_all[ti:ti + self.bs_itm].to(
                    self.device, dtype=model_dtype
                )
                ta = text_atts_all[ti:ti + self.bs_itm].to(self.device)
                bs = tf.shape[0]

                out = enc(
                    encoder_embeds=tf,
                    attention_mask=ta,
                    encoder_hidden_states=vf.expand(bs, -1, -1),
                    encoder_attention_mask=None,
                    return_dict=True,
                    mode="fusion",
                )
                s = self.itm_head(out.last_hidden_state[:, 0].float())[:, 1]
                scores[ti:ti + bs, vi] = s.cpu()

        return scores
