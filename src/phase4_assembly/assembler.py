"""
⑧d assembler.py — Phase 4 영상 조립 오케스트레이터

출처:
- MoviePy: https://github.com/Zulko/moviepy (MIT License)
  → Python 기반 영상 편집 라이브러리

- 자동 조립 파이프라인 설계: 자체 설계
  ① perception_sort: DINOv2 + 텍스트 유사도 기반 클립 순서 최적화
  ② select_transition: 유사도 기반 트랜지션 자동 결정
  ③ dreamcolour_normalize: 색감 통일
  ④ MoviePy로 최종 영상 렌더링

점수 산출:
  score = 0.6 * text_similarity + 0.4 * visual_similarity
  (텍스트 유사도 60% + 시각 유사도 40% → 인간 시지각 기반 순서)
"""

import os
import time
import subprocess
import tempfile
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2

from ..data_models import ClipResult, TransitionDecision, TransitionType
from .visual_scorer import VisualScorer
from .transition_selector import TransitionSelector
from .colour_normalizer import ColourNormalizer
from .morph_transition import MorphTransition


class VideoAssembler:
    """Phase 4 영상 조립 오케스트레이터

    assemble(clips, query) → output_path
    """

    def __init__(
        self,
        visual_scorer: Optional[VisualScorer] = None,
        transition_selector: Optional[TransitionSelector] = None,
        colour_normalizer: Optional[ColourNormalizer] = None,
        morph_transition: Optional[MorphTransition] = None,
        output_dir: str = "output",
        text_weight: float = 0.6,
        visual_weight: float = 0.4
    ):
        self.visual_scorer = visual_scorer or VisualScorer()
        self.transition_selector = transition_selector or TransitionSelector()
        self.colour_normalizer = colour_normalizer or ColourNormalizer()
        self.morph_transition = morph_transition or MorphTransition()
        self.output_dir = output_dir
        self.text_weight = text_weight
        self.visual_weight = visual_weight

    def assemble(
        self,
        clips: List[ClipResult],
        query: str,
        max_clips: int = 10,
        max_duration_sec: float = 120.0,
        sort_mode: str = "auto",
    ) -> Tuple[str, List[TransitionDecision], Dict[str, float]]:
        """클립들을 합성 영상으로 조립

        Args:
            clips: 재순위화된 클립 리스트 (Top-20)
            query: 원본 쿼리
            max_clips: 최대 사용 클립 수
            max_duration_sec: 최대 결과 영상 길이 (초)
            sort_mode: 정렬 모드 (v2 신규)
                - "auto": Perception-aware 자동 정렬 (기본, 기존 동작)
                - "manual": PD가 지정한 순서 그대로 유지
                - "hybrid": PD 순서 기반 + 시각 유사도 미세 조정

        Returns:
            (output_video_path, transitions, sub_latencies)
            sub_latencies: 세부 단계별 소요 시간 dict (ms)
        """
        os.makedirs(self.output_dir, exist_ok=True)
        sub_latencies: Dict[str, float] = {}

        # 사용할 클립 수 결정
        use_clips = clips[:max_clips]
        print(f"[Assembly] 영상 조립 시작: {len(use_clips)} clips for query='{query}' (sort_mode={sort_mode})")

        if len(use_clips) <= 1:
            print(
                f"[Assembly] ⚠️ 클립이 {len(use_clips)}개뿐입니다. "
                f"crossfade/morph 트랜지션 없이 단일 클립 렌더링합니다."
            )

        # Step 1: 정렬 (sort_mode에 따라 분기)
        t0 = time.time()
        if sort_mode == "manual":
            # PD가 지정한 순서 그대로 유지
            sorted_clips = use_clips
            print(f"[Assembly] Step 1 Manual Sort: PD 순서 유지 ({len(sorted_clips)} clips)")
        elif sort_mode == "hybrid":
            # PD 순서 기반이지만 인접 클립 간 시각 유사도로 미세 조정
            sorted_clips = use_clips  # TODO: hybrid 로직 구현
            print(f"[Assembly] Step 1 Hybrid Sort: PD 순서 기반 ({len(sorted_clips)} clips)")
        else:
            # auto: 기존 Perception-aware 정렬
            sorted_clips = self._perception_sort(use_clips, query)
        elapsed = (time.time() - t0) * 1000
        sub_latencies["4a_perception_sort"] = elapsed
        print(f"[Assembly] Step 1 Perception Sort: {elapsed:.1f}ms ({len(sorted_clips)} clips)")

        # Step 2: 키프레임 추출
        t0 = time.time()
        keyframes = self._extract_keyframes(sorted_clips)
        elapsed = (time.time() - t0) * 1000
        sub_latencies["4b_keyframe_extract"] = elapsed
        print(f"[Assembly] Step 2 Keyframe Extract: {elapsed:.1f}ms ({len(keyframes)} frames)")

        # Step 2.5: 시각 유사도 계산 (DINOv2)
        t0 = time.time()
        similarities = self.visual_scorer.compute_pairwise_similarities(keyframes)
        elapsed = (time.time() - t0) * 1000
        sub_latencies["4c_visual_similarity"] = elapsed
        print(f"[Assembly] Step 2.5 Visual Similarity (DINOv2): {elapsed:.1f}ms ({len(similarities)} pairs)")

        # Step 3: 트랜지션 결정 (PDF 원안: 순수 DINOv2 기반)
        t0 = time.time()
        clip_ids = [c.clip_id for c in sorted_clips]
        transitions = self.transition_selector.select_all(
            similarities, clip_ids
        )
        elapsed = (time.time() - t0) * 1000
        sub_latencies["4d_transition_select"] = elapsed
        # 트랜지션 타입별 카운트
        type_counts = {}
        for t_dec in transitions:
            tname = t_dec.transition_type.value if hasattr(t_dec.transition_type, 'value') else str(t_dec.transition_type)
            type_counts[tname] = type_counts.get(tname, 0) + 1
        print(f"[Assembly] Step 3 Transition Select: {elapsed:.1f}ms (types: {type_counts})")

        # Step 3.5: DreamColour 3D LUT 사전 계산 (렌더링 시 프레임 콜백으로 적용)
        t0 = time.time()
        self._clip_luts = {}  # clip_id → LUT3D 매핑
        if len(sorted_clips) >= 2:
            try:
                ref_frame = self._get_keyframe(sorted_clips[0])
                if ref_frame is not None:
                    import cv2 as _cv2
                    ref_bgr = _cv2.cvtColor(ref_frame, _cv2.COLOR_RGB2BGR)
                    for clip in sorted_clips[1:]:
                        target_frame = self._get_keyframe(clip)
                        if target_frame is not None:
                            target_bgr = _cv2.cvtColor(target_frame, _cv2.COLOR_RGB2BGR)
                            lut = self.colour_normalizer._build_3d_lut(target_bgr, ref_bgr)
                            self._clip_luts[clip.clip_id] = lut
                    print(
                        f"[Assembly] DreamColour 3D LUT 사전 계산 완료: "
                        f"{len(self._clip_luts)}개 클립 (레퍼런스: {sorted_clips[0].clip_id})"
                    )
            except Exception as e:
                print(f"[Assembly] ⚠️ DreamColour LUT 계산 실패 (원본 색감 사용): {e}")
                self._clip_luts = {}
        elapsed = (time.time() - t0) * 1000
        sub_latencies["4e_dreamcolour_lut"] = elapsed
        print(f"[Assembly] Step 3.5 DreamColour LUT: {elapsed:.1f}ms ({len(self._clip_luts)} LUTs)")

        # Step 4: FFmpeg 영상 렌더링
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(
            self.output_dir, f"result_{timestamp}.mp4"
        )

        t0 = time.time()
        self._render_video(sorted_clips, transitions, output_path, max_duration_sec)
        elapsed = (time.time() - t0) * 1000
        sub_latencies["4f_ffmpeg_render"] = elapsed
        print(f"[Assembly] Step 4 FFmpeg Render: {elapsed:.1f}ms")

        # 전체 Assembly 소요 시간 요약
        total_assembly = sum(sub_latencies.values())
        sub_latencies["4_total"] = total_assembly
        print(f"[Assembly] ===== 세부 단계 소요 시간 요약 =====")
        for step, ms in sub_latencies.items():
            pct = (ms / total_assembly * 100) if total_assembly > 0 else 0
            print(f"[Assembly]   {step}: {ms:.1f}ms ({pct:.1f}%)")
        print(f"[Assembly] 영상 조립 완료: {output_path} (총 {total_assembly:.1f}ms)")

        return output_path, transitions, sub_latencies

    def _perception_sort(
        self,
        clips: List[ClipResult],
        query: str
    ) -> List[ClipResult]:
        """Perception-aware 클립 순서 정렬 (PDF 원안: DINOv2 + 텍스트 유사도)

        PDF 설계 기준:
          score = 0.6 * text_sim + 0.4 * visual_sim
          → 순수 DINOv2 시각 유사도 + 텍스트 적합도만으로 클립 배열.
          → Scene bonus 없이 DINOv2가 시각적 연속성을 전적으로 판단.

        Greedy nearest-neighbor 방식:
          첫 번째 클립(최고 score) 배치 후, 매 단계마다 마지막 클립과의
          combined 점수가 가장 높은 후보를 선택.
        """
        if len(clips) <= 2:
            return clips

        # 텍스트 유사도 (ColBERT/WRRF 점수를 정규화하여 사용)
        max_score = max(c.score for c in clips) if clips else 1.0
        if max_score == 0:
            max_score = 1.0

        # PDF 원안 가중치: 텍스트 60% + DINOv2 시각 40%
        w_text = self.text_weight   # 0.6
        w_visual = self.visual_weight  # 0.4

        # 첫 번째 클립: 가장 높은 점수
        sorted_clips = [clips[0]]
        remaining = list(clips[1:])

        while remaining:
            last = sorted_clips[-1]
            last_kf = self._get_keyframe(last)

            if last_kf is not None:
                last_feat = self.visual_scorer.extract_features(last_kf)
            else:
                last_feat = None

            best_idx = 0
            best_combined = -1

            for i, candidate in enumerate(remaining):
                text_sim = candidate.score / max_score

                if last_feat is not None:
                    cand_kf = self._get_keyframe(candidate)
                    if cand_kf is not None:
                        cand_feat = self.visual_scorer.extract_features(cand_kf)
                        visual_sim = self.visual_scorer.compute_similarity(
                            last_feat, cand_feat
                        )
                    else:
                        visual_sim = 0.5
                else:
                    visual_sim = 0.5

                combined = w_text * text_sim + w_visual * visual_sim

                if combined > best_combined:
                    best_combined = combined
                    best_idx = i

            sorted_clips.append(remaining.pop(best_idx))

        return sorted_clips

    def _extract_keyframes(self, clips: List[ClipResult]) -> List[np.ndarray]:
        """클립별 키프레임 추출"""
        frames = []
        for clip in clips:
            kf = self._get_keyframe(clip)
            if kf is not None:
                frames.append(kf)
            else:
                # 더미 프레임
                frames.append(
                    np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                )
        return frames

    def _get_keyframe(self, clip: ClipResult) -> Optional[np.ndarray]:
        """클립에서 키프레임 가져오기"""
        import cv2

        # 키프레임 파일이 있는 경우
        if clip.keyframe_path and os.path.exists(clip.keyframe_path):
            frame = cv2.imread(clip.keyframe_path)
            if frame is not None:
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 영상에서 직접 추출
        if clip.video_path and os.path.exists(clip.video_path):
            try:
                cap = cv2.VideoCapture(clip.video_path)
                mid_ms = (clip.start_ms + clip.end_ms) / 2
                cap.set(cv2.CAP_PROP_POS_MSEC, mid_ms)
                ret, frame = cap.read()
                cap.release()
                if ret:
                    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            except Exception:
                pass

        return None

    def _render_video(
        self,
        clips: List[ClipResult],
        transitions: List[TransitionDecision],
        output_path: str,
        max_duration_sec: float
    ):
        """FFmpeg 직접 호출 기반 영상 렌더링 (v2)

        변경사항 (v2):
          이전: MoviePy → 파이썬 프레임 루프 → libx264 인코딩 (7-12초)
          현재: FFmpeg 직접 호출 → C 레벨 처리 → LUT는 .cube 필터 (1-2초)

        프로세스:
          1) 각 클립을 FFmpeg로 잘라서 중간 파일 생성 (LUT 적용 포함)
          2) Morph 전환이 필요한 구간은 optical flow 프레임을 생성
          3) FFmpeg concat demuxer로 최종 합성
        """
        tmp_dir = tempfile.mkdtemp(prefix="videorag_render_")

        try:
            segment_files = []  # 최종 concat에 사용될 세그먼트 파일 목록
            total_duration = 0
            fps = 25
            render_t0 = time.time()

            skipped_reasons = {}  # clip_id → skip 사유

            for i, clip in enumerate(clips):
                if total_duration >= max_duration_sec:
                    reason = f"max_duration 초과 ({total_duration:.1f}s >= {max_duration_sec}s)"
                    skipped_reasons[clip.clip_id] = reason
                    print(f"[Render] 클립 {i} ({clip.clip_id}) SKIP: {reason}")
                    break

                if not clip.video_path:
                    reason = "video_path가 비어있음 (None 또는 빈 문자열)"
                    skipped_reasons[clip.clip_id] = reason
                    print(f"[Render] ⚠️ 클립 {i} ({clip.clip_id}) SKIP: {reason}")
                    continue

                if not os.path.exists(clip.video_path):
                    reason = f"파일 없음: {clip.video_path}"
                    skipped_reasons[clip.clip_id] = reason
                    print(f"[Render] ⚠️ 클립 {i} ({clip.clip_id}) SKIP: {reason}")
                    continue

                start_s = clip.start_ms / 1000
                end_s = clip.end_ms / 1000
                duration_s = end_s - start_s
                if duration_s <= 0:
                    reason = f"duration <= 0 (start={clip.start_ms}ms, end={clip.end_ms}ms)"
                    skipped_reasons[clip.clip_id] = reason
                    print(f"[Render] ⚠️ 클립 {i} ({clip.clip_id}) SKIP: {reason}")
                    continue

                # 남은 시간 제한
                remaining = max_duration_sec - total_duration
                if duration_s > remaining:
                    duration_s = remaining
                    end_s = start_s + duration_s

                # ── 세그먼트 추출 (FFmpeg) ──────────────────────────
                seg_path = os.path.join(tmp_dir, f"seg_{i:03d}.mp4")

                # LUT .cube 파일 생성 (있으면)
                lut_filter = ""
                if hasattr(self, '_clip_luts') and clip.clip_id in self._clip_luts:
                    cube_path = os.path.join(tmp_dir, f"lut_{i:03d}.cube")
                    self.colour_normalizer.export_lut_cube(
                        self._clip_luts[clip.clip_id], cube_path
                    )
                    lut_filter = f",lut3d='{cube_path}'"

                cmd = [
                    'ffmpeg', '-y',
                    '-ss', f'{start_s:.3f}',
                    '-i', clip.video_path,
                    '-t', f'{duration_s:.3f}',
                    '-vf', f'scale=640:480:force_original_aspect_ratio=decrease,'
                           f'pad=640:480:(ow-iw)/2:(oh-ih)/2{lut_filter}',
                    '-c:v', 'libx264', '-preset', 'ultrafast',
                    '-pix_fmt', 'yuv420p',
                    '-an',  # 오디오 제거 (프로토타입)
                    '-r', str(fps),
                    seg_path
                ]

                try:
                    seg_t0 = time.time()
                    subprocess.run(
                        cmd, capture_output=True, timeout=30,
                        check=True
                    )
                    seg_elapsed = (time.time() - seg_t0) * 1000
                    segment_files.append(seg_path)
                    total_duration += duration_s
                    print(
                        f"[Render] 세그먼트 {i} ({clip.clip_id}): "
                        f"{duration_s:.2f}s 추출 → {seg_elapsed:.0f}ms"
                    )
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                    print(f"[Render] ⚠️ FFmpeg 세그먼트 추출 실패: {clip.clip_id}: {e}")
                    continue

                # ── Morph/Crossfade 전환 프레임 삽입 ──────────────
                if i < len(transitions):
                    t = transitions[i]
                    if t.transition_type == TransitionType.MORPH:
                        tr_t0 = time.time()
                        morph_seg = self._render_morph_segment(
                            clip, clips[i + 1] if i + 1 < len(clips) else None,
                            t, tmp_dir, i, fps
                        )
                        tr_elapsed = (time.time() - tr_t0) * 1000
                        if morph_seg:
                            segment_files.append(morph_seg)
                            total_duration += t.duration_sec
                            print(f"[Render] Morph 전환 {i}: {tr_elapsed:.0f}ms")
                        else:
                            print(f"[Render] ⚠️ Morph 전환 {i} 실패 ({tr_elapsed:.0f}ms)")
                    elif t.transition_type == TransitionType.CROSSFADE:
                        tr_t0 = time.time()
                        xfade_seg = self._render_crossfade_segment(
                            clip, clips[i + 1] if i + 1 < len(clips) else None,
                            t, tmp_dir, i, fps
                        )
                        tr_elapsed = (time.time() - tr_t0) * 1000
                        if xfade_seg:
                            segment_files.append(xfade_seg)
                            total_duration += t.duration_sec
                            print(f"[Render] Crossfade 전환 {i}: {tr_elapsed:.0f}ms")
                        else:
                            print(f"[Render] ⚠️ Crossfade 전환 {i} 실패 ({tr_elapsed:.0f}ms)")

            if not segment_files:
                self._create_dummy_output(output_path)
                return

            seg_total_ms = (time.time() - render_t0) * 1000
            print(
                f"[Render] 세그먼트 추출 완료: {len(segment_files)}개 성공 / "
                f"{len(skipped_reasons)}개 SKIP, "
                f"총 {total_duration:.1f}s, 소요 {seg_total_ms:.0f}ms"
            )
            if skipped_reasons:
                print(f"[Render] ⚠️ SKIP된 클립 {len(skipped_reasons)}개 상세:")
                for cid, reason in skipped_reasons.items():
                    print(f"[Render]   {cid}: {reason}")

            # ── FFmpeg concat demuxer로 최종 합성 ──────────────────
            concat_t0 = time.time()
            concat_list = os.path.join(tmp_dir, "concat.txt")
            with open(concat_list, 'w') as f:
                for seg in segment_files:
                    f.write(f"file '{seg}'\n")

            # 모든 세그먼트가 동일한 코덱/해상도/fps로 인코딩되어 있으므로
            # -c copy로 재인코딩 없이 합성 (16~30초 → <1초)
            cmd_concat = [
                'ffmpeg', '-y',
                '-f', 'concat', '-safe', '0',
                '-i', concat_list,
                '-c', 'copy',
                '-movflags', '+faststart',
                output_path
            ]

            try:
                result = subprocess.run(
                    cmd_concat, capture_output=True, timeout=60,
                    check=True
                )
                concat_elapsed = (time.time() - concat_t0) * 1000
                print(
                    f"[Render] FFmpeg concat 완료 (-c copy): {output_path} "
                    f"({total_duration:.1f}s 영상, concat {concat_elapsed:.0f}ms)"
                )
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                # -c copy 실패 시 재인코딩 폴백
                print(f"[Render] ⚠️ FFmpeg concat -c copy 실패, 재인코딩 폴백: {e}")
                if hasattr(e, 'stderr') and e.stderr:
                    print(f"[Render]   stderr: {e.stderr.decode(errors='replace')[:500]}")
                cmd_concat_reencode = [
                    'ffmpeg', '-y',
                    '-f', 'concat', '-safe', '0',
                    '-i', concat_list,
                    '-c:v', 'libx264', '-preset', 'ultrafast',
                    '-pix_fmt', 'yuv420p',
                    '-movflags', '+faststart',
                    output_path
                ]
                try:
                    subprocess.run(
                        cmd_concat_reencode, capture_output=True, timeout=60,
                        check=True
                    )
                    concat_elapsed = (time.time() - concat_t0) * 1000
                    print(
                        f"[Render] FFmpeg concat 완료 (재인코딩 폴백): {output_path} "
                        f"({total_duration:.1f}s 영상, concat {concat_elapsed:.0f}ms)"
                    )
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e2:
                    print(f"[Render] ❌ FFmpeg concat 재인코딩도 실패: {e2}")
                    self._render_video_moviepy_fallback(
                        clips, transitions, output_path, max_duration_sec
                    )

        finally:
            # 임시 파일 정리
            import shutil
            try:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception:
                pass

    def _render_morph_segment(
        self,
        clip_from: ClipResult,
        clip_to: Optional[ClipResult],
        transition: TransitionDecision,
        tmp_dir: str,
        index: int,
        fps: int
    ) -> Optional[str]:
        """Morph 전환 세그먼트 생성 (optical flow 기반)

        이론적 근거:
          Farneback Dense Optical Flow로 두 프레임 사이의 움직임 벡터를 추정하고,
          이를 기반으로 중간 프레임을 합성(warp + blend).
          단순 Crossfade와 달리 물체의 움직임 방향을 고려하므로
          시각적으로 더 자연스러운 전환을 만들어냄.
        """
        if clip_to is None:
            print(f"[Morph {index}] ❌ clip_to is None → SKIP")
            return None

        frame_a = self._get_last_frame_bgr(clip_from)
        frame_b = self._get_first_frame_bgr(clip_to)

        if frame_a is None or frame_b is None:
            print(
                f"[Morph {index}] ❌ 프레임 추출 실패: "
                f"frame_a={'OK' if frame_a is not None else 'None'} "
                f"(clip_from={clip_from.clip_id}, path={clip_from.video_path}, "
                f"start={clip_from.start_ms}ms, end={clip_from.end_ms}ms), "
                f"frame_b={'OK' if frame_b is not None else 'None'} "
                f"(clip_to={clip_to.clip_id}, path={clip_to.video_path}, "
                f"start={clip_to.start_ms}ms, end={clip_to.end_ms}ms)"
            )
            return None

        # 해상도 통일
        frame_a = cv2.resize(frame_a, (640, 480))
        frame_b = cv2.resize(frame_b, (640, 480))

        n_frames = max(1, int(transition.duration_sec * fps))
        print(
            f"[Morph {index}] 프레임 추출 성공: "
            f"frame_a={frame_a.shape}, frame_b={frame_b.shape}, "
            f"n_frames={n_frames}, duration={transition.duration_sec:.2f}s"
        )
        morph_frames = self.morph_transition.generate_morph_frames(
            frame_a, frame_b, n_frames=n_frames
        )

        if not morph_frames:
            print(f"[Morph {index}] ❌ generate_morph_frames 결과 빈 리스트")
            return None

        # cv2.VideoWriter로 세그먼트 생성
        seg_path = os.path.join(tmp_dir, f"morph_{index:03d}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(seg_path, fourcc, fps, (640, 480))
        for frame in morph_frames:
            out.write(frame)
        out.release()

        # mp4v → H.264 재인코딩 (concat 호환)
        h264_path = os.path.join(tmp_dir, f"morph_{index:03d}_h264.mp4")
        try:
            subprocess.run([
                'ffmpeg', '-y', '-i', seg_path,
                '-c:v', 'libx264', '-preset', 'ultrafast',
                '-pix_fmt', 'yuv420p', '-r', str(fps),
                h264_path
            ], capture_output=True, timeout=15, check=True)
            return h264_path
        except Exception as e:
            print(f"[Render] ⚠️ Morph 재인코딩 실패: {e}")
            return seg_path

    def _render_crossfade_segment(
        self,
        clip_from: ClipResult,
        clip_to: Optional[ClipResult],
        transition: TransitionDecision,
        tmp_dir: str,
        index: int,
        fps: int
    ) -> Optional[str]:
        """Crossfade 전환 세그먼트 생성 (투명도 블렌딩)"""
        if clip_to is None:
            return None

        frame_a = self._get_last_frame_bgr(clip_from)
        frame_b = self._get_first_frame_bgr(clip_to)

        if frame_a is None or frame_b is None:
            return None

        frame_a = cv2.resize(frame_a, (640, 480))
        frame_b = cv2.resize(frame_b, (640, 480))

        n_frames = max(1, int(transition.duration_sec * fps))
        seg_path = os.path.join(tmp_dir, f"xfade_{index:03d}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(seg_path, fourcc, fps, (640, 480))

        for i in range(n_frames):
            t = (i + 1) / (n_frames + 1)
            blended = cv2.addWeighted(frame_a, 1.0 - t, frame_b, t, 0)
            out.write(blended)

        out.release()

        # 재인코딩
        h264_path = os.path.join(tmp_dir, f"xfade_{index:03d}_h264.mp4")
        try:
            subprocess.run([
                'ffmpeg', '-y', '-i', seg_path,
                '-c:v', 'libx264', '-preset', 'ultrafast',
                '-pix_fmt', 'yuv420p', '-r', str(fps),
                h264_path
            ], capture_output=True, timeout=15, check=True)
            return h264_path
        except Exception:
            return seg_path

    def _get_last_frame_bgr(self, clip: ClipResult) -> Optional[np.ndarray]:
        """클립의 마지막 프레임을 BGR로 가져오기"""
        if not clip.video_path or not os.path.exists(clip.video_path):
            print(f"[FrameExtract] _get_last_frame: 파일 없음 clip={clip.clip_id} path={clip.video_path}")
            return None
        try:
            cap = cv2.VideoCapture(clip.video_path)
            if not cap.isOpened():
                print(f"[FrameExtract] _get_last_frame: VideoCapture 열기 실패 clip={clip.clip_id}")
                return None
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_fps = cap.get(cv2.CAP_PROP_FPS)

            # 실제 영상 길이 계산 (메타데이터 end_ms가 실제보다 클 수 있음)
            actual_duration_ms = (total_frames / video_fps * 1000) if video_fps > 0 else clip.end_ms
            effective_end_ms = min(clip.end_ms, actual_duration_ms)

            # 끝에서 약간 앞 시점 (최소 200ms 여유)
            seek_ms = max(effective_end_ms - 200, clip.start_ms)

            # 방법 1: 시간 기반 seek
            cap.set(cv2.CAP_PROP_POS_MSEC, seek_ms)
            ret, frame = cap.read()

            # 방법 2: 시간 seek 실패 시 프레임 번호 기반 폴백
            if not ret and total_frames > 0:
                fallback_frame = max(total_frames - 2, 0)
                cap.set(cv2.CAP_PROP_POS_FRAMES, fallback_frame)
                ret, frame = cap.read()
                if ret:
                    print(
                        f"[FrameExtract] _get_last_frame: 시간seek 실패 → "
                        f"프레임#{fallback_frame} 폴백 성공 clip={clip.clip_id}"
                    )

            cap.release()
            if not ret:
                print(
                    f"[FrameExtract] _get_last_frame: read 실패 clip={clip.clip_id} "
                    f"seek_ms={seek_ms}, actual_dur={actual_duration_ms:.0f}ms, "
                    f"total_frames={total_frames}, fps={video_fps}"
                )
            return frame if ret else None
        except Exception as e:
            print(f"[FrameExtract] _get_last_frame: 예외 clip={clip.clip_id}: {e}")
            return None

    def _get_first_frame_bgr(self, clip: ClipResult) -> Optional[np.ndarray]:
        """클립의 첫 프레임을 BGR로 가져오기"""
        if not clip.video_path or not os.path.exists(clip.video_path):
            print(f"[FrameExtract] _get_first_frame: 파일 없음 clip={clip.clip_id} path={clip.video_path}")
            return None
        try:
            cap = cv2.VideoCapture(clip.video_path)
            if not cap.isOpened():
                print(f"[FrameExtract] _get_first_frame: VideoCapture 열기 실패 clip={clip.clip_id}")
                return None
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            cap.set(cv2.CAP_PROP_POS_MSEC, clip.start_ms)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                print(
                    f"[FrameExtract] _get_first_frame: read 실패 clip={clip.clip_id} "
                    f"seek_ms={clip.start_ms}, total_frames={total_frames}, fps={video_fps}"
                )
            return frame if ret else None
        except Exception as e:
            print(f"[FrameExtract] _get_first_frame: 예외 clip={clip.clip_id}: {e}")
            return None

    def _render_video_moviepy_fallback(
        self,
        clips: List[ClipResult],
        transitions: List[TransitionDecision],
        output_path: str,
        max_duration_sec: float
    ):
        """MoviePy 폴백 렌더링 (FFmpeg 실패 시)"""
        try:
            import pkgutil
            if not hasattr(pkgutil, 'ImpImporter'):
                pkgutil.ImpImporter = type('ImpImporter', (), {})
            from moviepy.editor import VideoFileClip, concatenate_videoclips, ColorClip

            video_clips = []
            total_duration = 0
            for clip in clips:
                if total_duration >= max_duration_sec:
                    break
                if not clip.video_path or not os.path.exists(clip.video_path):
                    continue
                try:
                    vclip = VideoFileClip(clip.video_path)
                    start_s = clip.start_ms / 1000
                    end_s = min(clip.end_ms / 1000, vclip.duration)
                    if end_s <= start_s:
                        vclip.close()
                        continue
                    subclip = vclip.subclip(start_s, end_s)
                    remaining = max_duration_sec - total_duration
                    if subclip.duration > remaining:
                        subclip = subclip.subclip(0, remaining)
                    video_clips.append(subclip)
                    total_duration += subclip.duration
                except Exception:
                    continue

            if video_clips:
                final = concatenate_videoclips(video_clips)
                final.write_videofile(output_path, fps=25, codec='libx264', logger=None)
                for vc in video_clips:
                    try:
                        vc.close()
                    except Exception:
                        pass
            else:
                self._create_dummy_output(output_path)
        except ImportError:
            self._create_dummy_output(output_path)

    def _create_dummy_output(self, output_path: str):
        """더미 출력 (FFmpeg/MoviePy 모두 없을 때)"""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 25, (640, 480))
        black = np.zeros((480, 640, 3), dtype=np.uint8)
        for _ in range(25):  # 1초
            out.write(black)
        out.release()
