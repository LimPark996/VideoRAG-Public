# MSR-VTT 1k-A 평가 데이터

## 개요

MSR-VTT 벤치마크 데이터를 담는 폴더이다. Tier-1 검색 정확도 평가에 **1k-A split** (JSFusion, Yu et al. 2018)만 사용한다 — 테스트 영상 1,000개, 각 영상당 지정 캡션 1개.

## 폴더 구조

```
data/msrvtt/
  annotations/          # 1k-A 어노테이션 JSON (HuggingFace에서 다운로드)
    msrvtt_test_1k.json # [{video_id, caption, ...} x 1000]
  test_1ka_videos/      # 1k-A 서브셋 MP4 파일 (TestVideo.zip에서 추출)
    video7020.mp4
    video7021.mp4
    ...
  videos/               # (선택) 전체 TrainVal 7010개 영상
  keyframes/            # (선택) 추출된 키프레임
```

## 데이터 준비 방법

### 1. 어노테이션 (자동 — 노트북 Cell 2에서 처리)

`03_evaluation.ipynb`에서 HuggingFace로부터 다운로드:
```
https://huggingface.co/datasets/friedrichor/MSR-VTT/raw/main/msrvtt_test_1k.json
```

### 2. 테스트 영상 (수동 — 최초 1회 설정)

1k-A 영상은 MSR-VTT의 **TestVideo.zip** (테스트 영상 2,990개)에서 가져온다. 어노테이션 파일이 2,990개 중 1k-A split에 해당하는 1,000개를 지정한다.

**방법 A: MediaFire (원본 미러)**
```
https://www.mediafire.com/folder/h14iarbs62e7p/shared
→ "TestVideo.zip" 다운로드 → 압축 해제 → 1k-A 영상을 여기에 복사
```

**방법 B: Google Drive에 TestVideo가 이미 있는 경우**
```python
# Colab에서 Drive 마운트 후:
import shutil, json
with open("data/msrvtt/annotations/msrvtt_test_1k.json") as f:
    ann = json.load(f)
video_ids = {e["video_id"] for e in ann}  # 1000개 ID

src_dir = "/content/drive/MyDrive/YOUR_MSRVTT_PATH/"
dst_dir = "data/msrvtt/test_1ka_videos/"
for vid in video_ids:
    shutil.copy2(f"{src_dir}/{vid}.mp4", dst_dir)
```

**방법 C: yt-dlp (폴백 — 느림, 일부 영상 불가)**
```bash
pip install yt-dlp
# 어노테이션 JSON의 URL 사용, start/end time으로 트리밍
```

## 논문 참조

- InternVideo2-Stage2-1B, #F=4, MSR-VTT 1k-A T2V 제로샷:
  **R@1=51.9  R@5=74.6  R@10=81.7** (Table 24a, Supplementary)

---

# MSR-VTT 1k-A Evaluation Data

## Overview

This folder holds MSR-VTT benchmark data for Tier-1 retrieval evaluation.
Only the **1k-A split** (JSFusion, Yu et al. 2018) is used — 1,000 test videos
with 1 designated caption each.

## Folder Structure

```
data/msrvtt/
  annotations/          # 1k-A annotation JSON (downloaded from HuggingFace)
    msrvtt_test_1k.json # [{video_id, caption, ...} x 1000]
  test_1ka_videos/      # 1k-A subset MP4 files (from TestVideo.zip)
    video7020.mp4
    video7021.mp4
    ...
  videos/               # (optional) full TrainVal 7010 videos
  keyframes/            # (optional) extracted keyframes
```

## How to Populate

### 1. Annotation (automatic — done in notebook Cell 2)

Downloaded from HuggingFace in `03_evaluation.ipynb`:
```
https://huggingface.co/datasets/friedrichor/MSR-VTT/raw/main/msrvtt_test_1k.json
```

### 2. Test Videos (manual — one-time setup)

The 1k-A videos come from MSR-VTT's **TestVideo.zip** (2,990 test videos).
The annotation file specifies which 1,000 of those 2,990 are in the 1k-A split.

**Option A: MediaFire (original mirror)**
```
https://www.mediafire.com/folder/h14iarbs62e7p/shared
→ Download "TestVideo.zip" → unzip → copy 1k-A videos here
```

**Option B: If you already have TestVideo on Google Drive**
```python
# In Colab after Drive mount:
import shutil, json
with open("data/msrvtt/annotations/msrvtt_test_1k.json") as f:
    ann = json.load(f)
video_ids = {e["video_id"] for e in ann}  # 1000 IDs

src_dir = "/content/drive/MyDrive/YOUR_MSRVTT_PATH/"
dst_dir = "data/msrvtt/test_1ka_videos/"
for vid in video_ids:
    shutil.copy2(f"{src_dir}/{vid}.mp4", dst_dir)
```

**Option C: yt-dlp (fallback — slow, some may be unavailable)**
```bash
pip install yt-dlp
# Use URLs from annotation JSON, trim with start/end time
```

## Paper Reference

- InternVideo2-Stage2-1B, #F=4, MSR-VTT 1k-A T2V zero-shot:
  **R@1=51.9  R@5=74.6  R@10=81.7** (Table 24a, Supplementary)
