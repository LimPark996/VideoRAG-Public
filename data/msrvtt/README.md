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

- InternVideo2s2-1B, #F=4, MSR-VTT 1k-A T2V zero-shot:
  **R@1=51.9  R@5=74.6  R@10=81.7** (Table 24a, Supplementary)
