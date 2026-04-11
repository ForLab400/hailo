# Dataset Preprocessing & Merge — Journal Report
**Date:** 2026-04-11  
**Task:** Dangerous Object Detection — Hammer, Knife, Person, Scissors  
**Pipeline:** Roboflow v2 + OpenImages v7 → Merged 640×640 Dataset for Hailo-8L on Raspberry Pi 5

---

## 1. Motivation

The original model was trained on 359 images (Roboflow Knives & Scissors Training v2, train split only). While it achieved mAP50 = 0.897, the small dataset raised concerns:
- Hammer class was underrepresented (few diverse poses)
- Knife class had limited background variety
- No outdoor/surveillance-style scene diversity

A second source — OpenImages v7 — was downloaded with 1,219 additional annotated images across all four classes. Merging both sources was the next step to improve generalization and robustness.

---

## 2. Source Datasets

### 2.1 Roboflow — Knives & Scissors Training v2
| Property | Value |
|----------|-------|
| Source | https://universe.roboflow.com/eitan/knives-and-scissors-training/dataset/2 |
| License | CC BY 4.0 |
| Format | YOLO (normalized cx cy w h) |
| Classes | Hammer (0), Knife (1), Person (2), scissors (3) |
| Train images | 359 |
| Val images | 102 |
| Test images | 51 |
| **Total** | **512** |
| Image size | Varies (mostly 640×640 or close) |
| Image type | JPEG |

All existing Roboflow splits (train/val/test) were pooled together and re-split, so the merged dataset has a consistent random split across both sources.

### 2.2 OpenImages v7 — Custom Export
| Property | Value |
|----------|-------|
| Source | OIDv4 Toolkit, OpenImages v7 |
| License | CC BY 4.0 |
| Format | YOLO (Darknet — normalized cx cy w h per class subfolder) |
| Classes | hammer, knife, person, scissors (same 4 classes) |
| hammer images | 53 |
| knife images | 500 |
| person images | 500 |
| scissors images | 166 |
| **Total** | **1,219** |
| Image size | Varies (real-world photos, non-square, up to 4608×2592) |
| Image type | JPEG |

OpenImages images are diverse real-world photographs. Aspect ratios vary significantly — many are widescreen or portrait — making letterboxing essential.

---

## 3. Combined Dataset Summary

| Source | Images | Percentage |
|--------|--------|------------|
| Roboflow | 512 | 29.6% |
| OpenImages | 1,219 | 70.4% |
| **Total collected** | **1,731** | 100% |

**After preprocessing (1 image skipped — missing label):**
| Split | Images | Percentage |
|-------|--------|------------|
| Train | 1,383 | 79.9% |
| Val | 173 | 10.0% |
| Test | 174 | 10.1% |
| **Total** | **1,730** | 100% |

Split ratios: 80% / 10% / 10% (random shuffle with seed=42)

---

## 4. Preprocessing Pipeline

### 4.1 Design Requirements

The preprocessing was designed around three hardware constraints:

| Constraint | Requirement |
|------------|-------------|
| **Hailo-8L input** | Fixed 640×640 pixels, float32, HWC format |
| **Camera Module 3 (IMX708)** | Up to 4608×2592 resolution, 16:9 aspect ratio |
| **YOLOv8 training** | YOLO label format (normalized cx cy w h) must remain valid after resize |

A naive `cv2.resize(img, (640, 640))` would distort non-square images, shifting bounding box positions and aspect ratios. For example, a 4608×2592 image stretched to 640×640 compresses horizontally by 7.2× and vertically by 4.0×, making it impossible to detect objects reliably.

### 4.2 Letterbox Resize

**Letterboxing** preserves aspect ratio by:
1. Computing the uniform scale factor: `scale = 640 / max(H, W)`
2. Resizing to `(new_w, new_h)` = `(W * scale, H * scale)` using bilinear interpolation
3. Padding the shorter dimension symmetrically with grey (114, 114, 114) — YOLOv8's standard fill color

```python
def letterbox(img, size=640):
    h, w = img.shape[:2]
    scale = size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_w = (size - new_w) // 2
    pad_h = (size - new_h) // 2
    img_padded = cv2.copyMakeBorder(
        img_resized, pad_h, size-new_h-pad_h, pad_w, size-new_w-pad_w,
        cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return img_padded, scale, (pad_w, pad_h)
```

**Example:** A 1920×1080 image  
- scale = 640 / 1920 = 0.333  
- Resized to: 640×360  
- Pad top/bottom: 140px each side  
- Result: 640×640 (no horizontal padding needed)

### 4.3 Label Coordinate Adjustment

After letterboxing, YOLO label coordinates (normalized to original image size) must be recalculated to match the padded canvas:

```
cx_new = (cx_px * scale + pad_w) / 640
cy_new = (cy_px * scale + pad_h) / 640
bw_new = (bw_px * scale) / 640
bh_new = (bh_px * scale) / 640
```

Where `cx_px = cx * orig_w` converts from normalized to pixel coordinates. After transformation, all coordinates are clamped to [0.001, 0.999] to avoid boundary anomalies in YOLOv8's loss computation.

### 4.4 Label Validation

Each label line is validated before saving:
- Must have exactly 5 fields: `class cx cy w h`
- `class` must be an integer in [0, 3]
- All coordinate values must be in [0, 1]
- Lines failing validation are dropped; if all lines fail, the image is skipped entirely

Of 1,731 images, only 1 was skipped due to a completely missing label file (no annotations).

### 4.5 Filename Collision Prevention

Both datasets contain generic filenames (e.g., `image001.jpg`). Each output file is prefixed with its source:
- Roboflow: `roboflow_<original_stem>.jpg`
- OpenImages: `openimages_hammer_<stem>.jpg`, `openimages_knife_<stem>.jpg`, etc.

This prevents any filename collisions in the merged output directory.

---

## 5. Class Distribution (Train Split)

| Class ID | Class Name | Train Instances |
|----------|-----------|-----------------|
| 0 | Hammer | 150 |
| 1 | Knife | 731 |
| 2 | Person | 1,572 |
| 3 | scissors | 667 |
| **Total** | | **3,120** |

**Notes:**
- **Hammer (0)** is the most underrepresented class at only 150 instances. OpenImages provided only 53 hammer images total. This may limit hammer detection precision compared to knife/person.
- **Person (2)** dominates with 1,572 instances — many OpenImages images contain people with other objects, resulting in co-annotations.
- **Knife (1)** and **Scissors (3)** are well-represented. The 500 OpenImages knife images significantly improve knife diversity.
- Class imbalance is mild-to-moderate; YOLOv8's focal loss partially compensates, but hammer recall may remain lower.

---

## 6. Output Structure

```
/home/sixthdragon/hailo_project/merged_dataset/
├── data.yaml                    ← Training config for YOLOv8
├── train/
│   ├── images/                  ← 1,383 JPEG images (640×640)
│   └── labels/                  ← 1,383 YOLO .txt label files
├── val/
│   ├── images/                  ← 173 JPEG images (640×640)
│   └── labels/                  ← 173 YOLO .txt label files
└── test/
    ├── images/                  ← 174 JPEG images (640×640)
    └── labels/                  ← 174 YOLO .txt label files
```

**data.yaml:**
```yaml
train: /home/sixthdragon/hailo_project/merged_dataset/train/images
val:   /home/sixthdragon/hailo_project/merged_dataset/val/images
test:  /home/sixthdragon/hailo_project/merged_dataset/test/images

nc: 4
names: ['Hammer', 'Knife', 'Person', 'scissors']
```

---

## 7. Technical Decisions & Rationale

### Why grey padding (114, 114, 114)?
This is YOLOv8's standard letterbox fill value used during inference. Using the same colour for training preprocessing ensures the model learns to ignore grey-padded regions consistently, reducing false detections near padding borders.

### Why bilinear interpolation?
`cv2.INTER_LINEAR` is the standard choice for downscaling — it provides smoother results than nearest-neighbour and is computationally cheaper than Lanczos/bicubic. For detection tasks, fine texture detail is less critical than preserving object shape and scale.

### Why pool all Roboflow splits before re-splitting?
The Roboflow dataset's original val/test splits were small (102 and 51 images). After merging with OpenImages, the overall dataset is large enough that a fresh 80/10/10 split is more statistically valid than preserving the original tiny splits.

### Why seed=42?
Reproducibility. Any future re-run of the preprocessing script will produce identical train/val/test splits, making experiment comparisons valid.

### Why JPEG quality 95?
Quality 95 retains near-lossless detail while reducing file size by ~40% compared to quality 100. At 640×640, any JPEG artefacts at Q95 are well below the threshold detectable by YOLOv8's feature extractor.

---

## 8. Preprocessing Statistics

| Metric | Value |
|--------|-------|
| Total images collected | 1,731 |
| Images processed successfully | 1,730 |
| Images skipped (missing label) | 1 |
| Images skipped (corrupted) | 0 |
| Output image size | 640×640 JPEG |
| Total train instances (bboxes) | 3,120 |
| Random seed | 42 |
| Script | `preprocess_merge.py` |
| Output directory | `merged_dataset/` |

---

## 9. Next Steps

1. **Retrain YOLOv8s** using `merged_dataset/data.yaml` — expected mAP improvement due to 3.4× more training data
2. **Export ONNX** → parse HAR → optimize with PTQ → compile HEF
3. **Re-deploy** `best.hef` to Raspberry Pi 5 + Hailo-8L

---

*Preprocessing script: `/home/sixthdragon/hailo_project/preprocess_merge.py`*  
*Output dataset: `/home/sixthdragon/hailo_project/merged_dataset/`*  
*Hardware target: Hailo-8L on Raspberry Pi 5 + Camera Module 3 (IMX708)*
