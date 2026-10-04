# Personal Protective Equipment (PPE) Detection

**English** | [Français](README.fr.md)

End-to-end object detection project for workplace safety: dataset audit,
annotation normalization, training, evaluation, inference
(image / folder / video / webcam), compliance logic, REST API, local
interface and ONNX export.

Built with **Ultralytics YOLO26** and **PyTorch**, tested on **Windows 11**
with an **NVIDIA RTX 5080 Laptop (16 GB)**.

---

## Table of contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [Dataset and license](#3-dataset-and-license)
4. [Prerequisites](#4-prerequisites)
5. [Installation](#5-installation)
6. [Dataset audit](#6-dataset-audit)
7. [Annotation conversion](#7-annotation-conversion)
8. [Smoke test](#8-smoke-test)
9. [Full training](#9-full-training)
10. [Evaluation](#10-evaluation)
11. [Inference](#11-inference)
12. [PPE compliance](#12-ppe-compliance)
13. [REST API](#13-rest-api)
14. [Streamlit interface](#14-streamlit-interface)
15. [ONNX export](#15-onnx-export)
16. [Output structure](#16-output-structure)
17. [Code quality and tests](#17-code-quality-and-tests)
18. [Troubleshooting](#18-troubleshooting)
19. [Known limitations](#19-known-limitations)
20. [Future work](#20-future-work)

---

## 1. Overview

The current model (`artifacts/models/best.pt`, trained on the v3 dataset)
detects **9 classes** of equipment, people and PPE "look-alikes" in images of
construction or industrial sites:

| ID | Class               | Instances (v3) | Share  |
| -- | ------------------- | -------------- | ------ |
| 0  | Face Mask           | 788            | 1.09%  |
| 1  | Person              | 25,168         | 34.93% |
| 2  | Safety Gloves       | 2,172          | 3.01%  |
| 3  | Safety Harness      | 1,175          | 1.63%  |
| 4  | Safety Helmet       | 24,415         | 33.88% |
| 5  | Safety Shoes        | 5,875          | 8.15%  |
| 6  | Safety Vest         | 2,434          | 3.38%  |
| 7  | Non-Safety Headwear | 4,248          | 5.90%  |
| 8  | Uncovered Head      | 5,785          | 8.03%  |

Classes 0 to 6 come from the original Roboflow export. Classes 7 and 8 are
**negative** classes, added so that a bicycle helmet or a bare head is no
longer mistaken for a hard hat (see
[section 12](#telling-real-ppe-apart-from-look-alikes)). The English names are
the internal identifiers; the interface currently displays French labels.

### Model versions

| Version          | Classes                       | Weights                            | mAP@0.50 (test)                                |
| ---------------- | ----------------------------- | ---------------------------------- | ---------------------------------------------- |
| 7 classes        | 7 Roboflow classes            | `best_7classes.pt`               | 0.7992                                         |
| 8 classes        | + `Non-Safety Headwear`     | `best_8classes.pt`               | 0.7988 (on the 7 shared classes)               |
| **v3 (current)** | + `Uncovered Head`         | `best.pt` (= `best_v3.pt`)     | **0.8368** (v3 test, 9 classes)          |

The v3 figure is measured on the v3 dataset's test split, which is larger than
the original one: it is not directly comparable to the other two. On the
**same 578 original test images**, the fair comparison gives 0.7992
(7 classes), 0.7988 (8 classes) and **0.8077 (v3)**. Details in
[section 10](#current-model-v3-9-classes).

An **optional** business layer then associates PPE items with the detected
people to produce a compliance status. This association is a geometric
heuristic, not a certain measurement: see
[section 12](#12-ppe-compliance).

### Notable engineering decisions

| Topic                | Decision                                     | Rationale                                                                                                                                                                                         |
| -------------------- | -------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Model                | `yolo26s.pt` (Ultralytics 8.4)             | Roboflow advertises a "YOLO26" export: checked, **YOLO26 really exists** in Ultralytics 8.4 (`yolo26n/s/m/l/x`). The annotations remain standard YOLO.                                     |
| Python               | 3.12 in a dedicated venv                     | Python 3.14 is installed globally, but PyTorch does not publish wheels for that version yet.                                                                                                     |
| PyTorch              | `2.9.1+cu128`                              | The RTX 5080 is a **Blackwell (sm_120)** chip. CUDA builds older than 12.8 contain no kernels for this architecture. Verified via `torch.cuda.get_arch_list()`.                         |
| Polygons             | Converted to boxes in a **copy**       | 349 segmentation lines coexist with the boxes. The original dataset is never modified.                                                                                                           |
| `data.yaml` paths  | Multi-candidate resolution                   | The Roboflow export writes `../train/images`, which does not resolve correctly from the dataset root. The code tries several interpretations and keeps the one that exists.                   |
| ONNX verification    | **Functional** comparison              | YOLO26 exports "end-to-end" (output `(1, 300, 6)` already filtered). Comparing raw tensors element by element makes no sense: we compare the detections produced on a real image. |

---

## 2. Architecture

```
.
├── data.yaml                     # Original Roboflow export (never modified)
├── train/ valid/ test/           # Original images and labels (untouched)
│
├── configs/
│   ├── train.yaml                # Training hyperparameters
│   └── inference.yaml            # Inference thresholds + compliance rules
│
├── src/ppe_detection/
│   ├── annotations.py            # Label parsing/validation/conversion (no heavy dependency)
│   ├── config.py                 # Typed configuration dataclasses
│   ├── utils.py                  # Logging, seed, device, I/O, filename sanitization
│   ├── dataset_audit.py          # Full dataset audit
│   ├── dataset_cleaner.py        # Builds the normalized detection dataset
│   ├── dataset_merge.py          # Merges public datasets, remapping their classes
│   ├── pseudo_label.py           # Pre-annotates a missing class with an existing model
│   ├── taxonomy.py               # Extended class schema, counter-evidence, French labels
│   ├── train.py                  # Training
│   ├── evaluate.py               # Evaluation + error analysis
│   ├── calibrate.py              # Per-class threshold calibration (validation)
│   ├── pose.py                   # PPE/person association via keypoints
│   ├── predict.py                # Unified inference (image/folder/video/webcam/stream)
│   ├── video.py                  # Video and webcam loop
│   ├── compliance.py             # Geometric PPE <-> person association
│   ├── visualization.py          # Detection rendering and charts
│   ├── export.py                 # ONNX export + real verification
│   └── api.py                    # FastAPI REST API
│
├── app/streamlit_app.py          # Local interface
├── docs/plan_ecart_terrain.md    # Data collection plan for real-world deployment
├── docs/plan_donnees_epi_sosies.md  # Sources and annotation rules for negative classes
├── scripts/*.ps1                 # End-to-end PowerShell scripts
├── tests/                        # 221 unit and integration tests
└── artifacts/                    # Generated outputs (not tracked by Git)
    ├── dataset_detection/        # Normalized dataset (5-field labels)
    ├── models/                   # best.pt, last.pt
    ├── runs/                     # Ultralytics runs
    ├── reports/                  # JSON + Markdown reports
    ├── predictions/              # Inference results
    └── exports/                  # Exported models
```

---

## 3. Dataset and license

- **Source**: [Roboflow Universe — PPE Detection Project](https://universe.roboflow.com/ousmane-savadogo/ppe-detection-project-jeezl-p9ncg)
- **License**: **CC BY 4.0** — reuse allowed with attribution.
- **Export**: July 30, 2026, advertised format "YOLO26".
- **Size**: 7,000 images, 25,542 annotations, 857 MB.

| Split | Images | Labels | Annotations |
| ----- | ------ | ------ | ----------- |
| train | 4,903  | 4,903  | 17,873      |
| valid | 1,399  | 1,399  | 5,197       |
| test  | 698    | 698    | 2,472       |

Image ↔ label pairing is **perfect**: no orphan image, no label without an
image.

Sections 6 to 9 describe how this original dataset is processed.

### v3 dataset (current model)

The current model is trained on a merge of three sources, assembled by
[`dataset_merge.py`](src/ppe_detection/dataset_merge.py):

| Source                                                | License                                          | Images | Contribution                                                 |
| ----------------------------------------------------- | ------------------------------------------------ | ------ | ------------------------------------------------------------ |
| Roboflow — PPE Detection Project                      | CC BY 4.0                                        | 7,000  | The 7 original classes                                       |
| Open Images V7                                        | annotations CC BY 4.0, images under their own license | 1,853  | `Non-Safety Headwear` (4,248), `Person` (3,231, pseudo-labeled) |
| Voxel51 `hard-hat-detection` (Hugging Face)           | CC0                                              | 5,000  | `Uncovered Head` (5,785), `Safety Helmet` (18,966), `Person` (pseudo-labeled) |

| Split     | Images | Annotations |
| --------- | ------ | ----------- |
| train     | 9,942  | 51,151      |
| valid     | 2,648  | 13,968      |
| test      | 1,263  | 6,941       |
| **Total** | 13,853 | 72,060      |

v3 dataset audit: 0 errors, 0 polygon lines, 0 exact binary duplicates.
Report: `artifacts/reports/audit_v3.md`.

---

## 4. Prerequisites

- **Windows 10/11** with PowerShell (the code remains portable to Linux/macOS).
- **Python 3.10 to 3.13** (3.12 recommended). Python 3.14 is not yet
  supported by PyTorch.
- **NVIDIA GPU** optional but strongly recommended. CPU works, but full
  training would be unreasonably long.
- ~10 GB of disk space (original dataset + normalized copy + weights).

Validated reference environment:

```
Python        3.12.10
torch         2.9.1+cu128
ultralytics   8.4.112
opencv        5.0.0
onnxruntime   1.28.0
GPU           NVIDIA GeForce RTX 5080 Laptop (16 GB, sm_120)
Driver        596.36 (CUDA 13.2)
```

---

## 5. Installation

### Automatic installation (recommended)

```powershell
.\scripts\setup.ps1
```

The script creates the venv, detects the GPU, installs the matching PyTorch
variant, installs the project, then **checks that the PyTorch build actually
contains kernels for your GPU**.

Variants:

```powershell
.\scripts\setup.ps1 -Cuda cpu                    # machine without a GPU
.\scripts\setup.ps1 -PythonVersion 3.11 -Force   # other version, recreate
```

### Manual installation

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

**GPU (Blackwell / RTX 50xx — CUDA 12.8):**

```powershell
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

**GPU (Ampere / Ada — RTX 30xx, 40xx):**

```powershell
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

**CPU only:**

```powershell
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Then the project:

```powershell
python -m pip install -e ".[api,ui,export,audit,dev]"
```

Check:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_arch_list())"
```

---

## 6. Dataset audit

```powershell
python -m ppe_detection.dataset_audit --data data.yaml --output artifacts/reports/dataset_audit_original.json
```

Useful options:

```powershell
# Fast audit, without searching for visual near-duplicates
python -m ppe_detection.dataset_audit --data data.yaml --output artifacts/reports/audit.json --skip-perceptual-hash

# Fail if blocking errors or polygons remain (useful in CI)
python -m ppe_detection.dataset_audit --data data.yaml --output artifacts/reports/audit.json --fail-on-error --fail-on-polygon
```

The audit checks: existence of the splits, validity of `data.yaml`,
image/label pairing, extensions, corrupted images, dimensions and aspect
ratios, empty labels, class IDs, non-numeric values, coordinates outside
`[0, 1]`, zero or negative sizes, boxes overflowing the frame, tiny boxes,
5-field lines, polygon lines, per-class distribution, imbalance, exact binary
duplicates, perceptual near-duplicates and leakage between splits.

### Results on this dataset

| Finding                                      | Value                                         |
| -------------------------------------------- | --------------------------------------------- |
| Images / annotations                         | 7,000 / 25,542                                |
| Image ↔ label pairing                       | perfect on all 3 splits                       |
| Unreadable or corrupted images               | 0                                             |
| Malformed lines                              | 0                                             |
| **Polygon lines (segmentation)**       | **349** (285 train, 46 valid, 18 test)  |
| Tiny numerical drifts corrected              | 457                                           |
| Exact binary duplicates                      | 0                                             |
| Imbalance (max/min)                          | **9.71** (Person vs Face Mask)          |
| Small objects (area < 1% of the image)       | ~35% of boxes                                 |
| Distinct resolutions                         | 256 (train), from 55×87 to 5178×3884        |

> The 349 polygon lines are **not** counted as errors: they are valid
> segmentation annotations that must be converted to bounding boxes for a
> detection task.

### Leakage between splits — an important finding

The audit reveals a real and significant problem:

| Indicator                                                                          | Value                          |
| ---------------------------------------------------------------------------------- | ------------------------------ |
| Groups of images from the **same source photo** spread across several splits | **390** (1,252 files)    |
| Clusters of visually near-identical images                                         | 296 (2,654 images)             |
| of which clusters spanning several splits                                          | 173 (2,368 images)             |
| Largest cluster                                                                    | 1,032 images                   |
| Numbered sequences (`frame_000324`, …) spread across several splits             | 58 prefixes (3,241 images)     |

Two distinct mechanisms are at play:

1. **Augmented variants of the same photo.** Roboflow names files
   `photo_jpg.rf.<hash>.jpg`; the prefix before `.rf.` identifies the source
   photo. 390 source photos appear in several splits. A pixel-by-pixel
   comparison confirms that these are indeed geometric transformations of the
   same image (180° rotation verified on `101307074_544e234e97`), **even
   though the Roboflow README states "No pre-processing or augmentation was
   applied"**.
2. **Consecutive video frames.** 1,785 images are named `frame_NNNNNN` and
   come from video sequences. Two neighboring frames are nearly identical;
   randomly split between train and test, they make the evaluation
   optimistic.

**Consequence: metrics measured on this split overestimate real-world
performance on a never-seen site.** See
[section 7](#anti-leak-option) for the available mitigation.

---

## 7. Annotation conversion

The original dataset is **never** modified. A normalized copy is built,
containing only 5-field YOLO detection lines.

```powershell
python -m ppe_detection.dataset_cleaner --source data.yaml --output artifacts/dataset_detection --mode copy
```

Or, in a single command with a before/after audit:

```powershell
.\scripts\audit_dataset.ps1
```

### Anti-leak regrouping (enabled by default)

The Roboflow export places **augmented variants of the same photo** in
different splits: 390 source photos were spread across train, valid and
test. A model evaluated under these conditions is scored on images it has
already learned.

The cleaner therefore regroups, **by default**, all variants of the same
source photo into a single split. The chosen split is the one already holding
the majority of the files; ties are broken by a stable hash, so the result is
reproducible.

Measured effect on this project:

|                                         | Roboflow split    | Regrouped (default) |
| --------------------------------------- | ----------------- | ------------------- |
| Train/valid/test split                  | 4903 / 1399 / 698 | 5145 / 1277 / 578   |
| Source photos spanning several splits   | 390               | **0**         |
| mAP@0.50 reported on test               | 0.8319            | **0.7992**    |
| mAP@0.50:0.95 reported                  | 0.4696            | **0.4326**    |

The numbers on the right are the real ones. The +0.033 gap measured leakage,
not performance: 139 of the 698 test images (19.9%) were also in train.

To reproduce the original split — for example to compare with results
published on the Roboflow dataset — use `--allow-source-leak`, keeping in mind
that the resulting metrics will be optimistic.

**Accepted residual leakage**: 78 near-duplicate clusters still cross the
splits. These are consecutive video frames (`frame_000324`,
`frame_000325`...), formally distinct source images that name-based
regrouping cannot match. Removing them would require re-stratifying by video
sequence, which is a protocol decision.

### Conversion rule

For a polygon line `class_id x1 y1 x2 y2 …`:

```
xmin = min(x)      center_x = (xmin + xmax) / 2
ymin = min(y)      center_y = (ymin + ymax) / 2
xmax = max(x)      width    = xmax - xmin
ymax = max(y)      height   = ymax - ymin
```

### Correction policy

| Situation                                                                       | Handling                                                                         |
| ------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| Valid 5-field line                                                              | kept as is                                                                       |
| Polygon (≥ 3 points, paired coordinates)                                       | converted to a bounding box, logged                                              |
| Deviation outside `[0, 1]` ≤ 1e-3                                            | clamped to the bounds (numerical drift from the exporter)                        |
| Box overflowing the frame, valid center                                         | clipped to the image, logged                                                     |
| Center outside the image, zero/negative size, unknown class, non-numeric field | **line excluded** and logged — no plausible box is invented               |

### Options

```
--source          data.yaml of the original dataset
--output          directory of the derived dataset
--mode            copy | symlink
--overwrite       explicitly replaces an existing dataset
--dry-run         analyze without writing anything
--strict          fail if at least one line must be excluded
--allow-source-leak  disable anti-leak regrouping (original Roboflow split)
--regroup-by-source  no effect, kept for compatibility (regrouping is the default)
--no-audit        do not run the verification audit
```

Always do a dry run first:

```powershell
python -m ppe_detection.dataset_cleaner --source data.yaml --output artifacts/dataset_detection --dry-run
```

### Result obtained

```
25,542 lines read → 25,542 kept (0 lost)
   349 converted from a polygon
   457 numerical drifts corrected
     0 excluded
```

The automatically rerun verification audit confirms: **0 polygon lines,
0 malformed lines, 100% 5-field lines**.

### Anti-leak option

The leakage described in section 6 is removed **by default**: the conversion
command above already regroups all variants of the same source photo into a
single split (the one already holding the majority of the files; ties are
broken by a stable hash, so reproducibly). The legacy `--regroup-by-source`
flag is still accepted but no longer has any effect.

To go back to the original Roboflow split, for example to compare with
published results:

```powershell
python -m ppe_detection.dataset_cleaner --source data.yaml --output artifacts/dataset_detection_leak --mode copy --allow-source-leak
```

Measured effect of regrouping:

| Indicator                                       | Original splits       | After regrouping      |
| ----------------------------------------------- | --------------------- | --------------------- |
| Source groups spread across several splits      | 390 (1,252 files)     | **0**           |
| Numbered sequences spread across several splits | 58 prefixes           | 19 prefixes           |
| Train / valid / test split                      | 4,903 / 1,399 / 698   | 5,145 / 1,277 / 578   |

488 files are moved. The accepted trade-off: the resulting numbers are not
directly comparable with those published on the original Roboflow split, hence
the `--allow-source-leak` option to reproduce them.

> Limitation: regrouping by source photo does **not** solve leakage due to
> video sequences, since two neighboring frames are different source images.
> After regrouping, 78 near-duplicate clusters still span several splits.

---

## 8. Smoke test

**Run it systematically before any long training.**

```powershell
.\scripts\smoke_train.ps1
```

or:

```powershell
python -m ppe_detection.train --config configs/train.yaml --smoke
```

The smoke test runs 2 epochs on 4% of the data and checks that the full
pipeline does produce weights and metrics.

**Result on the reference machine: passed in 70 seconds**, weights written to
`artifacts/models/smoke_best.pt`. The associated metrics
(mAP@0.50 = 0.095) have no predictive value — this is expected after
2 epochs on 4% of the data.

---

## 9. Full training

```powershell
.\scripts\train.ps1
```

or, as a direct command:

```powershell
python -m ppe_detection.train --config configs/train.yaml
```

Common overrides:

```powershell
python -m ppe_detection.train --config configs/train.yaml --epochs 150 --batch 24
python -m ppe_detection.train --config configs/train.yaml --model yolo26m.pt --imgsz 768
python -m ppe_detection.train --config configs/train.yaml --device cpu
```

### Chosen configuration (`configs/train.yaml`)

| Parameter                              | Value          | Rationale                                                                  |
| -------------------------------------- | -------------- | -------------------------------------------------------------------------- |
| `model`                              | `yolo26s.pt` | Speed/accuracy trade-off as a baseline                                     |
| `imgsz`                              | 640            | Standard; ~35% of objects are small, a smaller size would degrade them     |
| `epochs`                             | 100            | With early stopping (`patience: 25`)                                     |
| `batch`                              | `-1` (auto)  | Ultralytics calibrates to ~60% of VRAM                                     |
| `workers`                            | 8              | On Windows, each worker is a full process                                  |
| `amp`                                | `true`       | Essential on Blackwell                                                     |
| `seed` / `deterministic`           | 42 / `true`  | Reproducibility                                                            |
| `degrees`, `flipud`                | 0.0            | PPE has a stable orientation (helmet on top)                               |
| `mixup`, `erasing`, `copy_paste` | 0.0            | Risk making small PPE items disappear                                      |
| `close_mosaic`                       | 10             | Disables mosaic at the end of training                                     |

### Resuming after an interruption

`Ctrl+C` stops cleanly; `last.pt` remains usable.

```powershell
python -m ppe_detection.train --resume artifacts/runs/ppe_yolo26s/weights/last.pt
```

### Items archived with each run

`best.pt`, `last.pt`, `resolved_train_config.yaml`, `run_metadata.json`
(environment, `pip freeze`, seed, device), `results.csv`, curves,
confusion matrix, prediction samples, `training_summary.json`.

---

## 10. Evaluation

```powershell
python -m ppe_detection.evaluate --weights artifacts/models/best.pt --data artifacts/dataset_detection/data.yaml --split test
```

or, validation then test:

```powershell
.\scripts\evaluate.ps1
```

> **Protocol**: thresholds and hyperparameters are chosen on the
> **validation** split. The test split is only used once the choices are
> final.

The report contains: precision, recall, mAP@0.50, mAP@0.50:0.95, mAP@0.75,
per-class metrics, confusion matrix, preprocessing / inference /
postprocessing times, throughput in images/s, weights size, parameter count,
error analysis (TP / FP / FN per class), confusions between classes, best and
worst examples, and known limitations.

### Current model (v3, 9 classes)

`yolo26s`, 640 px, v3 dataset (section 3). v3 test split: 1,263 images.

| Metric        | v3 test |
| ------------- | ------- |
| mAP@0.50      | 0.8368  |
| mAP@0.50:0.95 | 0.5077  |
| Precision     | 0.8360  |
| Recall        | 0.8009  |
| Inference     | 2.15 ms / image (RTX 5080 Laptop) |
| Throughput    | 340 img/s |

| Class               | Precision | Recall | mAP@0.50        | mAP@0.50:0.95 |
| ------------------- | --------- | ------ | --------------- | ------------- |
| Face Mask           | 0.905     | 0.932  | **0.958** | 0.583         |
| Uncovered Head      | 0.911     | 0.900  | 0.950           | 0.626         |
| Safety Helmet       | 0.909     | 0.908  | 0.948           | 0.591         |
| Person              | 0.858     | 0.907  | 0.917           | 0.678         |
| Safety Vest         | 0.851     | 0.864  | 0.903           | 0.552         |
| Safety Harness      | 0.837     | 0.693  | 0.796           | 0.405         |
| Non-Safety Headwear | 0.776     | 0.763  | 0.772           | 0.489         |
| Safety Shoes        | 0.808     | 0.694  | 0.762           | 0.419         |
| Safety Gloves       | 0.669     | 0.547  | **0.525** | 0.226         |

This v3 test set contains many `hard-hat-detection` images, on which the model
does markedly better: `Safety Helmet` reaches 0.948 mAP@0.50 there, versus
0.794 on the original images alone. The global score is therefore more
flattering than on the original domain. The fair comparison with the previous
models is done on the **same 578 original test images**, class by class:

|               | 7 classes | 8 classes | v3               |
| ------------- | --------- | --------- | ---------------- |
| mAP@0.50      | 0.7992    | 0.7988    | **0.8077** |
| Recall        | 0.7539    | 0.7451    | **0.7723** |
| Face Mask     | 0.9224    | 0.8931    | **0.9592** |
| Safety Vest   | 0.8777    | 0.8741    | **0.9077** |
| Safety Helmet | 0.7961    | **0.8067** | 0.7940          |

Reports: `artifacts/reports/evaluation_v3_test.md` and
`artifacts/reports/common_v3.md`. The results below concern the 7-class model,
which served as the baseline for every analysis in this section.

### Reference results of the 7-class model (`yolo26s`, 640 px, leak-free dataset)

| Metric        | Validation | Test   |
| ------------- | ---------- | ------ |
| mAP@0.50      | 0.7789     | 0.7992 |
| mAP@0.50:0.95 | 0.4294     | 0.4326 |
| Precision     | 0.7855     | 0.8086 |
| Recall        | 0.7333     | 0.7539 |

Per class, on test:

| Class          | Instances | Median size @640 | mAP@0.50        | mAP@0.50:0.95 |
| -------------- | --------- | ---------------- | --------------- | ------------- |
| Face Mask      | 788       | 46 px            | **0.922** | 0.560         |
| Person         | 7,649     | 200 px           | 0.893           | 0.531         |
| Safety Vest    | 2,434     | 136 px           | 0.878           | 0.535         |
| Safety Helmet  | 5,449     | 50 px            | 0.796           | 0.352         |
| Safety Harness | 1,175     | 155 px           | 0.782           | 0.400         |
| Safety Shoes   | 5,875     | 73 px            | 0.775           | 0.430         |
| Safety Gloves  | 2,172     | 50 px            | **0.548** | 0.221         |

**Performance follows object size, not frequency.** `Face Mask` is the rarest
class (3.1% of annotations) and the best detected; `Safety Helmet` is the
second most frequent (21.3%) and plateaus, because 25% of helmets are smaller
than 32 px at 640. Rebalancing the classes would therefore be useless here —
the lever is resolution.

### Retraining at 960 px: hypothesis tested, negative result

The hypothesis was that training at 960 px would improve the small-object
classes. It was tested all the way through — a full 3 h 36 min training run
(97 epochs, early stopping, best epoch 72) — and **it is not confirmed at the
global level**.

Comparison on the same test split, each model evaluated at its training
resolution:

|               | 640 px              | 960 px           | Delta    |
| ------------- | ------------------- | ---------------- | -------- |
| mAP@0.50      | **0.7992**    | 0.7980           | −0.0012 |
| mAP@0.50:0.95 | **0.4326**    | 0.4276           | −0.0050 |
| Precision     | **0.8086**    | 0.8059           | −0.0027 |
| Recall        | 0.7539              | **0.7572** | +0.0033  |
| Inference     | **2.76 ms**   | 5.83 ms          | ×2.1    |
| Throughput    | **271 img/s** | 130 img/s        | ÷2.1    |

Per class, the prediction is **partially** confirmed — the two classes
singled out by the analysis do improve:

| Class          | % objects < 32 px | mAP@0.50 640 | mAP@0.50 960     | Delta             |
| -------------- | ----------------- | ------------ | ---------------- | ----------------- |
| Safety Gloves  | 13%               | 0.5483       | **0.5757** | **+0.0274** |
| Person         | 0%                | 0.8927       | **0.9152** | +0.0225           |
| Safety Helmet  | 25%               | 0.7961       | **0.8096** | +0.0135           |
| Safety Vest    | 2%                | 0.8777       | 0.8733           | −0.0044          |
| Safety Shoes   | 9%                | 0.7751       | 0.7673           | −0.0078          |
| Safety Harness | 2%                | 0.7821       | 0.7553           | −0.0268          |
| Face Mask      | 18%               | 0.9224       | 0.8892           | −0.0332          |

`Safety Gloves`, the weakest class, gains 5% in relative terms. But the signal
remains weak and noisy: small-object classes gain +0.0026 on average, the
others lose −0.0041. Above all, **`Face Mask` regresses the most even though
18% of its objects are tiny**, which contradicts a purely size-based
explanation.

**Decision: `best.pt` remains the 640 px model.** It is better or equivalent
on every global metric and twice as fast. The 960 model is kept under
`artifacts/models/best_960.pt`: it may be justified if glove detection becomes
a priority, at the cost of throughput.

What this teaches: **resolution alone does not make up for a lack of
diversity in the data.** The remaining lever is collecting field data — see
[`docs/plan_ecart_terrain.md`](docs/plan_ecart_terrain.md).

Also note: evaluating the 640 px weights at 960 px without retraining degrades
the result (0.780 vs 0.799). **Increasing resolution at inference time alone
does not work** — the model expects the scale it was trained on.

---

### Per-class threshold calibration

A single threshold for all classes is a poor compromise: each class has its
own score distribution. The following command sweeps the thresholds and
keeps, for each class, the one that maximizes F1:

```powershell
python -m ppe_detection.calibrate --weights artifacts/models/best.pt `
  --data artifacts/dataset_detection/data.yaml --split valid
```

Add `--apply` to write the thresholds directly into
`configs/inference.yaml` (beware: rewriting the YAML removes the file's
comments; the report always provides the snippet to copy).

**Protocol**: calibration is performed on **validation** only. Choosing
thresholds on test would amount to fitting the model on the data meant to
evaluate it — the command actually refuses `--split test` unless
`--allow-test-split` is explicitly passed.

Results on this project (thresholds already applied in `inference.yaml`):

| Class              | Chosen threshold | F1 gain on **test** |
| ------------------ | ---------------- | ------------------------- |
| Face Mask          | 0.25             | +0.0000                   |
| Person             | 0.35             | +0.0076                   |
| Safety Gloves      | 0.30             | +0.0025                   |
| Safety Harness     | 0.40             | **+0.0235**         |
| Safety Helmet      | 0.30             | +0.0116                   |
| Safety Shoes       | 0.30             | −0.0001                  |
| Safety Vest        | 0.45             | +0.0127                   |
| **Macro F1**       |                  | **+0.0083**         |

In practice on the test split: **120 fewer false positives** for 52 true
positives lost. Thresholds chosen on validation therefore generalize well
(+0.0067 expected, +0.0083 observed).

> These thresholds were calibrated on the 7-class model and **have not been
> recomputed for the v3 model**. The `Non-Safety Headwear` and
> `Uncovered Head` classes use the global `conf` threshold. Rerunning
> `calibrate` on the v3 validation split is the logical next step.

A business constraint on minimum recall is available via `--min-recall`:
useful when missing a PPE item costs more than a false alarm.

## 11. Inference

Single interface for all sources.

```powershell
# Image
python -m ppe_detection.predict --weights artifacts/models/best.pt --source path\to\image.jpg --save --save-json

# Folder
python -m ppe_detection.predict --weights artifacts/models/best.pt --source path\to\folder --save --save-txt --save-json --save-csv

# Video
python -m ppe_detection.predict --weights artifacts/models/best.pt --source path\to\video.mp4 --save --save-json

# Webcam (real-time window, 'q' or Esc to quit)
python -m ppe_detection.predict --weights artifacts/models/best.pt --source 0 --show

# RTSP stream
python -m ppe_detection.predict --weights artifacts/models/best.pt --source "rtsp://user:password@192.168.1.10:554/stream" --show
```

Main options:

```
--conf / --iou       confidence and NMS thresholds
--device             auto | cpu | cuda | 0
--imgsz              inference size
--max-det            maximum detections per image
--half               FP16 (GPU only)
--save               annotated images/video
--save-txt           YOLO labels (class cx cy w h conf)
--save-json          structured JSON report
--save-csv           flat CSV report
--recursive          walk subfolders
--hide-labels        hide class names
--hide-conf          hide scores
--compliance         enable PPE compliance
--track              object tracking + temporal smoothing of verdicts
--tracker            bytetrack.yaml (fast) | botsort.yaml (occlusions)
--show               real-time window (video/webcam)
--frame-skip N       only run inference on one frame out of N+1
--max-frames N       limit the number of frames
```

**Per-class** thresholds are defined in `configs/inference.yaml`:

```yaml
inference:
  conf: 0.20
  class_conf:
    Face Mask: 0.35
    Safety Gloves: 0.40
```

> A per-class threshold can only **tighten** the global threshold:
> Ultralytics already filters at `conf` before these thresholds apply. To
> actually lower a threshold, lower `conf` and then raise the other classes.

Verified video behavior: the model is loaded only once, frame order is
preserved, FPS is shown as an overlay, the camera and the `VideoWriter` are
released in a `finally` block, and the output video's properties
(resolution, FPS, frame count) are preserved.

---

## 12. PPE compliance

> **Warning.** The model detects objects **independently**. Nothing in its
> outputs formally links a helmet to a person. The compliance layer applies a
> **geometric heuristic**: a PPE item is assigned to the person whose
> expected region contains the largest fraction of the PPE box. A
> "non-compliant" status is an **alert to be checked**, never an automatic
> finding.

```powershell
python -m ppe_detection.predict --weights artifacts/models/best.pt --source path\to\image.jpg --compliance --save --save-json
python -m ppe_detection.predict --weights artifacts/models/best.pt --source 0 --show --compliance --required-ppe "Safety Helmet" "Safety Vest"
```

Configuration (`configs/inference.yaml`):

```yaml
compliance:
  enabled: false
  person_class: Person
  required_ppe:
    - Safety Helmet
    - Safety Vest
  association:
    containment_threshold: 0.50   # fraction of the PPE box inside the expected region
    helmet_region: 0.35           # top 35% of the person
    shoes_region: 0.30            # bottom 30%
    torso_region: [0.20, 0.80]
  region_by_class:
    Safety Helmet: head
    Face Mask: head
    Safety Vest: torso
    Safety Harness: torso
    Safety Gloves: any
    Safety Shoes: feet

  # Observability: when can we ASSERT that a PPE item is missing?
  min_region_height_px: 24   # below this threshold, the object cannot be resolved
  edge_margin_px: 2          # region touching the edge = truncated

  # Temporal smoothing (--track option)
  temporal_window: 15
  temporal_min_ratio: 0.70
  temporal_min_observations: 5
```

### Association via body keypoints (recommended)

Fraction-based splitting assumes a person **standing and seen from the
front**. That assumption breaks as soon as the person is crouching, leaning,
sitting, or filmed from above — the usual case in video surveillance.

The `--pose` option replaces this splitting with the **actual** position of
body parts, obtained from a pose estimation model (17 COCO keypoints,
`yolo26n-pose.pt`, downloaded automatically):

```powershell
python -m ppe_detection.predict --weights artifacts/models/best.pt `
  --source path\to\image.jpg --compliance --pose --save --save-json
```

| Region    | Keypoints used                        | Replaces                  |
| --------- | ------------------------------------- | ------------------------- |
| `head`  | nose, eyes, ears + torso scale        | top 35% of the box        |
| `torso` | shoulders and hips                    | 20–80% band              |
| `feet`  | ankles                                | bottom 30%                |
| `hands` | wrists                                | whole box                 |

A helmet hides the skull: the "head" region is therefore extrapolated
**above** the face keypoints, based on torso length.

Measured on a crouching worker (real frame) — the torso region goes from
`x[387-1133] y[144-576]` (fractions) to `x[719-999] y[262-679]` (pose), a
re-centering consistent with her posture.

**Automatic fallback**: if the required keypoints are missing (person seen
from behind, too small, occluded), the system falls back to fraction-based
splitting for that region. The `association_method` field of each verdict
indicates the method actually used, `pose` or `bbox_fractions`.

Cost: a second model in memory and one extra inference per image.

### Three states, not two

A detector that does not see a vest does not prove its absence. Declaring a
person "non-compliant" when the relevant region is not observable produces
false alarms en masse. The system therefore distinguishes:

| Status            | Meaning                                                              | Color |
| ----------------- | -------------------------------------------------------------------- | ----- |
| `compliant`     | All required PPE is detected and assigned                            | green |
| `non_compliant` | A required PPE item is missing **in a truly observable region** | red   |
| `indeterminate` | The region is not observable — no conclusion                        | amber |

A region is deemed not observable in two cases: it **touches an edge of the
frame** (truncated person, typically the head sticking out at the top), or it
is **too small in pixels** for the detector to resolve an object in it.

Each person's `reasons` field states precisely why a PPE item is
indeterminate, for example `Safety Helmet : zone 'head' tronquee par le bord haut du cadre`
(i.e. "head region truncated by the top edge of the frame" — messages are
currently emitted in French).

The compliance rate is computed over **assessable** people only: including
indeterminate people in the denominator would artificially lower the rate
because of people we simply could not observe.

### Tracking and temporal smoothing (video)

```powershell
python -m ppe_detection.predict --weights artifacts/models/best.pt --source path\to\video.mp4 --compliance --track --save --save-json
```

With `--track`, each person gets a persistent ID (ByteTrack by default,
`botsort.yaml` available for more robustness to occlusions), and the verdict
is smoothed over a sliding window: it only flips after a clear majority of
consistent observations. An alert is raised **once per person**, not on every
frame.

The report then distinguishes two views:

- `tracked_compliance` — the **per-person** summary, the only view that is
  operationally meaningful;
- `per_detection_compliance` — the former per-detection count, kept for
  comparison.

Measured effect on a 122-frame construction site video:

| View                                     | Without tracking          | With `--track`              |
| ---------------------------------------- | ------------------------- | ----------------------------- |
| Units counted                            | 194 person detections     | **6 tracked people**    |
| Non-compliant                            | 147                       | 4                             |
| Indeterminate (held back by level 1)     | 35                        | 2                             |
| Alerts raised                            | 147                       | **5**                   |

### Telling real PPE apart from look-alikes

The 7-class model labeled a **bicycle helmet** as `Safety Helmet` with
**0.84 confidence** (verified on test images). It had not learned "hard hat"
but "rigid domed shell on a head".

This is not a lack of data: the schema only contains **positive** classes, so
no output can express "looks like a helmet but isn't one". Adding more hard
hats won't change that.

The project provides the tooling to fix this:

- [`taxonomy.py`](src/ppe_detection/taxonomy.py) defines an **extended
  11-class schema** (`Non-Safety Headwear`, `Uncovered Head`, `Non-Safety Vest`,
  `Non-Safety Footwear`). The v3 model learns 9 of them: the last two have no
  data yet.
  The seven original classes keep their IDs: an extended dataset stays
  backward compatible.
- [`dataset_merge.py`](src/ppe_detection/dataset_merge.py) assembles public
  datasets by remapping their classes, which avoids annotating from scratch.
- The **counter-evidence** mechanism distinguishes two levels of evidence in
  the verdict: `evidence: absence` (nothing detected, may be a false negative)
  and `evidence: observed` (non-compliant headwear seen — violation observed).
  Counter-evidence takes precedence over the observability test: seeing the
  object proves the region is visible.

Volumes to annotate, free sources and annotation rules:
[`docs/plan_donnees_epi_sosies.md`](docs/plan_donnees_epi_sosies.md).

#### Results of the 8-class model

Extended dataset: 8,204 images, 29,079 annotations, including **3,537
`Non-Safety Headwear` instances** from Open Images V7 — with no manual
annotation. Training took 2 h 09 min (91 epochs, early stopping, best
epoch 66).

**Test 1 — look-alikes.** This is the goal being pursued, and it is reached:

| Image              | 7 classes              | 8 classes                              |
| ------------------ | ---------------------- | -------------------------------------- |
| MTB helmet         | `Safety Helmet 0.32` | **`Non-Safety Headwear 0.94`** |
| Road bike helmet   | `Safety Helmet 0.84` | **`Non-Safety Headwear 0.50`** |
| Baseball cap       | *nothing*            | **`Non-Safety Headwear 0.41`** |
| Sports cap         | *nothing*            | **`Non-Safety Headwear 0.95`** |

No more false `Safety Helmet`. Caps, previously simply ignored, are now
**actively detected**, which lets counter-evidence work.

**Test 2 — non-regression on the same 578 images.** Comparing the global mAP
of two models with a different number of classes would be meaningless: the
average does not cover the same classes. The comparison is therefore done
class by class, on identical images.

| Class                        | mAP@0.50 7 cls   | mAP@0.50 8 cls   | Delta              |
| ---------------------------- | ---------------- | ---------------- | ------------------ |
| Safety Harness               | 0.7821           | **0.8091** | +0.0270            |
| Safety Gloves                | 0.5483           | **0.5607** | +0.0124            |
| Safety Helmet                | 0.7961           | **0.8067** | +0.0106            |
| Person                       | 0.8927           | **0.8960** | +0.0033            |
| Safety Vest                  | 0.8777           | 0.8741           | −0.0036           |
| Safety Shoes                 | 0.7751           | 0.7523           | −0.0228           |
| Face Mask                    | 0.9224           | 0.8931           | −0.0293           |
| **Global (7 classes)** | **0.7992** | **0.7988** | **−0.0004** |

The global gap is within noise. `Safety Helmet` **improves** by +0.0106,
contrary to the degradation one might have feared: discriminating did not
cost detection performance. Two classes drop by more than 0.02, `Face Mask`
and `Safety Shoes`, with no obvious link to headwear.

**Test 3 — quality of the new class**: `Non-Safety Headwear` reaches
**0.7373 mAP@0.50** with 0.829 precision, i.e. 5th class out of 8 —
ahead of `Safety Shoes` and far ahead of `Safety Gloves`. Reliable enough to
base an alert on, hence `counter_evidence` being enabled in
[`configs/inference.yaml`](configs/inference.yaml).

#### Known limitation: people not detected in sports contexts

On photos from Open Images — portraits of cyclists, sports scenes — the
8-class model **no longer detects people**. On the cyclist image, the 7-class
model saw `Person 0.886`; the new one only sees the headwear.

Cause: the Open Images pictures were downloaded with `only_matching=True`,
which only keeps the headwear labels. People therefore appear **unannotated**,
and the model learns not to detect them in that visual context.

Actual impact: **none on the target domain**. On the same 60 construction
site frames, the 8-class model even detects *more* people than the previous
one (72 vs 65). The regression is limited to sports imagery.

Fix applied in the v3 dataset: people in the Open Images pictures are
pre-annotated there through pseudo-labeling (3,231 people added, see below).

#### Results of the 9-class v3 model (current model)

**Why an `Uncovered Head` class.** No accessible public dataset annotates bump
caps, work caps or beanies as such. On the other hand, `hard-hat-detection`
(Voxel51, 5,000 images, CC0) provides 5,785 **unprotected** heads in an
industrial context. Rather than naming the type of headwear, the model
therefore answers the operational question: "is this head protected?".

**Incompatible annotation conventions.** Open Images boxes the hat, Voxel51
boxes the head. On a cap, both boxes would be nearly identical with opposite
labels. Hence a separate `Uncovered Head` class, and `ANNOTATION_CONVENTIONS`
in `taxonomy.py`, which documents what each box delimits so the two
conventions are never mixed.

**Pseudo-labeling people.** `hard-hat-detection` only annotates 751 people
across 5,000 images: merging it as is would have reproduced the disappearance
of people observed with the 8-class model.
[`pseudo_label.py`](src/ppe_detection/pseudo_label.py) pre-annotates the missing
class with an existing model, preserving the real annotations:

| Source               | Model used     | People added |
| -------------------- | -------------- | ------------ |
| `hard-hat-detection` | 8-class model  | 13,537       |
| Open Images          | 7-class model  | 3,231        |

Pitfall encountered: pre-annotating Open Images with the 8-class model only
yielded 14 people across 1,853 images — the regression was blocking its own
fix. The 7-class model, sound on this point, was used instead.

**Cleanup.** Football helmets (1,329 instances, 37.6% of the raw
`Non-Safety Headwear` class) were removed: they are irrelevant on a
construction site.

**Results.** On the 578 original test images, v3 is the best of the three
models (0.8077 mAP@0.50 vs 0.7992 and 0.7988), with higher recall (0.7723).
`Uncovered Head` reaches **0.950 mAP@0.50**, 2nd class out of 9. Full tables in
[section 10](#current-model-v3-9-classes).

**Measured limitation.** A white cap seen from behind is now detected as
`Safety Helmet 0.80`, whereas the 8-class model detected nothing: the 24,415
added helmets reinforce the prior "light dome on a head in an industrial
context = hard hat". Three other caps out of four are correctly classified as
`Non-Safety Headwear`.

**Counter-evidence status.** `taxonomy.py` lists `Non-Safety Headwear` and
`Uncovered Head` as counter-evidence for `Safety Helmet`, but
`configs/inference.yaml` currently only enables `Non-Safety Headwear`.
`Uncovered Head` is detected and displayed, but does not yet feed the
compliance verdict.

### Remaining limitations

The safeguards above remove a large share of false alarms, but the heuristic
remains fallible:

- **People close to or overlapping each other**: one person's helmet may be
  assigned to the other.
- **Strong high or low camera angle**: the "head is at the top of the box"
  assumption no longer holds. Only a pose-based approach would fix this case.
- **PPE not worn**: a helmet lying on a table inside the head region of a
  seated person will be counted as worn.
- **Detection false negative in an observable region**: a clearly visible
  vest that goes undetected still produces a wrong "non-compliant". This is
  currently the main remaining source of error, and it comes from the
  detector, not from the business rule. The `--pose` option does not fix it:
  by making the observability assessment more accurate, it actually tends to
  **expose** these failures rather than hide them behind an "indeterminate".
- **Unreliable classes**: `Safety Gloves` plateaus at 0.53 mAP@0.50 and 0.55
  recall (v3 model). It is listed in `unreliable_ppe`: adding it to `required_ppe`
  triggers a warning at load time, because the rule would produce a majority
  of false alarms.
- The `verdict_confidence` field relates to **detection**, not to the
  correctness of the business rule.
- A person who becomes compliant and then non-compliant again generates
  **two** alert events: this is intended, but it distinguishes `n_alerts`
  (events) from `persons_currently_alerted` (final state).

---

## 13. REST API

```powershell
.\scripts\run_api.ps1
.\scripts\run_api.ps1 -Weights artifacts/models/best.pt -Port 8080 -Compliance
```

or:

```powershell
python -m uvicorn ppe_detection.api:app --host 127.0.0.1 --port 8000
```

Interactive documentation: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

| Method  | Route              | Description                             |
| ------- | ------------------ | --------------------------------------- |
| GET     | `/health`        | Service status (`ok` / `degraded`)  |
| GET     | `/model-info`    | Metadata of the loaded model            |
| POST    | `/predict/image` | Inference on one image                  |
| POST    | `/predict/batch` | Inference on several images             |

Sample response:

```json
{
  "request_id": "3f1c...",
  "filename": "scene.jpg",
  "image": { "width": 640, "height": 640 },
  "detections": [
    {
      "class_id": 1,
      "class_name": "Person",
      "confidence": 0.8177,
      "bbox_xyxy": [411.99, 140.43, 605.24, 581.97]
    }
  ],
  "compliance": [],
  "timing_ms": { "preprocess": 1.48, "inference": 79.1, "postprocess": 0.27, "total_request": 103.22 }
}
```

Configuration via environment variables (see `.env.example`):
`PPE_API_WEIGHTS`, `PPE_API_DEVICE`, `PPE_API_CONF`, `PPE_API_IOU`,
`PPE_API_COMPLIANCE`, `PPE_API_MAX_FILE_MB`, `PPE_API_MAX_BATCH`.

### Security choices

- The model is loaded **only once**, at startup.
- Received files are decoded **entirely in memory**: no client-supplied
  content ever reaches the disk, which removes any risk of arbitrary writes
  through a hostile filename.
- The filename returned in the response is sanitized (`../../etc/passwd`
  becomes a harmless name).
- MIME type and size are validated **before** any decoding.
- Logs contain neither image content nor the raw submitted filename.
- An unhandled exception returns a JSON with an error ID, without leaking any
  internal stack trace.
- If the weights are missing, the service still starts: `/health` responds
  `degraded` and the inference routes return **503** with an actionable
  message, rather than making startup fail.

---

## 14. Streamlit interface

```powershell
.\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py
```

Then open [http://localhost:8501](http://localhost:8501).

The interface has four tabs (UI labels are currently in French).

| Tab                       | Purpose                                                                                                                                                              |
| ------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Image**           | Upload a photo, annotated image, detections table, compliance, JSON/JPEG download.                                                                                   |
| **Video**           | Upload a file, processing, **direct playback of the annotated video**, alert log, downloads.                                                                   |
| **Webcam (live)**   | **Continuous** inference on the local camera, with FPS, tracking, real-time alerts and optional recording. A "single snapshot" mode is also available.         |
| **Results**         | Replay of all annotated videos already produced in `artifacts/predictions/`.                                                                                       |

Shared settings in the sidebar: weights, device, confidence and IoU
thresholds, enabling compliance, choice of required PPE and enabling
temporal tracking.

### Live webcam

The camera is opened **server-side**, by the Streamlit process. This suits
local use, where the browser and the camera are on the same machine; a remote
deployment would require WebRTC (`streamlit-webrtc`).

The "Demarrer la camera" (Start camera) button launches a continuous
inference loop; "Arreter" (Stop) interrupts it. A duration safeguard (120 s
by default) automatically stops the loop, and the camera is always released
in a `finally` block, even on an error or a Streamlit rerun.

Throughput measured on the reference machine: **~14 FPS at 640×480** with
tracking and compliance enabled, annotation included.

### Playing annotated videos — codec

If your annotated videos stayed black in the browser, this was a codec issue,
now fixed.

OpenCV was writing with `mp4v`, which produces an **MPEG-4 Part 2** stream
(FOURCC `FMP4`) that no browser decodes natively. The project now requests
`avc1` (**H.264**), which plays everywhere.

Two precautions were necessary:

- OpenCV sometimes prints `Could not open codec libopenh264` on stderr and
  then **silently falls back to another H.264 encoder**. The produced file is
  valid: this message is harmless and can be ignored.
- OpenCV **substitutes a codec without reporting it** when the requested one
  is unavailable: `isOpened()` returns `True` even with a bogus FOURCC. The
  project therefore re-reads the file after closing it to find out which
  codec was actually written (`probe_video_codec`). The `output_codec` field
  of the summary reflects reality, and `browser_playable` follows from it.

Videos produced before this fix remain in `FMP4`: the "Results" tab
flags them explicitly and offers to download them.

**The interface never trains a model.** If no weights are available, it
displays the exact commands to run.

---

## 15. ONNX export

```powershell
python -m ppe_detection.export --weights artifacts/models/best.pt --format onnx --imgsz 640 --simplify
```

Optional formats: `torchscript`, `openvino`, `engine` (TensorRT, requires a
dedicated installation).

### Verification actually performed

An export is never considered successful just because a file exists. The
verification includes:

1. file present and non-empty;
2. valid graph (`onnx.checker`);
3. ONNX Runtime session loadable;
4. inference on a dummy input of the expected shape;
5. **comparison of detections** between PyTorch and ONNX Runtime on a real
   image, with IoU matching.

Result measured on the smoke test weights:

| Check                      | Result                    |
| -------------------------- | ------------------------- |
| ONNX graph (opset 20)      | valid                     |
| ONNX Runtime session       | loadable                  |
| Output shape               | `(1, 300, 6)`           |
| PyTorch vs ONNX detections | **9 / 9 matched**   |
| Mean IoU                   | **0.999999**        |
| Max confidence gap         | 6 × 10⁻⁶               |
| Max box offset             | 0.0002 px                 |
| Diverging classes          | 0                         |

### Post-processing differences to be aware of

- The exported model expects an image normalized to `[0, 1]`, in NCHW
  (`1×3×H×W`), RGB, letterbox-resized to the fixed size.
- YOLO26 exports **end-to-end**: the output already contains detections
  filtered and sorted by confidence. Comparing raw tensors element by element
  would make no sense, because a tiny numerical difference reorders the rows.
- Coordinates refer to the resized image: the letterbox (scale and offset)
  must be undone to map back to the original image.

---

## 16. Output structure

```
artifacts/
├── reports/
│   ├── dataset_audit_original.{json,md}    # Audit of the original dataset
│   ├── dataset_audit_detection.{json,md}   # Audit of the normalized dataset
│   ├── dataset_cleaning.{json,md}          # Conversion log
│   ├── evaluation_{valid,test}.{json,md}   # Evaluation reports
│   ├── export.{json,md}                    # Export report
│   └── *_assets/                           # Charts and annotated examples
├── dataset_detection/          # Normalized dataset (data.yaml + train/valid/test)
├── models/                     # best.pt (= v3), best_7classes.pt, best_8classes.pt, best_960.pt, smoke_best.pt
├── runs/
│   ├── <experiment>/           # Weights, curves, confusion matrix
│   └── val/                    # Validation outputs
├── predictions/<name>/
│   ├── images/                 # Annotated images
│   ├── labels/                 # Predicted YOLO labels
│   ├── predictions.{json,csv}
│   └── video_{summary,predictions}.json
├── exports/                    # *.onnx, *.torchscript
└── logs/                       # Per-command logs
```

---

## 17. Code quality and tests

```powershell
python -m pytest tests -q          # 221 tests
python -m ruff check src tests app # linting
python -m mypy                     # type checking
```

Verified state: **221 tests pass, ruff clean, mypy clean**.

The tests do not require any full training: they rely on synthetic fixtures
(a mini dataset covering every annotation edge case) and automatically skip
inference tests when no weights are present.

Coverage: annotation parsing and conversion, `data.yaml` path resolution
(including the Roboflow `../train/images` convention), audit, cleaning,
preservation of the source dataset, compliance geometry, filename
sanitization, output formats, taxonomy and counter-evidence, dataset merging,
keypoint association, threshold calibration, French labels, and the four API
routes via the FastAPI test client.

---

## 18. Troubleshooting

### CUDA unavailable even though a GPU is present

```powershell
nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

If `torch.__version__` does not contain `+cuXXX`, the CPU build is installed:

```powershell
python -m pip uninstall -y torch torchvision
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

### "no kernel image is available for execution on the device"

The PyTorch build contains no kernels for your GPU. Typical of
RTX 50xx (Blackwell, `sm_120`) with a build older than CUDA 12.8.

```powershell
python -c "import torch; print(torch.cuda.get_device_capability(0), torch.cuda.get_arch_list())"
```

If `sm_120` is missing from the list, reinstall with `cu128`.

### Insufficient GPU memory (CUDA out of memory)

In order of effectiveness:

```powershell
python -m ppe_detection.train --config configs/train.yaml --batch 16   # then 8
python -m ppe_detection.train --config configs/train.yaml --imgsz 512
python -m ppe_detection.train --config configs/train.yaml --model yolo26n.pt
```

Also check `cache: false` in `configs/train.yaml` and close other
applications using the GPU (`nvidia-smi`).

### Very slow training on Windows

Reduce `workers` (each worker is a full process) and enable `cache: disk` if
the disk allows it.

### Webcam not accessible

Check that no other application is using the camera and that access is
allowed in **Settings > Privacy > Camera**. Try another index
(`--source 1`).

### The annotated video is not produced

The `mp4v` codec may be missing depending on the OpenCV build. The message is
explicit and detections remain exportable as JSON.

### Paths containing spaces

Wrap them in quotes:

```powershell
python -m ppe_detection.predict --weights "artifacts/models/best.pt" --source "C:\My Images\site.jpg"
```

---

## 19. Known limitations

This section lists what is **actually** limiting. Nothing is swept under the
rug.

### Data

1. **Residual leakage through video sequences.** Regrouping by source photo
   is now enabled by default and removes the 390 affected groups. There remain
   **78 near-duplicate clusters spanning the splits**: consecutive video
   frames, formally distinct, that name-based regrouping cannot match. Metrics
   therefore remain slightly optimistic.
2. **Inaccurate source documentation.** The Roboflow README states "No
   pre-processing or augmentation was applied", which pixel analysis
   contradicts (rotated variants of the same photo).
3. **Real diversity lower than the advertised volume.** Out of 7,000 images,
   about 1,785 are frames extracted from a few video sequences, highly
   redundant with each other.
4. **Boxes derived from polygons.** 349 boxes come from a min/max conversion:
   the bounding box of a polygon is always at least as large as the actual
   object. This contributes to the low mAP@0.50:0.95 (0.433 vs 0.799 at
   IoU 0.50).
5. **Small objects — the main limiting factor.** 25% of helmets and 13% of
   gloves are smaller than 32 px at 640. This weighs more than class
   imbalance: `Face Mask`, the rarest class, is the best detected (0.922),
   while `Safety Helmet`, the second most frequent, plateaus at 0.796.
6. **Gap to the field not measured.** The dataset covers neither night, rain,
   backlight, nor video surveillance angles. On a real construction site
   video, `Safety Vest` (0.878 mAP@0.50 on test) was detected only 15 times
   over 122 images. Remediation plan:
   [`docs/plan_ecart_terrain.md`](docs/plan_ecart_terrain.md).
7. **Pseudo-labeled people in the v3 dataset.** 16,768 `Person` boxes
   (13,537 + 3,231) were produced by a model, without human review. They
   propagate that model's errors: missed people or imprecise boxes on the
   `hard-hat-detection` and Open Images pictures.

### Model and pipeline

8. **`Safety Gloves` is not production-ready**: 0.525 mAP@0.50 and 0.547
   recall on the v3 test — nearly one glove in two is missed. The class is listed in
   `unreliable_ppe` and triggers a warning if added to `required_ppe`.
9. **PPE compliance remains a heuristic**, even with `--pose`. Keypoints
   remove the "person standing, seen from the front" assumption, but neither
   pose nor geometry proves that a PPE item is actually **worn**: a helmet
   lying on a table in the head region will be counted as worn. Detailed
   limitations in [section 12](#12-ppe-compliance).
10. **ONNX verification covers a single image.** It proves the fidelity of the
   conversion, not equivalence over every input distribution.
11. **TensorRT and OpenVINO exports are not automatically verified** (only
    ONNX is) and have not been tested here.
12. **`symlink` mode silently falls back to copying** on Windows if
    permissions do not allow creating links (developer mode required).
13. **53% test coverage.** Heavy paths (training, export, full audit) are
    poorly covered: exercising them would require a GPU and the full dataset.
14. **Inconsistent `--output` option.** It refers to a **directory** in
    `evaluate`, `export`, `predict` and `dataset_cleaner`, but to a **file**
    in `dataset_audit`. A pitfall to avoid until it is unified.

---

## 20. Future work

### Already done

- Anti-leak regrouping by source photo, **enabled by default** (section 7).
- `indeterminate` state: do not blame a person we cannot observe.
- Multi-object tracking and temporal smoothing of verdicts (`--track`).
- Association via body keypoints (`--pose`).
- Per-class threshold calibration on validation (`calibrate`).
- Negative classes `Non-Safety Headwear` and `Uncovered Head` to tell a real
  hard hat apart from its look-alikes (v3 model, section 12).
- Merging public datasets and pseudo-labeling missing classes
  (`dataset_merge`, `pseudo_label`).

### Remaining priorities

**1. Finalize the v3 model.** Recalibrate per-class thresholds on the v3
validation split, and enable `Uncovered Head` as counter-evidence in
`configs/inference.yaml`.

**2. Measure the gap to the field.** Build a test set of 150 to 200 images
from real deployment conditions, never mixed into training. It is the only
honest judge of production performance, and the prerequisite to any other
optimization. See
[`docs/plan_ecart_terrain.md`](docs/plan_ecart_terrain.md).

**3. Resolution: tested lead, not to be rerun as is.** A full training run at
960 px brought no global gain (see section 10). No point going back to it
without changing something else. Still to try: multi-scale training
(`multi_scale: true`), which exposes the model to both regimes rather than
favoring one, and a higher-capacity model (`yolo26m`) at 640 px, cheaper at
inference than a `yolo26s` at 960 px.

**4. `Safety Gloves`.** A dedicated annotation campaign, or a deliberate
removal from the compliance rules. As it stands, the class supports no
decision.

**5. Stratification by video sequence.** Removing the 78 remaining
near-duplicate clusters requires splitting by source sequence, not by
filename.

**6. Advanced compliance.** Check that a PPE item is *worn* and not merely
present in the region (temporal consistency of wearing, helmet orientation).

**7. Production readiness.** INT8 quantization for embedded targets; verified
TensorRT export; containerizing the API; monitoring model drift.

**8. Technical debt.** Unify the semantics of `--output`; raise test coverage
on `train.py` and `export.py`.

---

## License and attribution

Code under the MIT license. The dataset comes from Roboflow Universe under the
**CC BY 4.0** license and must be attributed to its author:
[https://universe.roboflow.com/ousmane-savadogo/ppe-detection-project-jeezl-p9ncg](https://universe.roboflow.com/ousmane-savadogo/ppe-detection-project-jeezl-p9ncg)

This system is a detection aid. **It does not replace human safety
inspection** and must not be used as the sole compliance control mechanism on
a real site.
