# TA-RandBox: temporal random proposals for open-world object detection in video

> Bachelor's thesis work from 2024. The code was cleaned up and documented in 2026; the method and the reported
> results are those of the original work.

TA-RandBox extends [RandBox](https://github.com/Yanghao-Wang/RandBox) (ICCV 2023), an open-world detector trained on
random region proposals, to video. Instead of drawing fresh proposals for every frame, it re-seeds part of them
around the previous frame's detections and keeps the rest random to keep finding new, possibly unknown, objects.

On TAO-LaSOT it matches the known-class accuracy of RandBox's 10,000-proposal inference with only 500 proposals,
**~9x faster per frame**, with the best unknown-object F1 and the fewest open-set errors among the methods compared.

Bachelor's thesis in Telecommunication Engineering, University of the Basque Country (UPV/EHU), carried out at
[TECNALIA](https://www.tecnalia.com) (2023-2024). **Management Solutions Award for the best final project in Digital
Transformation.**

<p align="center">
  <img src="docs/figures/tao_lasot_basketball.gif" width="49%" alt="Detections on a TAO-LaSOT video, outdoor basketball court">
  <img src="docs/figures/tao_lasot_indoor.gif" width="49%" alt="Detections on a TAO-LaSOT video, indoor basketball court">
</p>

## Method

RandBox replaces the region proposal network with random boxes, which removes the bias towards the training classes
and lets it find unknown objects. At inference, however, it defaults to a fixed grid of 10,000 boxes per image:
accurate, but ~10x slower than 500 random boxes, and wasteful on video, where consecutive frames are nearly
identical. TA-RandBox uses that temporal correlation. For every frame after the first one of a video:

1. Draw 500 random proposals.
2. Take the previous frame's detections with score > 0.15 (at most 70).
3. Spawn 3 copies of each, shifted by Gaussian noise (the expected motion between frames).
4. Let each copy replace the random proposal it overlaps the most.

At most 210 proposals follow already-detected objects and at least 290 keep exploring. Only inference changes: the
weights are RandBox's released task-1 model (20 known PASCAL VOC classes).

## Results on images: random proposals at inference

RandBox uses random proposals only for training; at inference it switches to 10,000 predefined boxes. Evaluating
it with just 500 random boxes, the same way it is trained, keeps most of its advantage over earlier open-world
detectors at about a tenth of the cost. This is what motivated the video method.

PASCAL VOC + COCO, task 1 (20 known classes, 60 unknown), 4,952 test images. Earlier methods as reported in the
RandBox paper; RandBox rows from our own runs of its released model.

| Method | K-mAP ↑ | U-R ↑ | WI ↓ | A-OSE ↓ | Inference time ↓ |
|---|---:|---:|---:|---:|---:|
| Faster R-CNN | 56.4 | - | 0.0699 | 13396 | |
| Deformable DETR | 60.3 | - | 0.0608 | 33270 | |
| ORE | 56.0 | 4.9 | 0.0621 | 10459 | |
| OST | 56.2 | - | 0.0417 | 4889 | |
| OW-DETR | 59.3 | 7.5 | 0.0521 | 7964 | |
| RandBox, 10,000 predefined boxes (default) | **61.74** | **10.56** | **0.024** | 4498 | 45 min |
| **RandBox, 500 random boxes** | 60.20 | 8.31 | 0.035 | **2938** | **4 min** |

With 500 random boxes RandBox still has better unknown recall, Wilderness Impact and open-set error than every
earlier method, and a K-mAP within 0.1 of Deformable DETR (which detects no unknowns), while inference is ~11x
faster (2 GPUs).

## Results on TAO-LaSOT

Task-1 model on the annotated frames of the TAO-LaSOT training videos, every frame processed in order. K-mAP: mAP@0.5
over the known classes; U-PRE / U-REC / U-F1: unknown-class precision, recall and F1; A-OSE: unknown objects
detected as known; WI: Wilderness Impact; Tq: seconds per frame.

| Method | K-mAP ↑ | U-PRE ↑ | U-REC ↑ | U-F1 ↑ | A-OSE ↓ | WI ↓ | Tq (s) ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| ORE | 38.58 | **19.21** | 6.62 | 9.58 | 5291 | 0.192 | 0.096 |
| UnSniffer | 41.94 | 6.55 | **46.95** | 11.50 | 6790 | 0.121 | 0.233 |
| RandBox, 10k proposals | **42.32** | 11.58 | 44.83 | 18.41 | 3541 | **0.071** | 0.597 |
| RandBox, 500 random | 39.42 | 16.65 | 38.87 | 23.31 | 3019 | 0.094 | **0.058** |
| **TA-RandBox (500)** | 42.26 | 17.66 | 38.87 | **24.29** | **2925** | 0.111 | 0.067 |

<details>
<summary>Other TAO subsets (ArgoVerse, BDD, Charades, YFCC100M)</summary>

These subsets annotate only 6-7 of the 20 known classes and far fewer frames, so known-class numbers are low and
noisy for every method; TA-RandBox's gains there are small or absent.

| Subset | Method | K-mAP | U-PRE | U-REC | U-F1 | A-OSE | WI | Tq (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ArgoVerse | ORE | 24.42 | 2.15 | 2.80 | 2.43 | 952 | 0.114 | 0.199 |
| | UnSniffer | 26.08 | 2.52 | 25.60 | 4.59 | 230 | 0.093 | 0.320 |
| | RandBox-10k | 26.10 | 2.49 | 48.00 | 4.73 | 176 | 0.017 | 0.727 |
| | RandBox-500 | 27.36 | 3.34 | 40.40 | 6.17 | 154 | 0.021 | 0.060 |
| | TA-RandBox | 25.50 | 3.49 | 41.60 | 6.44 | 161 | 0.022 | 0.057 |
| BDD | ORE | 9.08 | 7.32 | 6.45 | 6.86 | 2501 | 0.098 | 0.179 |
| | UnSniffer | 12.63 | 1.17 | 70.09 | 2.30 | 749 | 0.015 | 0.490 |
| | RandBox-10k | 10.04 | 2.99 | 53.96 | 5.67 | 513 | 0.012 | 0.609 |
| | RandBox-500 | 10.10 | 3.71 | 40.91 | 6.80 | 470 | 0.020 | 0.063 |
| | TA-RandBox | 9.25 | 3.92 | 42.08 | 7.17 | 785 | 0.020 | 0.045 |
| Charades | ORE | 15.64 | 3.40 | 0.24 | 0.45 | 2967 | 0.176 | 0.079 |
| | UnSniffer | 16.46 | 3.46 | 16.47 | 5.72 | 1389 | 0.057 | 0.282 |
| | RandBox-10k | 18.15 | 4.86 | 12.83 | 7.05 | 785 | 0.041 | 0.576 |
| | RandBox-500 | 12.37 | 5.11 | 8.74 | 6.45 | 509 | 0.044 | 0.058 |
| | TA-RandBox | 15.50 | 5.25 | 9.21 | 6.69 | 538 | 0.045 | 0.054 |
| YFCC100M | ORE | 36.69 | 7.33 | 1.00 | 1.76 | 3766 | 0.302 | 0.134 |
| | UnSniffer | 40.08 | 1.17 | 19.18 | 2.21 | 381 | 0.005 | 0.198 |
| | RandBox-10k | 41.23 | 2.12 | 19.09 | 3.82 | 254 | 0.002 | 0.558 |
| | RandBox-500 | 40.34 | 2.87 | 16.09 | 4.87 | 203 | 0.002 | 0.052 |
| | TA-RandBox | 40.98 | 2.85 | 16.36 | 4.85 | 215 | 0.002 | 0.052 |

</details>

Results as reported in the thesis and its follow-up presentation; the ORE and UnSniffer baselines come from the same
study and their code is not included. TAO annotates one frame per second and only some objects in it, so detections
of unannotated objects count as false positives: compare methods with each other rather than by absolute precision.

> The original evaluation loop ran on a modified Detectron2 on a TECNALIA server that is no longer available; the
> one in this repository (`randbox/video_inference.py`, `randbox/tao_owod_evaluation.py`) was rebuilt from the
> thesis and the original evaluator.

## Getting started

Python 3.10+, PyTorch 2.1+ with CUDA, Detectron2 0.6. A Colab notebook that installs everything and runs the
detector on a video is in [`notebooks/colab_quickstart.ipynb`](notebooks/colab_quickstart.ipynb).

```bash
pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git'
pip install -r requirements.txt
```

Download RandBox's task-1 weights from the [RandBox README](https://github.com/Yanghao-Wang/RandBox#models) to
`models/randbox_t1.pth`, and TAO from [taodataset.org](https://taodataset.org) into `datasets/tao/`.

```bash
# Detections on your own video
python tools/demo_video.py --input clip.mp4 --output out.mp4 --method ta-randbox MODEL.WEIGHTS models/randbox_t1.pth

# Evaluate the three inference methods on a TAO subset
bash scripts/eval_tao.sh LaSOT models/randbox_t1.pth
```

`tools/eval_tao.py --max-videos N` runs a quick check on the first videos only. Image OWOD training and evaluation
(RandBox's own pipeline): `scripts/train_owod.sh`, `scripts/eval_owod.sh`.

## Citation

```bibtex
@thesis{olivares2024tarandbox,
  author = {Olivares Rad, Aitzol},
  title  = {Nuevos modelos de detecci{\'o}n de objetos en flujos de v{\'i}deo basados en la aleatorizaci{\'o}n de
            propuestas de regiones de inter{\'e}s y Aprendizaje Profundo},
  school = {University of the Basque Country (UPV/EHU)},
  type   = {Bachelor's thesis},
  year   = {2024}
}

@inproceedings{wang2023randbox,
  author    = {Wang, Yanghao and Yue, Zhongqi and Hua, Xian-Sheng and Zhang, Hanwang},
  title     = {Random Boxes Are Open-world Object Detectors},
  booktitle = {ICCV},
  year      = {2023},
  pages     = {6233-6243}
}
```

## Attribution

Built on [RandBox](https://github.com/Yanghao-Wang/RandBox), [DiffusionDet](https://github.com/ShoufaChen/DiffusionDet),
[Sparse R-CNN](https://github.com/PeizeSun/SparseR-CNN) and [Detectron2](https://github.com/facebookresearch/detectron2).
The temporal proposals (`RandBox.temporal_proposals`, `MODEL.RandBox.TEMPORAL`) and the TAO evaluation are this
work; everything else is RandBox. See [NOTICE.md](NOTICE.md) for details and terms.
