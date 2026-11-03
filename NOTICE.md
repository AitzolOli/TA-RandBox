# Notice: origin of the code and terms of use

This repository extends third-party research code. What comes from where:

| Files | Origin | Terms |
|---|---|---|
| `randbox/` (except the parts listed below), `configs/`, `split/`, `tools/train_net.py`, `scripts/train_owod.sh` | [RandBox](https://github.com/Yanghao-Wang/RandBox), Wang et al., ICCV 2023 | The RandBox repository does not state a license. RandBox is built on DiffusionDet. |
| Diffusion-style sampling and head in `randbox/detector.py`, `randbox/head.py`, `randbox/loss.py` | [DiffusionDet](https://github.com/ShoufaChen/DiffusionDet), Chen et al. | CC BY-NC 4.0 (non-commercial) |
| Dynamic head, mapper and training script structure | [Sparse R-CNN](https://github.com/PeizeSun/SparseR-CNN) | MIT |
| `randbox/swintransformer.py` | [Swin Transformer](https://github.com/microsoft/Swin-Transformer) | MIT |
| Pascal VOC AP code in `randbox/*evaluation.py` | [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn) (B. Hariharan), via ORE / OW-DETR | MIT |
| Detection framework | [Detectron2](https://github.com/facebookresearch/detectron2) (dependency, not vendored) | Apache 2.0 |
| Temporal proposals (`RandBox.temporal_proposals` and the `MODEL.RandBox.TEMPORAL` options), `randbox/video_inference.py`, `randbox/tao_owod_evaluation.py` (TAO adaptation), `tools/prepare_tao.py`, `tools/eval_tao.py`, `tools/demo_video.py`, `scripts/eval_tao.sh`, `notebooks/`, `docs/` | Aitzol Olivares Rad | see below |

Because the code it builds on includes CC BY-NC 4.0 material and code with no stated license, this repository is
shared **for research and non-commercial use only**, with attribution to the works above. The original
contributions of this repository are offered under the same CC BY-NC 4.0 terms.

Media in `docs/figures/` show frames of TAO / LaSOT videos ([taodataset.org](https://taodataset.org)), used here for
illustration under those datasets' terms.
