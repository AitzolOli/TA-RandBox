"""
Evaluate RandBox / TA-RandBox on TAO video frames (open-world metrics + seconds per frame).

    python tools/eval_tao.py --config-file configs/t1.yaml --frames datasets/tao/frames_lasot.pkl \
        --method ta-randbox MODEL.WEIGHTS models/randbox_t1.pth OUTPUT_DIR output/lasot_ta

Methods (same task-1 RandBox weights, only inference changes):
  randbox-10k  RandBox's default inference: 10,000 boxes from a fixed grid (20 passes of 500).
  randbox-500  500 random boxes per frame, as in training.
  ta-randbox   500 boxes per frame, up to 210 of them propagated from the previous frame's detections
               (MODEL.RandBox.TEMPORAL), the rest random.

Writes <OUTPUT_DIR>/results.json with K-mAP, U-PRE, U-REC, U-F1, A-OSE, WI and Tq (model seconds per frame).
"""
import json
import logging
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader
from detectron2.engine import default_argument_parser, default_setup
from detectron2.modeling import build_model

from randbox import RandBoxDatasetMapper, add_RandBox_config
from randbox.tao_owod_evaluation import TAOOpenWorldEvaluator
from randbox.util.model_ema import add_model_ema_configs
from randbox.video_inference import inference_on_video_dataset, video_of

METHODS = {
    "randbox-10k": ["MODEL.RandBox.SAMPLING_METHOD", "Random_", "MODEL.RandBox.TEMPORAL.ENABLED", "False"],
    "randbox-500": ["MODEL.RandBox.SAMPLING_METHOD", "Random", "MODEL.RandBox.TEMPORAL.ENABLED", "False"],
    "ta-randbox": ["MODEL.RandBox.SAMPLING_METHOD", "Random", "MODEL.RandBox.TEMPORAL.ENABLED", "True"],
}


def setup(args):
    cfg = get_cfg()
    add_RandBox_config(cfg)
    add_model_ema_configs(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(METHODS[args.method])
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def load_frames(path, max_videos):
    with open(path, "rb") as f:
        frames = pickle.load(f)
    if max_videos:
        keep, videos = [], []
        for record in frames:
            video = video_of(record["file_name"])
            if video not in videos:
                if len(videos) == max_videos:
                    break
                videos.append(video)
            keep.append(record)
        frames = keep
    return frames


def main(args):
    cfg = setup(args)
    logger = logging.getLogger("detectron2")

    frames = load_frames(args.frames, args.max_videos)
    logger.info("%d frames, %d annotated", len(frames), sum(r["annotations"] is not None for r in frames))

    model = build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=False)
    model.eval()

    loader = build_detection_test_loader(frames, mapper=RandBoxDatasetMapper(cfg, is_train=False), batch_size=1,
                                         num_workers=cfg.DATALOADER.NUM_WORKERS)
    evaluator = TAOOpenWorldEvaluator(frames, cfg)
    results = inference_on_video_dataset(model, loader, frames, evaluator,
                                         temporal=cfg.MODEL.RandBox.TEMPORAL.ENABLED)
    results["method"] = args.method
    results["frames_file"] = args.frames

    out = os.path.join(cfg.OUTPUT_DIR, "results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    owod, timing = results["owod"], results["timing"]
    print(f"{args.method}: K-mAP {owod['K-mAP']:.2f} | U-PRE {owod['U-PRE']:.2f} | U-REC {owod['U-REC']:.2f} | "
          f"U-F1 {owod['U-F1']:.2f} | A-OSE {owod['A-OSE']:.0f} | WI {owod['WI']:.3f} | "
          f"Tq {timing['model_s_per_frame']:.3f} s/frame -> {out}")
    return results


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--frames", required=True, help="frame index from tools/prepare_tao.py")
    parser.add_argument("--method", choices=sorted(METHODS), required=True)
    parser.add_argument("--max-videos", type=int, default=0, help="only the first N videos (quick checks)")
    args = parser.parse_args()
    assert args.num_gpus == 1, "video inference runs frame by frame on one GPU"
    torch.backends.cudnn.benchmark = False
    main(args)
