"""
Run RandBox / TA-RandBox on a video (a file or a folder of frames) and write the detections as a video.

    python tools/demo_video.py --config-file configs/t1.yaml --input clip.mp4 --output out.mp4 \
        --method ta-randbox MODEL.WEIGHTS models/randbox_t1.pth

Known classes are drawn with their name; everything the model calls "unknown" is drawn as `unknown`.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.data import transforms as T
from detectron2.modeling import build_model
from detectron2.utils.visualizer import Visualizer

from randbox import add_RandBox_config
from randbox.tao_owod_evaluation import OWOD_CLASS_NAMES
from randbox.util.model_ema import add_model_ema_configs
from tools.eval_tao import METHODS


def frames_of(path):
    if os.path.isdir(path):
        for name in sorted(os.listdir(path)):
            if name.lower().endswith((".jpg", ".jpeg", ".png")):
                yield cv2.imread(os.path.join(path, name))
        return
    cap = cv2.VideoCapture(path)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield frame
    cap.release()


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config-file", default="configs/t1.yaml")
    parser.add_argument("--input", required=True, help="a video file or a folder of frames")
    parser.add_argument("--output", required=True, help="output .mp4")
    parser.add_argument("--method", choices=sorted(METHODS), default="ta-randbox")
    parser.add_argument("--score", type=float, default=0.3, help="minimum score to draw")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("opts", nargs=argparse.REMAINDER, help="config overrides, e.g. MODEL.WEIGHTS path")
    args = parser.parse_args()

    cfg = get_cfg()
    add_RandBox_config(cfg)
    add_model_ema_configs(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(METHODS[args.method])
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    model = build_model(cfg)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    model.eval()
    resize = T.ResizeShortestEdge([cfg.INPUT.MIN_SIZE_TEST] * 2, cfg.INPUT.MAX_SIZE_TEST)
    metadata = MetadataCatalog.get("owod_t1_demo").set(thing_classes=list(OWOD_CLASS_NAMES))
    temporal = cfg.MODEL.RandBox.TEMPORAL.ENABLED

    writer, prev_boxes = None, None
    for frame in frames_of(args.input):
        h, w = frame.shape[:2]
        image = resize.get_transform(frame).apply_image(frame)
        inputs = [{"image": torch.as_tensor(image.astype("float32").transpose(2, 0, 1)), "height": h, "width": w}]
        outputs, tracked = model(inputs, prev_boxes=prev_boxes, return_tracked=True)
        prev_boxes = tracked if temporal else None

        instances = outputs[0]["instances"].to("cpu")
        instances = instances[instances.scores >= args.score]
        drawn = Visualizer(frame[:, :, ::-1], metadata).draw_instance_predictions(instances).get_image()[:, :, ::-1]
        if writer is None:
            writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
        writer.write(drawn)
    if writer is not None:
        writer.release()
    print(f"-> {args.output}")


if __name__ == "__main__":
    main()
