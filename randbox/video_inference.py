"""
Frame-by-frame inference over video datasets (TAO), carrying temporal proposals between frames.

This replaces the modified copies of detectron2's `engine/defaults.py` and `evaluation/evaluator.py` that the
original experiments used: the same loop as `detectron2.evaluation.inference_on_dataset`, except that

  * frames are processed one at a time and in video order (the dataset is sorted by video, then frame);
  * with temporal proposals enabled, the boxes the model keeps for frame t are passed in as proposals for
    frame t + 1, and reset to None on the first frame of every video;
  * the evaluator also receives the frame's ground-truth record, so it only scores annotated frames (TAO
    annotates about one frame in 30, but every frame is run so that the temporal chain is unbroken).
"""
import datetime
import logging
import os
import time
from collections import OrderedDict

import torch
from detectron2.utils.logger import log_every_n_seconds

logger = logging.getLogger(__name__)


def video_of(file_name: str) -> str:
    """A frame's video: the folder it lives in (TAO: frames/<split>/<source>/<video>/<frame>.jpg)."""
    return os.path.dirname(file_name)


@torch.no_grad()
def inference_on_video_dataset(model, data_loader, dataset_dicts, evaluator, temporal: bool):
    """
    Args:
        model: a RandBox model in eval mode.
        data_loader: a test loader over `dataset_dicts`, batch size 1, same order.
        dataset_dicts (list[dict]): the frames' records (`tools/prepare_tao.py`), `annotations` None when the
            frame is not annotated.
        evaluator: `TAOOpenWorldEvaluator`.
        temporal (bool): feed each frame's tracked boxes to the next frame of the same video.

    Returns:
        dict: the evaluator's results, plus "timing" with the model's seconds per frame (Tq).
    """
    assert len(data_loader) == len(dataset_dicts), "the loader must iterate the dataset once, one frame per batch"
    evaluator.reset()
    total = len(data_loader)
    logger.info("Start inference on %d frames (temporal proposals: %s)", total, "on" if temporal else "off")

    prev_boxes, prev_video = None, None
    model_seconds = 0.0
    start = time.perf_counter()
    for idx, (inputs, record) in enumerate(zip(data_loader, dataset_dicts)):
        assert len(inputs) == 1, "video inference runs one frame per batch"
        assert inputs[0]["file_name"] == record["file_name"], (inputs[0]["file_name"], record["file_name"])

        video = video_of(record["file_name"])
        if video != prev_video:
            prev_boxes, prev_video = None, video  # first frame of a video: random proposals only

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        outputs, tracked = model(inputs, prev_boxes=prev_boxes if temporal else None, return_tracked=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_seconds += time.perf_counter() - t0

        prev_boxes = tracked if temporal else None
        evaluator.process(inputs, outputs, record)

        done = idx + 1
        per_frame = model_seconds / done
        eta = datetime.timedelta(seconds=int((time.perf_counter() - start) / done * (total - done)))
        log_every_n_seconds(logging.INFO, f"Inference done {done}/{total}. Model: {per_frame:.4f} s/frame. ETA={eta}",
                            n=30, name=__name__)

    wall = time.perf_counter() - start
    logger.info("Total inference time: %s (%.4f s/frame of model time, %d frames)",
                datetime.timedelta(seconds=int(wall)), model_seconds / max(total, 1), total)

    results = evaluator.evaluate() or OrderedDict()
    results["timing"] = {"frames": total, "model_s_per_frame": model_seconds / max(total, 1), "wall_s": wall}
    return results
