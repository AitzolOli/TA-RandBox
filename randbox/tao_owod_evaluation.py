# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Open-world evaluation on TAO video frames (Pascal-VOC style AP at IoU 0.5).

Adapted from RandBox's `pascal_voc_evaluation.py` (itself from ORE / OW-DETR) to read the ground truth from the
in-memory frame records written by `tools/prepare_tao.py` instead of VOC XML files, and to score only the
annotated frames of a video (TAO annotates roughly one frame per second).

Metrics (IoU 0.5), as reported in the thesis and the follow-up talk:
  * K-mAP  -- mean AP over the known classes that appear in the evaluated frames (classes with no ground truth
              give NaN and are left out: TAO subsets contain only 6-14 of the 20 known classes);
  * U-PRE / U-REC / U-F1 -- precision, recall and F1 of the "unknown" class;
  * WI     -- Wilderness Impact at recall 0.8;
  * A-OSE  -- Absolute Open-Set Error: unknown objects detected as one of the known classes.
"""
import logging
import os
import tempfile
from collections import OrderedDict, defaultdict

import numpy as np
import torch
from detectron2.evaluation.evaluator import DatasetEvaluator
from detectron2.utils import comm

# Class order of the OWOD splits (task 1 = the first 20, the 20 PASCAL VOC classes); index 80 is "unknown".
OWOD_CLASS_NAMES = (
    "airplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
    "dining table", "dog", "horse", "motorcycle", "person", "potted plant", "sheep", "couch", "train", "tv",
    "truck", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "microwave", "oven", "toaster", "sink",
    "refrigerator", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "bed", "toilet", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "unknown",
)
UNKNOWN_ID = len(OWOD_CLASS_NAMES) - 1  # 80

# Detections under this score are not scored (same threshold as the temporal proposals).
SCORE_THRESHOLD = 0.15
WI_RECALL_LEVEL = 0.8


class TAOOpenWorldEvaluator(DatasetEvaluator):
    """
    Pascal-VOC style open-world evaluator over TAO frame records.

    `process(inputs, outputs, record)` takes the frame's record as well: predictions are kept only for annotated
    frames (`record["annotations"] is not None`), while unannotated frames are still run by the model so that the
    temporal proposals see every frame.
    """

    def __init__(self, dataset_dicts, cfg):
        self._logger = logging.getLogger(__name__)
        self._cpu_device = torch.device("cpu")
        self._class_names = list(OWOD_CLASS_NAMES)
        self._annotated = _annotated_frames(dataset_dicts)

        self.prev_intro_cls = cfg.TEST.PREV_INTRODUCED_CLS
        self.curr_intro_cls = cfg.TEST.CUR_INTRODUCED_CLS
        self.total_num_class = cfg.MODEL.RandBox.NUM_CLASSES
        self.unknown_class_index = self.total_num_class - 1
        self.num_seen_classes = self.prev_intro_cls + self.curr_intro_cls
        self.known_classes = self._class_names[:self.num_seen_classes]

    def reset(self):
        self._predictions = defaultdict(list)  # class id -> list of "image_id score x1 y1 x2 y2"

    def process(self, inputs, outputs, record):
        if record["annotations"] is None:
            return
        for input, output in zip(inputs, outputs):
            image_id = input["image_id"]
            instances = output["instances"].to(self._cpu_device)
            boxes = instances.pred_boxes.tensor.numpy()
            scores = instances.scores.tolist()
            classes = instances.pred_classes.tolist()
            for box, score, cls in zip(boxes, scores, classes):
                if score < SCORE_THRESHOLD or cls == -100:
                    continue
                xmin, ymin, xmax, ymax = box
                # The inverse of data loading logic in `datasets/pascal_voc.py`
                xmin += 1
                ymin += 1
                self._predictions[cls].append(f"{image_id} {score:.3f} {xmin:.1f} {ymin:.1f} {xmax:.1f} {ymax:.1f}")

    # -- Wilderness Impact and unknown precision at fixed recall levels (from ORE) ------------------------------

    def compute_avg_precision_at_many_recall_level_for_unk(self, precisions, recalls):
        return {r / 10: self.compute_avg_precision_at_a_recall_level_for_unk(precisions, recalls, r / 10)
                for r in range(1, 10)}

    def compute_avg_precision_at_a_recall_level_for_unk(self, precisions, recalls, recall_level=0.5):
        precs = {}
        for iou, recall in recalls.items():
            prec = []
            for cls_id, rec in enumerate(recall):
                if cls_id == self.unknown_class_index and len(rec) > 0:
                    p = precisions[iou][cls_id][min(range(len(rec)), key=lambda i: abs(rec[i] - recall_level))]
                    prec.append(p)
            precs[iou] = np.mean(prec) if prec else 0
        return precs

    def compute_WI_at_many_recall_level(self, recalls, tp_plus_fp_cs, fp_os):
        return {r / 10: self.compute_WI_at_a_recall_level(recalls, tp_plus_fp_cs, fp_os, r / 10) for r in range(1, 10)}

    def compute_WI_at_a_recall_level(self, recalls, tp_plus_fp_cs, fp_os, recall_level=0.5):
        wi_at_iou = {}
        for iou, recall in recalls.items():
            tp_plus_fps, fps = [], []
            for cls_id, rec in enumerate(recall):
                if cls_id in range(self.num_seen_classes) and len(rec) > 0:
                    index = min(range(len(rec)), key=lambda i: abs(rec[i] - recall_level))
                    tp_plus_fps.append(tp_plus_fp_cs[iou][cls_id][index])
                    fps.append(fp_os[iou][cls_id][index])
            wi_at_iou[iou] = np.mean(fps) / np.mean(tp_plus_fps) if tp_plus_fps else 0
        return wi_at_iou

    def evaluate(self):
        all_predictions = comm.gather(self._predictions, dst=0)
        if not comm.is_main_process():
            return
        predictions = defaultdict(list)
        for predictions_per_rank in all_predictions:
            for clsid, lines in predictions_per_rank.items():
                predictions[clsid].extend(lines)
        del all_predictions

        aps, recs, precs = defaultdict(list), defaultdict(list), defaultdict(list)
        all_recs, all_precs = defaultdict(list), defaultdict(list)
        unk_det_as_knowns, num_unks = defaultdict(list), defaultdict(list)
        tp_plus_fp_cs, fp_os = defaultdict(list), defaultdict(list)
        thresh = 50

        with tempfile.TemporaryDirectory(prefix="tao_owod_eval_") as dirname:
            res_file_template = os.path.join(dirname, "{}.txt")
            for cls_id, cls_name in enumerate(self._class_names):
                lines = predictions.get(cls_id, [""])
                self._logger.info("%s has %d predictions.", cls_name, len(lines))
                with open(res_file_template.format(cls_name), "w") as f:
                    f.write("\n".join(lines))
                rec, prec, ap, unk_det_as_known, num_unk, tp_plus_fp_closed_set, fp_open_set = voc_eval(
                    res_file_template, self._annotated, cls_id, cls_name, ovthresh=thresh / 100.0)
                aps[thresh].append(ap * 100)
                unk_det_as_knowns[thresh].append(unk_det_as_known)
                num_unks[thresh].append(num_unk)
                all_precs[thresh].append(prec)
                all_recs[thresh].append(rec)
                tp_plus_fp_cs[thresh].append(tp_plus_fp_closed_set)
                fp_os[thresh].append(fp_open_set)
                try:
                    recs[thresh].append(rec[-1] * 100)
                    precs[thresh].append(prec[-1] * 100)
                except (IndexError, TypeError):
                    recs[thresh].append(0)
                    precs[thresh].append(0)

        wi = self.compute_WI_at_many_recall_level(all_recs, tp_plus_fp_cs, fp_os)
        avg_precision_unk = self.compute_avg_precision_at_many_recall_level_for_unk(all_precs, all_recs)
        a_ose = float(np.sum(unk_det_as_knowns[thresh]))
        self._logger.info("Wilderness Impact: %s", wi)
        self._logger.info("avg_precision: %s", avg_precision_unk)
        self._logger.info("Absolute OSE (total_num_unk_det_as_known): %s", a_ose)
        self._logger.info("total_num_unk %s", num_unks[thresh][-1])
        self._logger.info("AP50: %s", ["%.1f" % x for x in aps[thresh]])
        self._logger.info("Precisions50: %s", ["%.1f" % x for x in precs[thresh]])
        self._logger.info("Recall50: %s", ["%.1f" % x for x in recs[thresh]])

        # Known classes absent from the evaluated frames have no ground truth (AP NaN): leave them out.
        seen = slice(0, self.num_seen_classes)
        keep = [i for i, ap in enumerate(aps[thresh][seen]) if not np.isnan(ap)]
        known_ap50 = float(np.mean([aps[thresh][i] for i in keep]))
        known_prec50 = float(np.mean([precs[thresh][i] for i in keep]))
        known_rec50 = float(np.nanmean(recs[thresh][seen]))
        u_pre, u_rec = float(precs[thresh][-1]), float(recs[thresh][-1])
        u_f1 = 2 * u_pre * u_rec / (u_pre + u_rec) if u_pre + u_rec > 0 else 0.0

        self._logger.info("Known AP50: %s", known_ap50)
        self._logger.info("Known Precisions50: %s", known_prec50)
        self._logger.info("Known Recall50: %s", known_rec50)
        self._logger.info("Unknown AP50: %s", aps[thresh][-1])
        self._logger.info("Unknown Precisions50: %s", u_pre)
        self._logger.info("Unknown Recall50: %s", u_rec)

        ret = OrderedDict()
        ret["owod"] = {
            "K-mAP": known_ap50,
            "U-PRE": u_pre,
            "U-REC": u_rec,
            "U-F1": u_f1,
            "A-OSE": a_ose,
            "WI": float(wi[WI_RECALL_LEVEL][thresh]),
            "known_classes_evaluated": len(keep),
        }
        return ret


def _annotated_frames(dataset_dicts):
    """The annotated frames as {"file_name", "image_id", "bboxes", "category_ids"} (boxes are x1, y1, x2, y2)."""
    frames = []
    for record in dataset_dicts:
        if record["annotations"] is None:
            continue
        frames.append({
            "file_name": record["file_name"],
            "image_id": record["image_id"],
            "bboxes": [ann["bbox"] for ann in record["annotations"]],
            "category_ids": [ann["category_id"] for ann in record["annotations"]],
        })
    return frames


def voc_ap(rec, prec, use_07_metric=False):
    """Compute VOC AP given precision and recall (VOC07 11-point if use_07_metric, else the area under the curve)."""
    if use_07_metric:
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            p = 0 if np.sum(rec >= t) == 0 else np.max(prec[rec >= t])
            ap = ap + p / 11.0
        return ap
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    i = np.where(mrec[1:] != mrec[:-1])[0]
    return np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])


def _ground_truth(frames, class_id):
    """Per frame (keyed by file name without extension): the boxes of one class, and how many there are."""
    class_recs, npos = {}, 0
    for frame in frames:
        mask = np.array(frame["category_ids"]) == class_id
        bbox = np.array([b for b, m in zip(frame["bboxes"], mask) if m])
        n = int(mask.sum())
        npos += n
        class_recs[frame["file_name"].replace(".jpg", "")] = {
            "bbox": bbox, "difficult": np.zeros(n, dtype=bool), "det": [False] * n}
    return class_recs, npos


def _overlaps(BBGT, bb):
    ixmin = np.maximum(BBGT[:, 0], bb[0])
    iymin = np.maximum(BBGT[:, 1], bb[1])
    ixmax = np.minimum(BBGT[:, 2], bb[2])
    iymax = np.minimum(BBGT[:, 3], bb[3])
    iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
    ih = np.maximum(iymax - iymin + 1.0, 0.0)
    inters = iw * ih
    uni = ((bb[2] - bb[0] + 1.0) * (bb[3] - bb[1] + 1.0)
           + (BBGT[:, 2] - BBGT[:, 0] + 1.0) * (BBGT[:, 3] - BBGT[:, 1] + 1.0) - inters)
    return inters / uni


def voc_eval(detpath, frames, class_id, classname, ovthresh=0.5, use_07_metric=False):
    """
    Pascal VOC evaluation of one class over the annotated frames (modified from
    https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/voc_eval.py, MIT License,
    by Bharath Hariharan), plus the open-set counts of ORE.

    Returns:
        rec, prec, ap, the number of unknown objects detected as this class (A-OSE term), the number of
        unknown ground-truth objects, and the cumulative TP+FP and open-set FP curves (for WI).
    """
    class_recs, npos = _ground_truth(frames, class_id)

    with open(detpath.format(classname), "r") as f:
        lines = f.readlines()
    splitlines = [x.strip().split(" ") for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines]).reshape(-1, 4)

    sorted_ind = np.argsort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    frame_of = {frame["image_id"]: frame["file_name"].split(".")[0] for frame in frames}

    for d in range(nd):
        R = class_recs[frame_of[int(image_ids[d])]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R["bbox"].astype(float)
        if BBGT.size > 0:
            overlaps = _overlaps(BBGT, bb)
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)
        if ovmax > ovthresh:
            if not R["difficult"][jmax]:
                if not R["det"][jmax]:
                    tp[d] = 1.0
                    R["det"][jmax] = 1
                else:
                    fp[d] = 1.0
        else:
            fp[d] = 1.0

    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)  # NaN when the class has no ground truth in these frames
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    # Absolute Open-Set Error and Wilderness Impact: detections of this (known) class lying on an unknown object.
    unknown_class_recs, n_unk = _ground_truth(frames, UNKNOWN_ID)
    if classname == "unknown":
        return rec, prec, ap, 0, n_unk, None, None

    is_unk = np.zeros(nd)
    for d in range(nd):
        R = unknown_class_recs[frame_of[int(image_ids[d])]]
        bb = BB[d, :].astype(float)
        BBGT = R["bbox"].astype(float)
        if BBGT.size > 0 and np.max(_overlaps(BBGT, bb)) > ovthresh:
            is_unk[d] = 1.0

    return rec, prec, ap, np.sum(is_unk), n_unk, tp + fp, np.cumsum(is_unk)
