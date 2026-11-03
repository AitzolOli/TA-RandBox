"""
Build the frame index used to evaluate on TAO: every frame of every video, in video order, with the OWOD labels
on the annotated ones.

    python tools/prepare_tao.py --tao-root datasets/tao --subsets LaSOT --out datasets/tao/frames_lasot.pkl

TAO (https://taodataset.org) annotates about one frame per second (one in 30), but the temporal proposals need the
whole video, so the index lists every frame on disk: annotated frames carry their boxes, the others
`annotations: None`. HACS and AVA are left out (their frames are not distributed with TAO).

Labels: TAO's LVIS category ids are mapped to COCO ids and then to the OWOD class order (task 1 = the 20 PASCAL
VOC classes, ids 0-19); every other object becomes "unknown" (id 80). Boxes become (x1, y1, x2, y2).

`--label-mapping thesis` (the default) reproduces the mapping the reported results used. In it, a TAO id with no
COCO equivalent is looked up in the COCO -> OWOD table by its raw value, so the few non-COCO TAO categories whose
id is below 80 end up as a known class instead of "unknown". `--label-mapping fixed` maps them to "unknown".
"""
import argparse
import os
import pickle
from collections import defaultdict

import cv2
from detectron2.data.datasets.coco import load_coco_json

EXCLUDED_SOURCES = ("HACS", "AVA")

# TAO (LVIS v0.5) category id -> COCO contiguous id (0-79).
TAO_TO_COCO = {91: 13, 58: 34, 621: 33, 747: 49, 118: 8, 221: 51, 95: 1, 126: 73, 1122: 79, 729: 27, 926: 48,
               1117: 61, 1038: 11, 1215: 40, 276: 74, 78: 21, 1162: 75, 699: 68, 185: 55, 13: 47, 79: 59, 982: 30,
               371: 60, 896: 65, 99: 14, 642: 63, 1135: 6, 717: 64, 829: 53, 1115: 70, 235: 67, 805: 0, 41: 32,
               452: 10, 1155: 25, 1144: 7, 625: 43, 60: 35, 502: 23, 4: 4, 779: 12, 1001: 57, 1099: 38, 34: 24,
               45: 46, 139: 45, 980: 36, 133: 39, 382: 16, 480: 29, 154: 50, 429: 20, 211: 2, 392: 54, 36: 28,
               347: 41, 544: 78, 1057: 37, 1132: 9, 1097: 62, 1018: 44, 579: 17, 714: 3, 1229: 22, 229: 15,
               1091: 77, 35: 26, 979: 71, 299: 66, 174: 5, 475: 42, 237: 56, 428: 72, 937: 76, 961: 18, 852: 58,
               993: 31, 81: 19}
# COCO contiguous id -> OWOD class id (see randbox/tao_owod_evaluation.py for the order).
COCO_TO_OWOD = {0: 14, 1: 1, 2: 6, 3: 13, 4: 0, 5: 5, 6: 18, 7: 20, 8: 3, 9: 21, 10: 22, 11: 23, 12: 24, 13: 25,
                14: 2, 15: 7, 16: 11, 17: 12, 18: 16, 19: 9, 20: 26, 21: 27, 22: 28, 23: 29, 24: 30, 25: 31, 26: 32,
                27: 33, 28: 34, 29: 40, 30: 41, 31: 42, 32: 43, 33: 44, 34: 45, 35: 46, 36: 47, 37: 48, 38: 49,
                39: 4, 40: 74, 41: 75, 42: 76, 43: 77, 44: 78, 45: 79, 46: 50, 47: 51, 48: 52, 49: 53, 50: 54,
                51: 55, 52: 56, 53: 57, 54: 58, 55: 59, 56: 8, 57: 17, 58: 15, 59: 60, 60: 10, 61: 61, 62: 19,
                63: 62, 64: 63, 65: 64, 66: 65, 67: 66, 68: 35, 69: 36, 70: 37, 71: 38, 72: 39, 73: 67, 74: 68,
                75: 69, 76: 70, 77: 71, 78: 72, 79: 73}
NUM_KNOWN = 20
UNKNOWN_ID = 80


def owod_label(tao_id: int, mapping: str) -> int:
    """OWOD label of a TAO category: its known class (0-19) or UNKNOWN_ID."""
    if tao_id in TAO_TO_COCO:
        coco_id = TAO_TO_COCO[tao_id]
    elif mapping == "thesis":
        coco_id = tao_id  # see the module docstring
    else:
        return UNKNOWN_ID
    owod_id = COCO_TO_OWOD.get(coco_id, UNKNOWN_ID)
    return owod_id if owod_id < NUM_KNOWN else UNKNOWN_ID


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tao-root", default="datasets/tao", help="holds frames/ and annotations/")
    parser.add_argument("--annotations", default="annotations/train.json", help="relative to --tao-root")
    parser.add_argument("--split", default="train", help="frames/<split>/<source>/<video>/")
    parser.add_argument("--subsets", default="", help="comma-separated sources (e.g. LaSOT); empty = all")
    parser.add_argument("--label-mapping", choices=("thesis", "fixed"), default="thesis")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    frames_root = os.path.join(args.tao_root, "frames")
    split_root = os.path.join(frames_root, args.split)
    wanted = {s.strip() for s in args.subsets.split(",") if s.strip()}

    # Annotated frames, keyed by path. load_coco_json joins image_root with TAO's "train/<source>/..." names.
    annotated = {}
    for record in load_coco_json(os.path.join(args.tao_root, args.annotations), frames_root):
        file_name = record["file_name"]
        if file_name.endswith(".jpeg"):  # the frames on disk are .jpg
            file_name = file_name[:-len(".jpeg")] + ".jpg"
        annos = []
        for ann in record.get("annotations", []):
            x, y, w, h = ann["bbox"]
            annos.append({"bbox": [x, y, x + w, y + h], "category_id": owod_label(ann["category_id"], args.label_mapping)})
        annotated[os.path.normpath(file_name)] = {"height": record["height"], "width": record["width"], "annotations": annos}

    records, image_id = [], 0
    counts = defaultdict(lambda: [0, 0])
    for source in sorted(os.listdir(split_root)):
        if source in EXCLUDED_SOURCES or (wanted and source not in wanted):
            continue
        for video in sorted(os.listdir(os.path.join(split_root, source))):
            video_dir = os.path.join(split_root, source, video)
            names = sorted(f for f in os.listdir(video_dir) if f.endswith((".jpg", ".jpeg")))
            size = next((annotated[os.path.normpath(os.path.join(video_dir, n))] for n in names
                         if os.path.normpath(os.path.join(video_dir, n)) in annotated), None)
            if size is None:  # a video with no annotated frame: read its size once
                h, w = cv2.imread(os.path.join(video_dir, names[0])).shape[:2]
                size = {"height": h, "width": w}
            for name in names:
                path = os.path.join(video_dir, name)
                ann = annotated.get(os.path.normpath(path))
                records.append({
                    "file_name": path,
                    "height": size["height"],
                    "width": size["width"],
                    "image_id": image_id,
                    "annotations": ann["annotations"] if ann else None,
                })
                image_id += 1
                counts[source][0] += 1
                counts[source][1] += ann is not None

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(records, f)
    for source, (n, n_ann) in sorted(counts.items()):
        print(f"{source}: {n} frames, {n_ann} annotated")
    print(f"{len(records)} frames -> {args.out}")


if __name__ == "__main__":
    main()
