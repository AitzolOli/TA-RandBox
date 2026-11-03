#!/bin/bash
# RandBox evaluation on the image OWOD splits (PASCAL VOC + COCO), RandBox's default 10,000-box inference.
# models/randbox_t1.pth is RandBox's released task-1 checkpoint (see README).

python tools/train_net.py --num-gpus 1 --task t1 --config-file configs/t1.yaml --eval-only MODEL.WEIGHTS models/randbox_t1.pth

# 500 random proposals instead of the 10,000-box grid (thesis Table 3):
# python tools/train_net.py --num-gpus 1 --task t1 --config-file configs/t1.yaml --eval-only MODEL.WEIGHTS models/randbox_t1.pth MODEL.RandBox.SAMPLING_METHOD Random
