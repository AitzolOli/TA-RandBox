from detectron2.config import CfgNode as CN


def add_RandBox_config(cfg):
    """
    Add config for RandBox
    """
    cfg.MODEL.RandBox = CN()
    cfg.MODEL.RandBox.NUM_CLASSES = 81
    cfg.MODEL.RandBox.NUM_PROPOSALS = 300

    # RCNN Head.
    cfg.MODEL.RandBox.NHEADS = 8
    cfg.MODEL.RandBox.DROPOUT = 0.0
    cfg.MODEL.RandBox.DIM_FEEDFORWARD = 2048
    cfg.MODEL.RandBox.ACTIVATION = 'relu'
    cfg.MODEL.RandBox.HIDDEN_DIM = 256
    cfg.MODEL.RandBox.NUM_CLS = 1
    cfg.MODEL.RandBox.NUM_REG = 3
    cfg.MODEL.RandBox.NUM_HEADS = 6

    # Dynamic Conv.
    cfg.MODEL.RandBox.NUM_DYNAMIC = 2
    cfg.MODEL.RandBox.DIM_DYNAMIC = 64

    # Loss.
    cfg.MODEL.RandBox.CLASS_WEIGHT = 2.0
    cfg.MODEL.RandBox.NC_WEIGHT = 0.1
    cfg.MODEL.RandBox.GIOU_WEIGHT = 2.0
    cfg.MODEL.RandBox.L1_WEIGHT = 5.0
    cfg.MODEL.RandBox.DEEP_SUPERVISION = True
    cfg.MODEL.RandBox.NO_OBJECT_WEIGHT = 0.1

    # Focal Loss.
    cfg.MODEL.RandBox.USE_FOCAL = True
    cfg.MODEL.RandBox.USE_FED_LOSS = False
    cfg.MODEL.RandBox.ALPHA = 0.25
    cfg.MODEL.RandBox.GAMMA = 2.0
    cfg.MODEL.RandBox.PRIOR_PROB = 0.01

    # Dynamic K
    cfg.MODEL.RandBox.OTA_K = 5
    cfg.MODEL.RandBox.FORWARD_K = 10
    
    
    # WARM_UP
    cfg.MODEL.RandBox.SIMILAR_THRESHOLD = 2.0
    # cfg.MODEL.RandBox.CHANGE_START = 500
    cfg.MODEL.RandBox.CHANGE_START = 0

    
    
    # RandBox
    cfg.MODEL.RandBox.SNR_SCALE = 2.0
    cfg.MODEL.RandBox.SAMPLE_STEP = 1

    # Inference
    cfg.MODEL.RandBox.USE_NMS = True
    cfg.MODEL.RandBox.M_STEP = 20
    # 'Random_' (upstream default): M_STEP x NUM_PROPOSALS boxes from a fixed 10^4 grid (10,000 boxes for 20 x 500).
    # 'Random': NUM_PROPOSALS boxes drawn at random, the same way the model is trained.
    cfg.MODEL.RandBox.SAMPLING_METHOD = 'Random_'

    # Temporal proposals for video streams (this repository's contribution).
    # On every frame after the first one of a video, each confident detection of the previous frame spawns
    # NUM_PER_BOX jittered copies (Gaussian offsets, std BASE_STD px scaled to the frame size relative to
    # BASE_SIZE), and each copy replaces the random proposal it overlaps the most. At most MAX_TRACKED
    # detections with score > SCORE_THRESH are carried over, so with 500 proposals at most 210 are temporal
    # and at least 290 stay random for exploration. Only used with SAMPLING_METHOD = 'Random'.
    cfg.MODEL.RandBox.TEMPORAL = CN()
    cfg.MODEL.RandBox.TEMPORAL.ENABLED = False
    cfg.MODEL.RandBox.TEMPORAL.NUM_PER_BOX = 3
    cfg.MODEL.RandBox.TEMPORAL.BASE_STD = 7.0
    cfg.MODEL.RandBox.TEMPORAL.BASE_SIZE = (1333, 750)
    cfg.MODEL.RandBox.TEMPORAL.SCORE_THRESH = 0.15
    cfg.MODEL.RandBox.TEMPORAL.MAX_TRACKED = 70

    
    # Swin Backbones
    cfg.MODEL.SWIN = CN()
    cfg.MODEL.SWIN.SIZE = 'B'  # 'T', 'S', 'B'
    cfg.MODEL.SWIN.USE_CHECKPOINT = False
    cfg.MODEL.SWIN.OUT_FEATURES = (0, 1, 2, 3)  # modify

    # Optimizer.
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 1.0

     
    # OW EVALUATION
    cfg.TEST.PREV_INTRODUCED_CLS = 0
    cfg.TEST.CUR_INTRODUCED_CLS = 20
    
    
    # TTA.
    cfg.TEST.AUG.MIN_SIZES = (400, 500, 600, 640, 700, 900, 1000, 1100, 1200, 1300, 1400, 1800, 800)
    cfg.TEST.AUG.CVPODS_TTA = True
    cfg.TEST.AUG.SCALE_FILTER = True
    cfg.TEST.AUG.SCALE_RANGES = ([96, 10000], [96, 10000], 
                                 [64, 10000], [64, 10000],
                                 [64, 10000], [0, 10000],
                                 [0, 10000], [0, 256],
                                 [0, 256], [0, 192],
                                 [0, 192], [0, 96],
                                 [0, 10000])
