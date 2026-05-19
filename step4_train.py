# ============================================================
# STEP 4 — Train Mask R-CNN on the kangaroo dataset
# Transfer learning from COCO weights (2 phases)
# ============================================================

import os
import sys
sys.path.append(os.path.join(os.getcwd(), 'mask_rcnn_lib'))

from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from step2_dataset import KangarooDataset


# ── Config ────────────────────────────────────────────────────
class KangarooConfig(Config):
    NAME = 'kangaroo'

    # 1 class + background
    NUM_CLASSES = 1 + 1

    # Adjust to your machine:
    #   GPU with 8 GB  → IMAGES_PER_GPU = 1
    #   No GPU / CPU   → GPU_COUNT = 1, IMAGES_PER_GPU = 1
    GPU_COUNT      = 1
    IMAGES_PER_GPU = 1

    # ~131 training images → 1 step per image per epoch
    STEPS_PER_EPOCH    = 131
    VALIDATION_STEPS   = 33

    # Smaller backbone for faster training on small dataset
    BACKBONE = 'resnet50'

    # Training hyperparameters
    LEARNING_RATE       = 0.001
    LEARNING_MOMENTUM   = 0.9
    WEIGHT_DECAY        = 0.0001

    # Anchor scales suited for kangaroo sizes
    RPN_ANCHOR_SCALES   = (32, 64, 128, 256, 512)

    # Max objects per image (kangaroo dataset has at most ~10)
    MAX_GT_INSTANCES    = 10
    DETECTION_MAX_INSTANCES = 10

    # Minimum confidence for detections
    DETECTION_MIN_CONFIDENCE = 0.9


# ── Main ──────────────────────────────────────────────────────
if __name__ == '__main__':
    config = KangarooConfig()
    config.display()

    # Prepare datasets
    train_set = KangarooDataset()
    train_set.load_dataset('dataset', is_train=True)
    train_set.prepare()
    print(f"Train: {len(train_set.image_ids)} images")

    test_set = KangarooDataset()
    test_set.load_dataset('dataset', is_train=False)
    test_set.prepare()
    print(f"Test : {len(test_set.image_ids)} images")

    # Build model in training mode
    model = MaskRCNN(mode='training', model_dir='logs', config=config)

    # Load COCO weights — exclude final classification heads
    # so they get randomly initialised for our 1 custom class
    COCO_WEIGHTS = 'mask_rcnn_coco.h5'
    print(f"\nLoading COCO weights from {COCO_WEIGHTS}...")
    model.load_weights(
        COCO_WEIGHTS,
        by_name=True,
        exclude=[
            'mrcnn_class_logits',
            'mrcnn_bbox_fc',
            'mrcnn_bbox',
            'mrcnn_mask',
        ]
    )

    # ── Phase 1: train only the head layers (fast) ────────────
    print("\n>>> Phase 1: training heads (5 epochs)...")
    model.train(
        train_set, test_set,
        learning_rate=config.LEARNING_RATE,
        epochs=5,
        layers='heads',
    )

    # ── Phase 2: fine-tune all layers (slower, better results) ─
    print("\n>>> Phase 2: fine-tuning all layers (5 more epochs)...")
    model.train(
        train_set, test_set,
        learning_rate=config.LEARNING_RATE / 10,
        epochs=10,
        layers='all',
    )

    print("\n✅ Training complete! Weights saved in logs/")
    print("Next: python step5_predict.py")
