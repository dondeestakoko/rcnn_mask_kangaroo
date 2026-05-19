# ============================================================
# STEP 5 — Predictions on unknown kangaroo images
# Runs inference + computes mAP on the test set
# ============================================================

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

sys.path.append(os.path.join(os.getcwd(), 'mask_rcnn_lib'))

from mrcnn.config import Config
from mrcnn.model import MaskRCNN
import mrcnn.model as modellib
from mrcnn.utils import compute_ap
from mrcnn import visualize
from step2_dataset import KangarooDataset
from step4_train import KangarooConfig


# ── Inference config (batch size 1, no augmentation) ─────────
class PredictionConfig(KangarooConfig):
    GPU_COUNT              = 1
    IMAGES_PER_GPU         = 1
    DETECTION_MIN_CONFIDENCE = 0.9


# ── Evaluate mAP on the whole test set ───────────────────────
def evaluate_model(dataset, model, config, save_path='predictions/map_results.txt'):
    APs = []
    for image_id in dataset.image_ids:
        image, image_meta, gt_class_id, gt_bbox, gt_mask = \
            modellib.load_image_gt(dataset, config, image_id)

        sample = np.expand_dims(image, 0)
        yhat = model.detect(sample, verbose=0)
        r = yhat[0]

        AP, _, _, _ = compute_ap(
            gt_bbox, gt_class_id, gt_mask,
            r['rois'], r['class_ids'], r['scores'], r['masks']
        )
        APs.append(AP)

    mAP = np.mean(APs)
    print(f"\n📊 mAP on test set: {mAP:.3f}  ({len(APs)} images)")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        f.write(f"mAP: {mAP:.4f}\n")
        for i, (img_id, ap) in enumerate(zip(dataset.image_ids, APs)):
            f.write(f"  [{i:03d}] {dataset.image_info[img_id]['path']}: AP={ap:.3f}\n")
    print(f"Results saved → {save_path}")
    return mAP


# ── Predict and visualize a single image ─────────────────────
def predict_and_save(model, image_path, class_names,
                     save_path='predictions/prediction.png'):
    import cv2
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Could not read image: {image_path}")
        return
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = model.detect([img], verbose=1)
    r = results[0]

    n = len(r['class_ids'])
    print(f"Detected {n} kangaroo(s)")
    if n:
        print(f"Confidence scores: {r['scores'].round(3)}")

    # Draw with matplotlib
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(img)
    colors = plt.cm.Set1(np.linspace(0, 0.9, max(n, 1)))

    for i in range(n):
        mask  = r['masks'][:, :, i]
        box   = r['rois'][i]       # [y1, x1, y2, x2]
        score = r['scores'][i]
        color = colors[i % len(colors)]

        overlay = np.zeros((*mask.shape, 4))
        overlay[mask] = [*color[:3], 0.45]
        ax.imshow(overlay)

        y1, x1, y2, x2 = box
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor=color, facecolor='none'
        )
        ax.add_patch(rect)
        ax.text(x1, y1 - 5, f'kangaroo {score:.2f}',
                color=color, fontsize=10, fontweight='bold')

    ax.set_title(f'{os.path.basename(image_path)} — {n} detection(s)', fontsize=12)
    ax.axis('off')

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved → {save_path}")


# ── Visualize test set predictions vs ground truth ────────────
def compare_predictions(dataset, model, config, n=4,
                        save_path='predictions/comparison.png'):
    ids = dataset.image_ids[:n]
    fig, axes = plt.subplots(n, 2, figsize=(12, n * 5))
    fig.suptitle('Ground truth (left)  vs  Predictions (right)', fontsize=13)

    for row, img_id in enumerate(ids):
        image = dataset.load_image(img_id)
        masks_gt, _ = dataset.load_mask(img_id)

        results = model.detect([image], verbose=0)
        r = results[0]

        colors_gt   = plt.cm.Greens(np.linspace(0.4, 0.9, max(masks_gt.shape[-1], 1)))
        colors_pred = plt.cm.Reds(np.linspace(0.4, 0.9, max(len(r['class_ids']), 1)))

        for ax, (masks, colors, title) in zip(
            axes[row],
            [(masks_gt, colors_gt, f'GT ({masks_gt.shape[-1]} kang.)'),
             (r['masks'] if r['masks'].ndim == 3 else np.zeros((*image.shape[:2], 0), dtype=bool),
              colors_pred, f'Pred ({len(r["class_ids"])} kang.)')],
        ):
            ax.imshow(image)
            for i in range(masks.shape[-1]):
                overlay = np.zeros((*masks[:, :, i].shape, 4))
                overlay[masks[:, :, i]] = [*colors[i % len(colors)][:3], 0.45]
                ax.imshow(overlay)
            ax.set_title(title, fontsize=10)
            ax.axis('off')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved → {save_path}")


# ── Main ──────────────────────────────────────────────────────
if __name__ == '__main__':
    pred_config = PredictionConfig()

    model = MaskRCNN(mode='inference', model_dir='logs', config=pred_config)

    # Auto-find the latest trained weights
    model_path = model.find_last()
    print(f"Loading weights: {model_path}")
    model.load_weights(model_path, by_name=True)

    # Load test set
    test_set = KangarooDataset()
    test_set.load_dataset('dataset', is_train=False)
    test_set.prepare()

    class_names = ['BG', 'kangaroo']

    # 1. Evaluate mAP
    evaluate_model(test_set, model, pred_config)

    # 2. Compare GT vs predictions on first 4 test images
    compare_predictions(test_set, model, pred_config, n=4)

    # 3. Predict on a specific image (put your own image here)
    sample_img = test_set.image_info[test_set.image_ids[0]]['path']
    predict_and_save(model, sample_img, class_names)

    print("\n🎉 All done! Check the predictions/ folder.")
