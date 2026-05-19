# ============================================================
# STEP 3 — Visualize an image with its mask overlay
# ============================================================

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')               # headless — saves to file
import matplotlib.pyplot as plt
import matplotlib.patches as patches

sys.path.append(os.path.join(os.getcwd(), 'mask_rcnn_lib'))

from step2_dataset import KangarooDataset


def visualize_sample(dataset, image_id, save_path='predictions/sample_with_mask.png'):
    image  = dataset.load_image(image_id)
    masks, class_ids = dataset.load_mask(image_id)
    info   = dataset.image_info[image_id]

    n_kang = masks.shape[-1]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Image: {os.path.basename(info['path'])}  |  Kangourous: {n_kang}",
                 fontsize=13, fontweight='bold')

    # Left — original image
    axes[0].imshow(image)
    axes[0].set_title('Original image', fontsize=11)
    axes[0].axis('off')

    # Right — image + masks + bounding boxes
    axes[1].imshow(image)
    colors = plt.cm.Set1(np.linspace(0, 0.9, max(n_kang, 1)))

    for i in range(n_kang):
        mask = masks[:, :, i]
        color = colors[i % len(colors)]

        # Colored mask overlay
        overlay = np.zeros((*mask.shape, 4))
        overlay[mask] = [*color[:3], 0.45]
        axes[1].imshow(overlay)

        # Bounding box from mask extent
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        rect = patches.Rectangle(
            (cmin, rmin), cmax - cmin, rmax - rmin,
            linewidth=2, edgecolor=color, facecolor='none'
        )
        axes[1].add_patch(rect)
        axes[1].text(cmin, rmin - 4, f'kangaroo {i+1}',
                     color=color, fontsize=9, fontweight='bold')

    axes[1].set_title(f'Masks ({n_kang} object{"s" if n_kang != 1 else ""})', fontsize=11)
    axes[1].axis('off')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved → {save_path}")


def visualize_grid(dataset, n=6, save_path='predictions/dataset_grid.png'):
    """Show a grid of n random training samples with masks."""
    ids = np.random.choice(dataset.image_ids, min(n, len(dataset.image_ids)), replace=False)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('Training samples with ground-truth masks', fontsize=13, fontweight='bold')
    axes = axes.flatten()

    for ax, img_id in zip(axes, ids):
        image = dataset.load_image(img_id)
        masks, _ = dataset.load_mask(img_id)
        ax.imshow(image)
        colors = plt.cm.Set1(np.linspace(0, 0.9, max(masks.shape[-1], 1)))
        for i in range(masks.shape[-1]):
            overlay = np.zeros((*masks[:, :, i].shape, 4))
            overlay[masks[:, :, i]] = [*colors[i % len(colors)][:3], 0.45]
            ax.imshow(overlay)
        info = dataset.image_info[img_id]
        ax.set_title(f"{os.path.basename(info['path'])}  ({masks.shape[-1]} kang.)", fontsize=8)
        ax.axis('off')

    # Hide unused axes
    for ax in axes[len(ids):]:
        ax.axis('off')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved → {save_path}")


if __name__ == '__main__':
    train_set = KangarooDataset()
    train_set.load_dataset('dataset', is_train=True)
    train_set.prepare()

    print(f"Loaded {len(train_set.image_ids)} training images")

    # Single sample with detailed view
    visualize_sample(train_set, train_set.image_ids[0])

    # Grid of 6 samples
    visualize_grid(train_set, n=6)

    print("\n✅ Visualizations done — run: python step4_train.py")
