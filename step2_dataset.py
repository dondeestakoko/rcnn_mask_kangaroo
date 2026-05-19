# ============================================================
# STEP 2 — Dataset configuration
# Read XML annotations and build the Mask R-CNN Dataset class
# ============================================================

import os
import sys
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image

# Add Mask R-CNN library to path
sys.path.append(os.path.join(os.getcwd(), 'mask_rcnn_lib'))

from mrcnn.utils import Dataset


class KangarooDataset(Dataset):
    """
    Loads the kangaroo dataset.
    Annotations are Pascal VOC XML files with <bndbox> elements.
    Since the dataset has no pixel-level masks, we build binary
    masks from the bounding boxes.
    """

    def load_dataset(self, dataset_dir, is_train=True):
        # Register the single class
        self.add_class('kangaroo', 1, 'kangaroo')

        images_dir = os.path.join(dataset_dir, 'images')
        annots_dir = os.path.join(dataset_dir, 'annots')

        all_imgs = sorted([f for f in os.listdir(images_dir) if f.endswith('.jpg')])

        # 80% train / 20% test split
        split = int(len(all_imgs) * 0.8)
        imgs = all_imgs[:split] if is_train else all_imgs[split:]

        for img_file in imgs:
            img_id = img_file.replace('.jpg', '')
            img_path = os.path.join(images_dir, img_file)
            xml_path = os.path.join(annots_dir, img_id + '.xml')

            # Read image dimensions from XML
            tree = ET.parse(xml_path)
            root = tree.getroot()
            size = root.find('size')
            width  = int(size.find('width').text)
            height = int(size.find('height').text)

            self.add_image(
                'kangaroo',
                image_id=img_id,
                path=img_path,
                annotation=xml_path,
                width=width,
                height=height,
            )

    def load_mask(self, image_id):
        """
        Returns:
          masks  : bool array (H, W, N) — one mask per bounding box
          labels : int array (N,)       — all 1s (kangaroo class)
        """
        info = self.image_info[image_id]
        tree = ET.parse(info['annotation'])
        root = tree.getroot()

        h, w = info['height'], info['width']
        masks = []

        for obj in root.findall('object'):
            name = obj.find('name').text
            if name != 'kangaroo':
                continue
            bndbox = obj.find('bndbox')
            xmin = int(float(bndbox.find('xmin').text))
            ymin = int(float(bndbox.find('ymin').text))
            xmax = int(float(bndbox.find('xmax').text))
            ymax = int(float(bndbox.find('ymax').text))

            mask = np.zeros((h, w), dtype=bool)
            mask[ymin:ymax, xmin:xmax] = True
            masks.append(mask)

        if not masks:
            return np.zeros((h, w, 0), dtype=bool), np.zeros((0,), dtype=np.int32)

        masks = np.stack(masks, axis=-1)
        labels = np.ones((masks.shape[-1],), dtype=np.int32)
        return masks, labels

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']


# ── Quick test ────────────────────────────────────────────────
if __name__ == '__main__':
    train_set = KangarooDataset()
    train_set.load_dataset('dataset', is_train=True)
    train_set.prepare()

    test_set = KangarooDataset()
    test_set.load_dataset('dataset', is_train=False)
    test_set.prepare()

    print(f"Train images : {len(train_set.image_ids)}")
    print(f"Test  images : {len(test_set.image_ids)}")

    # Check first sample
    image_id = train_set.image_ids[0]
    masks, class_ids = train_set.load_mask(image_id)
    print(f"First image  : {train_set.image_reference(image_id)}")
    print(f"Kangourous   : {masks.shape[-1]}")
    print(f"Mask shape   : {masks.shape}")
    print("\n✅ Dataset OK — run: python step3_visualize.py")
