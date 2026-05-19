#!/bin/bash
# ============================================================
# STEP 1 — Setup architecture for training
# ============================================================

echo ">>> [1/4] Cloning Mask R-CNN (TF2 compatible fork)..."
git clone https://github.com/akTwelve/Mask_RCNN.git mask_rcnn_lib
pip install -r mask_rcnn_lib/requirements.txt

echo ">>> [2/4] Installing extra dependencies..."
pip install tensorflow==2.12.0 keras==2.12.0 \
    numpy==1.23.5 scipy scikit-image \
    matplotlib opencv-python-headless \
    imgaug pycocotools

echo ">>> [3/4] Cloning kangaroo dataset..."
git clone https://github.com/experiencor/kangaroo.git kangaroo_raw
cp kangaroo_raw/images/*.jpg dataset/images/
cp kangaroo_raw/annots/*.xml dataset/annots/
echo "Dataset copied: $(ls dataset/images | wc -l) images, $(ls dataset/annots | wc -l) annotations"

echo ">>> [4/4] Downloading COCO pre-trained weights..."
wget -q --show-progress \
  https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5 \
  -O mask_rcnn_coco.h5
echo "Weights downloaded: $(du -h mask_rcnn_coco.h5 | cut -f1)"

echo ""
echo "✅ Setup complete! Run: python step2_dataset.py"
