# 🦘 Reconnaissance de Kangourous — Mask R-CNN

Instance segmentation of kangaroos using Mask R-CNN with transfer learning from COCO weights.

## Project structure

```
rcnn_mask_kangaroo/
├── dataset/
│   ├── images/          ← kangaroo .jpg images
│   └── annots/          ← Pascal VOC .xml annotations
├── logs/                ← saved model weights (created at training)
├── predictions/         ← output visualizations (created at inference)
├── mask_rcnn_lib/       ← Mask R-CNN library (cloned in step 1)
├── step1_setup.sh       ← install everything
├── step2_dataset.py     ← dataset class + XML parsing
├── step3_visualize.py   ← visualize images with ground-truth masks
├── step4_train.py       ← transfer learning from COCO
├── step5_predict.py     ← inference + mAP evaluation
├── requirements.txt
└── README.md
```

## Steps

### Step 1 — Setup
```bash
bash step1_setup.sh
```
Clones Mask R-CNN (TF2 fork), downloads the kangaroo dataset and COCO weights.

### Step 2 — Dataset
```bash
python step2_dataset.py
```
Parses XML annotations, builds binary masks, splits 80/20 train/test.

### Step 3 — Visualize
```bash
python step3_visualize.py
```
Saves sample images with ground-truth mask overlays to `predictions/`.

### Step 4 — Train
```bash
python step4_train.py
```
Two-phase transfer learning from COCO weights (10 epochs total).

### Step 5 — Predict
```bash
python step5_predict.py
```
Runs inference on the test set, computes mAP, saves visualizations.

## Requirements
- Python 3.9+  |  TensorFlow 2.12  |  GPU recommended

## Expected results
mAP ≈ 0.80+ at IoU 0.5 after 10 epochs.
