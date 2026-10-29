# Results from a fresh run

Run on 30 Oct 2026 (IST) on a CPU-only Linux box (8 cores, Python 3.13, PyTorch 2.15 CPU, latest Ultralytics).
The original notebooks and scripts were **not modified**; copies were executed with `jupyter nbconvert --execute`.

Environment-only adjustments: the Colab paths `/content/...` were recreated locally (media copied to `/content`,
`runs_dir` set to `/content/runs`, and `test1.jpeg` placed at `/content/runs/detect/test1.jpeg` as in the original Colab session).
The `.py` scripts contain Colab `!pip` lines, so they were run with `ipython` rather than plain `python`.

## Part 1 – YOLO11 tasks tutorial

| Task | Model | Input | Result |
|---|---|---|---|
| Detection | yolo11n.pt | test.jpeg | 1 person, 1 cup, 1 potted plant |
| Detection | yolo11n.pt | test.mp4 | annotated video → `videos/detect_test.mp4` |
| Segmentation | yolo11n-seg.pt | test.jpeg | 1 person, 1 wine glass, 1 bowl, 1 chair, 1 potted plant, 2 sinks |
| Pose | yolo11n-pose.pt | test.mp4 | annotated video → `videos/pose_test.mp4` |
| Pose | yolo11n-pose.pt | test.jpeg | 1 person |
| Classification | yolo11n-cls.pt | test.jpeg | shoe_shop 0.31, fur_coat 0.08, bakery 0.04, brassiere 0.04, hair_spray 0.03 |
| Classification | yolo11n-cls.pt | test1.jpeg | keeshond 0.28, Pekinese 0.24, Pomeranian 0.10, Shih-Tzu 0.05, maillot 0.03 |
| Oriented boxes | yolo11n-obb.pt | test.mp4 | annotated video → `videos/obb_test.mp4` |

Videos were re-encoded to H.264 at 640 px width to stay under 5 MB.

## Part 2 – Potato leaf disease classification (yolo11n-cls, 5 epochs, imgsz 224)

Dataset: Kaggle `faysalmiah1721758/potato-dataset` (downloaded without login, random 80/20 split, not committed).

| Epoch | train loss | top-1 | top-5 | val loss |
|---|---|---|---|---|
| 1 | 0.4209 | 0.9374 | 1.000 | 0.1798 |
| 2 | 0.1193 | 0.9907 | 1.000 | 0.0287 |
| 3 | 0.0910 | 0.9954 | 1.000 | 0.0122 |
| 4 | 0.0948 | 0.9907 | 1.000 | 0.0171 |
| 5 | 0.0499 | **0.9977** | 1.000 | 0.0066 |

Final top-1: **99.77 %** (original notebook: 99.5 %). The small difference comes from the random train/val split and CPU vs GPU training.
A second, separate run of the `.py` script reached 0.9861 → 0.9954 → 1.000 → 0.9977 → 0.9977 top-1.

Inference on a validation image (Early Blight): **Early_Blight 99.99 %, Late_Blight 0.01 %, Healthy 0.00 %**.

Files: `training/results.csv`, `training/results.png`, `training/confusion_matrix*.png`, batch previews in `training/`,
and the executed notebooks `01_…_executed.ipynb`, `02_…_executed.ipynb`.
