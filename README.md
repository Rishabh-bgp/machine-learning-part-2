<div align="center">

# Machine Learning Part 2 — Computer Vision with YOLO11: Detection, Segmentation, Pose & Potato Leaf Disease Classification

![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)
![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO11-111F68)
![PyTorch](https://img.shields.io/badge/PyTorch-2.11-EE4C2C?logo=pytorch&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-notebooks-F37626?logo=jupyter&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-blue)

</div>

Hands-on **computer vision in Python** with **Ultralytics YOLO11**: a tour of every YOLO11 task (object detection,
instance segmentation, pose estimation, image classification and oriented bounding boxes) on images and video, followed
by an end-to-end **transfer-learning project** that fine-tunes `yolo11n-cls` to classify **potato leaf diseases**
(Early Blight, Late Blight, Healthy).

> Practice work by **Rishabh Aryan** for the **6-Month Industrial Training & Certification Program in Artificial Intelligence and Machine Learning**, jointly offered by [IIT Ropar](https://www.iitrpr.ac.in/) and [NIELIT Ropar](https://www.nielit.gov.in/ropar/), under MeitY, Govt. of India.
> Previous part: [Machine Learning Part 1](https://github.com/Rishabh-bgp/machine-learning-part-1) · Next part: [Machine Learning Part 3 — road damage detection](https://github.com/Rishabh-bgp/machine-learning-part-3).

---

## 📚 Topics and notebooks

| # | Notebook / script | Task | Topics | Models |
|---|---|---|---|---|
| 1 | [01_yolo11_tasks_tutorial.ipynb](notebooks/01_yolo11_tasks_tutorial.ipynb) · [script](scripts/yolo11_tasks_tutorial.py) | Pretrained inference | Detection, segmentation, pose, classification and OBB on an image and a video; saving annotated outputs | `yolo11n`, `yolo11n-seg`, `yolo11n-pose`, `yolo11n-cls`, `yolo11n-obb` |
| 2 | [02_potato_disease_classification_yolo11.ipynb](notebooks/02_potato_disease_classification_yolo11.ipynb) · [script](scripts/potato_disease_classification.py) | 3-class image classification | Kaggle download with `kagglehub`, 80/20 train/val split into class folders, fine-tuning, inference with class probabilities, training curves | `yolo11n-cls` (fine-tuned) |

## 📈 Results (from the executed notebooks)

### Potato leaf disease — `yolo11n-cls`, 5 epochs, 224 × 224, CPU
1,721 training and 431 validation images in 3 classes.

| Epoch | Top-1 val accuracy | Top-5 val accuracy |
|---|---|---|
| 1 | 0.988 | 1.000 |
| 2 | 0.928 | 1.000 |
| 3 | **0.995** | 1.000 |
| 4 | 0.979 | 1.000 |
| 5 | 0.991 | 1.000 |
| Best weights (final validation) | **0.995** | 1.000 |

A sample Early Blight validation leaf was predicted as Early_Blight 100.00%, Healthy 0.00%, Late_Blight 0.00%.

### YOLO11 task tour (pretrained, no training)
| Task | Model | Example output |
|---|---|---|
| Detection (image) | `yolo11n` | 1 person, 1 cup, 1 potted plant (156.5 ms) |
| Detection (video, 1,616 frames) | `yolo11n` | 3 persons per frame |
| Segmentation (image) | `yolo11n-seg` | 1 person, 1 wine glass, 1 bowl, 1 chair, 1 potted plant, 2 sinks |
| Pose (image) | `yolo11n-pose` | 1 person |
| Classification (image) | `yolo11n-cls` | ImageNet top-1: shoe_shop 0.31 |
| Oriented boxes (video) | `yolo11n-obb` | no detections (DOTA aerial classes) |

## 🖼️ Output gallery

Outputs saved from the executed notebooks (in [`docs/outputs/`](docs/outputs/)).

<table>
  <tr>
    <td align="center" width="50%"><img src="docs/outputs/detection.jpg" alt="Object detection" width="100%"><br><sub>Object detection — <code>yolo11n</code> on <code>test.jpeg</code></sub></td>
    <td align="center" width="50%"><img src="docs/outputs/detection_video.jpg" alt="Object detection" width="100%"><br><sub>Object detection — <code>yolo11n</code> on a <code>test.mp4</code> frame</sub></td>
  </tr>
  <tr>
    <td align="center" width="50%"><img src="docs/outputs/segmentation.jpg" alt="Instance segmentation" width="100%"><br><sub>Instance segmentation — <code>yolo11n-seg</code></sub></td>
    <td align="center" width="50%"><img src="docs/outputs/pose.jpg" alt="Pose estimation" width="100%"><br><sub>Pose estimation — <code>yolo11n-pose</code> on <code>test.jpeg</code></sub></td>
  </tr>
  <tr>
    <td align="center" width="50%"><img src="docs/outputs/pose_video.jpg" alt="Pose estimation" width="100%"><br><sub>Pose estimation — <code>yolo11n-pose</code> on a <code>test.mp4</code> frame</sub></td>
    <td align="center" width="50%"><img src="docs/outputs/classification.jpg" alt="Image classification" width="100%"><br><sub>Image classification — <code>yolo11n-cls</code> on <code>test.jpeg</code></sub></td>
  </tr>
  <tr>
    <td align="center" width="50%"><img src="docs/outputs/classification_test1.jpg" alt="Image classification" width="100%"><br><sub>Image classification — <code>yolo11n-cls</code> on <code>test1.jpeg</code></sub></td>
    <td align="center" width="50%"><img src="docs/outputs/obb_video.jpg" alt="Oriented bounding boxes" width="100%"><br><sub>Oriented bounding boxes — <code>yolo11n-obb</code> on a <code>test.mp4</code> frame</sub></td>
  </tr>
  <tr>
    <td align="center" width="50%"><img src="docs/outputs/training_curves.png" alt="Potato leaf disease classifier" width="100%"><br><sub>Potato leaf disease classifier — training curves (5 epochs)</sub></td>
  </tr>
</table>

## 🗂️ Data
Nothing is bundled. Notebook 2 downloads the [Kaggle potato dataset](https://www.kaggle.com/datasets/faysalmiah1721758/potato-dataset)
(about 38 MB) with `kagglehub` at run time. Notebook 1 expects your own `test.jpeg` and `test.mp4` in the working folder;
its image outputs were removed from the committed notebook to keep it small and private.

## 🚀 How to run
```bash
git clone https://github.com/Rishabh-bgp/machine-learning-part-2.git
cd machine-learning-part-2
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab notebooks/
```
Pretrained YOLO11 weights download automatically on first use. Kaggle downloads may need a
[Kaggle API token](https://www.kaggle.com/docs/api) in `~/.kaggle/`. Both notebooks also run as-is in Google Colab.

## 📁 Repository layout
```
notebooks/   executed Jupyter notebooks
scripts/     plain-Python exports of the notebooks (Colab "!pip" lines included)
docs/outputs/ output images extracted from the notebooks
requirements.txt  LICENSE  README.md
```

## 🔗 Other repositories
- [Machine Learning Part 1](https://github.com/Rishabh-bgp/machine-learning-part-1)
- [Machine Learning Part 3 — road damage detection](https://github.com/Rishabh-bgp/machine-learning-part-3) · [live demo](https://rishabh-bgp.github.io/machine-learning-part-3/)
- [NIELIT Machine Learning Projects](https://github.com/Rishabh-bgp/NIELIT-MACHINE-LEARNING-PROJECTS)
- [All repositories](https://github.com/Rishabh-bgp?tab=repositories)

## 📄 License
[MIT](LICENSE) © 2026 Rishabh Aryan. The dataset belongs to its Kaggle authors; YOLO11 is by Ultralytics (AGPL-3.0).

## Results from a fresh run

All files were re-run on 30 Oct 2026 (IST) – see [results/RESULTS.md](results/RESULTS.md) for the numbers. Potato classifier: **99.77 % top-1** after 5 epochs (original: 99.5 %).

Videos: [detection](results/videos/detect_test.mp4) · [pose](results/videos/pose_test.mp4) · [oriented boxes](results/videos/obb_test.mp4)

### YOLO11 tasks

<table>
<tr><td align="center"><img src="results/images/classify_test.jpg" width="260" alt="classify_test.jpg"><br><sub>classify_test.jpg</sub></td><td align="center"><img src="results/images/classify_test1.jpg" width="260" alt="classify_test1.jpg"><br><sub>classify_test1.jpg</sub></td><td align="center"><img src="results/images/detect_test.jpg" width="260" alt="detect_test.jpg"><br><sub>detect_test.jpg</sub></td></tr>
<tr><td align="center"><img src="results/images/pose_test.jpg" width="260" alt="pose_test.jpg"><br><sub>pose_test.jpg</sub></td><td align="center"><img src="results/images/segment_test.jpg" width="260" alt="segment_test.jpg"><br><sub>segment_test.jpg</sub></td></tr>
</table>

### Potato classifier training

<table>
<tr><td align="center"><img src="results/training/confusion_matrix.png" width="260" alt="confusion_matrix.png"><br><sub>confusion_matrix.png</sub></td><td align="center"><img src="results/training/confusion_matrix_normalized.png" width="260" alt="confusion_matrix_normalized.png"><br><sub>confusion_matrix_normalized.png</sub></td><td align="center"><img src="results/training/results.png" width="260" alt="results.png"><br><sub>results.png</sub></td></tr>
<tr><td align="center"><img src="results/training/train_batch0.jpg" width="260" alt="train_batch0.jpg"><br><sub>train_batch0.jpg</sub></td><td align="center"><img src="results/training/train_batch1.jpg" width="260" alt="train_batch1.jpg"><br><sub>train_batch1.jpg</sub></td><td align="center"><img src="results/training/train_batch2.jpg" width="260" alt="train_batch2.jpg"><br><sub>train_batch2.jpg</sub></td></tr>
<tr><td align="center"><img src="results/training/val_batch0_labels.jpg" width="260" alt="val_batch0_labels.jpg"><br><sub>val_batch0_labels.jpg</sub></td><td align="center"><img src="results/training/val_batch0_pred.jpg" width="260" alt="val_batch0_pred.jpg"><br><sub>val_batch0_pred.jpg</sub></td><td align="center"><img src="results/training/val_batch1_labels.jpg" width="260" alt="val_batch1_labels.jpg"><br><sub>val_batch1_labels.jpg</sub></td></tr>
<tr><td align="center"><img src="results/training/val_batch1_pred.jpg" width="260" alt="val_batch1_pred.jpg"><br><sub>val_batch1_pred.jpg</sub></td><td align="center"><img src="results/training/val_batch2_labels.jpg" width="260" alt="val_batch2_labels.jpg"><br><sub>val_batch2_labels.jpg</sub></td><td align="center"><img src="results/training/val_batch2_pred.jpg" width="260" alt="val_batch2_pred.jpg"><br><sub>val_batch2_pred.jpg</sub></td></tr>
</table>

