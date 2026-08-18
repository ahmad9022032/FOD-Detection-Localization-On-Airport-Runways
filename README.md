# FOD Detection and Localization on Airport Runways

Foreign Object Debris (FOD) — loose hardware, pavement fragments, tools, wildlife remains and other stray objects — is a serious hazard on airport runways and costs the aviation industry billions of dollars every year. This project builds an automated, camera-based system that **detects and localizes FOD in real time** using modern deep-learning object detectors, with super-resolution enhancement to improve the detection of small and partially obscured objects.

## Highlights

- **Two detector families trained and compared:** YOLOv8 (nano) and RT-DETR (large) on the FOD-A dataset.
- **Real-time inference:** tested on live video via a laptop camera pipeline.
- **Super-resolution pipeline:** every *n*-th frame is enhanced before detection to boost small-object accuracy.
- **Localization:** detected debris is reported with bounding-box coordinates on the runway.
- **Reproducible experiments:** training notebooks, evaluation plots, and full run artifacts included.

## Results

RT-DETR-L fine-tuned for 10 epochs on the FOD-A dataset (400×400 images, batch 16):

| Metric | Value |
|---|---|
| Precision | 0.996 |
| Recall | 0.992 |
| mAP@50 | 0.990 |
| mAP@50–95 | 0.908 |

Full training curves, confusion matrices, PR/F1 curves, and validation batch visualizations are in [RT-DETR/dt-retr_10_epochs/dt-retr_fod_10/](RT-DETR/dt-retr_10_epochs/dt-retr_fod_10/).

A YOLOv8n model was also trained for 200 epochs; its weights are available on request (see below).

## Repository Structure

```
├── RT-DETR/
│   ├── DeTr.ipynb                  # DETR experiments
│   ├── dt-retr.ipynb               # RT-DETR training notebook
│   ├── DETR_bbox.ipynb             # Bounding-box inference / visualization
│   └── dt-retr_10_epochs/          # Full RT-DETR training run artifacts
│       └── dt-retr_fod_10/         # curves, confusion matrices, results.csv
├── Testing Script for Laptop Camera/
│   └── testing_pt_files.ipynb      # Run trained .pt models on a live camera feed
├── Plotting Script for Metrics/
│   └── plot_script.ipynb           # Plot training/validation metrics
└── README.md
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/ahmad9022032/FOD-Detection-Localization-On-Airport-Runways.git
cd FOD-Detection-Localization-On-Airport-Runways
```

### 2. Install dependencies

```bash
pip install ultralytics opencv-python matplotlib pandas jupyter
```

### 3. Run live detection

Open [testing_pt_files.ipynb](Testing%20Script%20for%20Laptop%20Camera/testing_pt_files.ipynb) and point it at a weights file (request the trained weights via the Contact section below), e.g.:

```python
from ultralytics import YOLO

model = YOLO("yolov8n_200.pt")
results = model.predict(source=0, show=True)  # source=0 → default webcam
```

### 4. Retrain / reproduce

- **RT-DETR:** run [dt-retr.ipynb](RT-DETR/dt-retr.ipynb) against the FOD-A dataset.
- **Metric plots:** run [plot_script.ipynb](Plotting%20Script%20for%20Metrics/plot_script.ipynb) on the `results.csv` produced by training.

## Dataset

The models are trained on **[FOD-A](https://github.com/FOD-UNOmaha/FOD-data)** (Foreign Object Debris in Airports), a public dataset of common runway debris categories captured under varying lighting and weather conditions, resized to 400×400 for training.

## Model Weights

| Model | Training | Availability |
|---|---|---|
| YOLOv8n | 200 epochs | **On request** — email me (see Contact) |
| RT-DETR-L | 10 epochs | **On request** — email me (see Contact) |

Trained checkpoints are not hosted in this repository. If you would like the YOLOv8 or RT-DETR weights, please email me and I'll share them.

## Evaluation Metrics

- **mAP@50 / mAP@50–95** — mean average precision at IoU 0.5 and averaged over IoU 0.5–0.95.
- **Precision / Recall** — detection correctness vs. coverage of actual FOD.
- **Validation box / classification loss** — bounding-box and class prediction error during validation.

## Contact

For model weights, dataset preparation details, or collaboration:

📧 **muhammadahmadkhan316@gmail.com**
