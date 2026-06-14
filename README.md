# COVID-19 Chest X-Ray Classification with Explainable AI

A deep learning system for automated diagnosis of COVID-19 and other respiratory conditions from chest X-ray images using masked segmentation, ResNet-50, and Grad-CAM explainability.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Explainable AI](#explainable-ai)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)
- [Author](#author)
- [License](#license)

## Overview

This project performs multi-class classification of chest X-ray images into four categories:

| Class | Description |
|-------|-------------|
| COVID | COVID-19 viral infection |
| Lung Opacity | Non-COVID lung abnormalities |
| Normal | Healthy lung appearance |
| Viral Pneumonia | Non-COVID viral infections |

Performance Summary:

- Validation Accuracy: 92.0%
- Macro F1-Score: 0.92
- Grad-CAM heatmaps for model interpretability
- Masked segmentation to focus on lung regions

## Dataset

Source: COVID-19 Radiography Database (Kaggle)

Dataset Composition:

- 1000 images per category (4000 total)
- 15% validation split (600 images)
- Paired image and mask files for each X-ray

Preprocessing Steps:

- Resize images to 224x224 pixels
- Apply mask overlay segmentation
- ResNet-50 preprocessing normalization
- Data augmentation: rotation (10 degrees), shift (10%), horizontal flip

## Methodology

Model Architecture:

| Component | Specification |
|-----------|---------------|
| Base Model | ResNet-50 with ImageNet weights |
| Trainable Layers | Last 100 layers |
| Pooling | Global Average Pooling 2D |
| Dense Layer | 512 units with Batch Normalization |
| Regularization | L2 (0.001) and Dropout (0.5) |
| Output Layer | Softmax with 4 classes |

Training Configuration:

| Parameter | Value |
|-----------|-------|
| Epochs | 40 (early stopping at 21) |
| Batch Size | 32 |
| Optimizer | Adam (learning rate: 0.0001) |
| Loss Function | Categorical Crossentropy |
| Learning Rate Schedule | ReduceLROnPlateau (factor 0.2, patience 3) |
| Early Stopping | Patience 8, restore best weights |

## Results

Classification Report:

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| COVID | 0.96 | 0.87 | 0.91 | 150 |
| Lung Opacity | 0.90 | 0.86 | 0.88 | 150 |
| Normal | 0.84 | 0.97 | 0.90 | 150 |
| Viral Pneumonia | 0.99 | 0.97 | 0.98 | 150 |

Overall Accuracy: 92.0%

Macro Average F1-Score: 0.92

## Project Structure

```
COVID-19-Chest-XRay-Classification/
│
├── notebooks/
│   ├── 01_data_loading_segmentation.ipynb
│   ├── 02_model_training_resnet50.ipynb
│   ├── 03_performance_analytics.ipynb
│   ├── 04_tsne_visualization.ipynb
│   ├── 05_xai_gradcam_block3.ipynb
│   └── 06_xai_gradcam_block2_sharp.ipynb
│
├── scripts/
│   ├── data_loader.py
│   ├── model_builder.py
│   ├── train.py
│   ├── xai_utils.py
│   └── inference.py
│
├── outputs/
│   ├── confusion_matrix.png
│   ├── accuracy_loss_curves.png
│   └── tsne_clusters.png
│
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

Prerequisites:

- Python 3.9 or higher
- CUDA-capable GPU (recommended for faster training)

Setup Steps:

1. Clone the repository

```bash
git clone https://github.com/yourusername/covid19-chest-xray-resnet50-gradcam.git
cd covid19-chest-xray-resnet50-gradcam
```

2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate  # Linux or Mac
venv\Scripts\activate     # Windows
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

4. Download the dataset

Download from Kaggle: COVID-19 Radiography Database

Update the base_path variable in the notebooks to point to your dataset location.

## Usage

Train the model:

```bash
jupyter notebook notebooks/02_model_training_resnet50.ipynb
```

Or using the script:

```bash
python scripts/train.py --epochs 40 --batch_size 32
```

Evaluate the model:

```bash
python scripts/evaluate.py --model_path models/msc_best_covid_model.h5
```

Run interactive XAI interface:

```bash
jupyter notebook notebooks/05_xai_gradcam_block3.ipynb
```

The XAI interface allows:

- Prediction on validation set samples (indices 0 to 599)
- Upload of custom chest X-ray images
- Grad-CAM heatmap visualization

## Explainable AI

Two Grad-CAM implementations are provided:

| Block | Layer Name | Description |
|-------|------------|-------------|
| Block 3 | conv5_block3_out | Final convolutional layer, provides coarse feature visualization |
| Block 2 | conv5_block2_out | Intermediate layer, provides sharper localization |

Grad-CAM produces heatmaps that highlight the regions the model uses for classification decisions. This helps clinicians understand and trust the model predictions.

## Citation

If you use this code or methodology in your research, please cite:

```bibtex
@misc{covid19-xray-resnet50,
  author = {Abinaya V},
  title = {COVID-19 Chest X-Ray Classification with Explainable AI},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/yourusername/covid19-chest-xray-resnet50-gradcam}
}
```

## Acknowledgments

This project uses the following publicly available resources:

- COVID-19 Radiography Dataset from Kaggle
- TensorFlow and Keras deep learning frameworks
- ResNet-50 architecture
- Grad-CAM visualization technique
##KAGGLE LINK :
https://www.kaggle.com/code/abikullachi/residual-rebels

## Author

**Abinaya V**

- MSc Integrated Data Science
- SASTRA Deemed to be University
- abinayavenkat403@gmail.com
- 127177002@sastra.ac.in
- https://www.linkedin.com/in/abinaya-v-2a53172aa/

## License

This project is licensed under the MIT License.

## Disclaimer

This system is designed for research purposes only. It is not intended for clinical diagnosis without proper validation and regulatory approval.

