# Alzheimer's MRI Classification: ANN vs. CNN Architectures
 
A comparative deep learning study evaluating whether Artificial Neural Networks (ANNs) or Convolutional Neural Network (CNN) architectures are better suited for classifying Alzheimer's disease stages from brain MRI scans.
 
## Problem Statement
 
Alzheimer's Disease requires early intervention, but distinguishing subtle stages — such as "Very Mild" vs. "Mild" dementia — from MRI scans is difficult even for the human eye. This project builds an automated deep learning system that classifies MRI scans into four categories:
 
- Non-Demented
- Very Mild Dementia
- Mild Dementia
- Moderate Dementia
The goal is higher diagnostic accuracy and faster clinical turnaround than manual review allows.
 
## Approach
 
Inspired by the interpretability goals of GKAN (C2C-8), this project implements a TensorFlow-based comparative framework. A baseline **ANN** was trained alongside several **CNN architectures** — LeNet, AlexNet, VGGNet, GoogLeNet, ResNet, and ZFNet — to evaluate how spatial feature extraction affects classification accuracy on MRI data.
 
**Use case:** Positioned as a Clinical Decision Support Tool, the system is meant to assist radiologists by highlighting subtle atrophy patterns, particularly in early-stage ("Very Mild") cases where clinical symptoms may not yet be fully apparent.
 
## Dataset
 
- **Source:** [OASIS Alzheimer's Detection dataset (Kaggle)](https://www.kaggle.com/datasets/ninadaithal/imagesoasis)
- **Classes:** Non-Demented, Very Mild, Mild, Moderate Demented
- **Input:** MRI scans, flattened to a 12,288-dimensional vector (64×64×3) for the ANN, and used as raw spatial input for CNN architectures
## Results Summary
 
| Model | Accuracy | Micro-Avg AUC | Notable Weakness |
|---|---|---|---|
| ANN (baseline) | 67.46% (peak val, Epoch 8) | — | High training volatility; no spatial awareness |
| ANN (tuned, lr=2e-4) | 75.08% | ~0.74 | F1 of only 0.10–0.13 on Mild/Moderate class |
| CNN (AlexNet) | 79.28% | 0.94 | Some overfitting remains; F1 0.35–0.38 on minority classes |
| CNN (GoogLeNet v3) | Higher than ANN | Higher than ANN | Some validation instability |
| CNN (ZFNet) | 80.5% | ~0.80 | Macro F1 only ~0.46; struggles on minority classes without pretrained weights |
 
### Key Findings
 
- **The "Accuracy Paradox":** The ANN's ~75% accuracy looks strong but is inflated by the dominant "Non-Demented" class. Its F1-score for Mild/Moderate dementia (~0.10–0.13) is near-random — a clinically dangerous failure mode.
- **Spatial blindness is the ANN's fundamental bottleneck:** Flattening a 64×64×3 MRI image into a 12,288-dimensional vector destroys the spatial relationships between pixels that are diagnostically essential in brain imaging.
- **CNNs consistently outperform the ANN on disease detection**, not just raw accuracy:
  - AlexNet nearly doubles ANN recall on "Very Mild" dementia (48% vs. 33%).
  - ZFNet beats the ANN on every clinical metric: accuracy (+5.5pp), macro F1 (+0.07), micro AUC (+0.06), and Mild/Moderate F1 (+0.09).
- **Class imbalance remains an open challenge.** Even the best-performing CNNs (AlexNet, ZFNet) show a strong bias toward the majority "Non-Demented" class, indicating a need for stronger regularization and class-weighting in future iterations.
- **Overall conclusion:** For Alzheimer's MRI classification, convolutional architectures structurally outperform flat ANNs. The ANN's only advantages are training speed and memory efficiency — of limited value when the model cannot reliably detect the disease.
## Architectures Compared
 
| Architecture | Type |
|---|---|
| Baseline ANN | Fully-connected (flattened input) |
| LeNet | CNN |
| AlexNet | CNN |
| VGGNet | CNN |
| GoogLeNet (Inception v3) | CNN |
| ResNet | CNN |
| ZFNet | CNN |
 
## Evaluation Metrics
 
Each model was evaluated using:
- Training/validation loss and accuracy curves
- Confusion matrices
- Per-class precision, recall, and F1-score
- ROC AUC (per-class and micro-average)
- Hyperparameter tuning results
## Tech Stack
 
- **Framework:** TensorFlow / Keras
- **Language:** Python
- **Domain:** Medical imaging, deep learning, computer vision
## Future Work
 
- Apply stronger regularization and class-weighting to address dataset imbalance
- Use pretrained weights / transfer learning to improve minority-class detection
- Expand evaluation to additional CNN architectures and ensemble methods
## Acknowledgements
 
Dataset: [OASIS Alzheimer's Detection](https://www.kaggle.com/datasets/ninadaithal/imagesoasis) via Kaggle.
