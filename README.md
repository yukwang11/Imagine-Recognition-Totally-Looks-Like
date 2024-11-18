# Totally-Looks-Like Challenge

## Overview

This project explores the challenge of visual similarity in the field of computer vision. The objective is to develop an algorithm capable of finding the most similar image from a pool of candidates for a given image pair, addressing multiple types of image similarities such as color, shape, texture, pose, and facial expression. By leveraging modern deep learning techniques, this project evaluates the effectiveness of various pre-trained models and methods for image retrieval tasks.

---

## Environmental Requirements
To run this project, ensure the following are installed in your environment:
  - Python 3.x
  - TensorFlow 2.x
  - Keras
  - scikit-learn
  - numpy
  - pandas

---

## Dataset

The dataset originates from the **Totally-Looks-Like (TLL)** collection and includes:
- **Training Set**: 2,000 image pairs with ground truth matches in `train.csv`.
- **Test Set**: 2,000 left images, each matched with 20 right candidate images, including one true match and 19 foils, detailed in `test_candidates.csv`.

### Key Features:
- Images are standardized to **200 × 245 pixels**.
- Dataset spans diverse domains: celebrity faces, logos, art sketches, household objects, and more.

---

## Methods

### 1. **VGG16 with Cosine Similarity**
- Extracted high-level features using a pre-trained VGG16 model.
- Used cosine similarity to calculate pairwise similarity.
- **Accuracy (Kaggle)**: **0.51**

### 2. **DenseNet121 with Siamese Network**
- Adopted DenseNet121 for feature extraction due to its efficient architecture.
- Employed a Siamese network to compare feature vectors using `cosine_triplet_loss`.
- **Accuracy (Kaggle)**: **0.524**

### 3. **DenseNet121 & MobileNet with Siamese Network**
- Combined features from DenseNet121 and MobileNet, leveraging their strengths.
- Fusion of features (individually extracted) improved performance significantly.
- **Accuracy (Kaggle)**: **0.563**

---

## Evaluation

- **DenseNet121 + MobileNet (individual extraction)** achieved the highest accuracy: **0.563**.
- Overlapping features from multiple models generally improved results, but combined feature extraction occasionally underperformed due to feature redundancy.
- Data augmentation and certain pre-trained models (e.g., ResNet50) did not significantly enhance accuracy, likely due to sufficient diversity in the dataset.

### Experiment Accuracy Summary:

| Pre-trained Model(s)             | Training Model                | Accuracy (Kaggle) |
|----------------------------------|-------------------------------|-------------------|
| VGG16                            | -                             | **0.51**          |
| DenseNet121                      | Siamese + cosine_triplet_loss | **0.524**         |
| DenseNet121 + MobileNet (indiv.) | Siamese + cosine_triplet_loss | **0.563**         |
| DenseNet121 + MobileNet (comb.)  | Siamese + cosine_triplet_loss | **0.512**         |
| DenseNet121 + VGG16 (comb.)      | Siamese + cosine_triplet_loss | **0.392**         |
| ResNet50 (w/o augmentation)      | Siamese + binary_crossentropy | **0.201**         |

---


## Files

The following files are part of the project:

- `vgg+cosinesim.ipynb`: Notebook for implementing the VGG16 model with cosine similarity.
- `siamese_net.ipynb`: Notebook for implementing the Siamese model with DenseNet121 and MobileNet.
- Data Folder: Contains raw training and test data, along with the images used in the project.

