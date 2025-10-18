# Waste Classification using Fine-Tuned ResNet50V2

This project is a research and implementation of a deep learning model to solve the problem of waste image classification. By utilizing a fine-tuning approach on the advanced **ResNet50V2** convolutional neural network architecture, the system can classify waste into predefined categories, contributing to the automation of the sorting process and promoting more effective recycling.

This project is based on the scientific paper "Waste Classification Using Fine-Tune ResNet50V2: A Deep Learning Approach".

## Table of Contents
- [Introduction](#introduction)
- [Key Features](#key-features)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)
- [Setup & Re-running the Experiment](#setup--re-running-the-experiment)
- [Authors](#authors)
- [References](#references)

## Introduction

Vietnam is facing significant challenges in waste management due to a rapidly increasing volume of municipal solid waste. Currently, about 70% of waste is disposed of in landfills, causing severe environmental pollution. A critical component of effective waste management is accurate classification, which optimizes recycling and reduces the burden on landfills.

This project proposes an automated solution using deep learning. The **ResNet50V2** model, with its ability to learn complex features from images, was chosen for waste classification to address challenges such as diverse appearances, varying lighting conditions, and occlusions.

## Key Features
- **State-of-the-Art Model:** Utilizes the **ResNet50V2** architecture with residual learning and pre-activation, which helps mitigate the vanishing gradient problem and achieve high accuracy.
- **Transfer Learning:** Leverages pre-trained weights from the large ImageNet dataset, then fine-tunes the last 38 layers to adapt the model to the specific task of waste classification.
- **Data Augmentation:** Applies various real-time data augmentation techniques (rotation, shifting, zooming, flipping, brightness adjustment) using Keras's `ImageDataGenerator` to combat overfitting.
- **Class Imbalance Handling:** Implements a class weight calculation method based on the frequency of each waste type, forcing the model to pay more attention to minority classes (like 'trash').
- **Optimized Training:** Uses Keras callbacks such as `ReduceLROnPlateau` for adaptive learning rate adjustment, `EarlyStopping` to prevent overfitting, and `ModelCheckpoint` to save the best-performing model.
- **Result Visualization:** Provides plots for Accuracy and Loss over epochs, along with a Confusion Matrix for detailed performance evaluation.

## Model Architecture

The core architecture is based on **ResNet50V2** and is customized for the waste classification task.

1.  **Base Model:**
    -   The **ResNet50V2** network pre-trained on ImageNet.
    -   This architecture consists of 50 layers with residual blocks that use a **pre-activation** scheme (Batch Normalization -> ReLU -> Convolution), which improves training.
    -   The original top classification layer is removed.

2.  **Custom Layers:**
    -   The output of the base model is connected to a new sequence of layers:
        -   `GlobalAveragePooling2D`: Reduces spatial dimensions and the number of parameters.
        -   `BatchNormalization`: Stabilizes and accelerates the training process.
        -   `Dense(512, activation='relu')`: A fully connected layer to learn high-level feature combinations.
        -   `Dropout(0.5)`: A regularization technique to prevent overfitting.
        -   `Dense(6, activation='softmax')`: The final output layer with 6 neurons (for 6 waste types) and a softmax activation function to produce prediction probabilities.

3.  **Fine-Tuning Strategy:**
    -   Initially, the layers of the base model are "frozen" (`trainable = False`).
    -   Subsequently, the **last 38 layers** of ResNet50V2 are "unfrozen" (`trainable = True`) and re-trained along with the custom layers to allow the model to learn more specific features from the waste dataset.

## Dataset
- **Source:** The dataset is from [Kaggle: Trash Type Image Dataset](https://www.kaggle.com/datasets/farzadnekouei/trash-type-image-dataset) by Farzad Nekouei.
- **Format:** JPEG images, 512x384 pixels.
- **Class Distribution:** The dataset contains 2527 images, divided into 6 categories:
    -   `cardboard`: 403 images
    -   `glass`: 501 images
    -   `metal`: 410 images
    -   `paper`: 594 images
    -   `plastic`: 482 images
    -   `trash`: 137 images
- **Challenge:** There is a significant class imbalance, with 'paper' being the majority class and 'trash' being the minority class.

## Methodology
The process is executed in the following steps within the notebook:
1.  **Data Splitting:** The data is split into a training set (80%) and a validation set (20%).
2.  **Data Augmentation:** The `ImageDataGenerator` is configured with parameters for:
    -   `rescale=1./255`: Normalizing pixel values.
    -   `rotation_range=45`: Random rotations.
    -   `width_shift_range=0.15`, `height_shift_range=0.15`: Horizontal/vertical shifts.
    -   `zoom_range=0.15`: Random zooming.
    -   `shear_range=0.05`: Shear transformations.
    -   And other techniques like flipping, brightness, and channel shifts.
3.  **Handling Class Imbalance:**
    -   Uses Scikit-learn's `compute_class_weight` function with the `'balanced'` mode.
    -   The weight formula is: `wj = n / (k * nj)`
    -   Where: `n` is the total number of samples, `k` is the number of classes, and `nj` is the number of samples in class `j`.
    -   These weights are passed to the `model.fit()` method to penalize misclassifications on minority classes more heavily.
4.  **Model Training:**
    -   **Optimizer:** Adam with a `learning_rate = 0.001`.
    -   **Loss Function:** `categorical_crossentropy`.
    -   **Metrics:** `accuracy`, `precision`, `recall`.
    -   The model is trained using the callbacks mentioned above.

## Results
-   The model achieved a validation accuracy of **92.09%**.
-   **Precision** and **Recall** were also high, averaging around 92% and 89%, respectively.
-   The **Confusion Matrix** shows that the model performs very well across most classes. The `trash` class (the minority class) had slightly lower accuracy but was still strong (76% precision, 81% recall), indicating that the class weighting method was effective.
-   The training plots show signs of **overfitting** as the training accuracy climbs high while the validation accuracy fluctuates. Regularization techniques and `EarlyStopping` helped mitigate this issue.
<img width="1145" height="395" alt="{FA69020A-2103-4878-A2A2-91335A1ED242}" src="https://github.com/user-attachments/assets/fe33fb70-5c4d-48c2-bf63-5317121a7485" />
<img width="1174" height="638" alt="{FCB5A04E-1061-4D94-8CC2-EC267C5BD357}" src="https://github.com/user-attachments/assets/4250fb47-4a2f-45a7-891e-6bb4693dc296" />

## Setup & Re-running the Experiment
This project is implemented as a Kaggle Notebook.

#### Environment
-   **Platform:** Kaggle Notebook
-   **Hardware:** GPU Tesla P100, 16GB VRAM
-   **Software:**
    -   Python 3.10.13
    -   TensorFlow 2.15.0
    -   NumPy 1.26.4
    -   Scikit-learn 1.2.2

#### How to Run
1.  **Load Data:**
    -   Create a new notebook on Kaggle.
    -   Add the [Trash Type Image Dataset](https://www.kaggle.com/datasets/farzadnekouei/trash-type-image-dataset) to the notebook. The data will be located at the path `/kaggle/input/trash-type-image-dataset`.
2.  **Copy Code:**
    -   Open the `dap391-garbageclassification.ipynb` file.
    -   Copy all the code from the notebook into your Kaggle notebook.
3.  **Execute:**
    -   Run the cells in the notebook sequentially.
    -   The training process will begin, and the results will be displayed.

## Authors
-   Ngo An
-   Nguyen Bao Han
-   Tran Hoang Nam
-   Nguyen Nguyen Phong

## References
-   **Paper:** *Waste Classification Using Fine-Tune ResNet50V2: A Deep Learning Approach*
-   **Dataset:** Farzad Nekouei. (2021). *Trash Type Image Dataset*. Kaggle.
-   **Architecture:** He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Identity Mappings in Deep Residual Networks*.
