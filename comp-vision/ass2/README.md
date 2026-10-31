# Data Splitting Strategy

To evaluate the models reliably, we use **5-fold cross-validation**.

However, a standard random split would cause **data leakage**, because multiple nodules can belong to the same patient.

To avoid this issue, we use **StratifiedGroupKFold** with the following constraints:

* **Grouping by patient_id** ensures that samples from the same patient never appear in both training and validation sets.
* **Stratification by label_id** preserves the class distribution across folds.

This approach allows us to:

1. Prevent patient-level data leakage.
2. Maintain balanced class distributions.
3. Obtain stable performance estimates.

Each fold uses:

* **80% of the data for training**
* **20% of the data for validation**

The reported performance metrics correspond to the **average Macro F1-score across the five folds**.

---

# Model Architectures

We compare three different approaches for processing the CT scans:

### 1. 2D CNN Model

The first approach uses a **2D convolutional neural network**.

Only the **central slice** of the cropped nodule region is used as input.
This slice is passed through a pretrained **EfficientNet-B3** model.

Advantages:

* Lower computational cost
* Faster training
* Ability to leverage pretrained ImageNet weights

However, this approach ignores most of the 3D spatial context of the CT scan.

---

### 2. 2.5D CNN Model

The second approach uses a **2.5D representation**.

Instead of a single slice, we extract **multiple adjacent slices** around the nodule and stack them as input channels.

In our implementation:

* **5/7 slices** are extracted around the center
* These slices form a multi-channel input
* The stacked slices are processed using **EfficientNet-B3**

This approach provides more spatial context while maintaining the efficiency of 2D CNN architectures.

---

### 3. 3D CNN Model

The third approach uses a **3D convolutional neural network**.

Instead of using slices, the model processes the entire **3D cropped volume**.

We use **ResNet50 with 3D convolutions** implemented in MONAI.

This model can fully exploit the spatial structure of the CT data but has significantly higher computational requirements.

---

# Ablation Study Results

The performance of each model is evaluated using **5-fold cross-validation**.

| Model   | Input Representation | Architecture    | Mean CV Macro F1 | Public
| ------- | -------------------- | --------------- | ---------------- | ---------------- |
| Model 1 | 2D central slice     | EfficientNet-B3 | 0.6351           |  0.708
| Model 2 | 2.5D (5 slices)      | EfficientNet-B3 | **0.6414**       |0.707
| Model 2 | 2.5D (7 slices)      | EfficientNet-B3 | 0.6119           |0.727
| Model 3 | 3D volume            | ResNet50 (3D)   | 0.6063           |0.660

---

# Results Analysis

The **2.5D model with five slices achieved the best cross-validation result**, with a Macro F1-score of **0.6414**. This suggests that using a small number of neighboring slices helps the model capture additional spatial information around the nodule.

At the same time, increasing the number of slices to **seven** leads to a lower cross-validation score (**0.6119**), although it achieves the **highest public leaderboard score (0.727)**. One possible explanation is that additional slices introduce more noise into the training process. However, they may still help the model generalize better on the hidden test data used in the leaderboard evaluation.

The **2D model**, which uses only the central slice of the nodule, performs surprisingly well. Its cross-validation score (**0.6351**) is close to the best result. This indicates that the central slice already contains most of the important visual information needed for classification.

The **3D model (ResNet50)** achieved the lowest performance (**0.6063**). Although 3D networks are designed to process volumetric data, they are also much more complex and require more training data. In this case, the dataset size may not be large enough to fully benefit from a 3D architecture.

Another important observation is the difference between **cross-validation results and public leaderboard scores**. While cross-validation helps estimate model performance locally, leaderboard scores depend on a hidden test dataset. Because of this, the ranking of models on the leaderboard may differ slightly from local validation results.

---

# Final Model Choice

Based on the experiments, the **2.5D EfficientNet-B3 model with five slices** was selected as the final model.

This model achieved:

* the **best cross-validation performance**
* strong leaderboard results
* a good balance between accuracy and computational efficiency.
