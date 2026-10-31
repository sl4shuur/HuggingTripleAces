# Computer Vision Assignment 4. Human Pose Estimation: COCO Keypoint Detection

## Assignment Overview and Motivation

- **Format:** Individual or group (max 3 students) with offline defense
- **Grading:** 17 points
- **Timeline:** Assignment #4 available on 15th April, code submission deadline till 28th April, and offline defense deadline till 1st May

While completing this assignment, you will develop and strengthen the following skills:

- Approaching a real-world keypoint detection problem with the standard COCO benchmark and OKS metric.
- Understanding the difference between top-down and bottom-up pose estimation pipelines.
- Using pretrained pose models such as YOLO-Pose, Keypoint R-CNN, and HRNet, and fine-tuning them.
- Implementing and understanding the Object Keypoint Similarity (OKS) evaluation metric.
- Exploring self-supervised representation learning, such as I-JEPA and DINOv2, for keypoint regression.
- Performing deep ablation studies on different models and methods to choose the best approach.

## Task

**Task:** Given an image and a person bounding box, predict the `(x, y, visibility)` for each of the 17 COCO body keypoints:

- nose
- eyes
- ears
- shoulders
- elbows
- wrists
- hips
- knees
- ankles

**Dataset:** [COCO 2017 Keypoints](https://www.kaggle.com/datasets/yanplayz08/coco-subset-for-pose-estimation) with human-annotated ground truth, available on Kaggle.

**Metric:** mean OKS, or Object Keypoint Similarity, which is the standard COCO evaluation metric for pose estimation.

Take part in the dedicated [Kaggle In-Class competition](https://www.kaggle.com/t/90a5bda7f3ed4e59828133cf76e8d6ff).

## EDA and Metric Analysis

- Perform EDA:
  - distribution of number of keypoints per annotation
  - person scales
  - occlusion patterns

- Analyze the OKS metric:
  - explain the formula
  - explain the role of per-keypoint sigmas
  - explain why face keypoints are harder than body keypoints

- Compute and visualize OKS for the bbox-center baseline.
  - Which keypoints contribute most to the score?

- Discuss:
  - AP@50
  - AP@75
  - AP@50:95

Explain what each threshold reveals about model quality.

## Validation

- Design a validation strategy.
  - Split by `image_id`, not by annotation, to avoid leakage.

- Use the provided `oks_scoring.py` to compute local OKS on your validation set.

- Discuss correlation between local OKS and Kaggle leaderboard RMSE.

## Approach A — Pretrained Pose Models

- Run inference with pretrained models:
  - YOLO-Pose
  - Keypoint R-CNN
  - HRNet

- Match model detections to test annotations via IoU with provided bounding boxes.

- Analyze per-keypoint accuracy:
  - Which keypoints are hardest?
  - Visualize failure cases.

- Prepare an ablation study across model sizes and architectures.

## Approach B — Fine-Tuning

- Fine-tune a keypoint detection model on the training data.

- Experiment with augmentation:
  - horizontal flip, with left/right keypoint swap
  - scale jitter
  - color jitter

- Implement Test-Time Augmentation, or TTA:
  - flip + average keypoints

- Compare fine-tuned vs pretrained results.

- Prepare an ablation study.

## Approach C — Self-Supervised Features (Bonus)

- Extract features from I-JEPA or DINOv2 on person crops.

- Train an MLP to regress normalized keypoints.

- Compare with Approaches A and B.

- Discuss when self-supervised features are competitive with task-specific pose models.

## Final Report

- Comprehensive report:
  - data insights
  - models
  - hyperparameters
  - OKS scores, CV + LB
  - ablation tables
  - failure analysis

- Reports may be prepared in LaTeX or Markdown, for example `README.md`.

## Notes

- You **may** use pretrained weights:
  - ImageNet
  - COCO

- You **may not** add additional training images beyond `train.csv`.

- You may use pseudo-labeling approaches, but only on training data, **not test data**.

- Your final report **must** include both local OKS and Kaggle LB scores for all experiments.

- If you do not reach the baseline on Private Leaderboard, the maximum score is **2.5/5 points**.

- Public LB is approximately **16%** of test data — **do not rely on it**.

- Do not hesitate to include failed ideas or experiments in your report.

Documenting unsuccessful attempts is a strong indication that you systematically explored different approaches and gradually arrived at the best-performing solution, rather than relying on a “lucky shot.”

Moreover, analyzing failed approaches can often lead to valuable insights about the dataset, the task itself, and the behavior of deep learning algorithms.

Such reflections may also highlight potential directions for future improvements and demonstrate a deeper understanding of the problem-solving process.

- Keep your EDA concise.

Avoid overloading it with excessive plots, histograms, or tables. Focus only on key aspects of the data, such as missing values and target distribution.

If a part of your EDA does not lead to meaningful conclusions, you likely do not need it. For example, full pairwise correlation matrices are unnecessary.

Do **not** follow the typical Kaggle-style EDA — most examples are far too verbose and unfocused.

- Prepare your presentation carefully.

A messy speech with too much code or too many files makes it nearly impossible for the audience to follow.

Think in advance about what you want to highlight.