# Hypotheses

## Part 1: CNNs

Written before model training. Setup-only W&B connectivity has been checked; no CNN scores were used to choose these hypotheses.

Data: 24 train soil samples. Use the original four sample folds, seed 42. Score one curve per sample. Tiles are 64 mm using estimated PPM. Train only on each fold's train samples. Test labels are not available.

Budget per run: up to 24 epochs, early stop after six epochs with no sample-level EMD improvement. Each GPU runs a separate fold. This uses both T4 GPUs without changing batch size through DataParallel.

The main input is 288 x 288 pixels. Use ImageNet mean/std with no extra pretrained centre crop. Safe augmentation uses 90-degree turns, flips, and mild colour/light changes. Use balanced sample/photo/tile sampling and the same three crop locations. The output is the soilcv mass-bin curve head. Its bias starts near the fold train-mean curve, using train labels only.

| ID | Experiment | Prediction before training |
| --- | --- | --- |
| H-CNN-00 | Two-epoch ResNet-18 smoke test, fold 0 | The GPU/AMP path gives finite gradients, valid curves, complete validation IDs, and a working online W&B run. This is a pipeline check, not a model ranking. |
| H-CNN-01 | `cnn-r18-frozen`: frozen ResNet-18, AdamW head, warmup + cosine, light augmentation, 288 px | Frozen image features should improve on a constant curve, but may miss fine-soil differences. |
| H-CNN-02 | `cnn-r18-adamw`: full ResNet-18 tuning, AdamW, frozen BN stats, warmup + cosine, light augmentation, 288 px | Tuning should help over the frozen encoder, but the gain may be small on only 18 train samples per fold. |
| H-CNN-03 | `cnn-r18-sgd`: same input and frozen BN, SGD + momentum with its own LR recipe | AdamW should learn faster in this short budget. This compares optimizer/LR recipes, not the optimizer alone. |
| H-CNN-04 | `cnn-r18-train-bn`: same as H-CNN-02 but update BN running stats | Updating BN on small, related tile batches may make validation less stable and worse. |
| H-CNN-05 | `cnn-r18-no-aug`: same as H-CNN-02 but no random augmentation | It may fit train tiles faster, but validation should be worse due to camera/light changes. |
| H-CNN-06 | `cnn-r18-constant-lr`: same as H-CNN-02 but constant LR, no warmup | Warmup + cosine should give more stable tuning; the constant recipe may overfit. |
| H-CNN-07 | `cnn-r18-224`: same as H-CNN-02 but resize the same 64 mm tile to 224 px | Lower resolution should cost some fine texture, but may reduce overfitting and compute. It may still be competitive. |
| H-CNN-08 | `cnn-convnext-tiny`: pretrained ConvNeXt-Tiny, AdamW, warmup + cosine, light augmentation, 288 px | Stronger features and LayerNorm may help, but extra capacity may overfit. Expect more compute than ResNet-18. |

Common fine-tuning LR: encoder 0.0001, head 0.001 for AdamW. SGD: encoder 0.001, head 0.01, momentum 0.9. Frozen head LR: 0.003. Weight decay: AdamW 0.01 for matrix weights; SGD 0.0001. No decay on bias/1D parameters. Three warmup epochs, cosine ending at 5% of the base LR. Batch size 16, fp16 AMP, clip gradient norm 5.

Exact pretrained weights: torchvision ResNet18_Weights.IMAGENET1K_V1 and ConvNeXt_Tiny_Weights.IMAGENET1K_V1. Save runtime library versions, weight-file hashes, configs, OOF curves, compute logs, and checkpoint artifact handles.

Use CV to choose the final CNN recipe. Public and private leaderboard scores remain unreported until a real competition submission is scored. No test-specific IDs or paths change model behaviour.

## Results

The block is complete. The predictions above are kept unchanged. The pre-run commit is `804c059`.

Best fixed-fold OOF EMD: **32.445**, fold standard deviation **14.044**, with `cnn-r18-constant-lr`. The fold median baseline is **85.213**. All 32 planned fold jobs and the separate smoke test completed. Each worker used one Tesla T4.

- Frozen ResNet EMD is 36.315; tuned ResNet EMD is 34.006. Tuning helps here, but the gain is small.
- AdamW recipe: 34.006; SGD recipe: 35.240. The score gap does not isolate optimizer choice or prove learning speed.
- Frozen BN stats: 34.006; updated BN stats: 39.271. This fits the concern about running statistics from related tile batches.
- Light augmentation: 34.006; no augmentation: 35.745. Light augmentation helps in this recipe.
- Warmup/cosine: 34.006; constant LR: 32.445. The schedule prediction was wrong. A full LR from the start works better in this short run; many best checkpoints are early.
- 288 px: 34.006; 224 px: 34.087. Compare this gap with fold variation before claiming that more pixels help.
- ConvNeXt-Tiny: 34.628; matched ResNet recipe: 34.006. The expected architecture gain did not hold. More capacity and different pretrained features do not ensure a gain on 18 train samples per fold.

H-CNN-00 passed: finite GPU training, valid curves, complete sample IDs, and online W&B tracking. H-CNN-03 gives the predicted score order, but this alone does not confirm faster learning. H-CNN-07 has only a 0.080 EMD gap; this is too small for a strong resolution claim.

These are one-seed observations on 24 samples. The fold standard deviation is not a confidence interval. Model and epoch selection can overfit these folds. The test ensemble averages four fold models; OOF scores evaluate one held-out model per sample.

The next Run All saves these files in Kaggle Outputs: `part1_cnn/results/cnn_report.md`, `part1_cnn/results/cnn_results.csv`, and `part1_cnn/results/cnn_checkpoints.csv`. Public LB: not submitted. Private LB: not available.

## Kaggle delivery

The CNN recipes now live in the code dataset under `soilcv/cnn/`. The notebook can run in a fresh Kaggle session with Run All. This changes the launch and file layout, not the recipes above. The output folder contains the full study and its environment. Each fold is tracked online; no extra CV-summary run is created. The earlier study was archived before changing this layout.
