# Hypotheses before the MIL run

The old tile-loss study is a reference. Its notebooks, results and checkpoints stay in the earlier Kaggle notebook versions and code dataset version 3. Local history is in archive/CNN_TILE_HISTORY.md. Its scores are not scores of the new MIL run.

Use the same four sample folds. Every photo of a sample stays in one fold.
One training item is one complete soil sample. Use all grid tiles of all its photos in train, validation and test. Do not randomly drop tiles or photos.

Use 64 mm crops and a full-image grid. Edge tiles overlap when needed. Check the union of rectangles for every photo. Use overlap-corrected area weights within each photo, then equal photo weights within each sample. Pool CNN features before the curve head. Compute EMD once per sample.

Each resolution gets its own cache directly from the same native rectangles with LANCZOS. There is no second resize. 224 and 288 use the same photos, bag weights, updates, folds, and physical area.

Budget: 16 epochs, early stop after 5 epochs without improvement. Two complete sample bags per optimizer update. Frozen BN statistics for all ResNet runs. AMP, checkpointed encoder chunks of 8 tiles, and deterministic algorithms. No extra split search or seed search in this block.

| ID | Recipe | Prediction before training |
| --- | --- | --- |
| H-MIL-01 | mil-r18-frozen-288 | Frozen pooled features can use the full view. They may be weaker than full tuning. |
| H-MIL-02 | mil-r18-288 | Sample loss and all-photo pooling may help mixed soils. Full coverage may also add background noise. The old tile score is a historical reference, not a clean loss-only ablation. |
| H-MIL-03 | mil-r18-224 | A direct native-to-224 resize may keep enough texture. Expect a small score gap and less memory, not a sure gain at 288. |
| H-MIL-04 | mil-r18-cosine | Warmup and cosine may help MIL stability. The old constant-LR result may not carry over to a new objective. |
| H-MIL-05 | mil-convnext-288 | Stronger features may help, but extra capacity may overfit 18 train samples per fold. Expect more GPU time. |
| H-MIL-06 | Paired camera check on held-out samples | Motorola and Samsung predictions may disagree. Measure each camera's label error and their prediction disagreement on the same paired samples. This cannot prove transfer to iPhone. |

Hypotheses stay separate from scores. Each run writes CNN_RESULTS.md and cnn_results.csv with a run ID and code/data hashes. The main numeric source is that run's CSV. No leaderboard submission is sent.
