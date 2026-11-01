# Hypotheses

## Hypotheses before the MIL run

This run uses full-coverage sample-level MIL. Only MIL checkpoints are supported by inference.

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

## Part 2 hypotheses before training

Keep the same four saved sample folds and seed 42 + fold. Do not train CNNs again.
Use full photos at 2 estimated pixels/mm for every camera. Keep the aspect ratio.
Resize once with LANCZOS. Pad only the right and bottom edge for patch alignment.
No photo area is cropped. Resizing still loses fine grain detail. The source PPM is
an estimate under the same no-crop assumption used in EDA.

Scratch ViT uses the Lecture 1 design: patch embedding, CLS, QKV attention,
pre-norm blocks and an MLP. Add 2-D sine/cosine positions for variable grids.
Use all photo CLS vectors, equal photo weights, and one curve/loss per sample.
Swin-T uses local shifted windows and the same full-photo scale. It pools spatial
features and then photo features. It does not use a CLS token.

| ID | Comparison | Prediction before training |
| --- | --- | --- |
| H-VIT-01 | Scratch ViT, patch 16, warmup + cosine | Few independent soils may cause overfit. Expect worse EMD than pretrained Swin and the best CNN MIL. Global context alone may not help. |
| H-VIT-02 | Scratch patch 32 vs patch 16 | Larger patches use fewer tokens and less time. They may lose useful grain detail. Only patch size changes. |
| H-VIT-03 | Scratch without warmup | Removing warmup may make early updates less stable. Only warmup changes. |
| H-VIT-04 | Frozen Swin-T | Pretraining may help on this small set. Frozen features may be more stable between cameras. |
| H-VIT-05 | Tune the last Swin stage | A small update may help texture features without large feature changes. It may beat frozen Swin, but this is uncertain. |
| H-VIT-06 | Tune all Swin stages with layer LR decay | More flexibility may lower train error but hurt held-out soils or cameras. Use a lower backbone LR. This is a tuning recipe comparison, not a one-factor ablation. |
| H-VIT-07 | Camera pair in checkpoint selection | Requiring no worse validation EMD and no worse paired camera disagreement may select an earlier model. It may stop too early if the two metrics disagree. Neither metric proves iPhone transfer. |
| H-VIT-08 | Motorola-only and Samsung-only train protocols | On the same paired held-out soils, cross-camera EMD may exceed same-camera EMD. The gap can be negative or noisy. Do not change the folds. |

Use EMD and camera disagreement as a pair, without a weighted sum. Save a new
best checkpoint only if neither metric gets worse and at least one gets better.
The first eligible checkpoint is epoch 4, after three training epochs. This fixed
burn-in avoids selecting the near-constant initial curve only because its camera
disagreement is near zero. Use the same burn-in in every recipe, also without warmup.
Stop after the patience budget without such an improvement. Keep both raw
metrics and the selected pair in W&B. This rule can reject a better EMD checkpoint
if camera disagreement gets worse. Report that limit with the results.

Plan three scratch recipes (32 epochs, patience 7) and three Swin recipes
(24 epochs, patience 6). Use AdamW, no decay on bias/1-D parameters, scale-safe
flips/quarter turns and mild colour jitter. Use two complete sample losses per
update, AMP, and photo encoder checkpointing. Each GPU runs an independent fold.

For each notebook, run two camera directions for its baseline recipe, four folds
each. Train on only the source camera in training folds. Validation contains both
cameras on paired held-out samples. Report E_same, E_cross and their difference
on exactly the same samples. Both cameras affect checkpoint selection; this is
a training-camera stress test, not an untouched device test or an iPhone proxy.

Save each recipe's complete OOF curves and a true four-fold test ensemble.
Do not rank camera diagnostic runs as ordinary CV recipes. Keep earlier outputs.
No automatic competition submissions or W&B CV-summary runs are planned.

## Part 2 revision: local detail and global context

This plan replaces the full-photo-only input and Pareto checkpoint rule above.
Keep the old plan and all earlier results as a record. Do not retrain CNNs.
Keep the same four soil folds and seed 42 + fold.

Each photo now has two views. All 64 mm grid tiles at 288 px give local detail
at about 4.5 px/mm. The full photo at 2 px/mm gives global context only.
Both views use the same encoder. Pool local features with overlap-area weights,
join local and global features, then average photos before one curve head.
Do not call the global view a high-detail image. Local tiles also do not keep
all native iPhone detail. They keep the physical scale used by the CNN study.

| ID | Comparison | Prediction before training |
| --- | --- | --- |
| H-VIT2-01 | Local + global vs the earlier full-photo-only plan | Local inputs may keep grain texture that 2 px/mm removed. Context may help large structures. This is a new input experiment, not proof that ViT beats CNN. |
| H-VIT2-02 | Scratch patch 16 vs 32, with the same two views | Patch 16 may keep more useful local texture, at a higher attention cost. |
| H-VIT2-03 | Scratch warmup vs no warmup | Warmup may help early updates. Keep all other input and training settings fixed. |
| H-VIT2-04 | Frozen, last-stage, full-tuned multiscale Swin | Pretraining may help with few soils. More tuning can still overfit. |
| H-VIT2-05 | Swin MIL, only 64 mm / 288 px tiles | Use the CNN physical input and sample pooling. Compare frozen, constant LR, and warmup/cosine. The constant and cosine recipes differ only in schedule. |
| H-VIT2-06 | Prediction after a small padding change | Replicate padding and valid-area handling may reduce artificial border effects. Swin can still use padding inside attention. Measure EMD between the two predictions. |
| H-VIT2-07 | Best EMD vs the selected epoch ensemble | The top two EMD checkpoints may be more stable than one epoch. Include rank three only when its camera disagreement is lower than rank two. This is a validation-based heuristic, not an independent robustness test. |

Best checkpoint and early stopping use validation EMD only, from epoch one.
Camera disagreement stays in W&B as a separate diagnostic. Save the top three
unique epochs by EMD. Use the top two for predictions. Use the third as well
only if its camera disagreement is strictly lower than the second's.
The same chosen epoch weights must produce OOF and test curves. Then average
four fold predictions. Save best-only OOF curves too, so the heuristic is visible.
This uses validation twice and can make the reported CV optimistic.

Report spatial coverage separately from resolution loss: cropped source area,
linear scale ratio and output/native pixel count ratio. Padding is not soil.
Scratch attention accounts for valid patch area. Keep a small padding stress
check for Swin instead of rewriting its pretrained attention blocks.

Keep the existing camera protocols and sample folds. Physical field of view,
photo overlap and foreground segmentation are deferred to the SSL stage.
Use a fresh output folder for this revision. Keep the three candidate weight
files; full optimizer snapshots are temporary while a fold is training.
No notebooks or datasets are published in this revision. Smoke tests are short,
offline for W&B, and use only a small part of the data.

## Observed results: full-photo v10

The completed full-photo study and its five public submissions are recorded in
[VIT_RESULTS_FULL_PHOTO_V10.md](VIT_RESULTS_FULL_PHOTO_V10.md).
These results do not belong to the later multiscale revision.

Pretrained Swin beat the submitted scratch ViTs on OOF and public EMD. This fits
the prediction that pretraining helps on few independent soils. It does not
isolate pretraining alone, since the architectures also differ.

Full Swin tuning had the best OOF EMD. Frozen Swin had the best public EMD,
47.37640. Camera disagreement did not rank these public results correctly.
The public set is small, so this does not prove why the rankings changed.
The proposed benefit of warmup was not supported by the scratch OOF ranking:
no warmup scored better, while patch 16 with warmup scored better on public LB.
There is no clean, stable win for warmup in this study.

The new local-detail inputs and EMD-only stopping remain separate experiments.
Do not treat these earlier public scores as results of those changes.
