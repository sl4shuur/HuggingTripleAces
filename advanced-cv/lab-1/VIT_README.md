# Part 2: local detail and global context

Use `vit_scratch.ipynb` and `vit_pretrained.ipynb` for the new study.
Earlier executed notebooks and all their outputs are in `archive/full_photo_v10/`.
The new notebooks use fresh `part2_scratch_multiscale` and
`part2_pretrained_multiscale` output folders. They do not change earlier results.

## Run on Kaggle

Build a new ZIP with `python package_soilcv.py`. Upload it as a new version of
soil-cv-utils. Version 10 contains the previous full-photo-only implementation.
Attach the new code version, competition data and `WANDB_API_KEY` Secret.
Select Internet and 2 x T4, then Save & Run All. The notebook checks the exact
source hash before training. This update does not publish any notebook or dataset.
W&B uses `public-team/soil-grain-size`, online. There are no CV-summary runs.

## Two image views

Each photo has all its 64 mm tiles at 288 px, about 4.5 pixels/mm, for local
texture. A separate full photo at 2 pixels/mm gives context. The same encoder
reads both views. Local features have overlap-corrected area weights. Local and
global features are joined before the curve head. Photos have equal weights.
One soil has one valid grain-size curve and one EMD loss.

All tiles and all photos are used. No grid area is skipped. Both views still
resize images; local tiles do not retain all native iPhone detail. Source PPM
still assumes the originals were resized without an unknown crop.
Physical field of view, overlap between photos and foreground masks are left
for the SSL stage, as requested.

`image_geometry.csv` records `spatial_coverage`, `cropped_source_area`,
`linear_scale_ratio` and `pixel_count_ratio` for the global view.
`tile_spatial_coverage.csv` records grid union coverage and local resize ratios.
Coverage means area seen. It does not mean all source information is retained.

Padding repeats the edge pixels. Scratch attention uses valid area as a key
weight and masks fully padded tokens. Partial patches can still be affected.
Swin keeps its pretrained window attention and uses valid area for final pooling.
Each fold tests one held-out soil with an extra padding strip, without changing
soil pixels or PPM. `results/padding_checks.csv` shows the prediction EMD change.
This is a sensitivity check, not a camera robustness score.

## Experiments

Scratch has three recipes: patch 16, patch 32, and patch 16 without warmup.
Pretrained Swin has frozen, last-stage and full tuning with both image views.
It also has three tile-only MIL recipes: frozen, constant LR and warmup/cosine.
The tile recipes follow the CNN 64 mm / 288 px input and sample pooling.
Constant and cosine use AdamW, backbone LR 1e-4, head LR 1e-3, weight decay .01,
16 epochs and patience 5; only the schedule changes. Frozen uses head LR .003.
These choices follow the earlier cosine/constant/frozen CNN study.

Keep the same four sample folds. For each notebook, the baseline also has
Motorola-only and Samsung-only training runs on those same folds. Same-camera
and cross-camera EMD use the same paired held-out soils. They are separate from
the main ranking. Both validation cameras still affect epoch selection.
CNNs are not retrained.

## Checkpoint rule

Best checkpoint and early stopping use validation EMD only, from epoch one.
Camera disagreement stays a separate W&B metric; it cannot reject a better EMD.
Keep the top three unique epochs by EMD, with their weights, validation curves
and SHA256 hashes. Use ranks 1 and 2 for predictions. Also include rank 3 only
when its camera disagreement is strictly below rank 2. A tie does not include it.

Use the same selected epoch weights for OOF, camera curves and test curves.
Then average all four folds of the same recipe. Best-only OOF and CV are also
saved for comparison. The optional third checkpoint is chosen with validation
data too; these scores are not an independent estimate of that choice.

Partial runs keep their optimizer checkpoints for explicit resume. A completed
fold keeps all three inference weight files, without a duplicate best-weight copy. Its large
best/last optimizer snapshots are removed after success to limit Outputs size.
Older studies are never pruned. Config and `checkpoint_summary.json` record this.

## Outputs and inference

Each study saves OOF and test curves, checkpoints, camera and padding checks,
configs, histories, compute records, source, environment and a SHA256 manifest.
`VIT_RESULTS.md` shows epoch-ensemble CV and best-only CV separately.
`results/inference_ensembles.json` records exact four-fold W&B artifact groups.
`/kaggle/working/submission.csv` uses the best standard recipe. It is not submitted.

Use `from soilcv.inference.runner import run_checkpoints` for W&B inference.
For a saved fold, `soilcv.vit.inference.predict_fold` applies the same 2/3-epoch
rule. Four-fold groups are checked for the same recipe, source and study.

The earlier full-photo v10 study has a separate
[results report](VIT_RESULTS_FULL_PHOTO_V10.md), including all five public scores.
