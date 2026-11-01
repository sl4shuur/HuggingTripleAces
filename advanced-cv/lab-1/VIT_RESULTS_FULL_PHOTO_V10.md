# Full-photo ViT / Swin results (utils v10)

The later local + global study and the combined CNN / ViT comparison are in
[PART2_RESULTS.md](PART2_RESULTS.md). Keep the v10 scores below separate.

Public scores were checked with the Kaggle competition API on 2026-11-01. All five submissions are COMPLETE. Lower EMD is better. Private scores are not available.

| Recipe | OOF EMD | Fold std | Public EMD | Camera disagreement | Fit GPU hours | Submission ID |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| swin-full | 37.75176 | 20.21450 | 57.66586 | 10.388 | 0.206 | 57882716 |
| swin-last-stage | 38.21990 | 19.38617 | 52.80828 | 14.645 | 0.137 | 57882730 |
| swin-frozen | 42.48306 | 21.34909 | 47.37640 | 16.994 | 0.119 | 57882825 |
| vit-scratch-no-warmup | 79.17321 | 15.68638 | 89.85861 | 9.293 | 0.103 | 57882831 |
| vit-scratch-p16 | 80.06121 | 14.27270 | 82.50369 | 3.111 | 0.117 | 57882838 |

## What happened

Frozen Swin has the best public EMD: 47.37640. Full tuning has the best OOF EMD: 37.75176, but its public EMD is 57.66586.
All three Swin recipes beat the two submitted scratch ViTs on both OOF and public EMD.
Lower camera disagreement did not give a better public score in these three Swin runs. It is a diagnostic, not proof of transfer to iPhone.
The OOF and public rankings differ. This small public set cannot explain the cause. Do not choose new settings only from these public scores.

## Exact prediction setup

Each CSV is an equal average of four sample-fold models of one recipe. All twenty exact W&B artifact versions are in submissions/full_photo_v10/provenance.json.
These are the full-photo inputs at 2 estimated px/mm, with the earlier Pareto checkpoint rule. They do not use the new local + global inputs or the new 2/3-epoch ensemble.
The training Outputs are from version 2 of the scratch and pretrained notebooks, with code dataset v10. The matching study IDs are 20261101T170306Z-eeb6d8f69e and 20261101T170501Z-a596a117ff.
Inference notebook: https://www.kaggle.com/code/denysmykhailov/soil-grain-size-inference?scriptVersionId=364651511
The verified inference run used CPU. Training-time test predictions used GPU AMP. Across all five files, the largest change was 0.02542 percentage points; mean prediction-to-prediction EMD was 0.01315 to 0.02056. IDs, columns and valid curves matched. These are numerical differences, not new leaderboard scores.
Kaggle filenames and submission IDs were matched through the API, not guessed from the screenshot order. Submitted CSV hashes are recorded separately from the earlier GPU CSV hashes.

## Limits

OOF covers 24 labelled soils. Every soil stays in one of the same four folds. Fold std is not a confidence interval.
The full-photo resize loses fine detail. The earlier checkpoint rule could reject a better validation EMD when camera disagreement grew.
These results belong to the earlier study. Keep them separate from the new multiscale and EMD-only selection runs.
The scratch patch-32 recipe was not submitted in this batch. Its OOF result remains in the archived executed notebook.

## Files

Executed notebooks with original outputs: archive/full_photo_v10/.
Submission mapping and hashes: submissions/full_photo_v10/provenance.json.
Machine-readable metrics: submissions/full_photo_v10/results.json.
