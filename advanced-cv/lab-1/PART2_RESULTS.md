# Part 2: ViT results and review

This report uses completed runs only. No new training or inference was run for
this report. Kaggle submission receipts were read on 2026-11-07.
Lower EMD is better in every table. Human text is kept short and simple.

| Part 2 requirement | Evidence / status |
| --- | --- |
| Own ViT, then pretrained model | Completed scratch ViT and Swin-T studies |
| Patch size and warmup experiments | Completed comparisons in both studies |
| Layer decay, weight decay, drop path, augmentation | Implemented; their effects were not tested one at a time |
| Predict before runs, explain after runs | Original hypotheses kept; outcomes below and in HYPOTHESES.md |
| CV ± std, LB and GPU cost | Tables below; missing LB stays empty |
| Compare CNN and ViT optimisation | Recipes, measured differences and reasons below |
| Ablation study | Patch size, schedules and best-only / epoch ensemble; limits are explicit |

## What is done

We trained our own small ViT from scratch, then used pretrained Swin-T.
We kept the same four soil folds. Photos of one soil never cross folds.
There are 24 labelled soils, with six validation soils per fold.
Many tiles do not mean many independent soils.

There are two Part 2 studies. Keep them separate:

- **Full-photo v10:** full photos at 2 estimated px/mm; one checkpoint per fold,
  chosen by EMD and camera disagreement after a three-epoch burn-in.
- **Local + global:** all 64 mm / 288 px tiles, plus a full photo at 2 px/mm.
  Tile-only Swin recipes use only the local view. Best epochs and early stopping
  use EMD. Each fold uses its top two epochs, with an optional third epoch.

A sample gets one curve and one loss. Local tiles use overlap-area weights.
Photos have equal weights. Test CSVs average all four folds of one recipe.
OOF uses only the held-out fold for each soil; it does not average four folds.
The new OOF uses the same selected epochs as test prediction.

## New study: all nine recipes

CV is OOF EMD. With four equal folds, it is also the mean fold EMD.
The ± value is fold standard deviation, not a confidence interval.
GPU hours sum fit time across four folds. Peak MB is the largest recorded GPU
allocation across those folds. These are not total wall time or all pipeline costs.
The camera-only diagnostic runs have a separate compute budget below.

| Recipe | Input | CV ± std | Public EMD | Camera disagreement | Fit GPU h | Peak MB |
| --- | --- | --- | --- | --- | --- | --- |
| swin-tiles-constant | tiles | 32.088 ± 18.257 | 74.77702 | 18.073 | 1.138 | 1845.0 |
| swin-tiles-cosine | tiles | 33.827 ± 18.663 | 74.01519 | 22.974 | 1.383 | 1845.2 |
| vit-scratch-p32 | multiscale | 33.918 ± 11.833 | 65.84300 | 40.658 | 1.151 | 521.2 |
| swin-full | multiscale | 34.053 ± 17.176 | 58.66346 | 11.876 | 1.543 | 1867.4 |
| swin-tiles-frozen | tiles | 34.596 ± 13.604 | 45.21894 | 17.393 | 0.592 | 534.0 |
| vit-scratch-no-warmup | multiscale | 34.887 ± 9.164 | — | 39.531 | 1.153 | 510.3 |
| swin-last-stage | multiscale | 34.940 ± 17.353 | — | 10.966 | 0.994 | 757.7 |
| swin-frozen | multiscale | 39.453 ± 14.627 | — | 14.951 | 0.578 | 588.3 |
| vit-scratch-p16 | multiscale | 39.520 ± 22.677 | — | 33.764 | 1.233 | 510.3 |

A dash means no matching four-fold submission was verified. It does not mean
zero error. Private LB is unavailable for every row.

Best new OOF: **swin-tiles-constant, 32.088**.
Best public score among the five new submissions: **swin-tiles-frozen, 45.21894**.
The earlier best public score was frozen full-photo Swin at 47.37640.
These are different models. Keep OOF as the model choice rule.
The public set is small and uses different cameras. The ranking difference alone
cannot tell us the cause. Do not tune new settings to these five public scores.

## Existing submissions: exact mapping

Kaggle API receipts link the files to scores. All ten Part 2 submissions are
complete. The five new receipts point to a notebook named `BirdCLEF submit`.
That is its title; the receipts were read from the soil competition.
The name does not make these bird competition results.

| Study | Recipe | Public EMD | Submission ID |
| --- | --- | --- | --- |
| full_photo_v10 | vit-scratch-no-warmup | 89.85861 | 57882831 |
| full_photo_v10 | vit-scratch-p16 | 82.50369 | 57882838 |
| full_photo_v10 | swin-full | 57.66586 | 57882716 |
| full_photo_v10 | swin-last-stage | 52.80828 | 57882730 |
| full_photo_v10 | swin-frozen | 47.37640 | 57882825 |
| local_global | vit-scratch-p32 | 65.84300 | 57924487 |
| local_global | swin-tiles-constant | 74.77702 | 57924477 |
| local_global | swin-tiles-cosine | 74.01519 | 57924485 |
| local_global | swin-full | 58.66346 | 57924509 |
| local_global | swin-tiles-frozen | 45.21894 | 57924515 |

New inference source: [Kaggle version 365048528](https://www.kaggle.com/code/denysmykhailov/birdclef-submit?scriptVersionId=365048528).
Earlier inference: [Kaggle version 364651511](https://www.kaggle.com/code/denysmykhailov/soil-grain-size-inference?scriptVersionId=364651511).
Full filenames, receipt IDs and UTC times are in [part2_results.json](part2_results.json).

## Comparison with CNN and the first ViT study

These are measured pipeline results on the same folds. They are not a pure
architecture test: inputs, epoch budgets and checkpoint rules differ.
The CNN uses local tiles and one best epoch. The first ViT study uses full photos.
The new study changes both inputs and selection. Its table uses epoch ensembles.

| Study / recipe | CV ± std | Public EMD | Fit GPU h |
| --- | --- | --- | --- |
| cnn_mil / mil-r18-cosine | 26.157 ± 9.500 | — | 0.842 |
| cnn_mil / mil-r18-288 | 30.977 ± 16.554 | — | 0.706 |
| cnn_mil / mil-convnext-288 | 32.825 ± 20.826 | — | 0.926 |
| cnn_mil / mil-r18-224 | 34.182 ± 16.168 | — | 0.450 |
| cnn_mil / mil-r18-frozen-288 | 37.783 ± 15.439 | — | 0.457 |
| full_photo_v10 / swin-full | 37.752 ± 20.215 | 57.66586 | 0.206 |
| full_photo_v10 / swin-last-stage | 38.220 ± 19.386 | 52.80828 | 0.137 |
| full_photo_v10 / swin-frozen | 42.483 ± 21.349 | 47.37640 | 0.119 |
| full_photo_v10 / vit-scratch-no-warmup | 79.173 ± 15.686 | 89.85861 | 0.103 |
| full_photo_v10 / vit-scratch-p16 | 80.061 ± 14.273 | 82.50369 | 0.117 |
| full_photo_v10 / vit-scratch-p32 | 83.470 ± 9.387 | — | 0.093 |
| local_global / swin-tiles-constant | 32.088 ± 18.257 | 74.77702 | 1.138 |
| local_global / swin-tiles-cosine | 33.827 ± 18.663 | 74.01519 | 1.383 |
| local_global / vit-scratch-p32 | 33.918 ± 11.833 | 65.84300 | 1.151 |
| local_global / swin-full | 34.053 ± 17.176 | 58.66346 | 1.543 |
| local_global / swin-tiles-frozen | 34.596 ± 13.604 | 45.21894 | 0.592 |
| local_global / vit-scratch-no-warmup | 34.887 ± 9.164 | — | 1.153 |
| local_global / swin-last-stage | 34.940 ± 17.353 | — | 0.994 |
| local_global / swin-frozen | 39.453 ± 14.627 | — | 0.578 |
| local_global / vit-scratch-p16 | 39.520 ± 22.677 | — | 1.233 |

The best measured CNN OOF is still 26.157. New Swin tile OOF is 32.088;
new scratch patch-32 OOF is 33.918. We have no result showing that a transformer
beats the best CNN on these folds.

The first scratch study was weak: patch-16 OOF was 80.061 and patch-32 was 83.470.
New scores are 39.520 and 33.918. This is a large pipeline improvement.
It does not measure the effect of local detail alone: padding, checkpoint
selection and epoch averaging also changed.

The old CNN notebook says its four-fold recipes were not submitted at that time.
The API has three earlier **single-fold** MIL submissions: ResNet-288 fold 2,
87.31631 (57853309); frozen ResNet-288 fold 2, 64.80630 (57853284); ResNet-224
fold 3, 73.91768 (57853264). These are not four-fold scores, so they are not
inserted into the comparison table as ensemble LB results.

## Ablations: what the results can say

Differences below use displayed scores rounded to three decimals.
They are observations from one seed and four folds, not significance tests.

| Comparison | Result | Limit |
| --- | --- | --- |
| Scratch patch 32 vs 16, new study | 33.918 vs 39.520; patch 32 is lower by 5.602 | Patch size changes token count and cost too. There is no new p16 LB. |
| Scratch no warmup vs warmup, new study | 34.887 vs 39.520; no warmup is lower by 4.633 | No seed repeat. This does not prove all ViTs should skip warmup. |
| Scratch patch 32 vs 16, full-photo v10 | 83.470 vs 80.061; patch 32 is worse by 3.409 | The patch-size result changes with the input and selection recipe. |
| Scratch no warmup vs warmup, v10 | 79.173 vs 80.061; no warmup has lower OOF by 0.888 | Public ranking is reversed: 89.85861 vs 82.50369. |
| Swin tiles constant vs warmup/cosine | 32.088 vs 33.827; constant is lower by 1.739 | This tests warmup + schedule together. Public slightly favours cosine. |
| Swin tiles frozen vs tuned constant | 34.596 vs 32.088; tuning lowers OOF by 2.508 | Head LR differs too. Frozen has much better public EMD: 45.21894 vs 74.77702. |
| Swin multiscale frozen / last-stage / full | 39.453 / 34.940 / 34.053 | LR and trainable parameters differ. This is a recipe comparison. |
| CNN constant vs warmup/cosine | 30.977 vs 26.157; warmup/cosine is lower by 4.820 | Warmup and schedule are changed together. |
| CNN 224 vs 288 px | 34.182 vs 30.977; 288 is lower by 3.205 | Same physical rectangles and one resize from source. |
| CNN frozen vs tuned | 37.783 vs 30.977 | Head LR differs. No isolated test of freezing alone. |
| CNN ResNet vs ConvNeXt | 30.977 vs 32.825 | More capacity did not improve OOF in this recipe. |

Full-photo vs local + global is not a one-change ablation.
Tile-only vs multiscale Swin also changes LR, weight decay, schedule, layer decay
and budget. We cannot claim that global context helps or hurts from that pair.

There is **no isolated ablation** for layer-wise LR decay, weight decay amount,
weight decay exclusions, stochastic depth or augmentation. They are implemented
settings, not measured improvements. No extra run is added to fill these gaps.

## Epoch ensemble: a result that did not always help

Both columns below already exist in the completed runs. No predictions were
recomputed. A negative change means the selected epoch ensemble has lower EMD.

| Recipe | Best-only OOF | Epoch ensemble OOF | Ensemble minus best |
| --- | --- | --- | --- |
| swin-tiles-constant | 31.691 | 32.088 | +0.397 |
| swin-tiles-cosine | 33.732 | 33.827 | +0.095 |
| vit-scratch-p32 | 31.945 | 33.918 | +1.973 |
| swin-full | 36.467 | 34.053 | -2.414 |
| swin-tiles-frozen | 36.637 | 34.596 | -2.041 |
| vit-scratch-no-warmup | 35.331 | 34.887 | -0.444 |
| swin-last-stage | 37.756 | 34.940 | -2.816 |
| swin-frozen | 40.164 | 39.453 | -0.711 |
| vit-scratch-p16 | 40.332 | 39.520 | -0.812 |

Epoch averaging helps six of nine recipes. It hurts scratch patch-32 and both
fully tuned tile-only Swin recipes. In particular, patch-32 best-only OOF is
31.945, while its submitted epoch ensemble uses 33.918 OOF.
There is no submitted best-only public score. Do not assign the ensemble LB to it.
This comparison also cannot isolate the optional third epoch or its camera rule.
Both choices use validation; a tiny validation set can make CV optimistic.

## Optimisation: CNN vs ViT

These are the actual recipe choices. Reasons in this section explain the design;
they are not extra experimental results.

| Setting | CNN MIL | Scratch ViT | Pretrained Swin |
| --- | --- | --- | --- |
| Optimizer | AdamW | AdamW | AdamW |
| Backbone / head LR | 1e-4 / 1e-3 when tuned | 3e-4 / 3e-4 | Multiscale last stage: 1e-4 / 1e-3; full: 3e-5 / 1e-3; tiles: 1e-4 / 1e-3 |
| Frozen head LR | .003 | Not used | .003 |
| Weight decay | .01 | .05 | .05 multiscale; .01 tiles |
| Layer LR decay | 1.0 | 1.0 | .75 multiscale; 1.0 tiles |
| Normalisation | Frozen ResNet BN statistics; ConvNeXt uses LayerNorm | LayerNorm | LayerNorm |
| LR schedule | Constant or warmup/cosine | Cosine with 3 or 0 warmup epochs | Multiscale: 3-epoch warmup/cosine; tiles: constant or warmup/cosine |
| Drop path | No sweep | Maximum .1 across blocks | Maximum .1 across blocks |
| Budget | Up to 16 epochs | Up to 32 epochs | Up to 24 multiscale; 16 tiles |

Bias and 1-D parameters have zero weight decay. For ViT encoders, CLS and Swin
relative position bias tables also have zero decay. Other trainable matrix
weights use the recipe decay. This avoids shrinking scale and bias parameters
in the same way as feature weights. We did not compare decay exclusions.
Layer decay gives earlier pretrained blocks smaller learning rates. It aims to
keep useful old features while later blocks adapt. Its benefit was not isolated.

A CNN applies the same local filters across an image. This helps it learn grain
texture with few soils. Its local rules reduce what it must learn from data.
Our scratch ViT must learn more relations from only 18 training soils per fold.
Pretraining gives Swin useful features before soil training. LayerNorm also avoids
noisy batch statistics when a batch has only a few independent soils.

Warmup limits early update size. That is useful in many transformer recipes,
but our measured scratch results did not show an OOF gain from it. CNN OOF
favoured warmup/cosine, while Swin tiles favoured constant LR. There is no single
schedule winner across these models. SGD vs AdamW was not tested in these studies.

All use scale-safe flips/quarter turns and mild colour changes. A random zoom
without the matching physical scale can change apparent grain size and harm
this task. Strong colour changes may also remove real soil cues. We did not
measure augmentation strength, so these are design reasons, not measured gains.

Pretraining should matter more for a plain ViT because it has fewer built-in
local image rules than a CNN. The first study fits this expectation: pretrained
Swin was much better than scratch ViT. The new scratch model is much closer to
Swin after the pipeline changes. These runs compare different architectures and
inputs; they cannot measure the effect of pretraining alone. We did not train
Swin from scratch or fine-tune a pretrained copy of the custom ViT.

## Camera results

Camera disagreement is prediction-to-prediction EMD on 21 paired held-out soils.
It is not error against labels. A model that predicts almost the same wrong curve
for every photo can have low disagreement. Full-photo v10 scratch patch-16 has
gap 3.111 but OOF 80.061. CNN cosine has gap 25.132 but the best CNN OOF, 26.157.
In the new study, Swin last-stage gap 10.966 is lower than tile-constant 18.073,
but its OOF is worse. Do not rank models by disagreement alone.

For camera-only training, define:

- E_same: label EMD on held-out soils, using the training camera.
- E_cross: label EMD on the same soils, using the other camera.
- Delta_domain = E_cross - E_same.

| Study / baseline | Train camera | Same EMD | Cross EMD | Delta | Fit GPU h |
| --- | --- | --- | --- | --- | --- |
| full_photo_v10 / vit-scratch-p16 | Motorola | 83.643 | 83.777 | +0.135 | 0.065 |
| full_photo_v10 / vit-scratch-p16 | Samsung | 84.646 | 84.630 | -0.016 | 0.072 |
| full_photo_v10 / swin-last-stage | Motorola | 40.701 | 44.542 | +3.840 | 0.082 |
| full_photo_v10 / swin-last-stage | Samsung | 39.821 | 37.917 | -1.904 | 0.077 |
| local_global / vit-scratch-p16 | Motorola | 37.955 | 37.832 | -0.123 | 0.690 |
| local_global / vit-scratch-p16 | Samsung | 52.887 | 48.454 | -4.433 | 0.609 |
| local_global / swin-last-stage | Motorola | 38.077 | 39.566 | +1.489 | 0.510 |
| local_global / swin-last-stage | Samsung | 37.475 | 36.865 | -0.610 | 0.784 |

All rows use 21 paired soils across four folds. Use the saved delta values;
subtracting rounded EMDs can differ by .001. Negative gaps show that another
camera is not always harder. View, light and image count can change too.
Both validation cameras affect checkpoint choice. This is not a blind test of an
unseen device, and it does not measure transfer to the iPhone test photos.

## Coverage and padding

The new input keeps local detail at about 4.5 px/mm and global context at 2 px/mm.
It uses all grid tiles and photos. Spatial coverage of one means no source area
is skipped. It does not mean no information is lost in resizing.
For example, the displayed iPhone-16 global pixel-count ratio is .010495;
its local-tile ratio is .053084. These are output/native pixel ratios, not
fractions of useful information. Native fine detail can still be lost.
Physical scale still assumes the source photos were not cropped in an unknown way.

The completed padding check adds one alignment strip at the right and bottom,
without changing soil pixels or PPM. It uses the **best-only** model on one
held-out soil per fold: F827, H371, H030 and H181. It is not the submitted
2/3-epoch ensemble and it is not a full-data robustness test.
Scratch uses valid-area attention masks. Swin keeps its pretrained window
attention; final valid-area pooling cannot remove all earlier padding effects.

| Recipe | Checked soils | Largest prediction EMD change |
| --- | --- | --- |
| swin-tiles-constant | 4 | 1.925392 |
| swin-tiles-cosine | 4 | 5.529893 |
| vit-scratch-p32 | 4 | 0.000004 |
| swin-full | 4 | 6.689860 |
| swin-tiles-frozen | 4 | 16.068089 |
| vit-scratch-no-warmup | 4 | 0.000000 |
| swin-last-stage | 4 | 6.355107 |
| swin-frozen | 4 | 11.205813 |
| vit-scratch-p16 | 4 | 0.000116 |

Scratch changes are close to zero at displayed precision. Swin can change a lot:
up to 16.068089 for frozen tiles. Border handling remains a real limitation.
This does not prove that padding caused any public score. There is no new test
or fix here, so the reported scores still describe the original completed runs.

## Hypotheses: what changed our view

- H-VIT-01: scratch was worse than pretrained Swin in v10, as expected. It was
  also worse than the best pretrained CNN. This supports the pipeline ranking,
  not a pure pretraining effect.
- H-VIT-02 / H-VIT2-02: larger patches hurt v10 OOF but help new OOF. The result
  depends on the pipeline. The new patch-32 GPU time is 1.151 h vs 1.233 h for
  patch-16; early stopping differs, so this is not a throughput benchmark.
- H-VIT-03 / H-VIT2-03: the expected warmup benefit was not seen in scratch OOF.
  Removing warmup improves displayed OOF in both studies. Only v10 has both
  public scores, and its LB ranking goes the other way.
- H-VIT-04: frozen Swin did not have the lowest camera disagreement in v10.
  Full tuning had lower disagreement and OOF, but frozen had better public EMD.
- H-VIT-07: the earlier two-metric checkpoint rule can reject a better EMD.
  The new rule removes that veto. Several other settings changed too; no
  isolated experiment measures the checkpoint-rule effect.
- H-VIT2-01: the new pipeline greatly improves scratch OOF, but input detail
  alone is not isolated. Global context has no proven independent gain.
- H-VIT2-05: the CNN schedule result did not carry over. Swin tiles constant LR
  beats warmup/cosine on OOF; CNN had the opposite order.
- H-VIT2-06: near-zero padding response is seen in scratch, but not in Swin.
  Valid-area final pooling is not a full padding solution for Swin attention.
- H-VIT2-07: epoch averaging helps six recipes and hurts three. It is not a
  general improvement, and the extra camera rule has not been isolated.

These are post-run conclusions. The original pre-run predictions in
[HYPOTHESES.md](HYPOTHESES.md) are kept. This report does not add predictions
retroactively or claim that they were committed before training.

## Reproducibility and remaining gaps

[part2_results.json](part2_results.json) stores 20 standard experiment rows,
eight camera-protocol rows, 36 padding rows, ten submission receipts and source
notebook SHA256 hashes. Values from displayed tables are rounded, usually to
three decimals. Full-precision OOF files were not recovered or re-scored here.
The six-decimal padding maxima above come from the existing notebook reports.

Study IDs:

- CNN: `20261031T222435Z-d834b6698f`.
- Full-photo scratch: `20261101T170306Z-eeb6d8f69e`.
- Full-photo Swin: `20261101T170501Z-a596a117ff`.
- New scratch: `20261101T183703Z-38390ab494`.
- New Swin: `20261101T183723Z-0bb7a836bb`.

Code identity in completed outputs: CNN `431a69bd...`, v10 `788545b0...`,
new study `4b60a00c...`. All use the same saved sample folds; their full SHA256
is in the JSON. The new source recipes are in
[soilcv/vit/experiments.py](soilcv/vit/experiments.py); implementation details are
in [VIT_README.md](VIT_README.md). Swin weights are torchvision
`Swin_T_Weights.IMAGENET1K_V1` (`swin_t-704ceda3.pth`). Scratch uses random weights.

Executed sources: [scratch](vit_scratch.ipynb), [Swin](vit_pretrained.ipynb),
[CNN](cnn.ipynb), and [v10 archive](archive/full_photo_v10/README.md).
The [W&B project](https://wandb.ai/public-team/soil-grain-size) holds run artifacts.
Kaggle training notebooks are
[scratch](https://www.kaggle.com/code/denysmykhailov/soil-grain-size-vit-from-scratch)
and [Swin](https://www.kaggle.com/code/denysmykhailov/soil-grain-size-pretrained-swin-t).
Use the study IDs to avoid mixing runs with the same recipe names.

The local executed inference notebook shows five GPU CSV outputs, ten rows each.
It requests W&B `:latest` aliases. Those aliases can move. The exact resolved
versions and file hashes for the new five submissions are not verified here.
Retrieve the existing inference manifest for exact reproduction; do not rerun
inference just to document it. Old v10 exact references remain in its
[provenance file](submissions/full_photo_v10/provenance.json).

Part 2 now has measured scratch/pretrained recipes, patch and schedule ablations,
CV, public scores for all ten verified Part 2 submissions, cost and failure analysis.
It still lacks isolated tests of layer decay, weight decay, stochastic depth and
augmentation. Four new recipes and v10 patch-32 have no verified public score.
Private scores are unavailable. These gaps stay explicit under the no-new-runs
constraint. Attention maps belong to Part 4 and are not claimed here.

AI helped write code and this report. Review the code and conclusions before
the defense. The assignment asks for disclosure of AI use and a chat transcript
or equivalent appendix; that submission appendix is not assembled here.
