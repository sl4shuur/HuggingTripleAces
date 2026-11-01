# Part 3 — DINOv3 + MAE

Run **[ssl.ipynb](ssl.ipynb)** for the complete study. The notebook explains the
protocol and displays the score matrix, paired adaptation changes, CKA and
training curves. Computation lives in `soilcv/ssl/`.

## Experiments

| Encoder / source | k-NN | Ridge | LP | LP-FT | Full FT |
|---|---|---|---|---|---|
| DINOv3 ViT-S/16 | ✓ | ✓ | ✓ | ✓ | ✓ |
| MAE ViT-B/16 | ✓ | ✓ | ✓ | ✓ | ✓ |
| DINOv3 after fold-local adaptation | ✓ | ✓ | ✓ | ✓ | ✓ |

Three DINOv3 ablations: CLS-only ridge, LP-FT without layer-wise LR decay, and
LP-FT without photometric jitter. All 18 recipes use all four saved sample folds.
MAE's public official checkpoint is ViT-B, larger than DINOv3-S; this is a
comparison of released encoders, not a parameter-matched architecture experiment.

Official weights, pinned by commit:

- [facebook/dinov3-vits16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m),
  `114c1379950215c8b35dfcd4e90a5c251dde0d32`.
- [facebook/vit-mae-base](https://huggingface.co/facebook/vit-mae-base),
  `25b184bea5538bf5c4c852c79d221195fdd2778d`.

## Physical and validation protocol

Every 64 mm native-image crop is resized directly to 224 px, without an
intermediate 288 px cache. Both encoders see the same 3.5 px/mm. The grid covers
the whole photo; overlap-corrected tile areas sum to one per photo. Photos have
equal sample weight. One labelled soil contributes one EMD loss. Background is
not masked; PPM still assumes the original photos were resized without cropping.

Descriptors join CLS and mean patch features. DINOv3's register tokens are
excluded. MAE uses zero masking and deterministic patch order in **both** training
and inference. No MAE decoder is loaded. DINOv3's random RoPE coordinate rescaling
is disabled during tuning.

Frozen ridge standardizes on training soils only, with alpha=100. k-NN uses
cosine similarity, k=3 and temperature=.07. LP trains a linear projection into
eleven mass bins; softmax and cumulative sum guarantee a valid curve. LP-FT
initializes that projection from the best LP checkpoint; full FT uses a fresh
projection. FT uses AdamW, backbone LR 1e-5, head LR .001, weight decay .05,
two warmup epochs, cosine decay and layer-wise LR decay .75. Biases, normalization
parameters and learned special tokens have no weight decay.

Validation EMD alone selects one best epoch. Camera curves are diagnostics.
There is no optional epoch ensemble. This differs from Part 2's final recipe.
Hyperparameters are declared in `soilcv.ssl.experiments.SETTINGS`; changes require
a fresh study directory. `uv.lock` records the available dependency resolution;
Kaggle's existing torch/CUDA build and actual library versions are saved separately.

## Domain adaptation and CKA

DINOv3 continues for ten fixed epochs separately inside each fold. Its unlabelled
loader rejects the held-out fold and all test photos. Sample/photo/area sampling
prevents soils with more photos from dominating. Two views preserve the entire
64 mm field of view and use quarter-turns, flips and mild color jitter. An EMA
teacher supplies centered, sharpened cross-view targets. This small DINO-style
continuation is not a reproduction of full DINOv3 training. The final teacher
encoder is evaluated with all five strategies, even if adaptation hurts.

Loss, teacher entropy, marginal entropy and feature standard deviation are
saved every epoch. CKA and centered RankMe compare the exact same held-out tile
IDs before/after tuning and adaptation. At most four grid tiles per photo and
256 total are used. These correlated tiles are a descriptive diagnostic, not
independent validation samples. Undefined CKA for constant features is recorded
as null. High CKA alone does not imply better accuracy.

Random zoom/crops without corrected PPM change apparent grain size. Strong blur
erases fine grains; MixUp/CutMix do not supply a justified soil mass target.
These augmentations are excluded. The color ablation tests a weaker assumption.

See [DINO](https://arxiv.org/abs/2104.14294),
[linear CKA](https://arxiv.org/abs/1905.00414), and the
[Transformers DINOv3 documentation](https://huggingface.co/docs/transformers/en/model_doc/dinov3).

## Kaggle

Attach competition `soil-grain-size-from-photos`, the updated private
`denysmykhailov/soil-cv-utils`, and private
`denysmykhailov/soil-ssl-official-weights`. The latter stores only official weights
and configs, never tokens. Select **2 × T4**, Internet, and the existing
`WANDB_API_KEY` secret. Online tracking uses `public-team/soil-grain-size`.

Build the code dataset with `python package_soilcv.py /tmp/soil-cv-utils.zip`.
The notebook checks the actual source hash. Eight workers (one encoder/fold each)
run two at a time. Frozen features and LP heads are reused within a source/fold.
Set `SMOKE=True` for a small GPU integration check without online W&B; smoke
results never enter the complete score table.

Outputs under `/kaggle/working/part3_ssl_v2` contain exact source/lockfile,
environment, model revisions and hashes, physical tile manifest, all 72 OOF
fold results, numerical regressors, inference weights, histories, compute logs,
camera curves and CKA. Each recipe/fold has its own W&B run. Each encoder/fold still has one model
artifact containing all its strategies, attached to the last full-FT run. The report and figures are generated from completed folds, not from
hard-coded results. The final submission averages the best main-matrix recipe's
four folds by CV. Competition submission is a separate action.

Feature extraction and adaptation are shared costs, listed separately in the
report. LP-FT additionally requires its LP stage. CPU Ridge/k-NN fits use zero
GPU-hours. Peak GPU memory records allocated memory. Public/private LB stay
empty until actual submission receipts are available.

## W&B metrics

A fresh study writes to `/kaggle/working/part3_ssl_v2`. Stop the earlier session
before starting it so old workers cannot recreate the deleted SSL logs.

Each recipe/fold is a separate run, grouped by `config.experiment_id`, for example
`dinov3-lpft`, `mae-full` or `dinov3-adapted-lp`. All supervised strategies reuse
`soilcv.tracking.log_epoch`, the logger used by CNNs and ViTs:

- `epoch` is the x-axis and starts at zero within each recipe/fold.
- `train/loss`, `train/lr_0`, optimizer steps, skipped steps and global step keep
  the existing names and units. Global step accumulates within the recipe.
- `val/emd`, `val/mae_pp`, `val/worst_point_pp`, D10/D50/D60 errors and the paired
  Motorola/Samsung metrics use the same names as Parts 1–2.
- `best/*` and `selected/*` describe the selected checkpoint, not the last epoch.
- Frozen Ridge/k-NN emit one validation point and best metrics, without invented
  training curves. Their camera predictions reuse the cached tile features.

DINO adaptation has a separate `dinov3-adaptation` run per fold. Its
`adaptation/*` metrics use the same `epoch` axis; its self-distillation loss is
kept separate from supervised EMD. The new study creates 72 recipe/fold runs
plus four adaptation runs. It does not combine several strategies in one curve.

## Inference and checks

`soilcv.ssl.inference.predict_fold(folder, photos, cache_root)` accepts the same
photo metadata as the existing pipeline: `sample_id`, `photo`, absolute `path`,
`ppm_est`, `device`. It loads numerical parameters and neural weights without
Hub access. `predict_ensemble` checks matching study/strategy and folds 0–3.

```bash
python -m soilcv.ssl.inference FOLD_RECIPE_FOLDER \
  --photos photos.csv --output predictions.csv --device cuda:0
python -m pytest tests -q
```

Current local validation: 90 tests passed. The earlier [Kaggle GPU smoke, version 2](https://www.kaggle.com/code/denysmykhailov/soil-ssl-part-3-gpu-smoke)
verified both official encoders on two T4 GPUs, real optimizer updates without
AMP skips, DINO-style adaptation and reload parity. MAE's peak allocated memory
was about 4.12 GiB. These small smoke runs are not CV results. Receipts are in
`part3_validation.json`.

The tests include a complete tiny source study, DINO adaptation and reload parity
for every strategy/ablation, as well as split exclusion, teacher stop-gradient,
CKA invariance, area pooling and deterministic MAE evaluation.

AI assistance: Codex helped implement and test Part 3. Retain this chat transcript
for the assignment's disclosure requirement and review the code for the defence.
The owner requested no commits; pre-run hypotheses remain in the working tree.
