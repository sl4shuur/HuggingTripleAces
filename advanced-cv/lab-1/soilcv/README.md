# soilcv

Shared code for soil grain-size experiments. Version 0.1.0.

Use the same data, folds, curve units, and scores for CNN, ViT, frozen features, and SSL fine-tuning. SSL pretraining keeps its own loop.

Importing `soilcv` does not read secrets, start W&B, or install packages. Core metrics need NumPy and pandas. Training needs PyTorch. Tracking needs W&B. Model libraries and image transforms belong to each experiment.

## Import on Kaggle

Upload the whole `soilcv/` folder as a code dataset. Keep this folder name so Python can import it. Attach the dataset to each notebook and pin its version.

```python
from pathlib import Path
import sys

roots = [p.parent for p in Path('/kaggle/input').rglob('soilcv')
         if (p / '__init__.py').is_file()]
if len(roots) != 1:
    raise ValueError('Attach exactly one soilcv code dataset.')
sys.path.insert(0, str(roots[0]))

import soilcv
print(soilcv.__version__)
```

In this repo, open the notebook in `advanced-cv/lab-1/`; `import soilcv` works there.

## Part 1 on Kaggle

The dataset also includes `soilcv.cnn`: models, the two-GPU queue, report code,
inference, and environment templates. Import `cnn.ipynb`, attach the latest code
dataset and soil data, enable Internet and the `WANDB_API_KEY` secret, then use
Save & Run All with two T4 GPUs. No helper files are needed next to the notebook.
Each fold is tracked online in `public-team/soil-grain-size`; no CV-summary runs
are created. The full study is saved under `/kaggle/working/part1_cnn`, with
`submission.csv` at the top level. Kaggle keeps these files in notebook Outputs.

## Data, folds, and cache

```python
from soilcv.data import load_metadata, load_folds, DEFAULT_FOLDS_PATH
from soilcv.tiles import make_tile_manifest, build_tile_cache

labels, photos, template = load_metadata(DATA_DIR)
folds = load_folds(DEFAULT_FOLDS_PATH, labels.sample_id)

manifest = make_tile_manifest(photos, tile_mm=64, output_px=288, image_root=DATA_DIR)
build_tile_cache(manifest, CACHE_DIR, image_root=DATA_DIR)
manifest = manifest.merge(folds[['sample_id', 'fold']], on='sample_id', how='left')
```

`DATA_DIR` is the competition folder. `CACHE_DIR` must be writable, for example `/kaggle/working/soil_tiles`.
The bundled `assets/sample_folds.csv` is the original EDA file: 24 train samples, four folds, six validation samples per fold. It is not regenerated on import.

The manifest has relative `tile_path` values. Move the cache and use its new root with `TileDataset`. `manifest.json` stays in its cache-ID folder. Read it with `pd.DataFrame(json.loads(path.read_text()))`.

PPM is still an estimate based on a full-image resize. Matching aspect ratios do not prove that no crop took place. Metadata keeps this assumption. Changing crop settings, input image bytes, or scale gives a new cache identity. `output_px=None` keeps native crop pixels.

## Load tiles

```python
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from soilcv.data import TileDataset, sample_weights
from soilcv.config import seed_worker

train_tiles = manifest[manifest.fold.notna() & manifest.fold.ne(fold)].copy()
valid_tiles = manifest[manifest.fold.eq(fold)].copy()

sampler = WeightedRandomSampler(
    sample_weights(train_tiles), num_samples=len(train_tiles), replacement=True,
    generator=torch.Generator().manual_seed(42),
)
train_loader = DataLoader(
    TileDataset(train_tiles, CACHE_DIR, labels, transform=train_transform),
    batch_size=16, sampler=sampler, worker_init_fn=seed_worker,
    generator=torch.Generator().manual_seed(43), persistent_workers=False,
)
valid_loader = DataLoader(
    TileDataset(valid_tiles, CACHE_DIR, labels, transform=valid_transform),
    batch_size=16, shuffle=False, persistent_workers=False,
)
```

Transforms take a PIL RGB image and return a CHW tensor. Use the encoder's required normalization. Validation transforms must be fixed. With no transform, the dataset returns float RGB pixels from 0 to 1. No extra image resize is done by the dataset.

Weights give equal expected weight to each sample, then each photo, then each tile. Use the repeated sampler only for training. Prediction and feature loaders must visit every tile once.

## Curves and metrics

```python
from soilcv.curves import aggregate_predictions, make_submission
from soilcv.metrics import evaluate_samples, summarize_cv

sample_predictions = aggregate_predictions(tile_predictions)
metrics = evaluate_samples(valid_labels, sample_predictions)
submission = make_submission(test_predictions, template)
cv = summarize_cv(labels, all_oof_predictions, folds)
```

`tile_predictions` has `sample_id`, `photo`, `tile_id`, and the 11 target columns. `aggregate_predictions` means tiles per photo, then photos per sample. OOF files have one row per sample and the saved `fold`.

Targets use percentages, not fractions. EMD uses 10 left-point errors and log10 interval widths. The last value has no interval weight but must be 100 in a valid curve. The metric never fixes curves. Use `repair_curves` explicitly for unconstrained Ridge or PCA outputs, with the same setting in validation and test.

`range_errors` and `group_errors` report separate diagnostics. For camera reports, first aggregate photos per sample and camera; one sample can belong to two camera reports. Those groups are not disjoint.

## Neural models and training

Use `MassCurveHead(feature_size)` after an encoder that returns `[batch, feature_size]`. Or use `MassCurveHead()` after 11 logits. It gives monotonic percentages ending at 100.

Build the model on the intended device before building its optimizer. `make_param_groups` supports explicit layer IDs for layer-wise learning rates. `freeze_bn_stats` freezes running statistics, not affine gradients.

```python
from soilcv.config import ExperimentConfig
from soilcv.nn import EMDLoss
from soilcv.training import fit_supervised_fold
from soilcv.tracking import start_run

config = ExperimentConfig(
    experiment_id='cnn-001', stage='cnn', model='your-exact-model-name', fold=fold,
    versions={'code': soilcv.__version__, 'data': DATA_VERSION, 'cache': CACHE_ID},
    params={'weights': WEIGHTS_REVISION, 'preprocessing': PREPROCESSING},
)
valid_labels = labels[labels.sample_id.isin(valid_tiles.sample_id)]

with start_run(config) as run:
    oof, history = fit_supervised_fold(
        model, train_loader, valid_loader, valid_labels, optimizer, EMDLoss(), config,
        epochs=20, device=device, amp=True, patience=5, run=run,
    )
```

The model constructor and transforms live in the experiment notebook. The shared loop checks sample leakage and fold IDs. It chooses the best checkpoint by sample-level EMD and returns its OOF curves. `train_one_epoch` needs a mean loss. Gradient accumulation handles small final batches correctly.

For a frozen encoder inside a supervised model, freeze its parameters and pass `frozen_modules=(encoder,)`. Set `freeze_batchnorm=True` only if that is part of the experiment. Scheduler policies are explicit: `epoch`, `batch`, or `metric` (for plateau schedulers).

Saved files include config, actual training settings, history, compute record, best/last checkpoints, and OOF curves. Checkpoints contain optimizer, scheduler, scaler, global random state, and loader/sampler generator states. Their identity includes the package source and input/config settings. Resume is at an epoch boundary with `persistent_workers=False`. Keep `best.pt` beside `last.pt`. Set `resume_from` to `last.pt` and `epochs` to the full desired epoch count. Load only your own trusted training checkpoints.

## Frozen features

`extract_features(encoder, loader)` expects `[batch, features]` output. Define pooling in an adapter. The function uses evaluation mode and restores the earlier module modes.

Use `feature_identity` with an exact weights revision, preprocessing settings, pooling, and tile cache ID. It adds a hash of the actual weights/buffers. For a fold-trained encoder, also pass `fold` and `train_ids`.

`save_features` stores ordered IDs, a NumPy array, and file hashes. `load_features` rejects a different identity or changed files. To fit Ridge, average feature columns with `aggregate_features`, then fit preprocessing and the regressor on train samples only.

## W&B online

Defaults:

- Entity: **public-team**
- Project: **soil-grain-size**
- Mode: **online**
- Secret label: **WANDB_API_KEY**

Attach the secret in Kaggle Add-ons -> Secrets and enable Internet. Each notebook needs access to the secret. `start_run` reads it at runtime, or uses the environment variable with the same name. The key is never saved in the experiment config.

Override the team with `ExperimentConfig(..., wandb_entity='another-team')`. One experiment ID groups fold runs. A config with `fold=None` starts a summary run. Use `log_evaluation` for combined CV metrics/tables, and `log_artifact` for explicit file uploads. Missing login or an offline run raises an error; there is no silent offline fallback.

For tracking resume, pass the checkpoint's run ID to `start_run(config, run_id=run_id, resume=True)`, and restore `last.pt` in the fit call. Run resume alone does not restore the model.

The default entity is passed explicitly to W&B. It does not rely on an account default.

No real online run is started by the test suite. Live authentication needs your attached secret and permission to write to the configured team/project.

## Checks

From `advanced-cv/lab-1/`, run `python -m pytest tests -q` in an environment with the core libraries, PyTorch, and pytest. Tests cover metric/loss parity, gradients, ID and fold failures, unequal tile aggregation, cache identity, checkpoint resume, a tiny CPU fit, and mocked online W&B setup.

## Build an upload ZIP

From `advanced-cv/lab-1/`, run:

```bash
python package_soilcv.py /tmp/soil-cv-utils-0.1.0.zip
```

The ZIP contains `soilcv/` and `release.json` with code/fold hashes and tracking defaults. It excludes test files, Python bytecode, and notebook outputs. Upload it as a Kaggle Dataset. This command does not publish anything.

## Tested CPU environments

- Local: Python 3.14, NumPy 2.5.4, pandas 3.1.0, PyTorch 2.15.0+cpu.
- Kaggle: Python 3.13, NumPy 2.1.3, pandas 2.3.3, PyTorch 2.11.0+cpu, W&B SDK 0.28.1.
- All 16 tests passed in both environments. W&B setup is mocked; a real online login needs your secret.

GPU AMP code is included. These checks used CPU.
