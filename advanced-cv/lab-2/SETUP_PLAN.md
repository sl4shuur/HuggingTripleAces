# Assignment 2 setup plan: segmentation and acquisition

Person 1 owns data, frozen encoders, segmentation training, evaluation, and the experiment runner. Person 2 owns uncertainty estimation, MaxHerding, UHerding, and adaptive acquisition parameters. Both parts use one contract and one runner so results remain comparable.

Requirements were checked against the local assignment PDF. Settings that are not prescribed by the assignment are proposed team conventions to freeze before the benchmark. Step 0 and the data/oracle portion of step 1 are implemented. A first frozen DINOv2 encoder and image/patch extraction are also implemented. A segmentation head, Lightning training, mIoU, W&B, cumulative Random rounds, and a shared fit/calibration split are implemented. DINO and iBOT ViT-B/16 use pinned ImageNet-1K teacher backbone weights. The next milestone is batch inference and teammate scoring/calibration integration.

## 1. Agree on the shared protocol

1. Work in `advanced-cv/lab-2`, independently of lab-1's task-specific code.
2. Share one ordered ID manifest, preprocessing specification, and task-model training recipe.
3. Use `Selector.select(request: SelectionRequest) -> list[int]` from `alseg/interfaces/`.
4. Follow cumulative budgets `[21, 42, 84, 168, 336, 672]`, with new batches `[21, 21, 42, 84, 168, 336]`.
5. Use one base seed and independent selection, training, and calibration RNG streams. For round index `r` starting at zero, selection uses `base_seed + selection_seed_offset + r`; training and calibration use fixed `base_seed + training_seed_offset` and `base_seed + calibration_seed_offset`. Do not include method names in seeds.
6. Start with an empty labeled set. Uncertainty and UHerding use the same Random cold-start fallback, matching Random's first 21 IDs. MaxHerding selects its own first batch.
7. Run the same deterministic synthetic fixtures and contract tests before integration.

The first deliverable is the contract and passing checks, not a trained segmentation model. Person 2 can start immediately with synthetic embeddings and logits.

## 2. Minimal structure and ownership

```text
advanced-cv/lab-2/
  AGENTS.md               # Persistent project and Kaggle conventions
  pyproject.toml          # Increase version before every source dataset update
  alseg/cli_common.py      # Shared rich-click settings and options
  script/setup_project.py # Setup entry point
  script/extract_features.py # Frozen image-embedding extraction
  script/publish_source.py # Kaggle source create/update entry point
  script/train_segmentation.py # One-budget head training and W&B
  script/run_experiment.py # Cumulative Random experiment entry point
  README.md, SETUP_PLAN.md
  kaggle/train.ipynb       # Declarative configuration and training
  kaggle/setup.ipynb       # Optional diagnostics
  experiments/cpu.py      # Example: independent ExperimentConfig instance
  alseg/
    core/
      pool.py             # PoolManifest
      selection.py        # SelectionRequest and output validation
      predictions.py      # LogitBatch
      calibration.py      # CalibrationBatch
      scores.py           # Canonical score alignment
      helpers.py          # Shared tensor/ID checks and CSV writer
      features.py         # EncoderFeatures
    interfaces/           # Acquisition, data and encoder ABCs
    data/voc.py           # VOC discovery, image/mask decoding and diagnostics
    adapters/voc.py       # VocImages, VocEvaluation, VocOracle implementations
    adapters/dinov2.py     # First frozen backbone
    data/preprocessing.py # Deterministic RGB/mask transforms
    nn/extraction.py      # Image-only batch extraction
    config.py             # Project root and shared TOML loading
    pipeline.py           # Cumulative Random orchestration and single-budget recipe
    metrics.py            # Dataset-wide mIoU
    nn/                   # Heads, training, extraction and evaluation
    runtime/              # W&B and notebook secret access
  tests/
    synthetic.py          # Shared fixtures; outside production core
    test_contracts.py, test_setup.py, test_cli.py
    test_acquisition.py   # Planned: algorithm sanity checks
    test_protocol.py      # Shared queried-label holdout checks
```

Use one trainer for every acquisition method. Notebooks display results and launch shared code; they must not contain a second benchmark implementation. Keep domain data in dataclasses and define ABCs at component boundaries. Core models do not import entry points or adapters. Adapters contain only classes implementing interfaces; standalone data helpers live in `alseg/data/voc.py`. The pipeline composes these components. Use separate rich-click entry-point files sharing `alseg/cli_common.py`, without a `cli/` package. Do not create `__init__.py` files; use namespace packages. Add metrics and model adapters when implemented, without empty placeholder modules.

All code, comments, configuration descriptions, and project documentation are written in English.

## 3. Kaggle runtime and reproducibility

Kaggle is the execution environment. Its input mounts are read-only; caches and results belong under `/kaggle/working/`. Local source files in VS Code are not automatically present in a remote kernel. Publish source as a Kaggle dataset and attach it to the notebook. Increase `[project].version` in `pyproject.toml` before every dataset update. Both teammates must attach the same source dataset version and restart their kernels after updating it. Import code directly from `/kaggle/input/datasets/denysmykhailov/alseg-source/source`; notebook bootstrap has no archive handling. Training does not require the setup notebook. Write outputs only to `/kaggle/working/`. Publish with `script/publish_source.py`; use its dry run to review the source-only archive and let its remote version check enforce increasing project versions. Launch notebook scripts with `%run` (preferred) or `!python`, never `subprocess`.

The supplied dataset mount is:

```text
/kaggle/input/datasets/gopalbhattrai/pascal-voc-2012-dataset/
```

The supplied nested directory is `VOC2012_test/VOC2012_test/`. Keep it as a reference, but require official segmentation train/val lists and masks for the assignment. `script/setup_project.py` discovers a single complete train/val root under the mount, or validates an explicit `data.voc_root` override. It fails if only test data is available or multiple valid roots are found.

Use Kaggle's preinstalled PyTorch for the initial contract checks; install rich-click for CLI commands; do not reinstall a different CUDA stack. Record Python, PyTorch, CUDA, GPU, and full installed package versions after a successful setup. Freeze the actual benchmark environment after a GPU smoke run. The initial `pyproject.toml` declares compatibility bounds, not an exact benchmark lockfile; attach the same saved Kaggle environment/dataset/model versions to both teammates' runs.

Use `tomllib` from Python 3.11+. Add torchvision, Pillow, plotting, and one checkpoint-loading library when their implementation steps require them. Do not install the entire lab-1 environment.

VS Code can use the supplied URL through the notebook kernel picker: Select Another Kernel, Existing Jupyter Server, then enter the URL and select the remote kernel. See the [official remote Jupyter instructions](https://code.visualstudio.com/docs/datascience/jupyter-notebooks#_connect-to-a-remote-jupyter-server). The URL contains access material: do not store it in source, config, notebooks, or Git.

## 4. Data identity and oracle

- Pool: official VOC 2012 segmentation train. Reporting test set: official segmentation val. Do not create a replacement random split.
- Record the Kaggle dataset version and archive checksum when available.
- Store `pool_ids` and `val_ids` as ordered VOC filename stems in a manifest. `image_id: int` means the permanent row index in `pool_ids`, never a current batch position.
- Include the manifest checksum in caches and run metadata; reject mismatches on load/resume.
- Masks use int64 class IDs `0..20`, with `255` ignored in training/evaluation. Resize masks with nearest-neighbor interpolation and pad them with 255.
- Selection preprocessing is deterministic and fixed per encoder. Store resize/padding, normalization, and image-vector pooling with cache metadata.

The runner owns `Oracle.query(ids)`. Validate the entire query atomically: reject duplicate, out-of-pool, or previously labeled IDs before consuming any budget. Log successful queries and store returned masks in a labeled store for subsequent access without querying twice.

Acquisition never receives an oracle or a dataset returning hidden masks. Official val masks are available only to the evaluator. Post-hoc class/cost analysis runs separately after selection, with no feedback into acquisition.

## 5. Trainer-to-selector contract

`alseg/core/selection.py` is the shared source of truth. Boundary tensors are CPU tensors; the acquisition implementation can move computation to a device internally. Requests and their tensor/list contents must not be mutated.

Use `AcquisitionMethod.MAXHERDING` and other enum members in domain code. Parse external strings with `AcquisitionMethod(config_value)` and serialize `.value` to config/results. Synthetic fixtures belong to `tests/synthetic.py`, outside core; the setup notebook imports them after adding `tests/` to `sys.path`.

| Field | Format | Meaning |
| --- | --- | --- |
| `method` | AcquisitionMethod | `RANDOM`, `UNCERTAINTY`, `MAXHERDING`, `UHERDING`; string values in config/results |
| `embeddings` | float32 `[N,D]` | Entire train pool in manifest order, L2-normalized rows |
| `labeled_ids` | list[int] | All queried IDs in acquisition order |
| `unlabeled_ids` | list[int] | Complement of L, in increasing canonical ID order |
| `n_select` | int | Number of **new** IDs, not the cumulative budget |
| `seed` | int | Selection seed resolved for this round |
| `uncertainty` | float32 `[N]` or None | Nonnegative image scores for all target rows; larger means more uncertain |
| `sigma` | positive float or None | Resolved kernel width, mandatory for herding methods |
| return value | list[int] | Exactly `n_select` unique IDs from U, in selection order |

Implemented API:

```python
from alseg.core.selection import AcquisitionMethod, SelectionRequest, validate_selection

request = SelectionRequest(
    method=AcquisitionMethod.MAXHERDING,
    embeddings=embeddings,
    labeled_ids=labeled_ids,
    unlabeled_ids=unlabeled_ids,
    n_select=n_select,
    seed=seed,
    uncertainty=uncertainty,
    sigma=sigma,
)
selected_ids = selector.select(request)  # Person 2 implements the Selector ABC.
validate_selection(request, selected_ids)  # Before Oracle.query.
```

Rules:

- L and U partition `range(N)`. Inputs must be finite; D can differ between encoders.
- `n_select=0` returns `[]`; a budget exceeding U is an error, never silently clipped.
- Ties use the smallest canonical pool ID. The runner validates IDs before the oracle query.
- Resolve cold-start fallback in the runner before constructing a model-based request. Do not silently substitute random selection inside UHerding.
- Kernel rows are all pool targets; columns are U candidates. Initialize coverage using L. UHerding weights belong to target rows, not candidate columns. A labeled target already has maximum self-coverage and no additional gain.
- Person 2 computes sigma/tau using agreed rules; the runner records the actual values used.

### Logits, scores, and calibration

Person 1 provides an iterator yielding `LogitBatch(ids, logits, valid_pixels)`. Person 2 implements scoring and temperature fitting:

```python
score_batch(batch: LogitBatch, *, method: str, temperature: float) -> Tensor
# CPU float32 [B], aligned with batch.ids

fit_temperature(batches: Iterable[CalibrationBatch], *, candidates, ece_bins) -> float
```

`LogitBatch` contains raw float32 `[B,21,H,W]` logits and a bool `[B,H,W]` geometry/padding mask. Run inference with `eval()` and inference mode, without random augmentation. Apply `softmax(logits / tau)` exactly once. Score all target rows, including L, and use `assemble_scores` to restore canonical order even when batches arrive out of order. Do not materialize full-resolution logits for the whole pool at once.

The geometry mask must not depend on hidden segmentation labels, including void=255. Unlabeled ground-truth void pixels cannot be filtered during image scoring. Every image must have at least one geometry-valid pixel.

`CalibrationBatch` adds int64 `[B,H,W]` targets from a held-out subset of queried L only. Ignore 255 in ECE; padding targets must be 255. Types validate shapes/values, but the runner remains responsible for ensuring calibration IDs belong to queried L and were excluded from model fitting.

Agreed calibration protocol: every reporting head trains from scratch on all queried L, identically for all four methods. Before a model-based query, split the previous L deterministically into fit/calibration using canonical IDs and the fixed calibration seed. Train an auxiliary acquisition head on fit only; choose its temperature by ECE on calibration and compute acquisition uncertainty with that same auxiliary head and temperature. Do not transfer temperature to the reporting head. The teammate notebook already implements the auxiliary fit but still needs this scoring adjustment; its notebook is maintained by the teammate.

Do not refit the auxiliary acquisition head on calibration masks after fitting tau. Fix ECE bins, tau candidates, tie-breaks, and aggregation before benchmark. If no valid calibration pixels exist, use tau=1 and log the fallback. Never calibrate or choose the primary uncertainty method using official val.

Patch features `[B,C,h,w]` are internal to encoder/head, not selector inputs. They are cached together with CLS embeddings; see the two-GPU workflow below.

## 6. One experiment loop

```text
load config, manifest, and image embeddings
L = []; U = entire pool; model = None
for target_budget in [21, 42, 84, 168, 336, 672]:
    n_select = target_budget - len(L)
    resolve common Random fallback if the method needs an unavailable model
    otherwise fit/calibrate the auxiliary acquisition head when needed and derive scores/sigma/tau from queried L
    create SelectionRequest; select; validate_selection
    query oracle; update L and U
    create a NEW reporting head and optimizer; keep the encoder frozen and in eval mode
    train on all queried L with the fixed recipe, without val-based checkpoint selection
    evaluate official val; save results and selection history
```

Scores for 21 -> 42 come from the model trained at budget 21. The budget-42 row records the sigma/tau actually used for that acquisition, not parameters for the next round.

Use identical head initialization seeds for the same encoder/round across methods. Reset optimizer, scheduler, and scaler as well as head weights. Keep encoder parameters out of the optimizer and set `requires_grad=False`.

Full-data upper bound is a separate reporting-head run on the entire official train split; no calibration/acquisition head is needed. The required grid is 2 * (4 * 6 + 1) = 50 reporting-head runs. Model-based methods additionally fit auxiliary acquisition heads.

## 7. Configuration, artifacts, and metrics

Before benchmarking, freeze the remaining defaults in `alseg/config.py`: encoder checkpoint and immutable revision/checksum, preprocessing, image-vector pooling, head, optimizer, LR, epochs, batch size, augmentations, calibration, uncertainty aggregation, sigma/tau rules, and metric conventions. DINO/iBOT ViT-B/16 checkpoints, the head recipe, and the queried-label split are configured and checked; acquisition/scoring and temperature calibration choices remain to be frozen.

Choose two encoders after checking dense feature availability, checkpoint access, licensing, pretraining provenance, absence of VOC fine-tuning, and actual GPU memory. Compare similar model sizes when possible.

Cache embeddings with manifest hash, checkpoint revision, preprocessing/pooling, dtype, and shape. Reject stale metadata. Deterministic feature caches must not silently replace stochastic training augmentations.

```text
runs/<encoder>/<method>/<run_id>/
  config.toml              # Fully resolved configuration
  metadata.json            # Source revision, dirty state, package/GPU versions, hashes
  selections.json          # Ordered new IDs and cumulative L for every round
  oracle_queries.jsonl     # Successful oracle queries
  results.csv              # One row per completed budget
  per_class_iou.json       # Per-class intersection, union, and IoU
```

CSV columns: `run_id, encoder, method, seed, round, budget, n_fit, n_calibration, miou, pixel_accuracy, mean_class_accuracy, selection_seconds, train_seconds, eval_seconds, sigma_used, tau_used, fallback`. Store metrics in `[0,1]`. JSON/CSV histories are replaced atomically. Current runs persist L/U and completed-round metrics but do not support process restart/resume; a new invocation requires a fresh output directory. Resume with manifest/config/oracle verification remains a later step.

Sum the confusion matrix over all official val images before computing IoU. Include background, ignore 255, and exclude zero-union classes from the mean while logging their count. Pixel accuracy and mean class accuracy use the same confusion matrix.

Define normalized AUBC as trapezoidal integration of mIoU over **linear** budgets `[21,672]`, divided by `672-21`. Do not extrapolate to zero or include the full-data point. A logarithmic plot x-axis does not change this calculation. Compare against Random at matching encoder/seed/budget.

Profile encoder parameters, extraction time, and peak device memory under one resolution/batch/dtype/device/warm-up/synchronization protocol. Compute annotation cost proxies only post hoc, for example instance counts from SegmentationObject excluding background/void.

Keep source, config, and tiny fixtures in Git; exclude real data, caches, and checkpoints. The root `.gitignore` ignores all CSV files: explicitly allow a final summary file or include results in a submission archive when needed.

## 8. Implementation order

| Step | Person 1 | Person 2 | Exit criterion |
| --- | --- | --- | --- |
| 0: shared setup | Contracts, fixture, config, Kaggle preflight | Review inputs required for herding/calibration | Same fixture and contract checks pass |
| 1: independent development | Manifest, oracle, one encoder/head, metrics, Random runner | Uncertainty and herding on synthetic inputs | Random completes two smoke rounds; selectors return valid IDs |
| 2: integration | Batch inference and held-out calibration data | Real-image scoring and full-pool selection | One encoder completes two rounds with MaxHerding and Uncertainty |
| 3: complete functionality | Second encoder, cache/profiling, full-data mode | UHerding, adaptive sigma/tau, trajectories, sanity checks | All four methods complete short end-to-end runs |
| 4: benchmark | Freeze recipe; run encoder 1 | Same runner/config; run encoder 2 | 50 required results with metadata and histories |
| 5: analysis | Encoder/model/oracle/cost sections | Acquisition/calibration/selection sections | Shared curves, AUBC, tables, and conclusions |

Mark reduced-budget/epoch runs as smoke tests and exclude them from benchmark tables. Measure one real round before scheduling all 50 runs. Do not change the recipe per method to fit the available time.

Current Random runner interface (`run_script` points to `script/run_experiment.py`; choose the encoder in the JSON config):

```ipython
%run "$run_script" --config "$config_path"
```

The teammate acquisition methods are not connected yet.

Use separate branches for segmentation and acquisition. Any contract change must update its caller, documentation, and fixture together.

## 9. Checks before benchmark

- Out-of-order inference preserves ID-to-score alignment; reject zero/NaN embeddings and invalid scores.
- Oracle rejects duplicates and invalid batches atomically; query count equals cumulative budget.
- Acquisition never reads oracle labels of U or official val; calibration uses held-out queried L only.
- Constant weights produce identical ordered UHerding/MaxHerding batches for identical sigma and tie-breaks.
- Tiny sigma with distinct features and unique scores makes UHerding match top-B uncertainty; test ties separately.
- UHerding uses minimum labeled pairwise distance; MaxHerding uses the agreed label-free median rule. Fix and log fallbacks for insufficient labeled points/zero distances. Seed any pair subsampling.
- Frozen encoder receives no parameter updates; create fresh head/optimizer state every round.
- Hand-computed masks verify confusion matrix, background, and ignore=255.
- Same-seed small runs reproduce selected IDs; define metric tolerances and record hardware. Do not promise bitwise equality across different GPUs.
- Full-pool selection fits memory; chunked gains match a small dense reference.

The assignment requires disclosure of AI assistance, including prompts/responses or transcripts. Preserve this interaction for the submission appendix. This setup supports your implementation and does not replace your own understanding or defense.

## 10. Current tracking and training scope

`TrackingConfig` uses Pydantic Settings with no environment prefix. A local `.env` can contain `WANDB_API_KEY`; on Kaggle enable the same secret in Add-ons > Secrets. `get_secret` detects Kaggle, Colab, or local runtime. Never save the typed `SecretStr` API key in run snapshots. Default entity is exactly `public-team`. Offline/disabled tracking does not request secrets.

`script/run_experiment.py` runs cumulative Random budgets `[21, 42, 84, 168, 336, 672]` using one manifest, frozen encoder, and Oracle. Round `r` samples only the missing count from canonical U with `seed + selection_seed_offset + r`. Every round initializes a fresh reporting head and AdamW optimizer with fixed `seed + training_seed_offset`. Official val is used only after the fixed training schedule.

The Random runner trains on all queried L (`n_fit=budget`, `n_calibration=0`). The shared `split_labeled_ids` helper is available for auxiliary calibration, with fraction=0.2 and minimum=4. Held-out labels must be excluded from the auxiliary head used for both temperature fitting and uncertainty scoring. Official val is never calibration data.

`script/train_segmentation.py` uses the same all-L reporting recipe for one standalone budget.

## Notebook configuration

Defaults live only in the Pydantic classes in `alseg/config.py`. `ExperimentConfig(...)` accepts partial section values; omitted values use class defaults. Specific experiments may export a `config` instance from an ordinary `experiments/*.py` module, as in `experiments/cpu.py`.

Notebooks call `.show()`, save a resolved JSON snapshot with `.save()`, then launch the training script through `%run ... --config ...`. They do not call training functions directly. CLI entry points load snapshots with `ExperimentConfig.load()`. Omitting the file means Python defaults. JSON captures a run's configuration; it does not define another set of defaults. Credentials are excluded from display and serialization and loaded at execution time.

Notebook dependency installation reads `pyproject.toml` with uv and targets `sys.executable`, preserving the current Kaggle kernel environment. Do not copy package/version lists into notebook cells.
