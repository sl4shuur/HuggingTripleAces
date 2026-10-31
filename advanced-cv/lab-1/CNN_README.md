# Part 1 on Kaggle

Use `cnn.ipynb` and the `denysmykhailov/soil-cv-utils` code dataset.
The CNN source is in `soilcv/cnn/`. There are no helper scripts next to the notebook.

1. Attach the soil competition data and code dataset version 3.
2. Select GPU T4 x2 and turn Internet on.
3. Add `WANDB_API_KEY` in Kaggle Secrets and allow notebook access.
4. Import `cnn.ipynb`, then choose Save Version → Save & Run All.

The notebook waits for all 32 fold jobs. Each GPU trains one fold at a time.
Each fold runs test inference on GPU. Seeds are fixed, but GPU math is not fully
deterministic, so scores can change on another run. Failed jobs stop the notebook and keep logs.
W&B tracks training, validation, compute, and model artifacts in
`public-team/soil-grain-size`. It creates no CV-summary runs.

Kaggle saves these files from `/kaggle/working` after a successful run:

- `submission.csv`: the selected recipe, averaged over four fold models.
- `part1_cnn/`: the full study, including all checkpoints, inference weights,
  tables, plots, report, configs, logs, and a source/environment copy.

Each fold folder has `best.pt`, `last.pt`, and `weights.pt`.
`part1_cnn/output_manifest.json` lists all study files and their hashes.
The code saves no API keys in these files. It does not submit to the competition.

To use the saved environment, download the study folder and run
`uv sync --project part1_cnn/source/environment` with a compatible CUDA setup.
The runtime versions and installed package list are saved there too.
The lock is a reproduction recipe; the Kaggle image may have other installed dependencies.

To predict from photos after downloading the study folder:

```bash
PYTHONPATH=part1_cnn/source python -m soilcv.cnn.inference \
  --photo photo.jpg --ppm 4.6 \
  --checkpoint-dir part1_cnn/runs/cnn-r18-constant-lr/fold_0 \
  --device cuda:0 --output prediction.csv
```

Use the actual PPM for each photo. For the full ensemble, pass all four fold folders.
The selected recipe is saved in `part1_cnn/results/selection.json`.
No old EDA results were removed. The earlier local CNN study was archived before this refactor.

To build a new code dataset version, run `python package_soilcv.py`.
Upload the resulting ZIP as a new version of `soil-cv-utils`, then update the input version in Kaggle.
