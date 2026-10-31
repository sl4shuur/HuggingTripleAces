# Computer Vision

Computer Vision coursework by the Hugging Triple Aces team at KSE.

## Assignments

| Folder | Topic | Contents |
| --- | --- | --- |
| [ass1](ass1/) | Coin and face counting | Shared EDA, coin detection, face regression, and Haar cascade |
| [ass2](ass2/) | Lung nodule classification | EDA, supervised and self-supervised notebooks, and [report](ass2/README.md) |
| [ass3](ass3/) | Semantic segmentation | EDA, DeepLabV3, supervised and semi-supervised experiments |
| [ass4](ass4/) | Human pose estimation | EDA, pose models, DINOv2, I-JEPA, and [assignment brief](ass4/README.md) |

## Local setup

Install Python 3.13 and [uv](https://docs.astral.sh/uv/), then run from the repository root:

```bash
cd comp-vision
uv sync --extra cpu
uv run jupyter notebook
```

For CUDA 12.8, use `uv sync --extra gpu` instead. The `cpu` and `gpu` extras are mutually exclusive. Select the kernel from `comp-vision/.venv` in Jupyter or your editor.

The Python project, lockfile, Python version, and shared `src/` helpers live in this folder. Shared configuration resolves `comp-vision/` as the project root and uses `comp-vision/data/` for local data.

## Data and notebook working directories

Datasets, trained weights, and generated submissions are not included. Open each notebook from its assignment folder so its relative paths resolve there:

- `ass1/eda.ipynb` reads CSV files and `train/` from `ass1/`.
- `ass1/faces.ipynb` reads `ass1/data/`; its Haar cascade is stored alongside the notebook. Its final ensemble also needs the coin submission referenced in the notebook.
- `ass1/coin_det.ipynb` reads `ass1/competitions/count-coins-and-faces/count-coins-and-faces/`.
- `ass2/eda2.ipynb` searches `comp-vision/data/archive/`, `comp-vision/archive/`, and `comp-vision/data/`, with additional paths relative to the notebook.
- `ass3/eda3.ipynb` uses `ass3/data/VOCdevkit/VOC2012/`; `ass4/eda4.ipynb` uses `ass4/data/coco2017/`.

Several training notebooks use Kaggle input and output paths and install additional packages in notebook cells. Run those on Kaggle with the referenced datasets attached, or adjust their data paths and install their additional dependencies for local execution. Existing outputs and experiment results are preserved.
