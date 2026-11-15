# Project conventions

- Write source code, comments, configuration descriptions, and documentation in English.
- Do not create `__init__.py` files. Use Python namespace packages.
- Split domain models into files by responsibility in `alseg/core/`; do not collect all models in one file. Group ABC contracts by responsibility in `alseg/interfaces/`.
- `alseg/adapters/` contains only classes implementing interfaces. Put standalone data helpers in `alseg/data/` or another relevant module outside adapters.
- Neural-network code belongs in `nn/`, runtime/tracking in `runtime/`, and shared file writers in `core/helpers.py`.
- Load local `.env` through Pydantic Settings without an environment prefix; read Kaggle credentials with `get_secret`. Exclude API keys from configuration artifacts.
- Keep entry points in root `script/` (next to `alseg/`). Share presentation/options through `alseg/cli_common.py`; keep that file there.
- Keep future experiment orchestration in `alseg/pipeline.py` and evaluation functions in `alseg/metrics.py` when implemented.
- Core models must not import adapters or entry-point code. Use ABCs at actual component boundaries, not for every helper.
- Use `rich_click` for all CLI commands.
- In notebooks, launch scripts using `%run` (preferred) or `!python`. Do not use `subprocess` to launch scripts or shell commands.
- Distribute project source to Kaggle as a source-code dataset and update that dataset for releases. Increase `[project].version` in `pyproject.toml` before every Kaggle dataset update; never publish changed source with the same version.
- Connect notebooks directly to the extracted `source/` input directory; do not add ZIP detection/extraction or make setup diagnostics a training prerequisite.
- Keep source on the read-only Kaggle input mount. Write caches, test scratch files, and run outputs only under `/kaggle/working/` or a temporary directory.
- Use the same source dataset version, project version, and experiment config for both teammates. Restart the kernel after switching source dataset versions.
- Never commit a Kaggle Jupyter server URL or access token.
