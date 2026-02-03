# Repository Name

Repository description goes here.

## Prerequisites for Local Setup

- **Python 3.13** installed on your system. You can download it from the [official Python website](https://www.python.org/downloads/). See `.python-version` file for the exact version.
- **Git** installed on your system. You can download it from the [official Git website](https://git-scm.com/downloads).
- **uv CLI** installed and on your `PATH` (see [Astral uv docs](https://docs.astral.sh/uv/)).

### How to install uv?

For all systems should work the following command:

```bash
pip install uv
```

See the [Astral uv docs](https://docs.astral.sh/uv/) for more installation options.

## How setup the project?

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Sync the dependencies:

   ```bash
   uv sync
   ```

3. Run what you want using `uv run <script>` command. For example, to update the README file with the countdown to the deadline, run:

   ```bash
   uv run src/scripts/script.py
   ```

   or directly with Python:

   ```bash
   # Activate your virtual environment first
   # (Windows)
   .venv\Scripts\Activate

   # (macOS/Linux)
   source .venv/bin/activate

   # Run the script
   python src/scripts/script.py
   ```

4. To run a Jupyter notebook, ensure you have selected the correct Python kernel in your Jupyter environment. You can do this by opening the notebook and selecting the kernel that corresponds to your virtual environment.

## Pulling updates to existing project

1. Add this repository as a remote if you haven't already:

   ```bash
   git remote add template https://github.com/sl4shuur/python-template
   ```

2. Fetch the latest changes from the template repository:

   ```bash
   git fetch template main
   ```

3. Merge the changes into your local main branch:

   ```bash
   git merge template/main --allow-unrelated-histories
   ```

4. Resolve any merge conflicts if they arise, then commit the changes.
5. Push the updated main branch to your remote repository:

   ```bash
   git push origin main
   ```

6. Remove the template remote if you no longer need it:

   ```bash
   git remote remove template
   ```
