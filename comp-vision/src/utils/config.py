try:
    from .config_helper import find_project_root
except ImportError:
    from config_helper import find_project_root

# Base of the project
BASE_DIR = find_project_root()
DATA_DIR = BASE_DIR / "data"

# check if the directories exist, if not create them
for directory in [DATA_DIR]:
    if directory.exists() and directory.is_file():
        directory.unlink()  # Remove the file if a file exists with the same name
    directory.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    print(f"Base directory: {BASE_DIR}")