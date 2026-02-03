import os
from dotenv import load_dotenv

try:
    from .config_helper import find_project_root
except ImportError:
    from config_helper import find_project_root

# Base of the project
BASE_DIR = find_project_root()
DATA_DIR = BASE_DIR / "data"

# # .env file loading
# dotenv_path = BASE_DIR / ".env"
# if not dotenv_path.exists():
#     raise FileNotFoundError(f"Environment file not found: {dotenv_path}")

# load_dotenv(dotenv_path)

# SOME_API_KEY = os.getenv("SOME_API_KEY")

# check if the directories exist, if not create them
for directory in [DATA_DIR]:
    if directory.exists() and directory.is_file():
        directory.unlink()  # Remove the file if a file exists with the same name
    directory.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    print(f"Base directory: {BASE_DIR}")