from pathlib import Path


def find_project_root(
    start_path: Path | None = None,
    markers: tuple[str, ...] = (
        ".git",
        ".gitignore",
        ".venv",
        "main.py",
        "requirements.txt",
        "pyproject.toml",
        "poetry.lock",
        "uv.lock",
        ".python-version"
    ),
    max_depth: int = 3,
) -> Path:
    r"""
    Find project root by searching for marker files/directories.

    Starts from the given path and walks up the directory tree until
    it finds a directory containing one of the marker files/directories.

    Args:
        start_path: Starting directory (defaults to current file's directory)
        markers: Tuple of file/directory names to search for
        max_depth: Maximum number of parent directories to check

    Returns:
        Path: Project root directory

    Raises:
        RuntimeError: If project root not found within max_depth

    Example:
        >>> root = find_project_root()
        >>> print(root)
        c:\GitHub\python-template
    """
    if start_path is None:
        start_path = Path(__file__).parent

    for _ in range(max_depth):
        # Check if any marker exists in current directory
        for marker in markers:
            if (start_path / marker).exists():
                return start_path

        # Move up one directory
        parent = start_path.parent

        # Stop if reached filesystem root
        if parent == start_path:
            break

        start_path = parent

    raise RuntimeError(
        f"Project root not found within {max_depth} parent directories. "
        f"Searched for markers: {markers}"
    )
