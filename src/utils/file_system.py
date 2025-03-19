from pathlib import Path


def remove_dirs_with_only_dirs(path: Path):
    """
    Recursively removes directories that contain only other directories.
    """
    if not path.is_dir():
        return

    # Process subdirectories first (post-order traversal)
    for subdir in list(path.iterdir()):
        if subdir.is_dir():
            remove_dirs_with_only_dirs(subdir)

    # Check if the directory now contains only other directories
    if all(item.is_dir() for item in path.iterdir()):
        try:
            path.rmdir()
        except OSError:
            pass  # Directory not empty due to permissions or race conditions


def fast_relative_to(path: Path, base_path: Path, allow_slow: bool = False) -> Path:
    """
    Get the relative path of a file or directory to a base path.
    """
    if allow_slow:
        return path.relative_to(base_path)
    else:
        base_parts = base_path.parts
        base_len = len(base_parts)
        path_parts = path.parts
        assert path_parts[:base_len] == base_parts
        return Path(*path_parts[base_len:])
