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
