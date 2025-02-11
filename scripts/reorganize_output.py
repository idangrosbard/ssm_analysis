import shutil
from pathlib import Path
from typing import Dict, List, Tuple

from src.consts import PATHS
from src.experiment_infra.base_config import BASE_OUTPUT_KEYS
from src.experiment_infra.output_path import OutputKey, OutputPath
from src.utils.file_system import remove_dirs_with_only_dirs


def reorganize_files(
    old_path_structure: OutputPath,
    new_path_structure: OutputPath,
    dry_run: bool = True,
    verbose: bool = True,
) -> None:
    """
    Reorganize files from old path structure to new path structure.

    Args:
        old_path_structure: The original OutputPath structure
        new_path_structure: The target OutputPath structure
        dry_run: If True, only print what would be done without making changes
        verbose: If True, print detailed information about file operations
    """
    # Verify that both path structures use the same keys
    old_keys = set(old_path_structure.get_key_names())
    new_keys = set(new_path_structure.get_key_names())

    if old_keys != new_keys:
        raise ValueError(
            f"Path structures have different keys:\n"
            f"Old structure only: {old_keys - new_keys}\n"
            f"New structure only: {new_keys - old_keys}"
        )

    def process_directory(
        current_path: Path, depth: int
    ) -> Tuple[List[Tuple[Path, Path, Dict[str, str]]], List[Tuple[Path, str]]]:
        """Process a directory at the given depth in the path structure.

        Args:
            current_path: The current directory path
            depth: Current depth in the path components
            collected_values: Values collected from parent directories

        Returns:
            List of moves to perform (old_path, new_path, values)
        """
        if not current_path.exists():
            return [], [(current_path, "Path not found")]

        if not current_path.is_dir():
            return [], [(current_path, "Not a directory, and not all components resolved")]

        sub_path = OutputPath(old_path_structure.base_path, old_path_structure.path_components[:depth])

        try:
            values = sub_path.extract_values_from_path(current_path)
        except ValueError as e:
            return [], [(current_path, str(e))]

        if depth == len(old_path_structure.path_components):

            class MockConfig:
                pass

            config = MockConfig()
            for key, value in values.items():
                setattr(config, key, value)

            new_base = new_path_structure.to_path(config)
            return [(current_path, new_base, values)], []
        else:
            moves: List[Tuple[Path, Path, Dict[str, str]]] = []
            errors: List[Tuple[Path, str]] = []
            for item in current_path.iterdir():
                res = process_directory(item, depth + 1)
                moves.extend(res[0])
                errors.extend(res[1])

            return moves, errors

    moves, errors = process_directory(old_path_structure.base_path, 0)
    if errors:
        print("Errors:")
        for path, error in errors:
            print(f"  {path.relative_to(old_path_structure.base_path)}: {error}")
    elif not moves:
        print("No files found to move.")
    else:
        print(f"Found {len(moves)} files to move:")
        for old_path, new_path, _ in moves:
            if verbose or dry_run:
                relative_old_path = old_path.relative_to(old_path_structure.base_path)
                relative_new_path = new_path.relative_to(new_path_structure.base_path)
                print(f"  {relative_old_path} -> {relative_new_path}")
            if not dry_run:
                shutil.move(str(old_path), str(new_path))

        if not dry_run:
            remove_dirs_with_only_dirs(old_path_structure.base_path)


if __name__ == "__main__":
    base_path = PATHS.PROJECT_DIR / "tests/src/experiments/baselines/full_pipeline/output"
    old_path_structure = OutputPath(
        base_path,
        [
            BASE_OUTPUT_KEYS.MODEL_ARCH,
            BASE_OUTPUT_KEYS.MODEL_SIZE,
            BASE_OUTPUT_KEYS.EXPERIMENT_NAME,
            OutputKey(key_name="variation", key_display_name=""),
            BASE_OUTPUT_KEYS.VARIATION,
        ],
    )
    new_path_structure = OutputPath(
        base_path,
        [
            BASE_OUTPUT_KEYS.EXPERIMENT_NAME,
            BASE_OUTPUT_KEYS.VARIATION,
            BASE_OUTPUT_KEYS.MODEL_ARCH,
            BASE_OUTPUT_KEYS.MODEL_SIZE,
        ],
    )
    dry_run = False
    reorganize_files(old_path_structure, new_path_structure, dry_run=dry_run)
    # reorganize_files(new_path_structure, old_path_structure, dry_run=dry_run)
