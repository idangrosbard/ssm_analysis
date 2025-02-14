import shutil
from typing import Dict, Optional

from src.consts import PATHS
from src.experiment_infra.base_config import BASE_OUTPUT_KEYS
from src.experiment_infra.output_path import OutputKey, OutputPath, dict_to_obj
from src.utils.file_system import remove_dirs_with_only_dirs


def is_filtered(values: Dict[str, str], filter_values: Dict[str, list[str]]) -> bool:
    return all(values[key] in filter_values[key] for key in filter_values)


def reorganize_files(
    old_path_structure: OutputPath,
    new_path_structure: OutputPath,
    filter_values: Optional[Dict[str, list[str]]] = None,
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

    in_pattern, out_of_pattern = old_path_structure.process_path()
    assert all([p.is_dir() for p, _ in in_pattern])
    moves = [(old_path, new_path_structure.to_path(dict_to_obj(values)), values) for old_path, values in in_pattern]
    if out_of_pattern:
        print("Errors:")
        for path, error in out_of_pattern:
            print(f"  {path.relative_to(old_path_structure.base_path)}: {error}")
    elif not moves:
        print("No files found to move.")
    else:
        print(f"Found {len(moves)} files to move:{' (dry run)' if dry_run else ''}")
        for old_path, new_path, values in moves:
            should_skip = not ((filter_values is None) or is_filtered(values, filter_values))
            if verbose or dry_run:
                relative_old_path = old_path.relative_to(old_path_structure.base_path)
                relative_new_path = new_path.relative_to(new_path_structure.base_path)
                print(f"  ({'Skipped' if should_skip else ''}) {relative_old_path} -> {relative_new_path}")
            if not dry_run and not should_skip:
                shutil.move(str(old_path), str(new_path))

        if not dry_run:
            remove_dirs_with_only_dirs(old_path_structure.base_path)


if __name__ == "__main__":
    base_path = PATHS.OUTPUT_DIR
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
        # base_path.parent/ "output.temp",
        base_path,
        # base_path,
        [
            BASE_OUTPUT_KEYS.EXPERIMENT_NAME,
            BASE_OUTPUT_KEYS.VARIATION,
            BASE_OUTPUT_KEYS.MODEL_ARCH,
            BASE_OUTPUT_KEYS.MODEL_SIZE,
        ],
    )
    dry_run = True
    # dry_run = True
    filter_values = {
        BASE_OUTPUT_KEYS.MODEL_ARCH.key_name: ["mamba1"],
        BASE_OUTPUT_KEYS.MODEL_SIZE.key_name: ["130M"],
        BASE_OUTPUT_KEYS.EXPERIMENT_NAME.key_name: ["evaluate"],
    }
    # filter_values = None
    reorganize_files(new_path_structure, new_path_structure, filter_values=filter_values, dry_run=dry_run)
    # reorganize_files(old_path_structure, new_path_structure, filter_values=filter_values, dry_run=dry_run)
    # reorganize_files(new_path_structure, old_path_structure, filter_values=filter_values, dry_run=dry_run)
