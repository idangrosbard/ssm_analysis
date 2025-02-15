# ruff: noqa: E731
def show_filter_results_text(filtered_count: int, all_count: int) -> str:
    return f"{filtered_count} / {all_count} ({filtered_count / all_count * 100}%)"


class HEATMAP_TEXTS:
    icon = "🔥"
    title = "Heatmap Creation"
    description = "Create heatmaps for the selected prompts"
    matching_counts = lambda len_filtered_prompts, len_all_prompts: (
        f"Found {show_filter_results_text(len_filtered_prompts, len_all_prompts)} prompts matching all criteria"
    )
