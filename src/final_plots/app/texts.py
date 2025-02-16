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
    show_combination = "Show Combination"
    show_prompts = "Show Prompts"
    active_models_str = lambda active_models: " ∩ ".join(active_models)
    run_selected_prompts = lambda count: f"🚀 Run {count} Selected Jobs"
    processing_status = lambda current, total: COMMON_TEXTS.processing_status(current, total, "jobs")
    success_status = lambda count: COMMON_TEXTS.success_status(count, "jobs")
    error_status = lambda count: COMMON_TEXTS.error_status(count, "jobs")
    submit_failed = lambda prompt_idx, error: f"Failed to submit job for prompt {prompt_idx}: {error}"
    run_selected_prompts_button = lambda count: f"🚀 Run {count} Selected Prompts"
    run_selected_prompts_success = "Successfully submitted jobs"
    run_selected_prompts_error = "Failed to submit jobs"
    show_models_with_correctness = (
        lambda is_correct, models: f"{'✅' if is_correct else '❌'} ({len(models)}): {', '.join(models)}"
    )
    skipping_running = (
        lambda model_arch, model_size: f"Skipping running {model_arch} {model_size} because it is already running"
    )


class INFO_FLOW_TEXTS:
    icon = "📈"
    title = "Info Flow Visualization"
    plot_config_title = "Plot Configuration"
    show_data_sources = "Show Data Sources Tree"
    total_experiments = lambda count: f"Total experiments: {count}"
    generate_plots = "Generate Plots"
    generating_plots = "Generating plots..."
    plots_generated = lambda count: f"Successfully generated {count} plots"
    no_plots_generated = "No plots could be generated. Please check the error details above."


class AppGlobalText:
    window_size = "Window Size"
    gpu_type = "GPU Type"
    variation = "Variation"


class RESULTS_BANK_TEXTS:
    icon = "📋"
    title = "Results Bank"


class DATA_REQUIREMENTS_TEXTS:
    icon = "📊"
    title = "Data Requirements and Overrides"
    update_requirements = "🔄 Update Latest Requirements"
    requirements_updated = "Requirements updated successfully!"
    save_overrides = "💾 Save Overrides"
    overrides_saved = "Overrides saved successfully!"


class COMMON_TEXTS:
    error_details = "Show Error Details"
    preparing_to_run = "Preparing to run"
    processing_status = lambda current, total, run_what: f"Processed {current} / {total} {run_what}..."
    success_status = lambda count, run_what: f"Successfully submitted {count} {run_what}..."
    error_status = lambda count, run_what: f"Failed to submit {count} {run_what}..."
    submit_failed = lambda prompt_idx, error, run_what: f"Failed to submit job for {prompt_idx} {run_what}: {error}"
