from src.data_defs import PlotPlans
from src.final_plots.plot_plan import EXPERIMENT_NAMES, ExperimentHyperParams, PlotPlan, PlotType


def get_default_plot_plans() -> PlotPlans:
    # Default plot plans
    default_plot_plans = PlotPlans({})

    # Architecture Knockout Plot
    default_plot_plans.add_plan(
        PlotPlan(
            title="Architecture Knockout Plot",
            description=(
                "Relative change in correct-token prediction probability"
                " when removing information flow to the last token from various source tokens."
                " The x-axis represents the relative depth of the first layer within the 9-layer attention"
                " knockout window, while the y-axis indicates the resulting performance change."
            ),
            plot_type=PlotType.ARCHITECTURE_KNOCKOUT,
            is_appendix=False,
            order=0,
            experiment_name=EXPERIMENT_NAMES.INFO_FLOW,
            lines=ExperimentHyperParams.model_arch_and_size,
            cols=ExperimentHyperParams.source,
            output_path="results/final_plots/architecture_knockout.png",
        )
    )

    # Model Size Knockout Plot
    default_plot_plans.add_plan(
        PlotPlan(
            title="Model Size Knockout Plot",
            description="""
            Comparison of different model sizes within the same architecture family when removing information flow
            to the last token from various source tokens.
            """,
            plot_type=PlotType.MODEL_SIZE_KNOCKOUT,
            is_appendix=False,
            order=0,
            experiment_name=EXPERIMENT_NAMES.INFO_FLOW,
            lines=ExperimentHyperParams.model_arch_and_size,
            cols=ExperimentHyperParams.source,
            output_path="results/final_plots/model_size_knockout.png",
        )
    )

    # Window Size Knockout Plot
    default_plot_plans.add_plan(
        PlotPlan(
            title="Window Size Knockout Plot",
            description="""
            Effect of different window sizes on the information flow from various source tokens to the last token.
            """,
            plot_type=PlotType.WINDOW_SIZE_KNOCKOUT,
            is_appendix=False,
            order=0,
            experiment_name=EXPERIMENT_NAMES.INFO_FLOW,
            lines=ExperimentHyperParams.window_size,
            cols=ExperimentHyperParams.source,
            output_path="results/final_plots/window_size_knockout.png",
        )
    )

    # Feature Knockout Plot
    default_plot_plans.add_plan(
        PlotPlan(
            title="Feature Knockout Plot",
            description=(
                "Comparison of model performance when using all features"
                " versus only context-dependent or context-independent features."
            ),
            plot_type=PlotType.FEATURE_KNOCKOUT,
            is_appendix=False,
            order=0,
            experiment_name=EXPERIMENT_NAMES.INFO_FLOW,
            lines=ExperimentHyperParams.feature_category,
            cols=ExperimentHyperParams.model_arch_and_size,
            output_path="results/final_plots/feature_knockout.png",
        )
    )

    # Shared Knockout Plot
    default_plot_plans.add_plan(
        PlotPlan(
            title="Shared Knockout Plot",
            description="""
            Comparison of the effect of knocking out shared components across different models.
            """,
            plot_type=PlotType.SHARED_KNOCKOUT,
            is_appendix=False,
            order=0,
            experiment_name=EXPERIMENT_NAMES.INFO_FLOW,
            lines=ExperimentHyperParams.source,
            cols=ExperimentHyperParams.model_arch_and_size,
            output_path="results/final_plots/shared_knockout.png",
        )
    )

    # Heatmap Plot
    default_plot_plans.add_plan(
        PlotPlan(
            title="Heatmap Plot",
            description="""
            Heatmap visualization of information flow between tokens for specific prompts.
            """,
            plot_type=PlotType.HEATMAP,
            is_appendix=False,
            order=0,
            experiment_name=EXPERIMENT_NAMES.HEATMAP,
            cols=ExperimentHyperParams.model_arch_and_size,
            output_path="results/final_plots/heatmap.png",
        )
    )

    return default_plot_plans
