from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import product
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    cast,
)

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.analysis.experiment_results.helpers import init_variant_params_from_values
from src.analysis.experiment_results.hyper_param_definition import (
    HyperParamDefinition,
    PossibleDerivedHPDTypes,
    PossibleHPDTypes,
    PromptFilterationHPD,
    TExperimentHyperParams,
    VirtualExperimentHyperParams,
    get_hyper_param_definition,
)
from src.analysis.plots.heatmaps import HeatmapPlotConfig
from src.analysis.plots.image_combiner import ImageGridParams
from src.analysis.plots.info_flow_confidence import InfoFlowPlotConfig
from src.core.names import (
    VARIANT_PARAM_NAME,
    BaseVariantParamName,
    ExperimentName,
    FinalPlotsPlanOrientation,
    ToClassifyNames,
)
from src.core.types import (
    MODEL_ARCH,
    MODEL_ARCH_AND_SIZE,
    TModelSize,
    TPlotID,
)
from src.data_ingestion.data_defs.data_defs import (
    DataReqiermentCollection,
    DataReqs,
    PlotPlans,
    ResultBank,
)
from src.utils.types_utils import str_enum_values


def get_experiment_orientations(
    experiment_name: ExperimentName,
) -> list[FinalPlotsPlanOrientation]:
    """Get the relevant parameters for a specific experiment type."""
    if experiment_name == ExperimentName.info_flow:
        return [
            FinalPlotsPlanOrientation.lines,
            FinalPlotsPlanOrientation.rows,
            FinalPlotsPlanOrientation.cols,
            FinalPlotsPlanOrientation.grids,
        ]
    elif experiment_name == ExperimentName.heatmap:
        return [
            FinalPlotsPlanOrientation.rows,
            FinalPlotsPlanOrientation.cols,
            FinalPlotsPlanOrientation.grids,
        ]
    else:
        return list(FinalPlotsPlanOrientation)


T_HPD_OPTION = VARIANT_PARAM_NAME | Literal[ToClassifyNames.prompt_filteration]


def get_experiment_hyper_param_hyper_param(
    experiment_name: ExperimentName,
) -> list[TExperimentHyperParams]:
    """Get the relevant parameters for a specific experiment type."""
    return [
        VirtualExperimentHyperParams.model_arch_and_size,
        *experiment_name.get_variant_cols(experiment_name),
    ]


@dataclass(frozen=True)
class Cell:
    grids: Optional[PossibleHPDTypes] = None
    rows: Optional[PossibleHPDTypes] = None
    cols: Optional[PossibleHPDTypes] = None

    @classmethod
    def from_orientation_combination(
        cls, orientation_combination: dict[FinalPlotsPlanOrientation, PossibleHPDTypes | None]
    ) -> Cell:
        """Create a Cell from an orientation combination dictionary."""
        return cls(
            grids=orientation_combination.get(FinalPlotsPlanOrientation.grids),
            rows=orientation_combination.get(FinalPlotsPlanOrientation.rows),
            cols=orientation_combination.get(FinalPlotsPlanOrientation.cols),
        )

    def get_field_display_name(self, field: str, plot_plan: PlotPlan) -> Optional[str]:
        """Get the display name for a field value using the plot plan's parameter definition."""
        value = getattr(self, field)
        if value is None:
            return None

        param_config = plot_plan.get_param_config_by_orientation(FinalPlotsPlanOrientation[field])
        if param_config is None:
            return str(value)

        return get_hyper_param_definition(param_config.param).get_display_name(value)

    def to_dict(self) -> dict[str, PossibleHPDTypes]:
        """Convert cell to dictionary for data requirements."""
        return {FinalPlotsPlanOrientation[field]: getattr(self, field) for field in ["grids", "rows", "cols"]}

    def get_display_name(self, plot_plan: PlotPlan) -> str:
        """Get the display name for the cell."""
        display_names = {
            field: self.get_field_display_name(field, plot_plan) or "None" for field in ["grids", "rows", "cols"]
        }

        # Create a unique identifier for the cell
        return "_".join(f"{value}" for key, value in display_names.items()).replace(" ", "_")


class ParamConfig(BaseModel):
    """Configuration for a hyperparameter in the plot plan."""

    param: TExperimentHyperParams
    orientation: Optional[FinalPlotsPlanOrientation] = None
    values: List[PossibleHPDTypes] = Field(default_factory=list)

    @property
    def fixed_value(self) -> PossibleHPDTypes:
        """Return the first value if this is a fixed parameter (orientation is None and exactly one value)."""
        assert self.is_fixed()
        return self.values[0]

    def get_param_def(self) -> HyperParamDefinition:
        """Get the HyperParamDefinition for this parameter."""
        return get_hyper_param_definition(self.param)

    @model_validator(mode="after")  # type: ignore
    def validate_param_values(self) -> "ParamConfig":
        """Validate that values match the expected type for the param."""
        # Skip validation during initialization
        # If orientation is None, values must have exactly one item or be empty
        if self.orientation is None and len(self.values) > 1:
            raise ValueError("Fixed parameters (orientation=None) must have exactly one or zero values")

        param_def = self.get_param_def()

        # Validate values if provided
        if self.values:
            for value in self.values:
                # Try to get the display name to ensure the value is valid for this param
                try:
                    param_def.get_display_name(value)
                except Exception as e:
                    raise ValueError(f"Invalid value {value} for parameter {self.param}: {str(e)}")

        return self

    def is_fixed(self) -> bool:
        """Check if this parameter is fixed (not variable across an orientation)."""
        return self.orientation is None and len(self.values) == 1

    def is_variable(self) -> bool:
        """Check if this parameter varies across an orientation."""
        return self.orientation is not None

    def get_values(self) -> List[dict[VARIANT_PARAM_NAME, PossibleDerivedHPDTypes]]:
        """Get the values for this parameter, either specified or from the result bank."""
        return [self.get_param_def().get_derived_hpds(value) for value in self.values]


class PlotPlan(BaseModel):
    """A plan for plotting experiment results in a grid layout."""

    model_config = ConfigDict(validate_assignment=True)

    plot_id: TPlotID
    title: str = Field(default="")
    description: str = Field(default="")
    order: int
    observation: str = Field(default="")
    notes: str = Field(default="")
    is_appendix: bool = Field(default=False)

    # Use proper type annotation for params
    experiment_name: ExperimentName
    params: List[ParamConfig] = Field(default_factory=list)

    # Plot configuration
    cell_plot_config: InfoFlowPlotConfig | HeatmapPlotConfig = Field(default_factory=dict)
    combine_plot_config: ImageGridParams = Field(default_factory=ImageGridParams)

    @model_validator(mode="after")  # type: ignore
    def validate_param_configs(self) -> PlotPlan:
        """Validate the parameter configurations."""
        # Skip validation for empty models or during initialization
        if not self.params:
            return self

        # Check for duplicate parameters with the same orientation
        orientation_to_param: Dict[FinalPlotsPlanOrientation, TExperimentHyperParams] = {}
        for config in self.params:
            if config.orientation is not None:
                if config.orientation in orientation_to_param:
                    raise ValueError(
                        f"Duplicate orientation {config.orientation} for parameters "
                        f"{orientation_to_param[config.orientation]} and {config.param}"
                    )
                orientation_to_param[config.orientation] = config.param

        # Validate that model_arch and model_size are present if needed
        has_model_arch = any(config.param == BaseVariantParamName.model_arch for config in self.params)
        has_model_size = any(config.param == BaseVariantParamName.model_size for config in self.params)
        has_model_arch_and_size = any(
            config.param == VirtualExperimentHyperParams.model_arch_and_size for config in self.params
        )

        if not (has_model_arch_and_size or (has_model_arch and has_model_size)):
            raise ValueError("Either model_arch_and_size or both model_arch and model_size must be specified")

        return self

    def get_param_config(self, param: TExperimentHyperParams) -> Optional[ParamConfig]:
        """Get the configuration for a specific parameter."""
        for config in self.params:
            if config.param == param:
                return config
        return None

    def get_param_config_by_orientation(self, orientation: FinalPlotsPlanOrientation) -> Optional[ParamConfig]:
        """Get the parameter configuration associated with a specific orientation."""
        for config in self.params:
            if config.orientation == orientation:
                return config
        return None

    def get_orientation_value_hpd(self, orientation: FinalPlotsPlanOrientation) -> Optional[HyperParamDefinition]:
        """Get the HyperParamDefinition for a specific orientation."""
        config = self.get_param_config_by_orientation(orientation)
        if config is None:
            return None
        return get_hyper_param_definition(config.param)

    def get_options_for_orientation(
        self, orientation: FinalPlotsPlanOrientation, result_bank: ResultBank
    ) -> List[PossibleHPDTypes] | tuple[None]:
        """Get the options for a specific orientation."""
        config = self.get_param_config_by_orientation(orientation)
        if config is None:
            return (None,)
        return config.values

    def get_summary(self) -> Dict[FinalPlotsPlanOrientation, list[str]]:
        """Get a summary of the plot structure."""
        result: Dict[FinalPlotsPlanOrientation, list[str]] = {}
        for orientation in str_enum_values(FinalPlotsPlanOrientation):
            config = self.get_param_config_by_orientation(orientation)
            if config is None:
                result[orientation] = []
                continue

            param_def = get_hyper_param_definition(config.param)
            result[orientation] = [param_def.get_display_name(option) for option in config.values or []]

        return result

    def get_data_requirements_per_cell(self, result_bank: ResultBank) -> dict[Cell, DataReqs]:
        """Generate data requirements for this plot plan based on the result bank."""
        data_reqs_per_cell: dict[Cell, DataReqiermentCollection] = defaultdict(DataReqiermentCollection)
        experiment_orientations = get_experiment_orientations(self.experiment_name)

        # Get options for each orientation
        orientation_options = {
            orientation: self.get_options_for_orientation(orientation, result_bank)
            for orientation in experiment_orientations
        }

        # Generate all combinations of orientation values
        combinations = product(*[orientation_options[orientation] for orientation in experiment_orientations])

        # Process each combination
        for combination in combinations:
            orientation_combination = {
                orientation: combination[i] for i, orientation in enumerate(experiment_orientations)
            }
            cell = Cell.from_orientation_combination(orientation_combination)

            # Collect all parameter values for this cell
            data_req_params: dict[VARIANT_PARAM_NAME, Any] = {
                BaseVariantParamName.experiment_name: self.experiment_name,
            }
            prompt_filterations: list[tuple[PromptFilterationHPD, PossibleHPDTypes]] = []

            for param_config in self.params:
                hpd = param_config.get_param_def()
                if param_config.is_fixed():
                    value = param_config.values[0]
                else:
                    assert param_config.orientation is not None
                    value = orientation_combination[param_config.orientation]
                    assert value is not None

                if isinstance(hpd, PromptFilterationHPD):
                    prompt_filterations.append((hpd, value))
                else:
                    for k, v in hpd.get_derived_hpds(value).items():
                        if k in data_req_params:
                            raise ValueError(f"Duplicate parameter {k}")
                        data_req_params[k] = v

            assert len(prompt_filterations) == 1
            prompt_filteration_hpd = prompt_filterations[0][0]
            prompt_filteration_value = prompt_filterations[0][1]

            prompt_filteration, more_values = prompt_filteration_hpd.get_derived_hpd_with_context(
                prompt_filteration_value, self.derive_model_arch_and_sizes_context()
            )

            data_req_params.update(more_values)

            data_reqs_per_cell[cell].add_data_req(init_variant_params_from_values(data_req_params), prompt_filteration)

        return {cell: DataReqs.from_data_reqs_collection(data_reqs) for cell, data_reqs in data_reqs_per_cell.items()}

    def get_data_requirements(self, result_bank: ResultBank) -> DataReqs:
        """Generate aggregated data requirements for the entire plot plan."""
        data_reqs_per_cell = self.get_data_requirements_per_cell(result_bank)
        data_reqs_collection = DataReqiermentCollection()

        for data_reqs_per_cell in data_reqs_per_cell.values():
            for data_req, prompt_filteration in data_reqs_per_cell.items():
                data_reqs_collection.add_data_req(data_req, prompt_filteration)

        return DataReqs.from_data_reqs_collection(data_reqs_collection)

    def derive_model_arch_and_sizes_context(self) -> list[MODEL_ARCH_AND_SIZE]:
        """Derive context model architectures and sizes."""
        models: List[MODEL_ARCH] = []
        sizes: List[TModelSize] = []

        # Check if model_arch_and_size is directly configured
        for config in self.params:
            if config.param == VirtualExperimentHyperParams.model_arch_and_size:
                if config.values:
                    # Filter to ensure we only return MODEL_ARCH_AND_SIZE values
                    return [value for value in config.values if isinstance(value, tuple) and len(value) == 2]  # type: ignore
                if config.fixed_value is not None:
                    return [config.fixed_value]  # type: ignore

        # Otherwise collect model arch and model size separately
        for config in self.params:
            if config.param == BaseVariantParamName.model_arch:
                if config.is_variable() and config.values:
                    models = [cast(MODEL_ARCH, model) for model in config.values]
                elif config.is_fixed():
                    models = [cast(MODEL_ARCH, config.fixed_value)]
            elif config.param == BaseVariantParamName.model_size:
                if config.is_variable() and config.values:
                    sizes = [cast(TModelSize, size) for size in config.values]
                elif config.is_fixed():
                    sizes = [cast(TModelSize, config.fixed_value)]

        # Generate all combinations
        return [MODEL_ARCH_AND_SIZE(model, size) for model, size in product(models, sizes)]

    def get_option_display_names_for_orientation(self, orientation: FinalPlotsPlanOrientation) -> list[str]:
        """Get display names for options for an orientation (compatibility method)."""
        config = self.get_param_config_by_orientation(orientation)
        if config is None:
            return []

        param_def = get_hyper_param_definition(config.param)
        return [param_def.get_display_name(option) for option in config.values or []]

    def get_options_for_param(self, orientation: FinalPlotsPlanOrientation) -> List[PossibleHPDTypes]:
        """Get the selected options for a parameter (compatibility method)."""
        config = self.get_param_config_by_orientation(orientation)
        if config is None:
            return []
        return config.values or []

    def save(self) -> None:
        """Save the parameter configuration."""
        PlotPlans.save_plot_plan(self)
