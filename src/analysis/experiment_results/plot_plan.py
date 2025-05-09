from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import (
    Any,
    Dict,
    ForwardRef,
    List,
    Literal,
    Optional,
    Sequence,
    Union,
    assert_never,
    cast,
)

from pydantic import BaseModel, Field, model_validator

from src.analysis.experiment_results.helpers import init_variant_params_from_values
from src.analysis.experiment_results.hyper_param_definition import (
    HyperParamDefinition,
    PossibleHPDTypes,
    PromptFilterationFactory,
    get_hyper_param_definition,
)
from src.analysis.plots.image_combiner import ImageGridParams
from src.core.consts import GRAPHS_ORDER
from src.core.names import (
    VARIANT_PARAM_NAME,
    BaseVariantParamName,
    ExperimentHyperParams,
    ExperimentName,
    FinalPlotsPlanOrientation,
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
    ResultBank,
)
from src.experiments.infrastructure.base_prompt_filteration import (
    SelectivePromptFilteration,
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


def get_experiment_hyper_param_hyper_param(
    experiment_name: ExperimentName,
) -> list[ExperimentHyperParams]:
    """Get the relevant parameters for a specific experiment type."""
    general = [
        ExperimentHyperParams.model_arch_and_size,
        ExperimentHyperParams.window_size,
        ExperimentHyperParams.model_arch,
        ExperimentHyperParams.model_size,
    ]
    match experiment_name:
        case ExperimentName.info_flow:
            general += [
                ExperimentHyperParams.source,
                ExperimentHyperParams.target,
                ExperimentHyperParams.feature_category,
            ]
        case ExperimentName.heatmap:
            pass
        case ExperimentName.evaluate_model | ExperimentName.full_pipeline:
            raise NotImplementedError(f"Not implemented for {experiment_name}")
        case _:
            assert_never(experiment_name)
    return general


@dataclass(frozen=True)
class Cell:
    grids: Optional[PossibleHPDTypes] = None
    rows: Optional[PossibleHPDTypes] = None
    cols: Optional[PossibleHPDTypes] = None

    @classmethod
    def from_orientation_combination(
        cls, orientation_combination: dict[FinalPlotsPlanOrientation, PossibleHPDTypes]
    ) -> Cell:
        """Create a Cell from an orientation combination dictionary."""
        return cls(
            grids=orientation_combination.get(FinalPlotsPlanOrientation.grids),
            rows=orientation_combination.get(FinalPlotsPlanOrientation.rows),
            cols=orientation_combination.get(FinalPlotsPlanOrientation.cols),
        )

    def get_display_name(self, field: str, plot_plan: PlotPlan) -> Optional[str]:
        """Get the display name for a field value using the plot plan's parameter definition."""
        value = getattr(self, field)
        if value is None:
            return None

        param_config = plot_plan.get_param_config_by_orientation(FinalPlotsPlanOrientation[field])
        if param_config is None:
            return str(value)

        return get_hyper_param_definition(param_config.param).get_display_name(value)

    def get_cache_path(self, plot_plan: PlotPlan, cache_dir: Path) -> Path:
        """Generate a unique cache path for this cell."""
        # Get display names for each field
        display_names = {
            field: self.get_display_name(field, plot_plan) or "None" for field in ["grids", "rows", "cols"]
        }

        # Create a unique identifier for the cell
        cell_id = "_".join(f"{value}" for key, value in display_names.items()).replace(" ", "_")
        return cache_dir / f"{cell_id}.png"

    def to_dict(self) -> dict[str, PossibleHPDTypes]:
        """Convert cell to dictionary for data requirements."""
        return {FinalPlotsPlanOrientation[field]: getattr(self, field) for field in ["grids", "rows", "cols"]}


# Use ForwardRef for self-referential types in ParamConfig
ParamConfigRef = ForwardRef("ParamConfig")


class ParamConfig(BaseModel):
    """Configuration for a hyperparameter in the plot plan."""

    param: ExperimentHyperParams
    orientation: Optional[FinalPlotsPlanOrientation] = None
    values: List[PossibleHPDTypes] = Field(default_factory=list)

    @property
    def fixed_value(self) -> Optional[PossibleHPDTypes]:
        """Return the first value if this is a fixed parameter (orientation is None and exactly one value)."""
        assert self.is_fixed()
        return self.values[0]

    @model_validator(mode="after")  # type: ignore
    def validate_param_values(self) -> "ParamConfig":
        """Validate that values match the expected type for the param."""
        # Skip validation during initialization
        if not hasattr(self, "param") or self.param is None:
            return self

        # If orientation is None, values must have exactly one item or be empty
        if self.orientation is None and len(self.values) > 1:
            raise ValueError("Fixed parameters (orientation=None) must have exactly one or zero values")

        param_def = get_hyper_param_definition(self.param)

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

    def get_values(self, result_bank: ResultBank) -> List[PossibleHPDTypes]:
        """Get the values for this parameter, either specified or from the result bank."""
        if self.values:
            return self.values

        # Use all available values from the result bank
        param_def = get_hyper_param_definition(self.param)
        return list(param_def.get_options(result_bank))


class PlotPlan(BaseModel):
    """A plan for plotting experiment results in a grid layout."""

    plot_id: TPlotID
    title: str
    description: str
    is_appendix: bool
    order: int
    experiment_name: ExperimentName

    # Use proper type annotation for params
    params: List[ParamConfig] = Field(default_factory=list)

    # Plot configuration
    cell_plot_config: Dict[str, Any] = Field(default_factory=dict)
    combine_plot_config: ImageGridParams = Field(default_factory=ImageGridParams)

    @model_validator(mode="after")  # type: ignore
    def validate_param_configs(self) -> PlotPlan:
        """Validate the parameter configurations."""
        # Skip validation for empty models or during initialization
        if not self.params:
            return self

        # Check for duplicate parameters with the same orientation
        orientation_to_param: Dict[FinalPlotsPlanOrientation, ExperimentHyperParams] = {}
        for config in self.params:
            if config.orientation is not None:
                if config.orientation in orientation_to_param:
                    raise ValueError(
                        f"Duplicate orientation {config.orientation} for parameters "
                        f"{orientation_to_param[config.orientation]} and {config.param}"
                    )
                orientation_to_param[config.orientation] = config.param

        # Validate that model_arch and model_size are present if needed
        has_model_arch = any(config.param == ExperimentHyperParams.model_arch for config in self.params)
        has_model_size = any(config.param == ExperimentHyperParams.model_size for config in self.params)
        has_model_arch_and_size = any(
            config.param == ExperimentHyperParams.model_arch_and_size for config in self.params
        )

        if not (has_model_arch_and_size or (has_model_arch and has_model_size)):
            raise ValueError("Either model_arch_and_size or both model_arch and model_size must be specified")

        return self

    def get_param_config(self, param: ExperimentHyperParams) -> Optional[ParamConfig]:
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

    def get_options_for_orientation(self, orientation: FinalPlotsPlanOrientation, result_bank: ResultBank) -> List[Any]:
        """Get the options for a specific orientation."""
        config = self.get_param_config_by_orientation(orientation)
        if config is None:
            return [None]
        return config.get_values(result_bank)

    def get_fixed_values(self) -> Dict[ExperimentHyperParams, PossibleHPDTypes]:
        """Get all fixed parameter values."""
        return {
            config.param: config.fixed_value
            for config in self.params
            if config.is_fixed() and config.fixed_value is not None
        }

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
        orientation_options: dict[FinalPlotsPlanOrientation, list[Any]] = {
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

            # Collect all parameter values for this cell
            params: dict[ExperimentHyperParams, Any] = {}

            # Add values from orientations
            for orientation, value in orientation_combination.items():
                if value is not None:
                    config = self.get_param_config_by_orientation(orientation)
                    if config is not None:
                        params[config.param] = value

            # Add fixed values
            params.update(self.get_fixed_values())

            # Handle the special case of model_arch_and_size
            if params.get(ExperimentHyperParams.model_arch_and_size) is not None:
                model_arch_and_size = cast(
                    MODEL_ARCH_AND_SIZE,
                    params[ExperimentHyperParams.model_arch_and_size],
                )
                params[ExperimentHyperParams.model_arch] = model_arch_and_size.arch
                params[ExperimentHyperParams.model_size] = model_arch_and_size.size
            else:
                assert params.get(ExperimentHyperParams.model_arch) is not None
                assert params.get(ExperimentHyperParams.model_size) is not None
                model_arch_and_size = MODEL_ARCH_AND_SIZE(
                    params[ExperimentHyperParams.model_arch],
                    params[ExperimentHyperParams.model_size],
                )
                if model_arch_and_size not in GRAPHS_ORDER:
                    continue
                params[ExperimentHyperParams.model_arch_and_size] = model_arch_and_size

            if self.experiment_name == ExperimentName.heatmap:
                prompt_filterations = SelectivePromptFilteration(
                    prompt_ids=tuple([params[ExperimentHyperParams.prompt_idx]])
                )
            elif self.experiment_name == ExperimentName.info_flow:
                item = params[ExperimentHyperParams.filteration_factory]
                assert isinstance(item, PromptFilterationFactory)
                prompt_filterations = item.get_filteration(self.derive_model_arch_and_sizes_context())
            else:
                raise NotImplementedError(f"Not implemented for {self.experiment_name}")

            data_req_params: dict[VARIANT_PARAM_NAME, Any] = {
                BaseVariantParamName.experiment_name: self.experiment_name,
            }

            for col in ExperimentName.get_variant_cols(self.experiment_name):
                if col in data_req_params:
                    continue
                if col in params:
                    data_req_params[col] = params[ExperimentHyperParams[col]]
                elif col in get_experiment_hyper_param_hyper_param(self.experiment_name):
                    hpd_col = ExperimentHyperParams(col)
                    fixed_values = self.get_fixed_values()
                    if hpd_col in fixed_values:
                        data_req_params[col] = fixed_values[hpd_col]

            cell = Cell.from_orientation_combination(orientation_combination)
            data_reqs_per_cell[cell].add_data_req(init_variant_params_from_values(data_req_params), prompt_filterations)

        return {cell: DataReqs.from_data_reqs_collection(data_reqs) for cell, data_reqs in data_reqs_per_cell.items()}

    def get_data_requirements(self, result_bank: ResultBank) -> DataReqs:
        """Generate aggregated data requirements for the entire plot plan."""
        data_reqs_per_cell = self.get_data_requirements_per_cell(result_bank)
        data_reqs_collection = DataReqiermentCollection()

        for data_reqs_per_cell in data_reqs_per_cell.values():
            for data_req, prompt_filteration in data_reqs_per_cell.items():
                data_reqs_collection.add_data_req(data_req, prompt_filteration)

        return DataReqs.from_data_reqs_collection(data_reqs_collection)

    def get_non_orientation_derived_params_params(
        self,
    ) -> set[
        Union[
            VARIANT_PARAM_NAME,
            Literal[ExperimentHyperParams.prompt_idx],
            Literal[ExperimentHyperParams.filteration_factory],
        ]
    ]:
        """Get all derived variant parameters across all orientations and fixed parameters."""
        result = set()
        lst = []

        for config in self.params:
            if config.orientation is None:
                continue
            param_def = get_hyper_param_definition(config.param)
            derived_variants_params = param_def.derived_variants_params()
            lst.append(derived_variants_params)
            if isinstance(derived_variants_params, Sequence):
                result.update(derived_variants_params)
            else:
                result.add(derived_variants_params)

        return result

    def derive_model_arch_and_sizes_context(self) -> list[MODEL_ARCH_AND_SIZE]:
        """Derive context model architectures and sizes."""
        models: List[MODEL_ARCH] = []
        sizes: List[TModelSize] = []

        # Check if model_arch_and_size is directly configured
        for config in self.params:
            if config.param == ExperimentHyperParams.model_arch_and_size:
                if config.values:
                    # Filter to ensure we only return MODEL_ARCH_AND_SIZE values
                    return [value for value in config.values if isinstance(value, tuple) and len(value) == 2]  # type: ignore
                if config.fixed_value is not None:
                    return [config.fixed_value]  # type: ignore

        # Otherwise collect model arch and model size separately
        for config in self.params:
            if config.param == ExperimentHyperParams.model_arch:
                if config.is_variable() and config.values:
                    models = [cast(MODEL_ARCH, model) for model in config.values]
                elif config.is_fixed():
                    models = [cast(MODEL_ARCH, config.fixed_value)]
            elif config.param == ExperimentHyperParams.model_size:
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


# Add this after the PlotPlan class definition to update forward references
ParamConfig.model_rebuild()
PlotPlan.model_rebuild()
