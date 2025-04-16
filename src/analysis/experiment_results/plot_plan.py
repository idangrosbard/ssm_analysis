from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from itertools import product
from pathlib import Path
from typing import Any, Dict, Generic, List, Optional, Sequence, TypeVar, assert_never, cast

from src.analysis.experiment_results.helpers import init_variant_params_from_values
from src.analysis.prompt_filterations import AnyExistingCompletePromptFilteration
from src.core.consts import ALL_VARIANT_PARAMETERS, GRAPHS_ORDER
from src.core.names import (
    VARIANT_PARAM_NAME,
    BaseVariantParamName,
    ExperimentHyperParams,
    ExperimentName,
    FinalPlotsPlanOrientation,
    PlotPlanCols,
    PlotPlanOptionCols,
    PlotType,
    ResultBankParamNames,
    map_final_plots_plan_orientation_to_options,
)
from src.core.types import (
    MODEL_ARCH,
    MODEL_ARCH_AND_SIZE,
    FeatureCategory,
    TModelSize,
    TokenType,
    TPlotID,
    TPromptOriginalIndex,
    TWindowSize,
)
from src.data_ingestion.data_defs.data_defs import DataReqiermentCollection, DataReqs, ResultBank
from src.experiments.infrastructure.base_prompt_filteration import SelectivePromptFilteration
from src.experiments.runners.heatmap import HeatmapRunner
from src.experiments.runners.info_flow import InfoFlowRunner
from src.utils.types_utils import str_enum_values

_T = TypeVar("_T")


class HyperParamDefinition(ABC, Generic[_T]):
    @abstractmethod
    def get_result_bank_options(self, result_bank: ResultBank) -> Sequence[_T]:
        pass

    @abstractmethod
    def get_static_options(self) -> Sequence[_T]:
        pass

    def get_options(self, result_bank: ResultBank) -> Sequence[_T]:
        return self.get_static_options()

    @abstractmethod
    def get_display_name(self, option: _T) -> str:
        pass

    def default_fix_value(self) -> _T:
        raise NotImplementedError(f"Default fix value not implemented for {self.__class__.__name__}")


# region Hyper Param Definitions


class ModelArchAndSizeHPD(HyperParamDefinition[MODEL_ARCH_AND_SIZE]):
    def get_result_bank_options(self, result_bank: ResultBank) -> list[MODEL_ARCH_AND_SIZE]:
        return list(
            [
                MODEL_ARCH_AND_SIZE(result.variant_params.model_arch, result.variant_params.model_size)
                for result in result_bank
            ]
        )

    def get_static_options(self) -> Sequence[MODEL_ARCH_AND_SIZE]:
        return list(GRAPHS_ORDER.keys())

    def get_display_name(self, option: MODEL_ARCH_AND_SIZE) -> str:
        return option.model_name


class ModelArchHPD(HyperParamDefinition[MODEL_ARCH]):
    def get_result_bank_options(self, result_bank: ResultBank):
        return list(set([result.variant_params.model_arch for result in result_bank]))

    def get_static_options(self):
        return str_enum_values(MODEL_ARCH)

    def get_display_name(self, option):
        return option


class ModelSizeHPD(HyperParamDefinition[TModelSize]):
    def get_result_bank_options(self, result_bank: ResultBank):
        return list(set([result.variant_params.model_size for result in result_bank]))

    def get_static_options(self):
        return list({size: size for _, size in GRAPHS_ORDER.keys()}.keys())

    def get_display_name(self, option):
        return option


class SourceHPD(HyperParamDefinition[TokenType]):
    def get_result_bank_options(self, result_bank):
        sources = set()
        for result in result_bank:
            if isinstance(result, InfoFlowRunner):
                sources.add(result.variant_params.source)
        return list(sources)

    def get_static_options(self):
        return str_enum_values(TokenType)

    def get_display_name(self, option: TokenType) -> str:
        return option


class TargetHPD(HyperParamDefinition[TokenType]):
    def get_result_bank_options(self, result_bank):
        targets = set()
        for result in result_bank:
            if isinstance(result, InfoFlowRunner):
                targets.add(result.variant_params.target)
        return list(targets)

    def get_static_options(self):
        return str_enum_values(TokenType)

    def get_display_name(self, option: TokenType) -> str:
        return option

    def default_fix_value(self) -> TokenType:
        return TokenType.last


class FeatureCategoryHPD(HyperParamDefinition[FeatureCategory]):
    def get_result_bank_options(self, result_bank):
        features = set()
        for result in result_bank:
            if isinstance(result, InfoFlowRunner):
                features.add(result.variant_params.feature_category)
        return list(features)

    def get_static_options(self):
        return str_enum_values(FeatureCategory)

    def get_display_name(self, option: FeatureCategory) -> str:
        return str(option)

    def default_fix_value(self) -> FeatureCategory:
        return FeatureCategory.ALL


class WindowSizeHPD(HyperParamDefinition[TWindowSize]):
    def get_result_bank_options(self, result_bank):
        window_sizes = set()
        for result in result_bank:
            if isinstance(result, InfoFlowRunner) or isinstance(result, HeatmapRunner):
                window_sizes.add(result.variant_params.window_size)
        return list(window_sizes)

    def get_static_options(self):
        return list([TWindowSize(i) for i in range(1, 20)])

    def get_display_name(self, option: TWindowSize) -> str:
        return f"{option}"

    def default_fix_value(self) -> TWindowSize:
        return TWindowSize(9)


class PromptIdxHPD(HyperParamDefinition[TPromptOriginalIndex]):
    def get_result_bank_options(self, result_bank: ResultBank) -> Sequence[TPromptOriginalIndex]:
        prompts: set[TPromptOriginalIndex] = set()
        for result in result_bank:
            if isinstance(result, HeatmapRunner):
                prompts.update(set(result.output_hdf5_path.get_existing_prompt_idx()))
        return sorted(prompts)

    def get_static_options(self):
        raise NotImplementedError("PromptIdxVariationOption does not have static options")

    def get_options(self, result_bank: ResultBank) -> Sequence[TPromptOriginalIndex]:
        return self.get_result_bank_options(result_bank)

    def get_display_name(self, option: TPromptOriginalIndex) -> str:
        return f"{option}"


# endregion


def get_hyper_param_definition(option: ExperimentHyperParams) -> HyperParamDefinition:
    match option:
        case ExperimentHyperParams.model_arch_and_size:
            return ModelArchAndSizeHPD()
        case ExperimentHyperParams.model_arch:
            return ModelArchHPD()
        case ExperimentHyperParams.model_size:
            return ModelSizeHPD()
        case ExperimentHyperParams.source:
            return SourceHPD()
        case ExperimentHyperParams.target:
            return TargetHPD()
        case ExperimentHyperParams.feature_category:
            return FeatureCategoryHPD()
        case ExperimentHyperParams.window_size:
            return WindowSizeHPD()
        case ExperimentHyperParams.prompt_idx:
            return PromptIdxHPD()
        case _:
            raise ValueError(f"Unsupported variation option: {option}")


def get_experiment_orientations(experiment_name: ExperimentName) -> list[FinalPlotsPlanOrientation]:
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


def get_experiment_hyper_param_hyper_param(experiment_name: ExperimentName) -> list[ExperimentHyperParams]:
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
    grids: Any
    rows: Any
    cols: Any

    @classmethod
    def from_orientation_combination(cls, orientation_combination: dict[FinalPlotsPlanOrientation, Any]) -> "Cell":
        """Create a Cell from an orientation combination dictionary."""
        return cls(
            grids=orientation_combination.get(FinalPlotsPlanOrientation.grids),
            rows=orientation_combination.get(FinalPlotsPlanOrientation.rows),
            cols=orientation_combination.get(FinalPlotsPlanOrientation.cols),
        )

    def get_display_name(self, field: str, plot_plan: "PlotPlan") -> Optional[str]:
        """Get the display name for a field value using the plot plan's parameter definition."""
        value = getattr(self, field)
        if value is None:
            return None

        param = plot_plan._get_orientation_value(FinalPlotsPlanOrientation[field])
        if param is None:
            return str(value)

        return get_hyper_param_definition(param).get_display_name(value)

    def get_cache_path(self, plot_plan: "PlotPlan", cache_dir: Path) -> Path:
        """Generate a unique cache path for this cell."""
        # Get display names for each field
        display_names = {
            field: self.get_display_name(field, plot_plan) or "None" for field in ["grids", "rows", "cols"]
        }

        # Create a unique identifier for the cell
        cell_id = f"{display_names['grids']}_{display_names['rows']}_{display_names['cols']}".replace(" ", "_")
        return cache_dir / f"{cell_id}.png"

    def to_dict(self) -> dict[str, Any]:
        """Convert cell to dictionary for data requirements."""
        return {FinalPlotsPlanOrientation[field]: getattr(self, field) for field in ["grids", "rows", "cols"]}


@dataclass
class PlotPlan:
    plot_id: TPlotID
    title: str
    description: str
    plot_type: PlotType
    is_appendix: bool
    order: int
    experiment_name: ExperimentName
    rows: Optional[ExperimentHyperParams] = None
    cols: Optional[ExperimentHyperParams] = None
    grids: Optional[ExperimentHyperParams] = None
    lines: Optional[ExperimentHyperParams] = None

    # Selected options for each parameter
    rows_options: list[Any] = field(default_factory=list)
    cols_options: list[Any] = field(default_factory=list)
    grids_options: list[Any] = field(default_factory=list)
    lines_options: list[Any] = field(default_factory=list)

    def _get_orientation_value(self, param_name: FinalPlotsPlanOrientation) -> Optional[ExperimentHyperParams]:
        return getattr(self, param_name)

    def _get_param_options_col(self, param_name: FinalPlotsPlanOrientation) -> list[Any]:
        return getattr(self, map_final_plots_plan_orientation_to_options(param_name))

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        # Convert Enum values to strings for JSON serialization
        result[PlotPlanCols.plot_type] = self.plot_type.name
        result[PlotPlanCols.experiment_name] = self.experiment_name.name

        for orientation in str_enum_values(FinalPlotsPlanOrientation):
            # Convert orientation parameters to strings
            param = self._get_orientation_value(orientation)
            if param:
                result[orientation] = param.name

                param_options_col = map_final_plots_plan_orientation_to_options(orientation)
                # Convert complex objects in options to serializable format
                if result[param_options_col]:
                    result[param_options_col] = self._serialize_options(self._get_param_options_col(orientation), param)

        return result

    def _serialize_options(self, options: List[Any], param_type: Optional[ExperimentHyperParams]) -> List[Any]:
        """Serialize options to a JSON-compatible format."""
        if not param_type or not options:
            return options

        serialized: list[Any] = []
        for option in options:
            if param_type == ExperimentHyperParams.model_arch_and_size and isinstance(option, MODEL_ARCH_AND_SIZE):
                serialized.append(
                    {ResultBankParamNames.model_arch: option.arch, ResultBankParamNames.model_size: option.size}
                )
            elif isinstance(option, (str, int, float, bool)) or option is None:
                serialized.append(option)
            else:
                serialized.append(str(option))
        return serialized

    def _deserialize_options(self, options: List[Any], param_type: Optional[ExperimentHyperParams]) -> List[Any]:
        """Deserialize options from a JSON format."""
        if not param_type or not options:
            return options

        deserialized = []
        for option in options:
            if param_type == ExperimentHyperParams.model_arch_and_size and isinstance(option, dict):
                if ResultBankParamNames.model_arch in option and ResultBankParamNames.model_size in option:
                    deserialized.append(
                        MODEL_ARCH_AND_SIZE(
                            option[ResultBankParamNames.model_arch], option[ResultBankParamNames.model_size]
                        )
                    )
            else:
                deserialized.append(option)
        return deserialized

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlotPlan":
        # Convert string values back to Enum values
        data_copy = data.copy()
        data_copy[PlotPlanCols.plot_type] = PlotType[data_copy[PlotPlanCols.plot_type]]
        data_copy[PlotPlanCols.experiment_name] = ExperimentName[data_copy[PlotPlanCols.experiment_name]]

        # Convert orientation parameters to ExperimentHyperParams
        for orientation in str_enum_values(FinalPlotsPlanOrientation):
            col_name = getattr(PlotPlanCols, orientation)
            if data_copy.get(col_name):
                data_copy[col_name] = ExperimentHyperParams[data_copy[col_name]]

        # Create the instance
        instance = cls(**{k: v for k, v in data_copy.items()})

        # Deserialize options for each orientation
        for orientation in str_enum_values(FinalPlotsPlanOrientation):
            options_col = getattr(PlotPlanOptionCols, f"{orientation}_options")
            if options_col in data_copy:
                param = instance._get_orientation_value(orientation)
                deserialized_options = instance._deserialize_options(data_copy[options_col], param)
                instance.set_options_for_orientation(orientation, deserialized_options)

        return instance

    def get_options_for_param(self, param: FinalPlotsPlanOrientation) -> List[Any]:
        """Get the selected options for a parameter."""
        match param:
            case FinalPlotsPlanOrientation.rows:
                return self.rows_options
            case FinalPlotsPlanOrientation.cols:
                return self.cols_options
            case FinalPlotsPlanOrientation.grids:
                return self.grids_options
            case FinalPlotsPlanOrientation.lines:
                return self.lines_options
            case _:
                assert_never(param)

    def set_orientation_value(self, param: FinalPlotsPlanOrientation, value: Optional[ExperimentHyperParams]) -> None:
        setattr(self, param.value, value)

    def set_options_for_orientation(self, param: FinalPlotsPlanOrientation, options: List[Any]) -> None:
        """Set the selected options for a parameter."""
        match param:
            case FinalPlotsPlanOrientation.rows:
                self.rows_options = options
            case FinalPlotsPlanOrientation.cols:
                self.cols_options = options
            case FinalPlotsPlanOrientation.grids:
                self.grids_options = options
            case FinalPlotsPlanOrientation.lines:
                self.lines_options = options
            case _:
                assert_never(param)

    def get_summary(self) -> Dict[FinalPlotsPlanOrientation, list[str]]:
        """Get a summary of the plot structure."""

        return {
            orientation: [
                get_hyper_param_definition(param).get_display_name(option)
                for option in self.get_options_for_param(orientation)
                if (param := self._get_orientation_value(orientation)) is not None
            ]
            for orientation in str_enum_values(FinalPlotsPlanOrientation)
        }

    def get_data_requirements_per_cell(self, result_bank: ResultBank) -> dict[Cell, DataReqs]:
        """Generate data requirements for this plot plan based on the result bank."""
        data_reqs_per_cell: dict[Cell, DataReqiermentCollection] = defaultdict(DataReqiermentCollection)
        experiment_orientations = get_experiment_orientations(self.experiment_name)
        experiment_hyper_param_defs = get_experiment_hyper_param_hyper_param(self.experiment_name)

        def get_options_for_orientation(orientation: FinalPlotsPlanOrientation) -> list[Any]:
            param = self._get_orientation_value(orientation)
            if not param:
                return [None]

            options = self.get_options_for_param(orientation)
            if not options:
                variation_option = get_hyper_param_definition(param)
                options = list(variation_option.get_options(result_bank))
            return options

        orientation_options: dict[FinalPlotsPlanOrientation, list[Any]] = {
            orientation: get_options_for_orientation(orientation) for orientation in experiment_orientations
        }

        # Generate all combinations

        combinations = product(*[orientation_options[orientation] for orientation in experiment_orientations])

        # Process each combination
        for combination in combinations:
            prompt_filterations = AnyExistingCompletePromptFilteration()
            orientation_combination = {
                orientation: combination[i] for i, orientation in enumerate(experiment_orientations)
            }

            params: dict["ExperimentHyperParams", Any] = {
                param: value
                for orientation, value in orientation_combination.items()
                if value is not None and (param := self._get_orientation_value(orientation)) is not None
            }

            # Handle the special case of model_arch_and_size
            if params.get(ExperimentHyperParams.model_arch_and_size) is not None:
                model_arch_and_size = cast(MODEL_ARCH_AND_SIZE, params[ExperimentHyperParams.model_arch_and_size])
                params[ExperimentHyperParams.model_arch] = model_arch_and_size.arch
                params[ExperimentHyperParams.model_size] = model_arch_and_size.size
            else:
                assert params.get(ExperimentHyperParams.model_arch) is not None
                assert params.get(ExperimentHyperParams.model_size) is not None
                model_arch_and_size = MODEL_ARCH_AND_SIZE(
                    params[ExperimentHyperParams.model_arch], params[ExperimentHyperParams.model_size]
                )
                if model_arch_and_size not in GRAPHS_ORDER:
                    continue
                params[ExperimentHyperParams.model_arch_and_size] = model_arch_and_size

            if self.experiment_name == ExperimentName.heatmap:
                prompt_filterations = SelectivePromptFilteration(
                    prompt_ids=tuple([params[ExperimentHyperParams.prompt_idx]])
                )

            data_req_params: dict[VARIANT_PARAM_NAME, Any] = {
                BaseVariantParamName.experiment_name: self.experiment_name,
            }

            for col in ALL_VARIANT_PARAMETERS:
                if col in data_req_params:
                    continue
                if col in params:
                    data_req_params[col] = params[ExperimentHyperParams[col]]
                elif col in experiment_hyper_param_defs:
                    data_req_params[col] = get_hyper_param_definition(ExperimentHyperParams[col]).default_fix_value()

            cell = Cell.from_orientation_combination(orientation_combination)
            data_reqs_per_cell[cell].add_data_req(init_variant_params_from_values(data_req_params), prompt_filterations)

        return {cell: DataReqs.from_data_reqs_collection(data_reqs) for cell, data_reqs in data_reqs_per_cell.items()}

    def get_data_requirements(self, result_bank: ResultBank) -> DataReqs:
        data_reqs_per_cell = self.get_data_requirements_per_cell(result_bank)
        data_reqs_collection = DataReqiermentCollection()
        for data_reqs_per_cell in data_reqs_per_cell.values():
            for data_req, prompt_filteration in data_reqs_per_cell.items():
                data_reqs_collection.add_data_req(data_req, prompt_filteration)
        return DataReqs.from_data_reqs_collection(data_reqs_collection)
