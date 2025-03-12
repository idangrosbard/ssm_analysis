from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from itertools import product
from pathlib import Path
from typing import Any, Dict, Generic, List, Optional, Sequence, TypeVar, assert_never, cast

from src.consts import GRAPHS_ORDER
from src.data_defs import DataReqs, ResultBank
from src.final_plots.data_reqs import DataReq
from src.final_plots.results_bank import HeatmapRecord, InfoFlowRecord
from src.names import (
    EXPERIMENT_NAMES,
    ExperimentHyperParams,
    PlotPlanCols,
    PlotPlanOptionCols,
    PlotType,
    ResultBankParamNames,
    map_final_plots_plan_orientation_to_options,
)
from src.types import (
    MODEL_ARCH,
    MODEL_ARCH_AND_SIZE,
    FeatureCategory,
    FinalPlotsPlanOrientation,
    TModelSize,
    TokenType,
    TPromptOriginalIndex,
    TWindowSize,
)
from src.utils.types_utils import str_enum_values

_T = TypeVar("_T")


class VariationOption(ABC, Generic[_T]):
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
        raise NotImplementedError("Default fix value not implemented")


# region Variation Options


class ModelArchAndSizeVariationOption(VariationOption[MODEL_ARCH_AND_SIZE]):
    def get_result_bank_options(self, result_bank: ResultBank) -> list[MODEL_ARCH_AND_SIZE]:
        return list([MODEL_ARCH_AND_SIZE(result.model_arch, result.model_size) for result in result_bank.to_rows()])

    def get_static_options(self) -> Sequence[MODEL_ARCH_AND_SIZE]:
        return list(GRAPHS_ORDER.keys())

    def get_display_name(self, option: MODEL_ARCH_AND_SIZE) -> str:
        return option.model_name


class ModelArchVariationOption(VariationOption[MODEL_ARCH]):
    def get_result_bank_options(self, result_bank: ResultBank):
        return list(set([result.model_arch for result in result_bank.to_rows()]))

    def get_static_options(self):
        return str_enum_values(MODEL_ARCH)

    def get_display_name(self, option):
        return option


class ModelSizeVariationOption(VariationOption[TModelSize]):
    def get_result_bank_options(self, result_bank: ResultBank):
        return list(set([result.model_size for result in result_bank.to_rows()]))

    def get_static_options(self):
        return list({size: size for _, size in GRAPHS_ORDER.keys()}.keys())

    def get_display_name(self, option):
        return option


class SourceVariationOption(VariationOption[TokenType]):
    def get_result_bank_options(self, result_bank):
        sources = set()
        for result in result_bank.to_rows():
            if isinstance(result, InfoFlowRecord):
                sources.add(result.source)
        return list(sources)

    def get_static_options(self):
        return str_enum_values(TokenType)

    def get_display_name(self, option: TokenType) -> str:
        return option


class TargetVariationOption(VariationOption[TokenType]):
    def get_result_bank_options(self, result_bank):
        targets = set()
        for result in result_bank.to_rows():
            if isinstance(result, InfoFlowRecord):
                targets.add(result.target)
        return list(targets)

    def get_static_options(self):
        return str_enum_values(TokenType)

    def get_display_name(self, option: TokenType) -> str:
        return option

    def default_fix_value(self) -> TokenType:
        return TokenType.last


class FeatureCategoryVariationOption(VariationOption[FeatureCategory]):
    def get_result_bank_options(self, result_bank):
        features = set()
        for result in result_bank.to_rows():
            if isinstance(result, InfoFlowRecord):
                features.add(result.feature_category)
        return list(features)

    def get_static_options(self):
        return str_enum_values(FeatureCategory)

    def get_display_name(self, option: FeatureCategory) -> str:
        return option


class WindowSizeVariationOption(VariationOption[TWindowSize]):
    def get_result_bank_options(self, result_bank):
        return list(set([result.window_size for result in result_bank.to_rows()]))

    def get_static_options(self):
        return list([TWindowSize(i) for i in range(1, 20)])

    def get_display_name(self, option: TWindowSize) -> str:
        return f"{option}"

    def default_fix_value(self) -> TWindowSize:
        return TWindowSize(9)


class PromptIdxVariationOption(VariationOption[TPromptOriginalIndex]):
    def get_result_bank_options(self, result_bank: ResultBank) -> Sequence[TPromptOriginalIndex]:
        prompts: set[TPromptOriginalIndex] = set()
        for result in result_bank.to_rows():
            if isinstance(result, HeatmapRecord):
                prompts.add(result.prompt_idx)
        return sorted(prompts)

    def get_static_options(self):
        raise NotImplementedError("PromptIdxVariationOption does not have static options")

    def get_options(self, result_bank: ResultBank) -> Sequence[TPromptOriginalIndex]:
        return self.get_result_bank_options(result_bank)

    def get_display_name(self, option: TPromptOriginalIndex) -> str:
        return f"{option}"


# endregion


def get_variation_option(option: ExperimentHyperParams) -> VariationOption:
    match option:
        case ExperimentHyperParams.model_arch_and_size:
            return ModelArchAndSizeVariationOption()
        case ExperimentHyperParams.model_arch:
            return ModelArchVariationOption()
        case ExperimentHyperParams.model_size:
            return ModelSizeVariationOption()
        case ExperimentHyperParams.source:
            return SourceVariationOption()
        case ExperimentHyperParams.target:
            return TargetVariationOption()
        case ExperimentHyperParams.feature_category:
            return FeatureCategoryVariationOption()
        case ExperimentHyperParams.window_size:
            return WindowSizeVariationOption()
        case ExperimentHyperParams.prompt_idx:
            return PromptIdxVariationOption()
        case _:
            raise ValueError(f"Unsupported variation option: {option}")


def get_experiment_params(experiment_name: EXPERIMENT_NAMES) -> list[FinalPlotsPlanOrientation]:
    """Get the relevant parameters for a specific experiment type."""
    if experiment_name == EXPERIMENT_NAMES.INFO_FLOW:
        return [
            FinalPlotsPlanOrientation.ROWS,
            FinalPlotsPlanOrientation.COLS,
            FinalPlotsPlanOrientation.GRIDS,
            FinalPlotsPlanOrientation.LINES,
        ]
    elif experiment_name == EXPERIMENT_NAMES.HEATMAP:
        return [
            FinalPlotsPlanOrientation.ROWS,
            FinalPlotsPlanOrientation.COLS,
            FinalPlotsPlanOrientation.GRIDS,
        ]
    else:
        return list(FinalPlotsPlanOrientation)


@dataclass
class PlotPlan:
    title: str
    description: str
    plot_type: PlotType
    is_appendix: bool
    order: int
    experiment_name: EXPERIMENT_NAMES
    rows: Optional[ExperimentHyperParams] = None
    cols: Optional[ExperimentHyperParams] = None
    grids: Optional[ExperimentHyperParams] = None
    lines: Optional[ExperimentHyperParams] = None
    output_path: Optional[str] = None

    # Selected options for each parameter
    rows_options: list[Any] = field(default_factory=list)
    cols_options: list[Any] = field(default_factory=list)
    grids_options: list[Any] = field(default_factory=list)
    lines_options: list[Any] = field(default_factory=list)

    def _get_param_type(self, param_name: FinalPlotsPlanOrientation) -> Optional[ExperimentHyperParams]:
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
            param = self._get_param_type(orientation)
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

        serialized = []
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
        data_copy[PlotPlanCols.experiment_name] = EXPERIMENT_NAMES[data_copy[PlotPlanCols.experiment_name]]

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
                param = instance._get_param_type(orientation)
                deserialized_options = instance._deserialize_options(data_copy[options_col], param)
                instance.set_options_for_param(orientation, deserialized_options)

        return instance

    def get_options_for_param(self, param: FinalPlotsPlanOrientation) -> List[Any]:
        """Get the selected options for a parameter."""
        match param:
            case FinalPlotsPlanOrientation.ROWS:
                return self.rows_options
            case FinalPlotsPlanOrientation.COLS:
                return self.cols_options
            case FinalPlotsPlanOrientation.GRIDS:
                return self.grids_options
            case FinalPlotsPlanOrientation.LINES:
                return self.lines_options
            case _:
                assert_never(param)

    def set_options_for_param(self, param: FinalPlotsPlanOrientation, options: List[Any]) -> None:
        """Set the selected options for a parameter."""
        match param:
            case FinalPlotsPlanOrientation.ROWS:
                self.rows_options = options
            case FinalPlotsPlanOrientation.COLS:
                self.cols_options = options
            case FinalPlotsPlanOrientation.GRIDS:
                self.grids_options = options
            case FinalPlotsPlanOrientation.LINES:
                self.lines_options = options
            case _:
                assert_never(param)

    def get_summary(self) -> Dict[FinalPlotsPlanOrientation, int]:
        """Get a summary of the plot structure."""
        summary = {
            FinalPlotsPlanOrientation.ROWS: len(self.rows_options) if self.rows and self.rows_options else 1,
            FinalPlotsPlanOrientation.COLS: len(self.cols_options) if self.cols and self.cols_options else 1,
            FinalPlotsPlanOrientation.GRIDS: len(self.grids_options) if self.grids and self.grids_options else 1,
            FinalPlotsPlanOrientation.LINES: len(self.lines_options) if self.lines and self.lines_options else 1,
        }
        return summary

    def get_data_requirements(self, result_bank: ResultBank) -> DataReqs:
        """Generate data requirements for this plot plan based on the result bank."""
        data_reqs = []

        # Get all orientation options
        orientation_options: Dict[FinalPlotsPlanOrientation, list[Any]] = {}
        for orientation in str_enum_values(FinalPlotsPlanOrientation):
            param = self._get_param_type(orientation)
            if not param:
                orientation_options[orientation] = [None]
                continue

            options = self.get_options_for_param(orientation)
            if not options:
                variation_option = get_variation_option(param)
                options = list(variation_option.get_options(result_bank))
            orientation_options[orientation] = options

        # Generate all combinations
        combinations = product(
            orientation_options[FinalPlotsPlanOrientation.ROWS],
            orientation_options[FinalPlotsPlanOrientation.COLS],
            orientation_options[FinalPlotsPlanOrientation.GRIDS],
            orientation_options[FinalPlotsPlanOrientation.LINES],
        )

        # Process each combination
        for row_opt, col_opt, grid_opt, line_opt in combinations:
            if self.experiment_name == EXPERIMENT_NAMES.INFO_FLOW:
                params: Dict[ExperimentHyperParams, Optional[Any]] = {
                    ExperimentHyperParams.source: None,
                    ExperimentHyperParams.target: None,
                    ExperimentHyperParams.feature_category: None,
                    ExperimentHyperParams.model_arch_and_size: None,
                    ExperimentHyperParams.window_size: None,
                }

                # Map options to parameters
                option_map = {
                    (FinalPlotsPlanOrientation.ROWS, row_opt): self.rows,
                    (FinalPlotsPlanOrientation.COLS, col_opt): self.cols,
                    (FinalPlotsPlanOrientation.GRIDS, grid_opt): self.grids,
                    (FinalPlotsPlanOrientation.LINES, line_opt): self.lines,
                }

                for (orientation, opt), param_type in option_map.items():
                    if param_type and opt is not None:
                        params[param_type] = opt

                # Apply default values for missing required parameters
                if params[ExperimentHyperParams.source] is None:
                    params[ExperimentHyperParams.source] = get_variation_option(
                        ExperimentHyperParams.source
                    ).default_fix_value()
                if params[ExperimentHyperParams.target] is None:
                    params[ExperimentHyperParams.target] = get_variation_option(
                        ExperimentHyperParams.target
                    ).default_fix_value()
                if params[ExperimentHyperParams.window_size] is None:
                    params[ExperimentHyperParams.window_size] = get_variation_option(
                        ExperimentHyperParams.window_size
                    ).default_fix_value()

                # Only add if we have a model_arch_and_size
                if params[ExperimentHyperParams.model_arch_and_size] is not None:
                    model_arch_and_size = cast(MODEL_ARCH_AND_SIZE, params[ExperimentHyperParams.model_arch_and_size])
                    data_reqs.append(
                        DataReq(
                            experiment_name=EXPERIMENT_NAMES.INFO_FLOW,
                            model_arch=model_arch_and_size.arch,
                            model_size=model_arch_and_size.size,
                            window_size=cast(TWindowSize, params[ExperimentHyperParams.window_size]),
                            is_all_correct=False,
                            source=cast(TokenType, params[ExperimentHyperParams.source]),
                            feature_category=cast(
                                Optional[FeatureCategory], params[ExperimentHyperParams.feature_category]
                            ),
                            target=cast(TokenType, params[ExperimentHyperParams.target]),
                            prompt_idx=None,
                        )
                    )

            elif self.experiment_name == EXPERIMENT_NAMES.HEATMAP:
                params: Dict[ExperimentHyperParams, Optional[Any]] = {
                    ExperimentHyperParams.model_arch_and_size: None,
                    ExperimentHyperParams.window_size: None,
                    ExperimentHyperParams.prompt_idx: None,
                }

                # Map options to parameters
                option_map = {
                    (FinalPlotsPlanOrientation.ROWS, row_opt): self.rows,
                    (FinalPlotsPlanOrientation.COLS, col_opt): self.cols,
                    (FinalPlotsPlanOrientation.GRIDS, grid_opt): self.grids,
                }

                for (orientation, opt), param_type in option_map.items():
                    if param_type and opt is not None:
                        params[param_type] = opt

                # Apply default values for missing required parameters
                if params[ExperimentHyperParams.window_size] is None:
                    params[ExperimentHyperParams.window_size] = get_variation_option(
                        ExperimentHyperParams.window_size
                    ).default_fix_value()
                if params[ExperimentHyperParams.prompt_idx] is None:
                    params[ExperimentHyperParams.prompt_idx] = TPromptOriginalIndex(0)

                # Only add if we have both model_arch_and_size and prompt_idx
                if (
                    params[ExperimentHyperParams.model_arch_and_size] is not None
                    and params[ExperimentHyperParams.prompt_idx] is not None
                ):
                    model_arch_and_size = cast(MODEL_ARCH_AND_SIZE, params[ExperimentHyperParams.model_arch_and_size])
                    data_reqs.append(
                        DataReq(
                            experiment_name=EXPERIMENT_NAMES.HEATMAP,
                            model_arch=model_arch_and_size.arch,
                            model_size=model_arch_and_size.size,
                            window_size=cast(TWindowSize, params[ExperimentHyperParams.window_size]),
                            is_all_correct=False,
                            source=None,
                            feature_category=None,
                            target=None,
                            prompt_idx=cast(TPromptOriginalIndex, params[ExperimentHyperParams.prompt_idx]),
                        )
                    )

        return DataReqs(set(data_reqs))


# Path for storing plot plans
PLOT_PLANS_PATH = Path(__file__).parent / "plot_plans.json"
