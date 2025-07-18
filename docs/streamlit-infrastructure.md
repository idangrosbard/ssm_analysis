# Streamlit Infrastructure Documentation

## Overview

The Streamlit application infrastructure is built on a sophisticated component-based architecture that emphasizes type safety, reusability, and modular design. This documentation covers the foundational base classes and patterns that form the core of the scientific application's UI system.

The infrastructure provides:
- **Type-safe component system** with generic `OutputType` support
- **Abstract base classes** for consistent component patterns
- **Profiling capabilities** for performance analysis
- **Session state management** with strong typing
- **Composition patterns** for building complex UIs from simple components

## Core Base Classes

### StreamlitComponent[OutputType]

The foundational abstract base class for all Streamlit components. This class provides the core architecture for type-safe, reusable UI components.

**Location**: `src/utils/streamlit/helpers/component.py`

#### Class Definition

```python
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import streamlit as st

OutputType = TypeVar("OutputType")

class StreamlitComponent(ABC, Generic[OutputType]):
    @abstractmethod
    def render(self) -> OutputType:
        pass

    def profile_render(self):
        from wfork_streamlit_profiler import Profiler

        with Profiler():
            try:
                self.render()
            finally:
                st.toast("Rendering complete, generating profile...")
```

#### Key Features

1. **Generic Type System**: Uses `OutputType` generic parameter for type-safe return values
2. **Abstract Render Method**: Enforces consistent interface across all components
3. **Profiling Support**: Built-in performance analysis capabilities
4. **Composition Ready**: Designed for easy composition and reuse

#### Usage Patterns

**Basic Component Implementation:**
```python
from src.utils.streamlit.helpers.component import StreamlitComponent

class MyComponent(StreamlitComponent[str]):
    def render(self) -> str:
        user_input = st.text_input("Enter text")
        return user_input
```

**Component with Complex Return Type:**
```python
from dataclasses import dataclass
from typing import List

@dataclass
class PlotConfig:
    title: str
    data: List[float]

class PlotComponent(StreamlitComponent[PlotConfig]):
    def render(self) -> PlotConfig:
        title = st.text_input("Plot title")
        data_points = st.multiselect("Select data", options=[1, 2, 3, 4, 5])
        return PlotConfig(title=title, data=data_points)
```

**Component with No Return Value:**
```python
class DisplayComponent(StreamlitComponent[None]):
    def render(self) -> None:
        st.write("This component displays information")
        st.button("Click me")
```

### StreamlitPage

A specialized component for full-page implementations that don't return values.

**Location**: `src/utils/streamlit/helpers/component.py`

#### Class Definition

```python
class StreamlitPage(StreamlitComponent[None]):
    pass
```

#### Usage Pattern

```python
from src.utils.streamlit.helpers.component import StreamlitPage

class HomePage(StreamlitPage):
    def render(self):
        st.title("Welcome to the Application")
        st.sidebar.success("Select a page above to explore different analyses.")
        
        # Global controls
        if st.button("Reset App"):
            st.session_state.clear()
```

## Type System Architecture

### OutputType Generic Pattern

The `OutputType` generic parameter enables type-safe component composition and data flow:

```python
# Component that returns a string
class TextInputComponent(StreamlitComponent[str]):
    def render(self) -> str:
        return st.text_input("Enter text")

# Component that returns a complex object
class ConfigComponent(StreamlitComponent[PlotConfig]):
    def render(self) -> PlotConfig:
        # ... configuration logic
        return PlotConfig(...)

# Component that returns nothing (display only)
class DisplayComponent(StreamlitComponent[None]):
    def render(self) -> None:
        st.write("Display only")
```

### Type Safety Benefits

1. **Compile-time Error Detection**: Type errors caught during development
2. **IDE Support**: Better autocomplete and refactoring support
3. **Documentation**: Types serve as inline documentation
4. **Composition Safety**: Ensures compatible data flow between components

## Session State Management

The application includes a sophisticated session state management system with strong typing and advanced features:

**Location**: `src/utils/streamlit/helpers/session_keys.py`

### SessionKey[TSessionKey] System

The core session state wrapper provides type safety and advanced functionality:

#### Core Features

```python
class SessionKey(Generic[TSessionKey]):
    def __init__(self, key: str, default_value: TSessionKey | None = None, allow_none: Optional[bool] = None):
        self._key = key
        self.default_value = default_value
        self._ever_changed = False
        self._allow_none = default_value is None if allow_none is None else allow_none

    @property
    def value(self) -> TSessionKey:
        """Get the current value with type safety"""
        if self.is_erroneous:
            if st.button("Reset Value"):
                self.reset_value()
            raise KeyError(f"Session key '{self.key}' not initialized and has no default value")
        return cast(TSessionKey, st.session_state[self.key] if self.exists() else self.default_value)

    @value.setter
    def value(self, new_value: TSessionKey):
        """Set the current value with type safety"""
        self._update(new_value)

    @property
    def key_for_component(self) -> str:
        """Get key for use with Streamlit components with external update support"""
        if self._key_need_external_update.value:
            self._update(self._key_next_external_update_value.value)
            self._key_need_external_update.value = False
        self._key_for_prev_value.value = self.value
        return self.key

    @property
    def is_changed(self) -> bool:
        """Check if the value has changed since last access"""
        return self._key_for_prev_value.value != self.value

    def post_external_update(self, value: TSessionKey | None, with_rerun: bool = True):
        """Update value externally after component rendering"""
        self._key_next_external_update_value.value = value
        self._key_need_external_update.value = True
        if with_rerun:
            st.rerun()
```

#### Advanced Features

1. **Type Safety**: Strong typing prevents runtime errors
2. **Change Detection**: Track when values change
3. **External Updates**: Update values after component rendering
4. **Error Handling**: Graceful handling of missing values
5. **Default Values**: Automatic initialization with defaults
6. **Component Integration**: Seamless integration with Streamlit widgets

#### Usage Examples

```python
from src.utils.streamlit.helpers.session_keys import SessionKey

# Typed session key with default
selected_model = SessionKey[ModelType]("selected_model", ModelType.GPT2)

# Component usage with change detection
model_choice = st.selectbox(
    "Select Model",
    options=list(ModelType),
    key=selected_model.key_for_component
)

# Check if value changed
if selected_model.is_changed:
    st.success("Model selection updated!")

# Type-safe access
current_model = selected_model.value  # Returns ModelType, not Any

# External update
selected_model.post_external_update(ModelType.LLAMA)
```

### SessionKeyDescriptor Pattern

For class-based session state management with automatic key generation:

```python
class SessionKeyDescriptor(Generic[TSessionKey]):
    def __init__(self, default_value: TSessionKey | None = None, allow_none: Optional[bool] = None):
        self.default_value = default_value
        self.allow_none = allow_none

    def __set_name__(self, owner: Any, name: str):
        # Add prefix based on class name
        prefix = owner.__name__.lower().strip("_")
        self.key = f"{prefix}_{name}"

    def __get__(self, obj: Any, objtype: Any = None) -> SessionKey[TSessionKey]:
        if obj is None:
            raise ValueError("SessionKeyDescriptor must be used as a class attribute")
        
        # Create or get SessionKey instance with automatic initialization
        if not hasattr(obj, self.instance_key):
            session_key = SessionKey(self.key, self.default_value, self.allow_none)
            if self.allow_none or self.default_value is not None:
                session_key.init(cast(TSessionKey, self.default_value))
            setattr(obj, self.instance_key, session_key)
        return cast(SessionKey[TSessionKey], getattr(obj, self.instance_key))
```

#### Usage Example

```python
class _AppSessionKeys(SessionKeysBase["_AppSessionKeys"]):
    # Each descriptor creates a SessionKey with the class name prefix
    code_version = SessionKeyDescriptor[TCodeVersionName](GLOBAL_APP_CONSTS.DEFAULT_CODE_VERSION)
    _selected_gpu = SessionKeyDescriptor[Union[SLURM_GPU_TYPE, Literal["smart"]]]("smart")
    window_size = SessionKeyDescriptor[TWindowSize](GLOBAL_APP_CONSTS.DEFAULT_WINDOW_SIZE)

# Singleton instance
AppSessionKeys = _AppSessionKeys()

# Usage
current_gpu = AppSessionKeys._selected_gpu.value
current_version = AppSessionKeys.code_version.value
```

### SessionKeysBase Pattern

For organized session key containers with singleton pattern:

```python
class SessionKeysBase(Generic[_T_SESSION_KEYS_BASE]):
    """Base class for session key containers that ensures singleton pattern."""
    
    _instance: ClassVar[dict[Type[Any], Any]] = {}

    def __new__(cls) -> _T_SESSION_KEYS_BASE:
        if cls not in cls._instance:
            cls._instance[cls] = super().__new__(cls)
        return cast(_T_SESSION_KEYS_BASE, cls._instance[cls])
```

## Caching Infrastructure

The application includes a sophisticated dependency-aware caching system:

**Location**: `src/utils/streamlit/helpers/cache.py`

### CacheWithDependencies Decorator

The core caching decorator with dependency tracking:

```python
class CacheWithDependencies:
    """Class decorator wrapping @st.cache_data with strong typing, dependency tracking, and UI rendering."""

    def __init__(self, *st_args, disable_cache: bool = False, is_resource: bool = False, **st_kwargs):
        self.st_args = st_args
        self.st_kwargs = st_kwargs
        self.disable_cache = disable_cache
        self.is_resource = is_resource

    def __call__(self, func: Callable[P, OutputType]) -> CachedFunction[P, OutputType]:
        if self.is_resource:
            cached_func = st.cache_resource(*self.st_args, **self.st_kwargs)(func)
        else:
            cached_func = st.cache_data(*self.st_args, **self.st_kwargs)(func)
        return CachedFunction(func, cached_func, is_disabled=self.disable_cache)
```

### CachedFunction Wrapper

Advanced wrapper with dependency tracking and UI rendering:

```python
class CachedFunction(Generic[P, OutputType]):
    """A strongly typed wrapper for a cached function with recursive clearing and UI rendering."""

    def __init__(self, func: Callable[P, OutputType], cached_func: Callable[P, OutputType], is_disabled: bool = False):
        self.func = func
        self.cached_func = cached_func
        self.func_name = func.__name__
        self.execution_time: dt.timedelta | None = None
        self.is_failed = False
        self.is_disabled = is_disabled
        # Register this instance
        self.global_store().add_instance(self.func_name, self)

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> OutputType:
        """Call the cached function and track dependencies."""
        caller_instance = _current_function.get()
        _current_function.set(self)  # Mark this function as active

        start_time = dt.datetime.now()
        with st.spinner(f"Running {self.func.__name__}...", show_time=True):
            func = self.func if self.is_disabled else self.cached_func
            result = func(*args, **kwargs)
        end_time = dt.datetime.now()
        execution_time = end_time - start_time

        if self.execution_time is None:
            self.execution_time = execution_time

        _current_function.set(caller_instance)  # Restore the previous caller

        # Register dependency if called within another cached function
        if caller_instance:
            self.global_store().add_dependency(caller_instance.func_name, self.func_name)

        return result

    def clear(self):
        """Clears this function's cache and all upstream dependencies recursively."""
        store = self.global_store()
        # Clear all downstream dependencies first
        for dep_name in store.get_downstream_deps(self.func_name):
            dep_instance = store.get_instance(dep_name)
            if dep_instance:
                dep_instance.clear()

        # Clear this function's cache
        self.cached_func.clear()
        store.reset_instance_deps(self.func_name)
        self.execution_time = None

    def render(self):
        """Renders Streamlit buttons for clearing caches in the dependency chain."""
        store = self.global_store()
        upstream_deps = store.get_upstream_deps(self.func_name)

        def recursively_build_items(deps: dict) -> list[Union[str, dict, sac.TreeItem]]:
            return [
                sac.TreeItem(
                    label=dep_name,
                    children=recursively_build_items(dep_upstream_deps),
                    icon="arrow-clockwise",
                    tag=(instance.execution_time_str if (instance := store.get_instance(dep_name)) else "Never run"),
                )
                for dep_name, dep_upstream_deps in deps.items()
            ]

        cols = st.columns([5, 2])
        with cols[1]:
            if instance := store.get_instance(self.selection_sk.value):
                if st.button(f"Clear Cache for {self.selection_sk.value}"):
                    instance.clear()
                    st.rerun()
        with cols[0]:
            sac.tree(
                items=recursively_build_items({self.func_name: upstream_deps}),
                label="Clear Dependencies",
                size="lg",
                open_all=True,
                checkbox_strict=True,
                key=self.selection_sk.key_for_component,
            )

    def call_and_render(self, *args: P.args, **kwargs: P.kwargs) -> OutputType:
        """Call the cached function and render the dependencies."""
        try:
            self.is_failed = False
            return self(*args, **kwargs)
        except Exception as e:
            self.is_failed = True
            raise e
        finally:
            self.render()
```

### Global Store Coordination

**Location**: `src/utils/streamlit/helpers/global_store.py`

The global store manages cache dependencies and instances:

```python
class StreamlitUtilsGlobalStore:
    def __init__(self):
        self._cache_dependencies: dict[str, set[str]] = defaultdict(set)
        self._instances: dict[str, CachedFunction] = {}

    def add_dependency(self, caller_name: str, callee_name: str):
        """Add a dependency where caller depends on callee."""
        if caller_name not in self._instances or callee_name not in self._instances:
            self.rebuild_instances()
        assert caller_name in self._instances
        assert callee_name in self._instances
        self._cache_dependencies[caller_name].add(callee_name)

    def get_upstream_deps(self, func_name: str, visited: set[str] | None = None) -> dict:
        """Get all functions that this function depends on (recursively)."""
        if visited is None:
            visited = set()

        if func_name in visited:
            return {}

        visited.add(func_name)
        return {dep_name: self.get_upstream_deps(dep_name, visited) for dep_name in self._cache_dependencies[func_name]}

    def get_downstream_deps(self, func_name: str, visited: set[str] | None = None) -> set[str]:
        """Get all functions that depend on this function (recursively)."""
        if visited is None:
            visited = set()

        if func_name in visited:
            return set()

        visited.add(func_name)
        deps = set()
        for caller, callees in self._cache_dependencies.items():
            if func_name in callees:
                deps.add(caller)
                deps.update(self.get_downstream_deps(caller, visited))
        return deps
```

### Usage Examples

#### Basic Caching

```python
from src.utils.streamlit.helpers.cache import CacheWithDependencies

@CacheWithDependencies()
def load_model_evaluations(code_version: TCodeVersionName, model_arch_and_size: MODEL_ARCH_AND_SIZE) -> TPromptData:
    return get_model_evaluations(code_version, [model_arch_and_size])[model_arch_and_size]

# Usage with UI rendering
evaluations = load_model_evaluations.call_and_render(code_version, model_arch_and_size)
```

#### Resource Caching

```python
@CacheWithDependencies(is_resource=True, max_entries=1)
def load_unique_tokenizers(model_arch_and_sizes: list[MODEL_ARCH_AND_SIZE]) -> Tokenizers:
    return Tokenizers.from_unique_tokenizers(model_arch_and_sizes)
```

#### Disabled Caching

```python
@CacheWithDependencies(disable_cache=True)
def load_results_bank() -> ResultBank:
    return get_experiment_results_bank()
```

### Integration Between Session State and Caching

The session state and caching systems work together seamlessly:

```python
# Session state provides configuration
code_version = AppSessionKeys.code_version.value
model_arch = AppSessionKeys._selected_gpu.value

# Caching uses session state values
evaluations = load_model_evaluations(code_version, model_arch)

# Changes in session state automatically invalidate dependent caches
if AppSessionKeys.code_version.is_changed:
    # Caches will be automatically cleared when dependencies change
    st.success("Code version updated, caches cleared!")
```

### Advanced Features

1. **Dependency Tracking**: Automatic tracking of function dependencies
2. **Recursive Clearing**: Clearing a cache clears all dependent caches
3. **UI Integration**: Built-in UI for cache management
4. **Performance Monitoring**: Execution time tracking
5. **Error Handling**: Graceful handling of cache failures
6. **Type Safety**: Full type safety throughout the caching system

## Component Composition Patterns

### Hierarchical Composition

Components can be composed hierarchically for complex UIs:

```python
class ComplexPage(StreamlitPage):
    def render(self):
        # Compose multiple components
        config = ConfigComponent().render()
        data = DataComponent().render()
        
        # Use component outputs
        if config and data:
            PlotComponent(config=config, data=data).render()
```

### Conditional Rendering

Components support conditional rendering based on state:

```python
class ConditionalComponent(StreamlitComponent[Optional[str]]):
    def render(self) -> Optional[str]:
        if st.session_state.get("show_input", False):
            return st.text_input("Conditional input")
        return None
```

### Data Flow Patterns

Components can pass data between each other:

```python
class DataProvider(StreamlitComponent[List[int]]):
    def render(self) -> List[int]:
        return [1, 2, 3, 4, 5]

class DataConsumer(StreamlitComponent[None]):
    def __init__(self, data: List[int]):
        self.data = data
    
    def render(self) -> None:
        st.write(f"Received data: {self.data}")

# Composition
data = DataProvider().render()
DataConsumer(data).render()
```

## Profiling and Performance

### Built-in Profiling

All components support performance profiling:

```python
class ProfiledComponent(StreamlitComponent[str]):
    def render(self) -> str:
        return st.text_input("Input")

# Profile the component
component = ProfiledComponent()
component.profile_render()  # Automatically profiles and shows toast notification
```

### Performance Best Practices

1. **Lazy Loading**: Only render components when needed
2. **State Minimization**: Minimize session state usage
3. **Component Reuse**: Reuse components across pages
4. **Efficient Updates**: Use session keys for efficient state updates

## Real-World Examples

### Plot Plan Selector Component

**Location**: `src/app/components/plot_plans.py`

```python
class PlotPlanSelector(StreamlitComponent[None]):
    """Component for selecting a plot plan from the list of available plans."""

    def __init__(self, plot_plans: PlotPlans, selected_plot_id_sk: SessionKey[TPlotID], new_label: TPlotID):
        self.plot_plans = plot_plans
        self.selected_plot_id_sk = selected_plot_id_sk
        self.new_plot_id = new_label

    def render(self):
        # Complex UI logic with type safety
        if not self.plot_plans.is_plan_exists(self.selected_plot_id_sk.value):
            self.selected_plot_id_sk.value = self.new_plot_id

        # Group plans by appendix/main
        main_plans = [p for p in self.plot_plans.values() if not p.is_appendix]
        appendix_plans = [p for p in self.plot_plans.values() if p.is_appendix]

        # Create menu items with type safety
        menu_items: List[Union[str, dict, sac.MenuItem]] = []
        # ... menu creation logic
```

### Plot Plan Editor Component

```python
@dataclass
class PlotPlanEditor(StreamlitComponent[Optional[PlotPlan]]):
    """Component for editing or creating a plot plan."""

    plot_plans: PlotPlans
    plan_id: Optional[TPlotID]

    def render(self) -> Optional[PlotPlan]:
        # Complex form logic with type-safe return
        if self.plan_id:
            existing_plan = self.plot_plans.get_plan(self.plan_id)
        else:
            existing_plan = PlotPlan(...)

        # Form rendering logic
        # ...

        if submit_button:
            return existing_plan
        return None
```

### Home Page Implementation

**Location**: `src/app/page/p01_home.py`

```python
class HomePage(StreamlitPage):
    def render(self):
        # Create navigation
        st.sidebar.success("Select a page above to explore different analyses.")

        # Global variables
        if st.button("Reset App"):
            st.session_state.clear()

        # Compose other components
        select_gpu_type()
        AppSessionKeys.code_version.create_input_widget("CodeVersion")
        select_window_size()
```

### AgGrid Usage Example

**Location**: `src/app/components/data_requirements.py`

```python
class RequirementsDisplay(StreamlitComponent):
    def render(self) -> DataReqs | None:
        original_df = self.summarized_data_fulfilled_reqs.to_df()
        data_reqs_df = original_df[DataReqConsts.DATA_REQS_FILTER_COLUMNS]

        # Use AgGrid with advanced features
        df, grid_builder = base_grid_builder(
            data_reqs_df, 
            self.selection_mode, 
            hide_columns=self.hide_columns
        )
        
        # Configure columns
        for col in DataReqConsts.DATA_REQS_FILTER_COLUMNS:
            grid_builder.configure_column(col, type=["textColumn"])
        
        # Apply default filters
        set_aagrid_apply_default_filters(
            grid_builder,
            {SummarizedDataFulfilledReqsCols.AvailableOptions: ["0"]},
        )
        
        # Display with AgGrid
        grid_response = AgGrid(
            df,
            gridOptions=grid_builder.build(),
            height=self.height,
            fit_columns_on_grid_load=True,
            floatingFilter=True,
            key=self.key,
            update_mode=GridUpdateMode.SELECTION_CHANGED,
            data_return_mode=DataReturnMode.FILTERED,
            allow_unsafe_jscode=True,
        )
        
        return self.process_selection(grid_response)
```

### Pydantic v2 Usage Example

**Location**: `src/app/components/multi_plots.py`

```python
class HeatmapPlotGenerationComponent(StreamlitComponent):
    def render(self):
        # Use Pydantic v2 for complex configuration
        image_grid_params = ImageGridParams.model_validate(
            pydantic_input(key="my_form", model=ImageGridParams)
        )
        
        # Create grid organizer with UI-configurable properties
        grid_organizer = GridOrganizer.model_validate(
            pydantic_input(key="grid_organizer", model=GridOrganizer)
        )
        
        # Use the validated configuration
        grid_organizer.col_order = default_col_order
        # ... rest of plotting logic
```

## Best Practices

### Component Design

1. **Single Responsibility**: Each component should have one clear purpose
2. **Type Safety**: Always use appropriate generic types
3. **Composition**: Build complex UIs from simple components
4. **Reusability**: Design components to be reusable across pages

### Session State Management

1. **Use SessionKey**: Always use typed session keys instead of raw session state
2. **Default Values**: Provide sensible defaults for all session keys
3. **Error Handling**: Handle missing session state gracefully
4. **Cleanup**: Clean up session state when no longer needed

### Performance Optimization

1. **Profile Components**: Use `profile_render()` for performance analysis
2. **Lazy Loading**: Only render components when needed
3. **Efficient Updates**: Use session keys for efficient state updates
4. **Memory Management**: Clean up resources properly

### Code Organization

1. **Component Separation**: Keep components in separate files
2. **Type Definitions**: Define types in appropriate modules
3. **Documentation**: Document complex component behavior
4. **Testing**: Test components in isolation

## Extended UI Components and Utilities

### AgGrid Integration

The application includes sophisticated AgGrid integration for advanced data table functionality:

**Location**: `src/utils/streamlit/components/aagrid.py`

#### Core Features

```python
from src.utils.streamlit.components.aagrid import (
    SelectionMode,
    base_grid_builder,
    set_aagrid_apply_default_filters,
    set_pre_selected_rows,
    FIT_STRATEGY
)

# Grid builder with advanced features
df, grid_builder = base_grid_builder(
    df,
    selection_mode=SelectionMode.MULTIPLE,
    hide_columns=["id", "internal_data"],
    fit_strategy=FIT_STRATEGY.FIT_CELL_CONTENTS,
    hide_singular_columns=True,
    pre_selected_rows=["row1", "row2"]
)

# Apply default filters
set_aagrid_apply_default_filters(
    grid_builder,
    {"status": ["active"], "category": ["important"]}
)

# Configure grid options
grid_options = grid_builder.build()
```

#### Selection Modes

```python
class SelectionMode(StrEnum):
    DISABLED = "disabled"
    SINGLE = "single"
    MULTIPLE = "multiple"
```

#### Advanced Features

1. **Auto-sizing**: Automatic column sizing based on content
2. **Filtering**: Built-in filtering with default values
3. **Selection**: Single/multiple row selection with pre-selection
4. **Pagination**: Automatic pagination with configurable page sizes
5. **JavaScript Integration**: Custom JavaScript for advanced interactions

#### Usage Example

```python
from st_aggrid import AgGrid, DataReturnMode, GridUpdateMode

# Create grid with advanced features
grid_response = AgGrid(
    df,
    gridOptions=grid_options,
    height=400,
    fit_columns_on_grid_load=True,
    floatingFilter=True,
    key="my_grid",
    update_mode=GridUpdateMode.SELECTION_CHANGED,
    data_return_mode=DataReturnMode.FILTERED,
    allow_unsafe_jscode=True,
)

# Access selected data
selected_data = grid_response["selected_data"]
```

## Extended UI Components and Utilities

The Streamlit application includes sophisticated extended UI components that enhance the user experience for scientific computing applications. These components provide advanced functionality beyond basic Streamlit widgets.

### Background Task Management

The application includes a comprehensive background task management system for handling long-running operations:

**Location**: `src/utils/streamlit/helpers/background_task.py`

#### Core Components

```python
from src.utils.streamlit.helpers.background_task import (
    BackgroundTask, TasksManager, TaskStatus, show_task_status, show_tasks_manager_summary
)

# Task status enumeration
class TaskStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"
    ERROR = "ERROR"
```

#### BackgroundTask[InputType, ResultType]

Represents a single background task with full lifecycle management:

```python
@dataclass
class BackgroundTask(Generic[T, R]):
    name: str
    func: Callable[[T, threading.Event], R]
    input_data: T
    cancellation_event: threading.Event
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: TaskStatus = TaskStatus.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[R] = None
    error: Optional[Exception] = None
    thread: Optional[threading.Thread] = None
```

#### TasksManager[InputType, ResultType]

Manages multiple background tasks with coordination and monitoring:

```python
@dataclass
class TasksManager(Generic[T, R]):
    tasks: Dict[str, BackgroundTask[T, R]] = field(default_factory=dict)
    cancellation_event: threading.Event = field(default_factory=threading.Event)

    def create_and_start_task(self, name: str, func: Callable[[T, threading.Event], R], input_data: T) -> BackgroundTask[T, R]:
        """Create and immediately start a new task."""
        task = BackgroundTask(name, func, input_data, self.cancellation_event)
        self.tasks[task.task_id] = task
        task.start()
        return task

    def get_progress_percentage(self) -> float:
        """Get overall progress as a percentage (completed/total)."""
        total = len(self.tasks)
        if total == 0:
            return 0.0
        completed = sum(1 for task in self.tasks.values() 
                       if task.status in (TaskStatus.COMPLETED, TaskStatus.CANCELLED, TaskStatus.ERROR))
        return (completed / total) * 100
```

#### UI Integration

```python
# Display individual task status
def show_task_status(task: BackgroundTask, show_progress: bool = True) -> None:
    status_colors = {
        TaskStatus.PENDING: "blue",
        TaskStatus.RUNNING: "blue", 
        TaskStatus.COMPLETED: "green",
        TaskStatus.CANCELLED: "orange",
        TaskStatus.ERROR: "red",
    }
    
    st.write(f"Task: **{task.name}** - :{status_colors[task.status]}[{task.status}]")
    
    if show_progress and task.status == TaskStatus.RUNNING:
        st.progress(0.5)
    
    if task.is_done() and task.get_duration() is not None:
        st.write(f"Duration: {task.get_duration():.2f} seconds")
    
    if task.status == TaskStatus.ERROR and task.error:
        st.error(f"Error: {str(task.error)}")

# Display tasks manager summary with controls
def show_tasks_manager_summary(
    get_manager: Callable[[], TasksManager],
    auto_start: bool = False,
    run_every: float = 4,
    on_cancel_click: Optional[Callable[[], Any]] = None,
    on_start_click: Optional[Callable[[], Any]] = None,
    get_additional_metrics: Callable[[TasksManager], tuple[Dict[str, str], Optional[float]]] = lambda _: ({}, None),
    button_keys_prefix: str = "",
) -> None:
    # Displays progress bar, summary statistics, and control buttons
    # Automatically refreshes based on run_every parameter
```

#### Usage Example

```python
from src.app.components.tokenization import AnalysisJob, TokenizationResults

# Create analysis job
analysis_job = AnalysisJob(prompts, unique_tokenizers)

# Start background processing
analysis_job.start(existing_results)

# Display progress with custom metrics
def get_additional_metrics(task: AnalysisJob) -> Tuple[Dict[str, str], float]:
    total_pairs = task.total_pairs_needed
    completed_pairs = len(task.results.processed_pairs)
    progress = completed_pairs / total_pairs if total_pairs > 0 else 0.0
    
    metrics = {
        "Pairs": f"{completed_pairs}/{total_pairs}",
        "Tokenizers": str(len(task.unique_tokenizers))
    }
    
    return metrics, progress

# Display in UI
show_tasks_manager_summary(
    get_manager=lambda: analysis_job,
    auto_start=True,
    run_every=2.0,
    get_additional_metrics=get_additional_metrics,
    button_keys_prefix="tokenization_"
)
```

### AgGrid Integration

Advanced data grid integration with selection, filtering, and JavaScript customization:

**Location**: `src/utils/streamlit/components/aagrid.py`

#### Core Features

```python
from src.utils.streamlit.components.aagrid import (
    SelectionMode, base_grid_builder, set_aagrid_apply_default_filters,
    set_pre_selected_rows, FIT_STRATEGY
)

# Selection modes
class SelectionMode(StrEnum):
    DISABLED = "disabled"
    SINGLE = "single" 
    MULTIPLE = "multiple"

# Auto-sizing strategies
class FIT_STRATEGY(StrEnum):
    FIT_CELL_CONTENTS = "fitCellContents"
    FIT_CONTENTS = "fitGridWidth"
```

#### Grid Builder Pattern

```python
def base_grid_builder(
    df: pd.DataFrame,
    selection_mode: SelectionMode,
    hide_columns: list[str],
    fit_strategy: FIT_STRATEGY = FIT_STRATEGY.FIT_CELL_CONTENTS,
    hide_singular_columns: bool = False,
    pre_selected_rows: list[str] | None = None,
) -> tuple[pd.DataFrame, GridOptionsBuilder]:
    # Configure pagination, selection, filtering, and auto-sizing
    # Returns processed dataframe and configured grid builder
```

#### Advanced Features

1. **Default Filters**: Apply default filters using JavaScript
2. **Pre-selection**: Pre-select specific rows with pagination handling
3. **Auto-sizing**: Automatic column sizing based on content or grid width
4. **Column Hiding**: Hide specific columns or columns with single values
5. **JavaScript Integration**: Custom JavaScript for advanced interactions

#### Usage Example

```python
# Create grid with advanced features
df, grid_builder = base_grid_builder(
    df,
    selection_mode=SelectionMode.MULTIPLE,
    hide_columns=["id", "internal_data"],
    fit_strategy=FIT_STRATEGY.FIT_CELL_CONTENTS,
    hide_singular_columns=True,
    pre_selected_rows=["row1", "row2"]
)

# Apply default filters
set_aagrid_apply_default_filters(
    grid_builder,
    {"status": ["active"], "category": ["important"]}
)

# Configure grid options
grid_options = grid_builder.build()

# Render with AgGrid
from st_aggrid import AgGrid, DataReturnMode, GridUpdateMode

grid_response = AgGrid(
    df,
    gridOptions=grid_options,
    height=400,
    fit_columns_on_grid_load=True,
    floatingFilter=True,
    key="my_grid",
    update_mode=GridUpdateMode.SELECTION_CHANGED,
    data_return_mode=DataReturnMode.FILTERED,
    allow_unsafe_jscode=True,
)

# Access selected data
selected_data = grid_response["selected_data"]
```

### Extended Pydantic Form Rendering

Enhanced Pydantic integration with custom type support and advanced form features:

**Location**: `src/utils/streamlit/components/extended_streamlit_pydantic.py`

#### Core Components

```python
from src.utils.streamlit.components.extended_streamlit_pydantic import (
    CustomInputUI, pydantic_input, annotate_dict_with_literal_values
)

class CustomInputUI(InputUI):
    """Extended version of InputUI that supports custom type renderers."""
    
    _custom_type_renderers: Dict[Type, Callable] = {}
    
    @classmethod
    def register_type_renderer(cls, type_cls: Type[T], renderer: Callable[[Any, str, Dict], T]) -> None:
        """Register a custom renderer for a specific type."""
        cls._custom_type_renderers[type_cls] = renderer
```

#### Advanced Type Support

```python
from pydantic_extra_types.color import Color
from src.utils.streamlit.components.extended_streamlit_pydantic import CustomInputUI

# Color picker support
def render_color_input(streamlit_app, key, property):
    color_value = property.get("init_value", "#000000")
    return streamlit_app.color_picker(
        property.get("title", "Select Color"),
        value=color_value,
        key=key
    )

# Register custom renderer
CustomInputUI.register_type_renderer(Color, render_color_input)

# Use in Pydantic model
class PlotConfig(BaseModel):
    title: str = Field(description="Plot title")
    color: Color = Field(default=Color("#FF0000"))
    data_points: List[int] = Field(default=[1, 2, 3])
```

#### Dictionary with Literal Keys

```python
# Create dictionary type with literal keys
def annotate_dict_with_literal_values(keys: list[str], value_type: type, default_factory: Callable[[], dict] = dict):
    result_type = dict[
        create_literal_value(keys) if len(keys) > 0 else str,
        value_type,
    ]
    
    def f_json_schema_extra(schema: dict):
        schema["maxItems"] = len(keys)
        if len(keys) == 0:
            schema["readOnly"] = True
    
    return Annotated[
        result_type,
        Field(default_factory=default_factory, json_schema_extra=f_json_schema_extra)
    ]

# Usage in model
class ConfigModel(BaseModel):
    parameters: annotate_dict_with_literal_values(
        ["alpha", "beta", "gamma"], 
        float, 
        default_factory=lambda: {"alpha": 0.1}
    )
```

#### Form Integration

```python
# Render Pydantic model as form
config_data = pydantic_input(
    "config_key", 
    ConfigModel, 
    with_form=True,
    group_optional_fields=GroupOptionalFieldsStrategy.NO,
    lowercase_labels=False,
    ignore_empty_values=False
)

# Access form data
if config_data:
    st.success(f"Configuration: {config_data}")
```

### Pydantic v2 Integration

The application includes custom Pydantic v2 integration for type-safe form rendering:

**Location**: `src/utils/streamlit/st_pydantic_v2/`

#### Core Features

```python
from src.utils.streamlit.st_pydantic_v2.input import pydantic_ui
from src.utils.streamlit.components.extended_streamlit_pydantic import pydantic_input

# Render Pydantic model as UI
@dataclass
class ConfigModel:
    title: str = Field(description="Plot title")
    data_points: List[int] = Field(default=[1, 2, 3])
    is_active: bool = Field(default=True)

# Using pydantic_ui (v2)
config = pydantic_ui("config_key", ConfigModel)

# Using pydantic_input (extended)
config_data = pydantic_input("config_key", ConfigModel, with_form=True)
```

#### Advanced Type Support

```python
from pydantic_extra_types.color import Color
from src.utils.streamlit.ui_pydantic_v2.extra_types import PercentageCrop

class AdvancedConfig(BaseModel):
    # Color picker
    plot_color: Color = Field(default=Color("#FF0000"))
    
    # Crop configuration
    image_crop: PercentageCrop = Field(
        default=PercentageCrop(left=0, top=0, width=100, height=100),
        json_schema_extra={"image": some_image}
    )
    
    # Enum support
    plot_type: Literal["line", "bar", "scatter"] = Field(default="line")
    
    # Nested models
    grid_config: ImageGridParams = Field(default_factory=ImageGridParams)
```

#### Form Integration

```python
# Form-based rendering
with st.form("config_form"):
    config = pydantic_ui("config", ConfigModel, use_form=True)
    submitted = st.form_submit_button("Save Configuration")
    
    if submitted:
        st.success(f"Configuration saved: {config.title}")
```

#### Custom Type Renderers

```python
from src.utils.streamlit.components.extended_streamlit_pydantic import CustomInputUI

# Register custom renderer for specific types
@CustomInputUI.register_type_renderer
def render_custom_type(streamlit_app, key, property):
    # Custom rendering logic
    return streamlit_app.selectbox(
        property.get("title", "Select option"),
        options=property.get("enum", []),
        key=key
    )
```

#### Backend Protocol

The Pydantic v2 integration uses a backend protocol for flexible UI rendering:

```python
from src.utils.streamlit.st_pydantic_v2.backend import BackendProtocol, StreamlitBackend

class BackendProtocol(Protocol):
    def text_input(self, label: str, key: str, **kw) -> str | None: ...
    def selectbox(self, label: str, options: Sequence[str], key: str, **kw) -> Optional[str]: ...
    def checkbox(self, label: str, key: str, **kw) -> bool: ...
    # ... other UI methods

class StreamlitBackend(BackendProtocol):
    def __init__(self, container: Optional[DeltaGenerator] = None):
        self.dg = container or st
    
    def text_input(self, label: str, key: str, **kw) -> str | None:
        return self.dg.text_input(label, key=key, **kw)
    # ... implementation of all protocol methods
```

#### Advanced Features

1. **Type-safe Validation**: Automatic Pydantic validation with error display
2. **Session State Integration**: Automatic session state management
3. **Form Support**: Seamless form integration with submit handling
4. **Custom Components**: Extensible component system for custom types
5. **Nested Models**: Support for complex nested Pydantic models
6. **Field Metadata**: Rich field metadata support for UI customization

### Directory Structure

The extended UI components are organized in a structured directory hierarchy:

```
src/utils/streamlit/
├── helpers/
│   ├── background_task.py      # Background task management
│   ├── component.py            # Base component classes
│   └── session_keys.py         # Session state management
├── components/
│   ├── aagrid.py              # AgGrid integration utilities
│   └── extended_streamlit_pydantic.py  # Extended Pydantic forms
└── st_pydantic_v2/
    ├── input.py               # Pydantic v2 input rendering
    └── backend.py             # Backend protocol implementation
```

### Component Selection Guidelines

#### When to Use Background Task Management

**Use BackgroundTask/TasksManager when:**
- Processing large datasets that would block the UI
- Running computationally intensive operations
- Need to show progress and allow cancellation
- Multiple operations need coordination
- Long-running analysis tasks

**Example Use Cases:**
- Tokenization analysis across multiple models
- Data preprocessing pipelines
- Model training or evaluation
- File upload and processing

#### When to Use AgGrid Integration

**Use AgGrid when:**
- Displaying large datasets with complex interactions
- Need advanced filtering and sorting capabilities
- Require row selection with pagination
- Need custom JavaScript interactions
- Displaying tabular data with multiple data types

**Example Use Cases:**
- Results data tables with filtering
- Model comparison tables
- Configuration parameter grids
- Data exploration interfaces

#### When to Use Extended Pydantic Forms

**Use Extended Pydantic Forms when:**
- Need type-safe form validation
- Complex nested data structures
- Custom type rendering (colors, files, etc.)
- Form-based configuration interfaces
- Data entry with validation

**Example Use Cases:**
- Plot configuration forms
- Model parameter configuration
- Data preprocessing settings
- User preference forms

#### When to Use Pydantic v2 Integration

**Use Pydantic v2 when:**
- Need the latest Pydantic features
- Complex type validation requirements
- Backend protocol flexibility
- Advanced form rendering capabilities
- Integration with modern Python type system

**Example Use Cases:**
- Complex configuration models
- API parameter validation
- Data transformation pipelines
- Schema-driven UI generation

### Integration Patterns

#### Combining Multiple Utilities

```python
# Example: Background task with AgGrid display
class DataAnalysisComponent(StreamlitComponent[None]):
    def render(self) -> None:
        # Background task for data processing
        analysis_job = AnalysisJob(data, models)
        
        # Display progress
        show_tasks_manager_summary(
            get_manager=lambda: analysis_job,
            auto_start=True,
            run_every=2.0
        )
        
        # Display results in AgGrid when complete
        if analysis_job.status == TaskStatus.COMPLETED:
            results_df = pd.DataFrame(analysis_job.get_results())
            df, grid_builder = base_grid_builder(
                results_df,
                selection_mode=SelectionMode.MULTIPLE,
                hide_columns=["internal_id"]
            )
            grid_response = AgGrid(df, gridOptions=grid_builder.build())
```

#### Form-Based Configuration

```python
# Example: Pydantic form with background task
class ConfigFormComponent(StreamlitComponent[ConfigModel]):
    def render(self) -> ConfigModel:
        # Render configuration form
        config_data = pydantic_input(
            "analysis_config",
            AnalysisConfig,
            with_form=True
        )
        
        if config_data and st.button("Start Analysis"):
            # Start background task with configuration
            analysis_job = AnalysisJob(config_data)
            analysis_job.start()
            
        return config_data
```

### Extension Mechanisms

#### Custom Type Renderers

```python
# Register custom type renderer
@CustomInputUI.register_type_renderer
def render_file_input(streamlit_app, key, property):
    uploaded_file = streamlit_app.file_uploader(
        property.get("title", "Upload File"),
        type=["txt", "csv", "json"],
        key=key
    )
    return uploaded_file

# Use in Pydantic model
class FileConfig(BaseModel):
    data_file: Optional[UploadedFile] = Field(description="Upload data file")
```

#### Custom Grid Features

```python
# Extend grid builder with custom features
def custom_grid_builder(df: pd.DataFrame, **kwargs) -> tuple[pd.DataFrame, GridOptionsBuilder]:
    df, grid_builder = base_grid_builder(df, **kwargs)
    
    # Add custom JavaScript
    custom_js = JsCode("""
    function(params) {
        // Custom grid behavior
        params.api.addEventListener('cellClicked', function(event) {
            console.log('Cell clicked:', event.data);
        });
    }
    """)
    
    grid_builder.configure_grid_options(onGridReady=custom_js)
    return df, grid_builder
```

#### Custom Task Types

```python
# Extend task manager for specific use cases
class ModelTrainingJob(TasksManager[TrainingConfig, TrainingResults]):
    def __init__(self, model_config: TrainingConfig):
        super().__init__()
        self.model_config = model_config
    
    def start_training(self):
        self.create_and_start_task(
            "model_training",
            train_model_function,
            self.model_config
        )
```

## Architectural Patterns and Best Practices

The Streamlit application follows established architectural patterns that promote code reusability, maintainability, and type safety. These patterns are designed to support the complex requirements of scientific computing applications.

### Core Architectural Principles

#### 1. Component Composition Over Inheritance

The application emphasizes composition over inheritance, using the `StreamlitComponent[OutputType]` base class as a foundation for all UI components:

**Principle**: Components should be composed from smaller, reusable pieces rather than inheriting complex hierarchies.

**Implementation**: Each component implements the abstract `render()` method and can be composed with other components to build complex UIs.

**Benefits**: 
- Easier testing and maintenance
- Better code reusability
- Clearer separation of concerns
- Reduced coupling between components

#### 2. Single Responsibility Pattern

Each component has a single, well-defined responsibility:

**Principle**: A component should have only one reason to change.

**Examples**:
- `ShowResultsBank`: Displays and manages result bank data
- `TokenizationComponent`: Handles tokenization analysis and visualization
- `PromptFilterComponent`: Manages prompt filtering and selection

**Benefits**:
- Easier to understand and modify
- Better testability
- Reduced complexity
- Clear interfaces

#### 3. Type-Safe Interface Design

The application uses generics and strong typing throughout:

**Principle**: All component interfaces should be type-safe and self-documenting.

**Implementation**: 
- `StreamlitComponent[OutputType]` uses generics for return types
- `SessionKey[T]` provides type-safe session state management
- Background tasks use `TasksManager[InputType, ResultType]`

**Benefits**:
- Compile-time error detection
- Better IDE support
- Self-documenting code
- Reduced runtime errors

#### 4. Separation of Concerns

Clear separation between pages, components, and business logic:

**Principle**: Pages handle navigation and composition, components handle specific functionality, and utilities provide reusable services.

**Structure**:
- **Pages** (`src/app/page/`): Handle navigation and page-level composition
- **Components** (`src/app/components/`): Reusable UI components with specific functionality
- **Utilities** (`src/utils/streamlit/`): Shared utilities and base classes

### Best Practices

#### Component Design Guidelines

1. **Consistent Interface**: All components should implement the `render()` method consistently
2. **Type Safety**: Use generics and strong typing for all component interfaces
3. **Session State Management**: Use `SessionKey` instead of direct session state access
4. **Error Handling**: Implement graceful error handling with user-friendly messages
5. **Performance**: Use profiling capabilities for performance-critical components

#### State Management Best Practices

1. **Use SessionKey**: Always use `SessionKey[T]` for type-safe session state management
2. **Avoid Direct Access**: Never access `st.session_state` directly
3. **Centralized Keys**: Use centralized key management through `SessionKeyDescriptor`
4. **Change Detection**: Use `is_changed` property to detect state changes
5. **External Updates**: Use `post_external_update()` for programmatic state changes

#### Naming Conventions

1. **Components**: Use descriptive names ending with "Component" (e.g., `TokenizationComponent`)
2. **Pages**: Use descriptive names ending with "Page" (e.g., `ResultsBankPage`)
3. **Session Keys**: Use descriptive names with consistent prefixes
4. **Functions**: Use verb-noun format for action functions
5. **Constants**: Use UPPER_CASE for constants and configuration values

#### Error Handling Patterns

1. **Graceful Degradation**: Components should handle errors gracefully without crashing
2. **User Feedback**: Provide clear error messages and recovery options
3. **Logging**: Use appropriate logging levels for debugging
4. **Validation**: Validate inputs and provide helpful error messages
5. **Recovery**: Provide mechanisms to recover from errors

### Anti-Patterns to Avoid

#### 1. Direct Session State Manipulation

**Anti-Pattern**:
```python
# DON'T: Direct session state access
st.session_state["my_key"] = value
value = st.session_state.get("my_key", default)
```

**Correct Pattern**:
```python
# DO: Use SessionKey for type-safe access
session_key = SessionKey[str]("my_key", "default")
session_key.value = value
value = session_key.value
```

#### 2. Bypassing Component Patterns

**Anti-Pattern**:
```python
# DON'T: Bypass component architecture
def render_page():
    st.title("My Page")
    # Direct widget creation without components
    user_input = st.text_input("Input")
    return user_input
```

**Correct Pattern**:
```python
# DO: Use component architecture
class MyPage(StreamlitPage):
    def render(self):
        st.title("My Page")
        return InputComponent().render()
```

#### 3. Hardcoded Configurations

**Anti-Pattern**:
```python
# DON'T: Hardcode configuration values
def render_component():
    st.selectbox("Model", ["gpt2", "llama", "mamba"])
```

**Correct Pattern**:
```python
# DO: Use centralized configuration
def render_component():
    st.selectbox("Model", list(ModelType))
```

#### 4. Complex Component Hierarchies

**Anti-Pattern**:
```python
# DON'T: Create deep inheritance hierarchies
class BaseComponent:
    pass

class IntermediateComponent(BaseComponent):
    pass

class SpecificComponent(IntermediateComponent):
    pass
```

**Correct Pattern**:
```python
# DO: Use composition and single inheritance
class SpecificComponent(StreamlitComponent[OutputType]):
    def __init__(self, dependencies):
        self.dependencies = dependencies
    
    def render(self) -> OutputType:
        # Compose functionality from dependencies
        pass
```

### Scientific Computing Integration

The architectural patterns are specifically designed to support scientific computing requirements:

#### 1. Data Flow Management

**Pattern**: Components handle data transformation and visualization with clear input/output contracts.

**Example**: `TokenizationComponent` takes raw data and produces visualization results with type-safe interfaces.

#### 2. Experiment Coordination

**Pattern**: Background tasks coordinate complex experiments while maintaining UI responsiveness.

**Example**: `AnalysisJob` manages tokenization analysis across multiple models with progress tracking.

#### 3. Configuration Management

**Pattern**: Type-safe configuration through Pydantic models with validation.

**Example**: Plot configuration forms use Pydantic models for validation and type safety.

#### 4. Result Visualization

**Pattern**: Reusable visualization components with consistent interfaces.

**Example**: `ShowResultsBank` provides consistent result display across different analysis types.

### Extension Guidelines

#### Adding New Components

1. **Inherit from Base Class**: Always inherit from `StreamlitComponent[OutputType]`
2. **Implement Render Method**: Provide a clear `render()` implementation
3. **Use Type Safety**: Define clear input/output types
4. **Handle Errors**: Implement graceful error handling
5. **Document Interface**: Provide clear documentation for component usage

#### Adding New Pages

1. **Inherit from StreamlitPage**: Use `StreamlitPage` as the base class
2. **Compose Components**: Build pages from smaller, reusable components
3. **Handle Navigation**: Implement clear navigation patterns
4. **Manage State**: Use appropriate session state management
5. **Follow Naming**: Use consistent naming conventions

#### Extending Utilities

1. **Maintain Interface**: Preserve existing interfaces when extending
2. **Add Type Safety**: Ensure new utilities are type-safe
3. **Document Changes**: Update documentation for new features
4. **Test Thoroughly**: Ensure new utilities work with existing components
5. **Follow Patterns**: Use established patterns for consistency

### Performance Considerations

#### Component Optimization

1. **Lazy Loading**: Load data only when needed
2. **Caching**: Use appropriate caching strategies for expensive operations
3. **Profiling**: Use built-in profiling capabilities to identify bottlenecks
4. **Background Processing**: Use background tasks for long-running operations
5. **Efficient Rendering**: Minimize unnecessary re-renders

#### Memory Management

1. **Session State Cleanup**: Clean up session state when no longer needed
2. **Resource Disposal**: Properly dispose of resources (files, connections)
3. **Background Task Cleanup**: Cancel background tasks when appropriate
4. **Cache Management**: Implement appropriate cache invalidation strategies

## Cross-References

- **Session State Management**: See [docs/streamlit-session-state.md](streamlit-session-state.md) for detailed session state patterns
- **Extended UI Components**: This document covers the extended UI components and utilities
- **Architectural Patterns**: See [docs/streamlit-architectural-patterns.md](streamlit-architectural-patterns.md) for design patterns
- **Utilities**: See [docs/utilities-and-patterns.md](utilities-and-patterns.md) for supporting utilities

## Troubleshooting

### Common Issues

1. **Type Errors**: Ensure all components properly implement the abstract `render()` method
2. **Session State Issues**: Use `SessionKey` instead of direct session state access
3. **Performance Problems**: Use `profile_render()` to identify bottlenecks
4. **Component Composition**: Ensure compatible types between components

### Debugging Tips

1. **Check Types**: Verify generic type parameters are correct
2. **Session State**: Use `st.write(st.session_state)` to debug session state
3. **Component Isolation**: Test components in isolation
4. **Profiling**: Use built-in profiling to identify performance issues 
