# Streamlit Infrastructure

Base classes and patterns for Streamlit component architecture. Focus on what LLMs need to know about UI component design.

## Core Base Classes

### StreamlitComponent[OutputType]

**File**: `src/utils/streamlit/helpers/component.py`

**Purpose**: Foundational abstract base class for all Streamlit components with type-safe generic support.

**Key Features**:
- **Generic Type System**: Uses `OutputType` generic parameter for type-safe return values
- **Abstract Render Method**: Enforces consistent interface across all components
- **Profiling Support**: Built-in performance analysis capabilities
- **Composition Ready**: Designed for easy composition and reuse

**Usage Pattern**:
```python
from src.utils.streamlit.helpers.component import StreamlitComponent

class MyComponent(StreamlitComponent[str]):
    def render(self) -> str:
        user_input = st.text_input("Enter text")
        return user_input
```

### StreamlitPage

**File**: `src/utils/streamlit/helpers/component.py`

**Purpose**: Specialized component for full-page implementations that don't return values.

**Semantic Distinction**: 
- **StreamlitComponent**: Reusable UI components that return values for composition
- **StreamlitPage**: Full-page implementations that handle navigation and page-level composition without returning values

**Usage Pattern**:
```python
class MyPage(StreamlitPage):
    def render(self) -> None:
        st.title("My Page")
        st.write("Page content")
```

## Component Architecture Patterns

### Type-Safe Component System

All components inherit from `StreamlitComponent[OutputType]` with generic type support:

```python
# Component with complex return type
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

### Component Composition Patterns

Components are designed for composition and reuse:

```python
# Component with no return value
class DisplayComponent(StreamlitComponent[None]):
    def render(self) -> None:
        st.write("This component displays information")
        st.button("Click me")

# Composing components
class ComplexPage(StreamlitPage):
    def render(self) -> None:
        config = PlotComponent().render()
        DisplayComponent().render()
        # Use config for further processing
```

### Profiling and Performance

Built-in profiling support for performance analysis:

```python
# Profiling is automatically available
component = MyComponent()
component.profile_render()  # Generates performance profile
```

## Session State Management

### AppSessionKeys

**File**: `src/app/app_consts.py`

**Purpose**: Centralized session state management with strong typing.

**Key Patterns**:
- **Global Configuration**: Code version, GPU selection, window size
- **Type Safety**: Strongly typed session state keys
- **Centralized Access**: Single source of truth for app state

### Session State Patterns

```python
# Setting session state
st.session_state[AppSessionKeys.code_version] = "v1.0"

# Accessing session state
code_version = st.session_state.get(AppSessionKeys.code_version, "default")
```

## Navigation Architecture

### Page Factory Pattern

**File**: `src/app/entry_point.py`

**Purpose**: Centralized page factory with enum-based navigation.

**Key Pattern**:
```python
def get_page(page_order: PAGE_ORDER) -> StreamlitPage:
    match page_order:
        case PAGE_ORDER.HOME:
            return p01_home.HomePage()
        case PAGE_ORDER.RESULTS_BANK:
            return p02_results_bank.ResultsBankPage()
        # ... other pages
```

### PAGE_ORDER Enum

**File**: `src/app/app_consts.py`

**Purpose**: Defines navigation structure and page metadata.

## Component Categories

### 1. Data Display Components
- **ResultBank Components**: `src/app/components/result_bank.py`
- **Data Requirements**: `src/app/components/data_requirements.py`
- **Data Visualization**: `src/app/components/plot_generation.py`

### 2. Analysis Components
- **Model Analysis**: `src/app/components/model_analysis.py`
- **Tokenization**: `src/app/components/tokenization.py`
- **Info Flow**: `src/app/components/info_flow.py`

### 3. Configuration Components
- **Input Widgets**: `src/app/components/inputs.py`
- **Prompt Filtering**: `src/app/components/prompt_filter.py`
- **Plot Plans**: `src/app/components/plot_plans.py`

### 4. Visualization Components
- **Plot Generation**: `src/app/components/plot_generation.py`
- **Multi Plots**: `src/app/components/multi_plots.py`
- **Heatmaps**: Integrated in plot generation

## File Structure

```
src/app/
├── entry_point.py              # Page factory and navigation
├── app_consts.py              # Session keys and page enums
├── app_utils.py               # App utilities
├── data_store.py              # Data management
├── texts.py                   # UI text constants
├── components/                # Reusable UI components
│   ├── result_bank.py         # Results display
│   ├── data_requirements.py   # Data requirements UI
│   ├── model_analysis.py      # Model analysis components
│   ├── tokenization.py        # Tokenization UI
│   ├── plot_generation.py     # Plot generation
│   ├── prompt_filter.py       # Prompt filtering
│   ├── info_flow.py           # Info flow analysis
│   ├── inputs.py              # Input widgets
│   ├── plot_plans.py          # Plot planning
│   └── multi_plots.py         # Multi-plot components
└── page/                      # Page implementations
    ├── p01_home.py            # Home page
    ├── p02_results_bank.py    # Results bank page
    ├── p03_data_requirements.py # Data requirements page
    ├── p04_heatmap_creation.py # Heatmap creation
    ├── p05_info_flow_analysis.py # Info flow analysis
    ├── p06_final_plots.py     # Final plots page
    ├── p07_prompts_comparison.py # Prompts comparison
    ├── p08_mamba_analysis.py  # Mamba analysis
    └── p09_prompt_filteration_presets.py # Prompt filtering presets
```

## Best Practices

1. **Type Safety**: Always use generic types for component return values
2. **Composition**: Build complex UIs from simple, reusable components
3. **Session State**: Use centralized session state management
4. **Profiling**: Use built-in profiling for performance analysis
5. **Navigation**: Follow the page factory pattern for consistent navigation

## Cross-References

- **Core Modules**: See [docs/core-modules.md](core-modules.md) for data object patterns
- **Infrastructure**: See [docs/infrastructure.md](infrastructure.md) for base class patterns
- **Utilities**: See [docs/utilities-and-patterns.md](utilities-and-patterns.md) for utility patterns
- **Data Interfaces**: See [docs/data-relationships-interface.md](data-relationships-interface.md) for data object patterns
- **Setup and Environment**: See [docs/setup-and-environment.md](setup-and-environment.md) for environment setup
- **Analysis and Plotting**: See [docs/analysis-and-plotting.md](analysis-and-plotting.md) for plotting components
