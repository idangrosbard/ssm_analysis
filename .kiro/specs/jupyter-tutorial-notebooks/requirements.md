# Requirements Document

## Introduction

This feature will create a comprehensive set of Jupyter notebooks that serve as an interactive tutorial for the SSM Analysis project. The notebooks will provide extensive explanations of the project's core concepts, demonstrate key functionality, and guide users through the complete research workflow from data ingestion to analysis and visualization.

## Requirements

### Requirement 1: Comprehensive Project Tutorial Notebook

**User Story:** As a researcher or developer new to the SSM Analysis project, I want a comprehensive tutorial notebook that covers the project overview, architecture, and basic usage, so that I can understand the project's purpose, structure, and how to get started.

#### Notebook to Create:
- `01_ssm_analysis_comprehensive_tutorial.ipynb` - Complete project introduction, architecture overview, and basic usage guide

#### Acceptance Criteria

1. WHEN a user opens the tutorial notebook THEN the system SHALL provide a clear explanation of the research purpose, Mamba State-Space Models, and knockout methodology
2. WHEN a user reads the architecture section THEN the system SHALL explain the critical 3-file coordination pattern (names.py → types.py → consts.py), BaseRunner pattern, and ModelInterface abstraction
3. WHEN a user explores the project structure THEN the system SHALL demonstrate the src/ directory organization, module relationships, and data object hierarchy (DataReqs, ResultBank, FulfilledReqs)
4. WHEN a user learns about supported models THEN the system SHALL provide examples of all supported architectures (Mamba-1, Mamba-2, GPT-2, Llama, Mistral, Qwen) with model size variants
5. WHEN a user follows setup instructions THEN the system SHALL demonstrate environment setup, package installation using UV, and basic configuration
6. IF a user wants to understand the web interface THEN the system SHALL show Streamlit application navigation and key features

### Requirement 2: Complete Experimental Workflow Notebook

**User Story:** As a researcher conducting experiments, I want a comprehensive notebook that walks through the complete experimental workflow from data ingestion to analysis and visualization, so that I can reproduce results and conduct my own analyses.

#### Notebook to Create:
- `02_complete_experimental_workflow.ipynb` - End-to-end experimental pipeline with detailed explanations

#### Acceptance Criteria

1. WHEN a user follows the data ingestion section THEN the system SHALL demonstrate CounterFact dataset downloading, preprocessing steps, and train/test splitting procedures
2. WHEN a user runs knockout experiments THEN the system SHALL show parameter configuration, model loading, and experiment execution for different architectures (Mamba, GPT-2, Llama)
3. WHEN a user analyzes information flow THEN the system SHALL demonstrate subject-token information emergence tracking, layer-wise dynamics analysis, and feature disentanglement techniques
4. WHEN a user creates visualizations THEN the system SHALL show heatmap generation, interpretation methods, and customization options
5. WHEN a user executes the full pipeline THEN the system SHALL demonstrate the complete workflow from raw data to final research insights
6. WHEN a user performs comparative analysis THEN the system SHALL show cross-architecture comparison techniques and statistical analysis methods
7. IF a user wants to interpret results THEN the system SHALL demonstrate identifying universal vs model-specific phenomena and extracting research insights

### Requirement 3: Advanced Development and Extension Notebook

**User Story:** As a developer extending the project or conducting advanced research, I want a comprehensive notebook that demonstrates how to add new components, create custom experiments, and implement sophisticated analysis techniques, so that I can contribute to the project and conduct cutting-edge research.

#### Notebook to Create:
- `03_advanced_development_and_research.ipynb` - Advanced development patterns, extension techniques, and research applications

#### Acceptance Criteria

1. WHEN a user adds new model architectures THEN the system SHALL demonstrate the complete process including core module updates (names.py → types.py → consts.py), ModelInterface implementation, and knockout mechanism integration
2. WHEN a user creates custom experiments THEN the system SHALL show how to implement BaseRunner subclasses with proper dependency management, parameter handling, and result storage
3. WHEN a user extends analysis components THEN the system SHALL demonstrate integration with the existing plotting framework, data processing pipelines, and visualization tools
4. WHEN a user develops web interface extensions THEN the system SHALL show how to create new Streamlit pages following established patterns and integrate with existing components
5. WHEN a user implements advanced techniques THEN the system SHALL demonstrate sophisticated intervention strategies, multi-layer analysis, and advanced interpretation methods
6. WHEN a user ensures reproducibility THEN the system SHALL show best practices for reproducible research, result validation, and experimental design
7. WHEN a user creates publication materials THEN the system SHALL demonstrate creating high-quality visualizations suitable for academic publication
8. IF a user adapts to new datasets THEN the system SHALL show how to extend the framework for new datasets and research questions beyond CounterFact
