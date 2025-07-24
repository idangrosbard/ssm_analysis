---
description: Dataset Processing Flow and Patterns
inclusion: manual
---


# Dataset Processing Flow and Patterns

## Data Processing Flow

### Raw Data Processing
- **File**: `src/data_ingestion/datasets/download_dataset.py`
- **Purpose**: Raw data downloading and initial processing
- **Output**: Raw datasets in standardized format

### Dataset Splitting
- **File**: `src/data_ingestion/datasets/splitting.py`
- **Purpose**: Dataset splitting logic for train/test/validation
- **Output**: Split datasets ready for experiments

## Processing Patterns

### Data Flow Overview
1. **Raw data loading** → download_dataset.py
2. **Dataset splitting** → splitting.py
3. **Experiment usage** → through data interface objects

### Key Components
- **Download utilities** for raw data acquisition
- **Splitting logic** for dataset partitioning
- **Standardized formats** for consistent processing

### Critical Rules
- **FOLLOW standardized data formats** throughout processing
- **USE consistent splitting logic** across all datasets
- **MAINTAIN data integrity** through processing pipeline

## File Locations

### Primary Files
- `src/data_ingestion/datasets/download_dataset.py` - Raw data downloading
- `src/data_ingestion/datasets/splitting.py` - Dataset splitting logic

### Processing Standards
- **Consistent data formats** across all processing steps
- **Reproducible splitting** with fixed seeds
- **Error handling** for data processing failures

**Reference**: docs/dataset-processing-flow.md for complete patterns and examples
