# SSM Analysis

This repository contains the code for the analysis of State Space Models (SSMs). The main goal of this project is to be a fully open-source and reproducible research project.

## Environment Setup

To set up the environment, we recommend using `uv`, a fast Python package installer and resolver.

1.  Install `uv`:

    ```bash
    pip install uv
    ```

2.  Create a virtual environment:

    ```bash
    uv venv
    ```

3.  Activate the virtual environment:

    ```bash
    source .venv/bin/activate
    ```

4.  Install the dependencies:
    ```bash
    uv pip install -r requirements.txt
    ```

## Reproducing Results

To reproduce the plots from the paper, run the `notebooks/reproduce_plots.ipynb` notebook. This notebook will:

1. **Run experiments** to generate the necessary data (model evaluation, heatmaps, and information flow analysis)
2. **Generate plots** using the experimental data and save them to the `final_plots/combined_plots/` directory

### Quick Start

```bash
# First, ensure you have installed the dependencies and activated the environment
# Then, start the Jupyter Notebook server
jupyter notebook
```

Then open and run the `notebooks/reproduce_plots.ipynb` notebook.

### Hardware Requirements

- **GPU**: Recommended for running experiments. The notebook will automatically use GPU if available.
- **Memory**: At least 8GB RAM recommended. Larger models may require more memory.
- **Storage**: Several GB of free space for storing experimental results and generated plots.

### Configuration

The notebook includes configuration options to customize the reproduction:

- **Models**: By default, runs on smaller models (MAMBA1-130M, MAMBA2-130M, GPT2-355M) for faster execution
- **Sample Size**: Uses first 100 prompts by default for faster testing (can be changed to use all prompts)
- **Experiment Types**: Configurable knockout experiments and window sizes

### Note on Execution Time

Running the full experiments may take significant time depending on your hardware:
- **With GPU**: 30 minutes to several hours depending on the models and sample size
- **CPU only**: Much longer (not recommended for full reproduction)

The notebook includes progress indicators and will skip already-computed experiments on subsequent runs.

## Project Structure

The project has been simplified for easy reproduction. Here is an overview of the key directories:

```
ssm_analysis/
├── final_plots/          # Contains plot plans and generated figures
│   └── plot_plans.json
├── notebooks/            # Jupyter notebooks for analysis and reproduction
│   └── reproduce_plots.ipynb
├── src/                  # Source code for the analysis
│   ├── analysis/
│   ├── core/
│   ├── data_ingestion/
│   ├── data_loading/     # New module for loading data
│   ├── experiments/
│   ├── plotting/         # New module for generating plots
│   └── utils/
├── requirements.txt      # Project dependencies
└── README.md             # This file
```
