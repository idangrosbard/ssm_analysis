from setuptools import find_packages, setup

setup(
    name="ssm_analysis",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "pyrallis",
        "submitit",
        "torch",
        "numpy",
        "pandas",
        "scipy",
        "tqdm",
        "beartype",
        "jaxtyping",
        "transformers",
        "datasets",
    ],
    extras_require={
        "dev": [
            "pre-commit",
            "ruff>=0.3.0",
        ],
        "streamlit": ["streamlit"],
    },
)
