import os
from typing import Any, Type

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from zenml.enums import ArtifactType, VisualizationType
from zenml.materializers.base_materializer import BaseMaterializer


class NumpyArrayMaterializer(BaseMaterializer):
    """Materializer for numpy arrays with visualization capabilities."""

    ASSOCIATED_TYPES = (np.ndarray,)
    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.DATA

    def handle_input(self, data_type: Type[Any]) -> bool:
        """Handles input validation."""
        return issubclass(data_type, np.ndarray)

    def load(self, data_type: Type[Any]) -> np.ndarray:
        """Load the numpy array from disk."""
        array_path = os.path.join(self.uri, "array.npy")
        return np.load(array_path)

    def save(self, data: np.ndarray) -> None:
        """Save the numpy array and generate visualization."""
        # Save the array
        array_path = os.path.join(self.uri, "array.npy")
        np.save(array_path, data)

    def save_visualizations(self, data: np.ndarray) -> dict[str, VisualizationType]:
        """Generate and save visualizations for the array."""
        visualizations = {}

        # Generate and save visualization
        plt.figure(figsize=(10, 6))

        # If the array is 1D, create a line plot
        if data.ndim == 1:
            plt.plot(data)
            plt.title("Array Values")
            plt.xlabel("Index")
            plt.ylabel("Value")

        # If the array is 2D, create a heatmap
        elif data.ndim == 2:
            sns.heatmap(data, cmap="RdYlGn", center=0)
            plt.title("Array Heatmap")

        # Save the plot
        plt.tight_layout()
        p = os.path.join(self.uri, "visualization.png")
        plt.savefig(p)
        visualizations[p] = VisualizationType.IMAGE
        plt.close()

        p = os.path.join(self.uri, "array_info.txt")
        # Save array info
        with open(p, "w") as f:
            f.write(f"Array Shape: {data.shape}\n")
            f.write(f"Array Type: {data.dtype}\n")
            f.write(f"Array Size: {data.size}\n")
            f.write(f"Array Mean: {data.mean():.4f}\n")
            f.write(f"Array Std: {data.std():.4f}\n")
            f.write(f"Array Min: {data.min():.4f}\n")
            f.write(f"Array Max: {data.max():.4f}\n")

        visualizations[p] = VisualizationType.MARKDOWN

        # Show first few elements (head) of the array
        head_size = min(5, data.size)
        p = os.path.join(self.uri, "array_head.txt")
        with open(p, "w") as f:
            f.write(f"Array Head (first {head_size} elements):\n")
            if data.ndim == 1:
                f.write(str(data[:head_size]))
            else:
                f.write(str(data.flatten()[:head_size]))

        visualizations[p] = VisualizationType.MARKDOWN

        return visualizations
