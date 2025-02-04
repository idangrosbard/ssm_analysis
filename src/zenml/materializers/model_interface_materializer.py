import json
from pathlib import Path
from typing import Any, Type

from src.models.model_interface import ModelInterface, get_model_interface
from src.types import MODEL_ARCH
from zenml.enums import ArtifactType
from zenml.materializers.base_materializer import BaseMaterializer


class ModelInterfaceMaterializer(BaseMaterializer):
    ASSOCIATED_TYPES = (ModelInterface,)
    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.DATA

    @property
    def config_path(self) -> Path:
        return Path(self.uri) / "config.json"

    def load(self, data_type: Type[Any]) -> Any:
        if not self.config_path.exists():
            raise FileNotFoundError("ModelInterface configuration file not found.")

        config = json.load(self.config_path.open("r"))

        model_arch_str = config["model_arch"]
        # Convert model_arch_str back to MODEL_ARCH enum
        try:
            model_arch = next(m for m in MODEL_ARCH if str(m) == model_arch_str or m.value == model_arch_str)
        except StopIteration:
            raise ValueError(f"Unknown model_arch: {model_arch_str}")

        model_size = config["model_size"]

        # Use from_arch_and_size to create a new ModelInterface instance
        return get_model_interface(model_arch, model_size)

    def save(self, data: ModelInterface) -> None:
        # Use to_config to get the configuration
        config = data.to_config()

        json.dump(config, self.config_path.open("w"))
