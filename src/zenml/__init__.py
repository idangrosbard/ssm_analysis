import importlib.abc
import sys
from typing import Any, NewType, Optional, Type, Union, cast

from zenml.materializers.materializer_registry import MaterializerRegistry
from zenml.utils.typing_utils import get_origin as original_get_origin

# region: Patch MaterializerRegistry.__getitem__
"""
Make MaterializerRegistry compatible with NewType
"""


def resolve_newtype(type_):
    if not hasattr(type_, "__mro__") and isinstance(type_, NewType):  # Handle NewType case
        type_ = type_.__supertype__  # Extract the original type
    return type_


original_getitem = MaterializerRegistry.__getitem__


def patched_getitem(self, key):
    return original_getitem(self, resolve_newtype(key))  # type: ignore


MaterializerRegistry.__getitem__ = patched_getitem  # type: ignore
# endregion


# region: Patch get_origin

"""
Make get_origin compatible with NewType
"""


# Function to patch `get_origin` dynamically
def patch_get_origin():
    """Patch `get_origin` if ZenML is already imported."""
    if "zenml.utils.typing_utils" in sys.modules:
        from zenml.utils.typing_utils import get_origin as original_get_origin

        def patched_get_origin(tp: Type[Any]) -> Optional[Type[Any]]:
            """Patched version that handles NewType."""
            if not hasattr(tp, "__mro__") and isinstance(tp, NewType):
                return tp.__supertype__  # type: ignore
            return original_get_origin(tp)

        # Apply patch
        sys.modules["zenml.utils.typing_utils"].get_origin = patched_get_origin  # type: ignore


# Try to patch immediately (if ZenML is already imported)
patch_get_origin()

# Hook into future imports of ZenML
import importlib


class ZenMLImportHook(importlib.abc.MetaPathFinder):
    """Intercepts imports and patches ZenML when it's imported."""

    def find_spec(self, fullname, path, target=None):
        if fullname == "zenml.utils.typing_utils":
            # ZenML is about to be imported—wait for it to load, then patch
            importlib.import_module(fullname)  # Ensure it loads
            patch_get_origin()  # Apply the patch
        return None  # Continue normal import process


# Install the import hook
sys.meta_path.insert(0, ZenMLImportHook())
# endregion
