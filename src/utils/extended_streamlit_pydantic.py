from enum import Enum
from typing import Any, Callable, Dict, Type, TypeVar

from pydantic import BaseModel
from streamlit_pydantic import schema_utils
from streamlit_pydantic.ui_renderer import GroupOptionalFieldsStrategy, InputUI

# Also, need to fix a bug in the library:
# see  https://github.com/lukasmasuch/streamlit-pydantic/issues/69


T = TypeVar("T")


class ExtendedSchemaUtils:
    """Extension of schema utils with additional type checks."""

    @staticmethod
    def is_multi_enum_property_extended(property: Dict, references: Dict) -> bool:
        """Check if property is a list of enums, extending the original check."""
        # First check original implementation
        if schema_utils.is_multi_enum_property(property, references):
            return True

        if property.get("type") != "array":
            return False

        if property.get("items", {}).get("$ref") is None:
            return False

        try:
            # Get the reference item
            reference = schema_utils.resolve_reference(property["items"]["$ref"], references)
            # Check if it's an enum
            return bool(reference.get("enum"))
        except Exception:
            return False


class CustomInputUI(InputUI):
    """Extended version of InputUI that supports custom type renderers."""

    # Registry to store custom type renderers
    _custom_type_renderers: Dict[Type, Callable] = {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._schema_properties = self._input_class.model_json_schema(by_alias=True).get("properties", {})
        self._schema_references = self._input_class.model_json_schema(by_alias=True).get("$defs", {})

    @classmethod
    def register_type_renderer(cls, type_cls: Type[T], renderer: Callable[[Any, str, Dict], T]) -> None:
        """Register a custom renderer for a specific type.

        Args:
            type_cls: The type class to register a renderer for
            renderer: A function that takes (streamlit_app, key, property) and returns the rendered value
        """
        cls._custom_type_renderers[type_cls] = renderer

    def _render_property(self, streamlit_app: Any, key: str, property: Dict) -> Any:
        """Override _render_property to support custom type renderers."""
        # Check if we have a custom renderer for this type
        if property.get("init_value") is not None:
            value_type = type(property["init_value"])
            if value_type in self._custom_type_renderers:
                return self._custom_type_renderers[value_type](streamlit_app, key, property)

        # # Check if this is a list of enums using our extended check
        # if ExtendedSchemaUtils.is_multi_enum_property_extended(property, self._schema_references):
        #     return self._render_multi_enum_input_extended(streamlit_app, key, property)

        # Fallback to parent class implementation
        return super()._render_property(streamlit_app, key, property)

    def _render_multi_enum_input_extended(self, streamlit_app: Any, key: str, property: Dict) -> Any:
        """Render any list of enums input."""
        streamlit_kwargs = self._get_default_streamlit_input_kwargs(key, property)
        overwrite_kwargs = self._get_overwrite_streamlit_kwargs(key, property)

        # Get the enum values either from direct enum or reference
        select_options = []
        if property.get("items", {}).get("enum"):
            select_options = property["items"]["enum"]
        else:
            # Get from reference
            reference_item = schema_utils.resolve_reference(property["items"]["$ref"], self._schema_references)
            select_options = reference_item["enum"]

        # Get current value or default
        current_value = []
        if property.get("init_value"):
            current_value = [opt.value if isinstance(opt, Enum) else opt for opt in property["init_value"]]
        elif property.get("default"):
            try:
                current_value = property.get("default")
            except Exception:
                pass

        # Create multiselect
        selected_values = streamlit_app.multiselect(
            **{**streamlit_kwargs, "options": select_options, "default": current_value, **overwrite_kwargs}
        )

        # If we have a reference, convert back to enum values
        if property.get("items", {}).get("$ref"):
            # Import the enum dynamically based on the reference name
            ref_name = property["items"]["$ref"].split("/")[-1]
            enum_class = None

            # Try to find the enum class in the references
            reference_item = self._schema_references.get(ref_name)
            if reference_item and reference_item.get("enum"):
                # Get module path from the reference if available
                module_path = reference_item.get("module_path")
                if module_path:
                    import importlib

                    try:
                        module = importlib.import_module(module_path)
                        enum_class = getattr(module, ref_name)
                    except (ImportError, AttributeError):
                        pass

            if enum_class and issubclass(enum_class, Enum):
                return [enum_class(val) for val in selected_values]

        return selected_values


def pydantic_input(
    key: str,
    model: Type[BaseModel],
    group_optional_fields: GroupOptionalFieldsStrategy = GroupOptionalFieldsStrategy.NO,
    lowercase_labels: bool = False,
    ignore_empty_values: bool = False,
) -> Dict:
    """Extended version of pydantic_input that uses CustomInputUI."""
    return CustomInputUI(
        key,
        model,
        group_optional_fields=group_optional_fields,
        lowercase_labels=lowercase_labels,
        ignore_empty_values=ignore_empty_values,
    ).render_ui()


# Example of how to register a custom renderer:
# def render_my_custom_type(streamlit_app: Any, key: str, property: Dict) -> Any:
#     # Custom rendering logic here
#     return streamlit_app.text_input(**property)
#
# CustomInputUI.register_type_renderer(MyCustomType, render_my_custom_type)
