import sys
from contextlib import contextmanager
from io import StringIO
from typing import Any, Callable, Generic, TypeVar, cast, get_args, get_origin

import streamlit as st
from streamlit_pydantic.ui_renderer import GroupOptionalFieldsStrategy, InputUI

TSessionKey = TypeVar("TSessionKey")


class SessionKey(Generic[TSessionKey]):
    """A strongly typed wrapper around streamlit session state values."""

    def __init__(self, key: str, default_value: TSessionKey | None = None):
        self.key = key
        self.default_value = default_value
        self._ever_changed = False

    def exists(self) -> bool:
        return self.key in st.session_state

    def delete(self):
        if self.exists():
            del st.session_state[self.key]

    def update(self, value: TSessionKey):
        self._ever_changed = True
        st.session_state[self.key] = value

    def init(self, value: TSessionKey):
        if not self.exists():
            st.session_state[self.key] = value

    @property
    def value(self) -> TSessionKey:
        """Get the current value. Raises KeyError if not initialized and no default."""
        if not self.exists() and self.default_value is None:
            raise KeyError(f"Session key '{self.key}' not initialized and has no default value")
        return cast(TSessionKey, st.session_state[self.key] if self.exists() else self.default_value)

    @value.setter
    def value(self, new_value: TSessionKey):
        """Set the current value."""
        self.update(new_value)

    def __str__(self) -> str:
        """Return the current value as string, useful for streamlit widgets."""
        return str(self.value)

    @property
    def ever_changed(self) -> bool:
        """Whether the value has ever been changed from its default."""
        return self._ever_changed

    def equal_if_exists(self, func: Callable[[TSessionKey], bool]) -> bool:
        if self.exists():
            return func(self.value)
        return False

    def exists_and_not_none(self) -> bool:
        return self.equal_if_exists(lambda val: val is not None)

    def update_button(self, value: TSessionKey, label: str):
        st.button(label=label, key=label, on_click=lambda: self.update(value))

    def create_input_widget(
        self,
        label: str,
        streamlit_container: Any = st,
        group_optional_fields: GroupOptionalFieldsStrategy = GroupOptionalFieldsStrategy.NO,
        lowercase_labels: bool = False,
        ignore_empty_values: bool = False,
    ) -> None:
        """Create an input widget for this session key using streamlit_pydantic's UI renderer.

        Args:
            label: Label for the input widget
            streamlit_container: Streamlit container to render in (default: st)
            group_optional_fields: How to group optional fields (default: NO)
            lowercase_labels: Whether to lowercase labels (default: False)
            ignore_empty_values: Whether to ignore empty values (default: False)
        """
        # Create a minimal Pydantic model for this single value
        from pydantic import BaseModel, Field, create_model

        # Get the actual type of the value by inspecting the generic parameters
        value_type = type(self.value) if self.value is not None else Any
        if hasattr(value_type, "__origin__"):  # Handle generic types like List, Dict etc
            origin = get_origin(value_type)
            args = get_args(value_type)
            if origin is not None and args:
                value_type = origin[args]

        # Create model dynamically to preserve type information
        SingleValueModel = create_model(
            "SingleValueModel", value=(value_type, Field(title=label, default=self.value)), __base__=BaseModel
        )

        # Use InputUI to render the widget
        input_ui = InputUI(
            key=self.key,
            model=SingleValueModel,
            streamlit_container=streamlit_container,
            group_optional_fields=group_optional_fields,
            lowercase_labels=lowercase_labels,
            ignore_empty_values=ignore_empty_values,
        )

        # Render and update value
        result = input_ui.render_ui()
        if result and "value" in result:
            self.value = result["value"]


class SessionKeyDescriptor(Generic[TSessionKey]):
    """A descriptor that creates SessionKey instances with automatic prefixing."""

    def __init__(self, default_value: TSessionKey | None = None):
        self.default_value = default_value
        self.key: str | None = None

    def __set_name__(self, owner: Any, name: str):
        # Add prefix based on class name
        prefix = owner.__name__.lower().strip("_")
        self.key = f"{prefix}_{name}"

    def __get__(self, obj: Any, objtype: Any = None) -> SessionKey[TSessionKey]:
        if obj is None:
            raise ValueError("SessionKeyDescriptor must be used as a class attribute")
        # Create or get SessionKey instance
        if not hasattr(obj, f"_{self.key}_instance"):
            assert self.key is not None, "SessionKeyDescriptor not properly initialized with __set_name__"
            session_key = SessionKey(self.key, self.default_value)
            if self.default_value is not None:
                session_key.init(self.default_value)
            setattr(obj, f"_{self.key}_instance", session_key)
        return getattr(obj, f"_{self.key}_instance")


@contextmanager
def st_redirect(src, dst, placeholder, overwrite):
    output_func = getattr(placeholder.empty(), dst)

    with StringIO() as buffer:
        old_write = src.write

        def new_write(b):
            is_newline = b == "\n"
            if is_newline:
                return

            old_write(b)
            buffer.write(b + "\r\n")

            # Without this condition, will cause infinite loop because we can't write to the streamlit from thread
            # TODO: st.script_run_context not found, fix this
            # if getattr(current_thread(), st.script_run_context.SCRIPT_RUN_CONTEXT_ATTR_NAME, None) is None:
            #     if overwrite:
            #         buffer.truncate(0)
            #         buffer.seek(0)
            #     return

            output_func(buffer.getvalue())

        try:
            src.write = new_write
            yield
        finally:
            src.write = old_write


@contextmanager
def st_stdout(dst, placeholder, overwrite):
    "this will show the prints"
    with st_redirect(sys.stdout, dst, placeholder, overwrite):
        yield


@contextmanager
def st_stderr(dst, placeholder, overwrite):
    "This will show the logging"
    with st_redirect(sys.stderr, dst, placeholder, overwrite):
        yield
