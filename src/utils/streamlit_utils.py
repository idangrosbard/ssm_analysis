import contextvars
import sys
from abc import ABC, abstractmethod
from collections import defaultdict
from contextlib import contextmanager
from io import StringIO
from typing import (
    Any,
    Callable,
    ClassVar,
    Generic,
    Optional,
    ParamSpec,
    Type,
    TypeVar,
    Union,
    cast,
    get_args,
    get_origin,
)

import streamlit as st
import streamlit_antd_components as sac
from streamlit_pydantic.ui_renderer import GroupOptionalFieldsStrategy, InputUI

import src.final_plots.app  # noqa: F401

TSessionKey = TypeVar("TSessionKey")


# region SessionKey
class SessionKey(Generic[TSessionKey]):
    """A strongly typed wrapper around streamlit session state values."""

    def __init__(self, key: str, default_value: TSessionKey | None = None, allow_none: Optional[bool] = None):
        self._key = key
        self.default_value = default_value
        self._ever_changed = False
        self._allow_none = default_value is None if allow_none is None else allow_none

    def exists(self) -> bool:
        return self.key in st.session_state

    def delete(self):
        if self.exists():
            del st.session_state[self.key]

    def _update(self, value: TSessionKey | None):
        self._ever_changed = True
        if self._allow_none or value is not None:
            st.session_state[self.key] = value
        else:
            self.delete()

    def init(self, value: TSessionKey):
        if not self.exists():
            st.session_state[self.key] = value

    @property
    def key(self) -> str:
        return self._key

    @property
    def _key_need_external_update(self) -> "SessionKey[bool]":
        sk = SessionKey(f"{self.key}_need_external_update")
        sk.init(False)
        return sk

    @property
    def _key_next_external_update_value(self) -> "SessionKey[TSessionKey | None]":
        sk = SessionKey(f"{self.key}_next_external_update_value")
        sk.init(None)
        return sk

    @property
    def _key_for_prev_value(self) -> "SessionKey[TSessionKey | None]":
        sk = SessionKey(f"{self.key}_prev_value")
        sk.init(None)
        return sk

    @property
    def is_changed(self) -> bool:
        """
        You need to check this value *before* the call for the component.
        """
        return self._key_for_prev_value.value != self.value

    @property
    def prev_value(self) -> TSessionKey | None:
        return self._key_for_prev_value.value

    @property
    def key_for_component(self) -> str:
        """
        This is a workaround to allow external updates to the value.
        And allow is_changed to work.
        Use this as the key for components that need this functionality.
        """
        if self._key_need_external_update.value:
            self._update(self._key_next_external_update_value.value)
            self._key_need_external_update.value = False
            self._key_next_external_update_value.value = None

        self._key_for_prev_value.value = self.value
        return self.key

    def post_external_update(self, value: TSessionKey | None, with_rerun: bool = True):
        """
        This is a workaround to allow external updates to the value after the component has been rendered.
        Use this method only for post render updates, else use update.
        """
        self._key_next_external_update_value.value = value
        self._key_need_external_update.value = True
        if with_rerun:
            st.rerun()

    @property
    def value(self) -> TSessionKey:
        """Get the current value. Raises KeyError if not _initialize and no default."""
        if not self.exists() and not self._allow_none:
            raise KeyError(f"Session key '{self.key}' not initialized and has no default value")
        return cast(TSessionKey, st.session_state[self.key] if self.exists() else self.default_value)

    @value.setter
    def value(self, new_value: TSessionKey):
        """Set the current value."""
        self._update(new_value)

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
        st.button(label=label, key=label, on_click=lambda: self._update(value))

    def reset_value(self):
        self._update(self.default_value)

    def post_external_reset_value(self, with_rerun: bool = True):
        self.post_external_update(self.default_value, with_rerun=with_rerun)

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

    def __init__(self, default_value: TSessionKey | None = None, allow_none: Optional[bool] = None):
        self.default_value = default_value
        self.key: str | None = None
        self.allow_none = allow_none

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
            session_key = SessionKey(self.key, self.default_value, self.allow_none)
            if self.allow_none or self.default_value is not None:
                session_key.init(cast(TSessionKey, self.default_value))
            setattr(obj, f"_{self.key}_instance", session_key)
        return getattr(obj, f"_{self.key}_instance")


_T_SESSION_KEYS_BASE = TypeVar("_T_SESSION_KEYS_BASE", bound="SessionKeysBase[Any]")


class SessionKeysBase(Generic[_T_SESSION_KEYS_BASE]):
    """Base class for session key containers that ensures singleton pattern."""

    _instance: ClassVar[dict[Type[Any], Any]] = {}

    def __new__(cls) -> _T_SESSION_KEYS_BASE:
        if cls not in cls._instance:
            cls._instance[cls] = super().__new__(cls)
        return cast(_T_SESSION_KEYS_BASE, cls._instance[cls])


# endregion


# region Redirect stdout and stderr to streamlit
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


# endregion

# region Streamlit components
OutputType = TypeVar("OutputType")


class StreamlitComponent(ABC, Generic[OutputType]):
    @abstractmethod
    def render(self) -> OutputType:
        pass


class StreamlitPage(StreamlitComponent[None]):
    pass


# endregion


# region Cached functions
P = ParamSpec("P")


class CachedFunction(Generic[P, OutputType]):
    """A strongly typed wrapper for a cached function with recursive clearing and UI rendering."""

    def global_store(self):
        return _get_global_store()

    def __init__(self, func: Callable[P, OutputType], cached_func: Callable[P, OutputType]):
        self.func = func
        self.cached_func = cached_func
        self.func_name = func.__name__
        # Register this instance
        self.global_store().add_instance(self.func_name, self)

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> OutputType:
        """Call the cached function and track dependencies."""
        caller_instance = _current_function.get()
        _current_function.set(self)  # Mark this function as active

        with st.spinner(f"Running {self.func.__name__}...", show_time=True):
            result = self.cached_func(*args, **kwargs)

        _current_function.set(caller_instance)  # Restore the previous caller

        # Register dependency if called within another cached function
        if caller_instance:
            self.global_store().add_dependency(caller_instance.func_name, self.func_name)

        return result

    def clear(self):
        """Clears this function's cache and all upstream dependencies recursively."""
        store = self.global_store()
        # Clear all downstream dependencies first
        for dep_name in store.get_downstream_deps(self.func_name):
            dep_instance = store.get_instance(dep_name)
            if dep_instance:
                dep_instance.cached_func.clear()  # type: ignore
                store.reset_instance_deps(dep_name)
        # Clear this function's cache
        self.cached_func.clear()  # type: ignore
        store.reset_instance_deps(self.func_name)

    def render(self):
        """Renders Streamlit buttons for clearing caches in the dependency chain."""
        store = self.global_store()

        # Show upstream dependencies (functions that this one depends on)
        upstream_deps = store.get_upstream_deps(self.func_name)

        def recursively_build_items(deps: dict) -> list[sac.TreeItem]:
            return [
                sac.TreeItem(label=dep_name, children=recursively_build_items(dep_upstream_deps))
                for dep_name, dep_upstream_deps in deps.items()
            ]

        items: list[Union[str, dict, sac.TreeItem]] = [
            sac.TreeItem(label=self.func_name, children=recursively_build_items(upstream_deps))
        ]

        selected_item = sac.tree(
            items=items,
            label="Clear Dependencies",
            size="lg",
            open_all=True,
            key=f"upstream_dependencies_tree_{self.func_name}",
        )

        if selected_item and isinstance(selected_item, str) and (instance := store.get_instance(selected_item)):
            if st.button(f"Clear Cache for {selected_item}"):
                instance.clear()
                st.rerun()


class CacheWithDependencies:
    """Class decorator wrapping @st.cache_data with strong typing, dependency tracking, and UI rendering."""

    def __init__(self, *st_args, **st_kwargs):
        self.st_args = st_args
        self.st_kwargs = st_kwargs

    def __call__(self, func: Callable[P, OutputType]) -> CachedFunction[P, OutputType]:
        cached_func = st.cache_data(*self.st_args, **self.st_kwargs)(func)
        return CachedFunction(func, cached_func)


# Thread-safe storage for tracking current function execution
_current_function = contextvars.ContextVar[Optional[CachedFunction]]("current_function", default=None)


# endregion


# region StreamlitUtilsGlobalStore
class StreamlitUtilsGlobalStore:
    def __init__(self):
        self._cache_dependencies: dict[str, set[str]] = defaultdict(set)
        self._instances: dict[str, CachedFunction] = {}

    def add_dependency(self, caller_name: str, callee_name: str):
        """Add a dependency where caller depends on callee."""
        if caller_name not in self._instances or callee_name not in self._instances:
            self.rebuild_instances()
        assert caller_name in self._instances
        assert callee_name in self._instances
        self._cache_dependencies[caller_name].add(callee_name)

    def get_upstream_deps(self, func_name: str, visited: set[str] | None = None) -> dict:
        """Get all functions that this function depends on (recursively)."""
        if visited is None:
            visited = set()

        if func_name in visited:
            return {}

        visited.add(func_name)
        return {dep_name: self.get_upstream_deps(dep_name, visited) for dep_name in self._cache_dependencies[func_name]}

    def get_downstream_deps(self, func_name: str, visited: set[str] | None = None) -> set[str]:
        """Get all functions that depend on this function (recursively)."""
        if visited is None:
            visited = set()

        if func_name in visited:
            return set()

        visited.add(func_name)
        deps = set()
        for caller, callees in self._cache_dependencies.items():
            if func_name in callees:
                deps.add(caller)
                deps.update(self.get_downstream_deps(caller, visited))
        return deps

    def add_instance(self, func_name: str, instance: CachedFunction):
        self._instances[func_name] = instance

    def reset_instance_deps(self, func_name: str):
        self._cache_dependencies[func_name] = set()

    def get_instance(self, func_name: str) -> Optional[CachedFunction]:
        """Get instance by function name, falling back to module search if needed."""
        return self._instances[func_name]

    def rebuild_instances(self):
        import inspect
        import sys

        for module in list(sys.modules.values()):
            if module is None:
                continue
            try:
                for _, obj in inspect.getmembers(module):
                    if isinstance(obj, CachedFunction):
                        # Update the instances map for future use
                        self._instances[obj.func_name] = obj
            except Exception:
                pass

    def clear_all_instances(self):
        """Clears all cached functions in the system."""
        # Clear all instances we know about
        for instance in self._instances.values():
            instance.clear()
        self._instances.clear()


@st.cache_resource(show_spinner=False)
def _get_global_store() -> StreamlitUtilsGlobalStore:
    """Get or create the cache store for dependencies and instances."""
    return StreamlitUtilsGlobalStore()


# endregion
