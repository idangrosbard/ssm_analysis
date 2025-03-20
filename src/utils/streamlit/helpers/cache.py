import contextvars
import datetime as dt
from typing import Callable
from typing import Generic
from typing import Optional
from typing import ParamSpec
from typing import Union

import humanize
import streamlit_antd_components as sac

from src.utils import streamlit as st
from src.utils.streamlit.helpers.component import OutputType
from src.utils.streamlit.helpers.global_store import _get_global_store

P = ParamSpec("P")


class CachedFunction(Generic[P, OutputType]):
    """A strongly typed wrapper for a cached function with recursive clearing and UI rendering."""

    def global_store(self):
        return _get_global_store()

    def __init__(self, func: Callable[P, OutputType], cached_func: Callable[P, OutputType]):
        self.func = func
        self.cached_func = cached_func
        self.func_name = func.__name__
        self.execution_time: dt.timedelta | None = None
        # Register this instance
        self.global_store().add_instance(self.func_name, self)

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> OutputType:
        """Call the cached function and track dependencies."""
        caller_instance = _current_function.get()
        _current_function.set(self)  # Mark this function as active

        start_time = dt.datetime.now()
        with st.spinner(f"Running {self.func.__name__}...", show_time=True):
            result = self.cached_func(*args, **kwargs)
        end_time = dt.datetime.now()
        execution_time = end_time - start_time

        if self.execution_time is None:
            self.execution_time = execution_time

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
                dep_instance.clear()

        # Clear this function's cache
        self.cached_func.clear()  # type: ignore
        store.reset_instance_deps(self.func_name)
        self.execution_time = None

    @property
    def execution_time_str(self) -> str:
        """Get the execution time as a human-readable string."""
        if self.execution_time is None:
            return "Never run"
        return humanize.precisedelta(
            self.execution_time,
            minimum_unit="milliseconds",
            suppress=["milliseconds"],
            format="%0.2f",
        )

    def render(self):
        """Renders Streamlit buttons for clearing caches in the dependency chain."""
        store = self.global_store()

        # Show upstream dependencies (functions that this one depends on)
        upstream_deps = store.get_upstream_deps(self.func_name)

        def recursively_build_items(deps: dict) -> list[Union[str, dict, sac.TreeItem]]:
            return [
                sac.TreeItem(
                    label=dep_name,
                    children=recursively_build_items(dep_upstream_deps),
                    icon="arrow-clockwise",
                    tag=(instance.execution_time_str if (instance := store.get_instance(dep_name)) else "Never run"),
                )
                for dep_name, dep_upstream_deps in deps.items()
            ]

        selected_item = sac.tree(
            items=recursively_build_items({self.func_name: upstream_deps}),
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


_current_function = contextvars.ContextVar[Optional[CachedFunction]]("current_function", default=None)
