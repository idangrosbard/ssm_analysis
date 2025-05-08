from __future__ import annotations

# mypy: ignore-errors
import datetime as _dt
from typing import (
    Any,
    List,
    Optional,
    Protocol,
    Sequence,
    Union,
)

import streamlit as st
from streamlit.delta_generator import DeltaGenerator


class BackendProtocol(Protocol):
    def text_input(self, label: str, key: str, **kw) -> str: ...

    def text_area(self, label: str, key: str, **kw) -> str: ...

    def number_input(self, label: str, key: str, **kw) -> Union[int, float]: ...

    def checkbox(self, label: str, key: str, **kw) -> bool: ...

    def button(self, label: str, key: Optional[str] = None, **kw) -> bool: ...

    def selectbox(self, label: str, options: Sequence[str], key: str, **kw) -> Optional[str]: ...

    def multiselect(self, label: str, options: Sequence[str], key: str, **kw) -> List[str]: ...

    def date_input(self, label: str, key: str, **kw) -> _dt.date: ...

    def color_picker(self, label: str, key: str, **kw) -> str: ...

    def subheader(self, txt: str) -> None: ...

    def markdown(self, txt: str) -> None: ...

    def info(self, txt: str) -> None: ...

    def warning(self, txt: str) -> None: ...

    def columns(self, spec: Sequence[int]) -> List[Any]: ...

    def expander(self, label: str, **kw) -> BackendProtocol: ...

    def form(self, key: str, clear_on_submit: bool = False) -> Any: ...

    def form_submit_button(self, label: str) -> bool: ...

    def text(self, txt: str) -> None: ...

    def toast(self, txt: str) -> None: ...

    def container(self, **kw) -> BackendProtocol: ...

    def cropper(self, img, key, is_relative_coords: bool = False, **kw) -> Any: ...

    def __getattr__(self, item):
        # This is a fallback for unknown attributes
        pass


class StreamlitBackend(BackendProtocol):  # Implementation of BackendProtocol
    def __init__(self, container: Optional[DeltaGenerator] = None):
        self.dg = container or st

    # input components
    def text_input(self, label: str, key: str, **kw) -> str:
        result = self.dg.text_input(label, key=key, **kw)
        return str(result) if result is not None else ""

    def text_area(self, label: str, key: str, **kw) -> str:
        result = self.dg.text_area(label, key=key, **kw)
        return str(result) if result is not None else ""

    def number_input(self, label: str, key: str, **kw) -> Union[int, float]:
        result = self.dg.number_input(label, key=key, **kw)
        return result if result is not None else 0

    def checkbox(self, label: str, key: str, **kw) -> bool:
        result = self.dg.checkbox(label, key=key, **kw)
        return bool(result) if result is not None else False

    def button(self, label: str, key: Optional[str] = None, **kw) -> bool:
        result = self.dg.button(label, key=key, **kw)
        return bool(result) if result is not None else False

    def selectbox(self, label: str, options: Sequence[str], key: str, **kw) -> Optional[str]:
        return self.dg.selectbox(label, options, key=key, **kw)

    def multiselect(self, label: str, options: Sequence[str], key: str, **kw) -> List[str]:
        result = self.dg.multiselect(label, options, key=key, **kw)
        return list(result) if result is not None else []

    def date_input(self, label: str, key: str, **kw) -> _dt.date:
        result = self.dg.date_input(label, key=key, **kw)
        return result if result is not None else _dt.date.today()

    def color_picker(self, label: str, key: str, **kw) -> str:
        result = self.dg.color_picker(label, key=key, **kw)
        return result if result is not None else "#000000"

    def slider(self, label: str, min_value: float, max_value: float, key: str, **kw) -> float:
        result = self.dg.slider(label, min_value=min_value, max_value=max_value, key=key, **kw)
        return float(result) if result is not None else min_value

    # containers + other components
    def subheader(self, txt: str) -> None:
        self.dg.subheader(txt)

    def markdown(self, txt: str) -> None:
        self.dg.markdown(txt)

    def info(self, txt: str) -> None:
        self.dg.info(txt)

    def warning(self, txt: str) -> None:
        self.dg.warning(txt)

    def columns(self, spec: Sequence[int]) -> List[BackendProtocol]:
        return [StreamlitBackend(col) for col in self.dg.columns(spec)]

    def expander(self, label: str, **kw) -> BackendProtocol:
        return StreamlitBackend(self.dg.expander(label, **kw))

    def container(self, **kw) -> BackendProtocol:
        return StreamlitBackend(self.dg.container(**kw))

    def form(self, key: str, clear_on_submit: bool = False) -> Any:
        return self.dg.form(key, clear_on_submit=clear_on_submit)

    def form_submit_button(self, label: str) -> bool:
        return self.dg.form_submit_button(label)

    def text(self, txt: str) -> None:
        self.dg.text(txt)

    def toast(self, txt: str) -> None:
        self.dg.toast(txt)

    def cropper(self, img, key, is_relative_coords: bool = False, **kw) -> Any:
        """Render a streamlit-cropper component."""
        try:
            from streamlit_cropper import st_cropper

        except ImportError:
            self.warning("streamlit-cropper is not installed. Run 'pip install streamlit-cropper'.")
            return

        @st.dialog(title="Interactive Crop", width="large")
        def _crop_dialog(img, key: str, **kw):
            img_w, img_h = img.size

            def box_algorithm(*args, **kwargs) -> dict[str, int]:
                return {
                    side: st.session_state[f"{key}.{side}"] * (img_w if side in ["left", "width"] else img_h)
                    for side in ["left", "top", "width", "height"]
                    if f"{key}.{side}" in st.session_state
                }

            box = st_cropper(
                img,
                box_algorithm=box_algorithm,
                return_type="box",
                key=f"{key}_cropper",
                **kw,
            )

            # If save button is clicked, update the session state
            if st.button("Close", key=f"{key}_close"):
                assert isinstance(box, dict)
                for side in ["left", "top", "width", "height"]:
                    value = box[side]
                    divide_by = 1
                    if is_relative_coords:
                        if side in ["left", "width"]:
                            divide_by = img_w
                        elif side in ["top", "height"]:
                            divide_by = img_h

                    val = value / divide_by
                    st.session_state[f"{key}.{side}"] = val
                st.rerun()  # Rerun to update the UI

        if self.dg.button("Open Crop Dialog", key=f"{key}_open_crop_dialog"):
            _crop_dialog(img, key, **kw)

    def __getattr__(self, item):
        try:
            return getattr(self.dg, item)
        except AttributeError:
            raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{item}'")
