from abc import ABC
from abc import abstractmethod
from typing import Generic
from typing import TypeVar

from src.utils import streamlit as st

OutputType = TypeVar("OutputType")


class StreamlitComponent(ABC, Generic[OutputType]):
    @abstractmethod
    def render(self) -> OutputType:
        pass

    def profile_render(self):
        from wfork_streamlit_profiler import Profiler

        with Profiler():
            self.render()
            st.success("Rendering complete, generating profile...")


class StreamlitPage(StreamlitComponent[None]):
    pass
