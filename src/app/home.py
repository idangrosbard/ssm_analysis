import streamlit as st

from src.app.app_consts import AppSessionKeys
from src.app.components.inputs import select_gpu_type
from src.app.components.inputs import select_window_size
from src.utils.streamlit.helpers.component import StreamlitPage

st.set_page_config(page_title="SSM Analysis Results", page_icon="📊", layout="wide")
st.title("SSM Analysis")


class HomePage(StreamlitPage):
    def render(self):
        st.sidebar.success("Select a page above to explore different analyses.")
        st.session_state

        # Global variables
        st.session_state
        if st.button("Reset App"):
            st.session_state.clear()

        select_gpu_type()
        AppSessionKeys.variation.create_input_widget("Variation")
        select_window_size()


if __name__ == "__main__":
    HomePage().render()
