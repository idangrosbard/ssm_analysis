import json

import streamlit as st
from st_aggrid import GridOptionsBuilder, JsCode


def set_aagrid_apply_default_filters(
    grid_builder: GridOptionsBuilder, filter_defaults: dict[str, list], with_st_code: bool = False
):
    """
    Applies default filters to AG Grid using JavaScript.

    Args:
        grid_builder: The AG GridOptionsBuilder instance.
        filter_defaults: A dictionary where keys are column names and values are the default filter values.

    Notice you have to have allow_unsafe_jscode=True in the AgGrid call.
    """

    # Convert Python dictionary to JavaScript object format
    filter_model_js = json.dumps(
        {
            col: {
                "filterType": "set",
                "values": values,
            }
            for col, values in filter_defaults.items()
        }
    )
    code = f"""
    function onFirstDataRendered(params) {{
        params.api.setFilterModel({filter_model_js});
        params.api.onFilterChanged();
    }}
    """

    if with_st_code:
        st.code(filter_model_js)

    # JavaScript function to set the filter on first render
    onFirstDataRendered = JsCode(code)

    # Apply the generated JavaScript code to AG Grid
    grid_builder.configure_grid_options(onFirstDataRendered=onFirstDataRendered.js_code)
