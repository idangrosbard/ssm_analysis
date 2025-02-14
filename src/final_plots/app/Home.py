import streamlit as st

st.set_page_config(page_title="SSM Analysis Results", page_icon="📊", layout="wide")

st.title("SSM Analysis Results 📊")

st.markdown("""
Welcome to the SSM Analysis Results Dashboard! 
This application provides visualizations and analysis of various experiments conducted on State Space Models.

### Available Pages:
- **📋 Results Bank**: View detailed information about all experiments and their results
- **📊 Data Requirements**: Manage data requirements and overrides for experiments
""")

st.sidebar.success("Select a page above to explore different analyses.")
