import sys
import importlib.util
import streamlit as st

st.write("Python version:", sys.version)
st.write("App location:", __file__)
st.write("joblib location:", importlib.util.find_spec("joblib"))

import joblib
