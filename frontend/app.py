import streamlit as st

st.set_page_config(page_title="MedVision AI", page_icon="🩺", layout="wide")

st.title("🩺 MedVision AI")
st.markdown("### Production-grade Medical Imaging Platform")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Upload & Predict", "Prediction History", "Dataset Explorer"])

if page == "Upload & Predict":
    from frontend.components import upload_view
    upload_view()
elif page == "Prediction History":
    from frontend.history import history_view
    history_view()
else:
    st.info("Dataset Explorer coming in v1.1")
