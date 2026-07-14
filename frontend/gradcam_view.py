import streamlit as st

def show_gradcam(image_path: str):
    st.subheader("Explainable AI: Grad-CAM")
    st.write("Heatmap highlighting the regions indicating Pneumonia.")
    # Dummy implementation - in a real scenario we'd call the backend Grad-CAM generator
    st.info("Grad-CAM visualization would appear here.")
