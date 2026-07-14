import streamlit as st
import requests

API_URL = "http://localhost:8000"

def upload_view():
    st.write("Upload a Chest X-Ray image to detect Pneumonia.")
    
    uploaded_file = st.file_uploader("Choose an X-Ray image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
        
        if st.button("Predict"):
            with st.spinner("Analyzing image..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                try:
                    res = requests.post(f"{API_URL}/upload/", files=files)
                    if res.status_code == 200:
                        st.success("Image uploaded successfully!")
                        # In a real implementation, call predict endpoint here
                        st.info("Prediction API call would happen here.")
                    else:
                        st.error("Upload failed.")
                except Exception as e:
                    st.error(f"Error connecting to backend: {e}")
