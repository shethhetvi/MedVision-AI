import streamlit as st
import requests

API_URL = "http://localhost:8000"

def upload_view():
    st.write("Upload a Chest X-Ray image to detect Pneumonia.")
    
    uploaded_file = st.file_uploader("Choose an X-Ray image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        
        if st.button("Predict", type="primary"):
            with st.spinner("Analyzing image..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                try:
                    # Upload
                    res = requests.post(f"{API_URL}/api/upload/", files=files)
                    if res.status_code == 200:
                        upload_data = res.json()
                        file_id = upload_data["file_id"]
                        
                        # Predict
                        predict_res = requests.post(f"{API_URL}/api/predict/{file_id}")
                        if predict_res.status_code == 200:
                            predict_data = predict_res.json()["result"]
                            
                            st.success("Prediction complete!")
                            
                            # Display Metrics
                            col1, col2, col3 = st.columns(3)
                            col1.metric("Diagnosis", predict_data["prediction"])
                            col2.metric("Confidence", f"{predict_data['confidence'] * 100:.2f}%")
                            col3.metric("Model Version", predict_data["model_version"])
                            
                            st.markdown("---")
                            
                            # Display Images side by side
                            img_col1, img_col2 = st.columns(2)
                            
                            # Original Image
                            img_col1.subheader("Original X-Ray")
                            original_image_url = f"{API_URL}/static/uploads/{file_id}_{uploaded_file.name}"
                            img_col1.image(original_image_url, use_column_width=True)
                            
                            # Heatmap
                            heatmap_path = predict_data.get("heatmap_path")
                            if heatmap_path:
                                img_col2.subheader("Grad-CAM Heatmap")
                                # Convert local path data/uploads/... to static url
                                heatmap_filename = heatmap_path.split("/")[-1]
                                heatmap_url = f"{API_URL}/static/uploads/{heatmap_filename}"
                                img_col2.image(heatmap_url, use_column_width=True)
                            else:
                                img_col2.info("No heatmap generated for this model version.")
                            
                            # PDF Report Download
                            report_url = f"{API_URL}/api/report/{file_id}"
                            st.markdown(f"[Download Clinical Report (PDF)]({report_url})")
                        else:
                            st.error(f"Prediction failed: {predict_res.text}")
                    else:
                        st.error(f"Upload failed: {res.text}")
                except Exception as e:
                    st.error(f"Error connecting to backend: {e}")
