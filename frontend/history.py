import streamlit as st
import requests
import pandas as pd

API_URL = "http://localhost:8000"

def history_view():
    st.subheader("Prediction History")
    
    try:
        res = requests.get(f"{API_URL}/api/logs")
        if res.status_code == 200:
            logs = res.json().get("logs", [])
            if not logs:
                st.info("No prediction history available yet.")
            else:
                # Convert to DataFrame for nice visualization
                df = pd.DataFrame(logs)
                
                # Format timestamp if it exists
                if "timestamp" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.strftime('%Y-%m-%d %H:%M:%S')
                    
                # Format confidence as percentage
                if "confidence" in df.columns:
                    df["confidence"] = df["confidence"].apply(lambda x: f"{x * 100:.1f}%")
                    
                # Rename columns for better readability
                df = df.rename(columns={
                    "id": "Log ID",
                    "image_id": "File ID",
                    "prediction": "Diagnosis",
                    "confidence": "Confidence",
                    "model_version": "Model",
                    "latency_ms": "Latency (ms)",
                    "timestamp": "Time"
                })
                
                # Reorder columns
                cols = ["Log ID", "Time", "Diagnosis", "Confidence", "Model", "File ID", "Latency (ms)"]
                df = df[[c for c in cols if c in df.columns]]
                
                st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.error("Failed to load history.")
    except Exception as e:
        st.error(f"Error connecting to backend: {e}")
