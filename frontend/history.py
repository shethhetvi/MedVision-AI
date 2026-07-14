import streamlit as st

def history_view():
    st.header("Prediction History")
    st.write("Here you can see the log of previous predictions.")
    
    # In a real app, we'd fetch this from the database via API
    import pandas as pd
    dummy_data = pd.DataFrame({
        "Image ID": ["img_001", "img_002", "img_003"],
        "Prediction": ["Pneumonia", "Normal", "Pneumonia"],
        "Confidence": [0.95, 0.98, 0.88],
        "Timestamp": ["2026-07-14 10:00", "2026-07-14 10:05", "2026-07-14 10:15"]
    })
    st.dataframe(dummy_data)
