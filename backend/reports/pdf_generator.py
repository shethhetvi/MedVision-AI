from fpdf import FPDF
import os

def generate_pdf_report(patient_info: dict, prediction_result: dict, original_image_path: str = None, heatmap_path: str = None):
    pdf = FPDF()
    pdf.add_page()
    
    # Title
    pdf.set_font("Helvetica", 'B', 16)
    pdf.cell(200, 10, text="MedVision AI Clinical Report", new_x="LMARGIN", new_y="NEXT", align='C')
    pdf.ln(10)
    
    # Patient Info
    pdf.set_font("Helvetica", 'B', 12)
    pdf.cell(200, 10, text="Patient Information:", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", '', 12)
    pdf.cell(200, 10, text=f"ID: {patient_info.get('id', 'N/A')}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(200, 10, text=f"Name: {patient_info.get('name', 'Anonymous Patient')}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(10)
    
    # Prediction Result
    pdf.set_font("Helvetica", 'B', 12)
    pdf.cell(200, 10, text="Prediction Result:", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", '', 12)
    pdf.cell(200, 10, text=f"Diagnosis: {prediction_result.get('prediction', 'Unknown')}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(200, 10, text=f"Confidence: {prediction_result.get('confidence', 0.0) * 100:.2f}%", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(200, 10, text=f"Model Version: {prediction_result.get('model_version', 'N/A')}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(10)
    
    # Images side by side
    if original_image_path and os.path.exists(original_image_path):
        pdf.set_font("Helvetica", 'B', 12)
        pdf.cell(95, 10, text="Original X-Ray", new_x="RIGHT", new_y="TOP", align='C')
        if heatmap_path and os.path.exists(heatmap_path):
            pdf.cell(95, 10, text="Grad-CAM Heatmap", new_x="LMARGIN", new_y="NEXT", align='C')
        else:
            pdf.ln(10)
            
        y_before_img = pdf.get_y()
        
        try:
            pdf.image(original_image_path, x=10, y=y_before_img, w=85)
            if heatmap_path and os.path.exists(heatmap_path):
                pdf.image(heatmap_path, x=105, y=y_before_img, w=85)
        except Exception as e:
            pdf.set_font("Helvetica", 'I', 10)
            pdf.cell(200, 10, text=f"Error embedding images: {e}", new_x="LMARGIN", new_y="NEXT")

    report_dir = "data/reports"
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, f"report_{patient_info.get('id', 'dummy')}.pdf")
    
    pdf.output(report_path)
    
    return report_path
