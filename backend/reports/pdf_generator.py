from fpdf import FPDF
import os

def generate_pdf_report(patient_info: dict, prediction_result: dict, original_image_path: str = None, heatmap_path: str = None):
    pdf = FPDF()
    pdf.add_page()
    
    # Title
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="MedVision AI Clinical Report", ln=True, align='C')
    pdf.ln(10)
    
    # Patient Info
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="Patient Information:", ln=True)
    pdf.set_font("Arial", '', 12)
    pdf.cell(200, 10, txt=f"ID: {patient_info.get('id', 'N/A')}", ln=True)
    pdf.cell(200, 10, txt=f"Name: {patient_info.get('name', 'Anonymous Patient')}", ln=True)
    pdf.ln(10)
    
    # Prediction Result
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="Prediction Result:", ln=True)
    pdf.set_font("Arial", '', 12)
    pdf.cell(200, 10, txt=f"Diagnosis: {prediction_result.get('prediction', 'Unknown')}", ln=True)
    pdf.cell(200, 10, txt=f"Confidence: {prediction_result.get('confidence', 0.0) * 100:.2f}%", ln=True)
    pdf.cell(200, 10, txt=f"Model Version: {prediction_result.get('model_version', 'N/A')}", ln=True)
    pdf.ln(10)
    
    # Images side by side
    if original_image_path and os.path.exists(original_image_path):
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(95, 10, txt="Original X-Ray", ln=False, align='C')
        if heatmap_path and os.path.exists(heatmap_path):
            pdf.cell(95, 10, txt="Grad-CAM Heatmap", ln=True, align='C')
        else:
            pdf.ln(10)
            
        y_before_img = pdf.get_y()
        
        try:
            pdf.image(original_image_path, x=10, y=y_before_img, w=85)
            if heatmap_path and os.path.exists(heatmap_path):
                pdf.image(heatmap_path, x=105, y=y_before_img, w=85)
        except Exception as e:
            pdf.set_font("Arial", 'I', 10)
            pdf.cell(200, 10, txt=f"Error embedding images: {e}", ln=True)

    report_dir = "data/reports"
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, f"report_{patient_info.get('id', 'dummy')}.pdf")
    
    pdf.output(report_path)
    
    return report_path
