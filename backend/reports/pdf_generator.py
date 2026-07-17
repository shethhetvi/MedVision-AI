def generate_pdf_report(patient_info: dict, prediction_result: dict, heatmap_path: str):
    # This is a dummy implementation
    # A real implementation would use ReportLab or FPDF
    report_content = f"Patient: {patient_info.get('name', 'Unknown')}\n"
    report_content += f"Prediction: {prediction_result.get('prediction')}\n"
    report_content += f"Confidence: {prediction_result.get('confidence')}\n"
    
    report_path = f"reports/report_{patient_info.get('id', 'dummy')}.pdf"
    
    with open(report_path, "w") as f:
        f.write(report_content)
        
    return report_path
