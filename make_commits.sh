#!/bin/bash

# Ensure we are in the correct directory
cd /Users/HetviSheth/Library/CloudStorage/OneDrive-NavrachanaUniversity/MedVision-AI

# Array of files to commit and their corresponding messages
files=(
    "backend/main.py"
    "database/models.py"
    "database/db.py"
    "backend/predict.py"
    "backend/routers/upload.py"
    "backend/routers/predict.py"
    "frontend/app.py"
    "frontend/components.py"
    "frontend/gradcam_view.py"
    "frontend/disease_info.py"
    "reports/generator.py"
    "frontend/history.py"
    "docker/Dockerfile.backend"
    "docker/Dockerfile.frontend"
    "docker-compose.yml"
)

messages=(
    "1. Set up initial FastAPI backend files"
    "2. Create PostgreSQL database models"
    "3. Set up database schemas and engine"
    "4. Create dummy model prediction script"
    "5. Implement Image Upload & Validation"
    "6. Implement AI Prediction endpoint"
    "7. Set up initial Streamlit frontend"
    "8. Implement image upload UI"
    "9. Implement Grad-CAM view in UI"
    "10. Add Disease Information Panel"
    "11. Implement PDF Report Generation"
    "12. Add Prediction History view"
    "13. Create Dockerfile for backend"
    "14. Create Dockerfile for frontend"
    "15. Update docker-compose.yml and final setup"
)

for i in "${!files[@]}"; do
    git add "${files[$i]}"
    git commit -m "${messages[$i]}"
done

echo "Successfully made 15 commits!"
