# MedVision AI
### Intelligent Medical Imaging Diagnosis & Explainable AI Platform

> AI-powered medical image analysis with explainable predictions for clinicians and researchers.

## Problem Statement
Medical image diagnosis often requires specialized expertise and can be time-consuming. MedVision AI assists clinicians by providing AI-based predictions along with visual explanations of what influenced the model's decision. It is intended as a decision-support tool, **not** as a replacement for medical professionals.

## Target Users
* Doctors
* Medical students
* Researchers
* Hospitals
* AI researchers

## Supported Diseases (Staged Release Plan)
We are building MedVision AI as a platform, rolling out capabilities over time.

### Version 1: Chest X-ray
* **Classes:** Normal vs Pneumonia
* **Why:** Large public datasets, straightforward preprocessing, and an excellent starting point for deployment.

### Future Versions
* **Version 2:** Brain Tumor (MRI)
* **Version 3:** Skin Cancer (Lesion Classification)
* **Version 4:** Malaria Detection (Blood Smear)

## Key Features

### 1. Image Upload & Preview
* Drag & Drop interface.
* Support for JPG, PNG, and JPEG.
* Image Preview displays resolution and file size.
* **Image Quality Check:** Warns users if the uploaded image is blurry or low resolution before inference.

### 2. AI Prediction & Confidence Score
* Clear display of predicted class (e.g., Pneumonia).
* Confidence meter using a progress bar (e.g., 97.4%).

### 3. Explainable AI (Grad-CAM)
* Visual explanations showing a heatmap of the highlighted infected region.
* **Original X-ray -> Grad-CAM -> Highlighted Image** to help clinicians trust the model.

### 4. Disease Information Panel
* Displays Symptoms, Causes, Risk Factors, Recommended Medical Tests, When to consult a doctor, Emergency warning signs, and Treatment overview.
* Clear disclaimer that it is **not personalized medical advice**.

