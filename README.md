# 🩺 AI-Based Skin Disease Classifier

This Streamlit web app allows users to upload a skin lesion image and get:
- Predicted skin disease
- Confidence score
- Clinical risk bar
- Grad-CAM visualization
- Basic treatment advice
- Option to search clinics in your city
- Ask skin-related health questions via live web search

## 🚀 Live App
👉 ( https://skindiseaseclassifier.streamlit.app/ )

## 🧠 Model
- Trained on HAM10000 dataset
- 7 skin disease classes
- CNN-based model (TensorFlow, Grad-CAM enabled)

## 🗂 Files
- `app.py` — Streamlit app
- `requirements.txt` — Python dependencies
- `.h5` model file hosted on Google Drive and downloaded on launch
