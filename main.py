import os
import gdown
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import requests
from bs4 import BeautifulSoup
import tensorflow as tf
import matplotlib.cm as cm

# ======== Download Model from Google Drive if not present ========
MODEL_PATH = "skin_disease_model_balanced.h5"
DRIVE_FILE_ID = "14W-HiBATLcptID4yf0tNjDoOW7WTQgn_"  # <-- PUT YOUR GOOGLE DRIVE FILE ID HERE!

if not os.path.exists(MODEL_PATH):
    gdown.download(
        f"https://drive.google.com/uc?id={DRIVE_FILE_ID}",
        MODEL_PATH,
        quiet=False
    )

model = load_model(MODEL_PATH)

# Force model build for Grad-CAM compatibility
_ = model.predict(np.zeros((1, 112, 150, 3)))

# ======== Labels, Colors, Advice =========
class_names = [
    "Melanocytic nevi", "Melanoma", "Benign keratosis",
    "Basal cell carcinoma", "Actinic keratoses",
    "Vascular lesions", "Dermatofibroma"
]

severity_color = {
    "Melanocytic nevi": "#4CAF50", "Benign keratosis": "#4CAF50", "Dermatofibroma": "#4CAF50",
    "Vascular lesions": "#FFC107", "Actinic keratoses": "#FF9800",
    "Basal cell carcinoma": "#F44336", "Melanoma": "#B71C1C"
}

disease_advice = {
    "Melanocytic nevi": ["Use sunscreen daily.", "Monitor for changes.", "Annual dermatologist visit."],
    "Melanoma": ["Seek urgent medical attention.", "Avoid sun exposure.", "Biopsy may be needed."],
    "Benign keratosis": ["Apply moisturizer.", "Avoid scratching.", "Cryotherapy may help."],
    "Basal cell carcinoma": ["May require surgery.", "Avoid sun exposure.", "Consult dermatologist."],
    "Actinic keratoses": ["Use SPF 50+.", "Topical treatments may help.", "Cryotherapy is common."],
    "Vascular lesions": ["Laser treatment can help.", "Avoid trauma to area.", "Consult specialist."],
    "Dermatofibroma": ["Usually harmless.", "Monitor for growth.", "Remove if painful."]
}

# ======== Grad-CAM Function =========
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # Build model if needed for .input/.output attributes
    try:
        _ = model.output
    except Exception:
        _ = model.predict(np.zeros((1, 112, 150, 3)))
    grad_model = tf.keras.models.Model(
        [model.input],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def overlay_heatmap_on_image(img, heatmap):
    heatmap = np.uint8(255 * heatmap)
    heatmap = Image.fromarray(heatmap).resize(img.size)
    heatmap = np.array(heatmap)
    heatmap_color = cm.jet(heatmap / 255.0)[:, :, :3]
    heatmap_img = np.uint8(255 * heatmap_color)
    blended = Image.blend(img.convert("RGB"), Image.fromarray(heatmap_img), alpha=0.5)
    return blended

# ======== DuckDuckGo Clinic Search ========
def duckduckgo_search(query):
    url = f"https://html.duckduckgo.com/html/?q={query}"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    results = []
    for link in soup.find_all('a', class_='result__a', limit=3):
        title = link.get_text()
        href = link['href']
        results.append((title, href))
    return results

# ======== Q&A Web Fetch ========
def fetch_answer_from_web(query):
    url = f"https://html.duckduckgo.com/html/?q={query}"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    snippets = soup.find_all('a', class_='result__snippet', limit=3)
    return "\n\n".join([s.get_text() for s in snippets]) or "Sorry, I couldn't find an answer."

# ======== Streamlit UI =========
st.set_page_config(page_title="Skin Disease Detector", layout="centered")
st.title("🩺 AI Based Skin Disease Classifier")
st.write("Upload a skin lesion image to get a prediction, clinical risk level, treatment advice, and Grad-CAM visualization.")

uploaded_file = st.file_uploader("Upload a skin image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)
    image_resized = image.resize((150, 112))
    img_array = np.array(image_resized) / 255.0
    img_array = img_array.reshape(1, 112, 150, 3)

    # Basic validity check
    mean_val = np.mean(img_array)
    std_val = np.std(img_array)

    if mean_val < 0.2 or mean_val > 0.9 or std_val < 0.05:
        st.error("❌ This doesn't seem to be a valid skin image.")
    else:
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)
        confidence = float(np.max(prediction))
        confidence_percent = int(confidence * 100)
        disease = class_names[predicted_class]
        color = severity_color[disease]

        st.markdown("## 🧪 Prediction Results")
        if confidence < 0.65:
            st.warning(f"⚠️ Low confidence ({confidence_percent}%). Not certain.")
        else:
            st.success(f"✅ Prediction: {disease} ({confidence_percent}% confidence)")

        st.progress(confidence_percent)

        # Risk bar
        st.markdown("**Clinical Risk Level:**")
        st.markdown(f"<div style='height: 25px; width: 100%; background-color: {color}; border-radius: 5px;'></div>", unsafe_allow_html=True)
        st.markdown("""
        <p style='font-size: 13px; margin-top: 5px;'>
        🟢Low Risk | 🟡Medium | 🟠Elevated | 🔴High | 🟥Critical
        </p>
        """, unsafe_allow_html=True)

        if disease in ["Melanocytic nevi", "Benign keratosis", "Dermatofibroma"] and confidence_percent > 70:
            st.success("🎉 Your skin looks good. No serious concern detected.")

        st.markdown("---")
        st.markdown(f"### 💡 Basic Precautions & Treatments for {disease}:")
        for advice in disease_advice[disease]:
            st.markdown(f"- {advice}")

        # Grad-CAM
        if st.checkbox("Show Grad-CAM Explanation"):
            st.info("Visualizing model attention...")
            last_conv = next(
                (layer.name for layer in reversed(model.layers) if isinstance(layer, tf.keras.layers.Conv2D)),
                None
            )
            heatmap = make_gradcam_heatmap(img_array, model, last_conv)
            gradcam_img = overlay_heatmap_on_image(image, heatmap)

            col1, col2, col3 = st.columns(3)
            col1.image(image, caption="Original", use_container_width=True)
            col2.image(heatmap, caption="Grad-CAM", clamp=True, use_container_width=True)
            col3.image(gradcam_img, caption="Overlay", use_container_width=True)

        # Clinic Search
        st.markdown("---")
        st.subheader("🏥 Find Skin Clinics")
        city = st.text_input("Enter your city to search for nearby clinics:")
        if city:
            clinics = duckduckgo_search(f"skin care clinics in {city}")
            for title, href in clinics:
                st.markdown(f"- [{title}]({href})")

# Q&A Chatbot
st.markdown("---")
st.subheader("💬 Ask a Skin Health Question")
query = st.text_input("Ask about skin care (e.g. best cream for eczema):")
if query:
    answer = fetch_answer_from_web(query)
    st.success(f"**Answer:** {answer}")
