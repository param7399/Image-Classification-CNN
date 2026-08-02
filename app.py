import os
import tempfile

import streamlit as st
from PIL import Image

from predict import predict_image

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Image Classification (CNN)",
    page_icon="🖼️",
    layout="wide"
)

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.title("📌 Project Information")

st.sidebar.markdown("""
### Image Classification using CNN

**Dataset**
- CIFAR-10

**Deep Learning**
- Convolutional Neural Network (CNN)

**Classes**
- ✈ Airplane
- 🚗 Automobile
- 🐦 Bird
- 🐱 Cat
- 🦌 Deer
- 🐶 Dog
- 🐸 Frog
- 🐴 Horse
- 🚢 Ship
- 🚚 Truck
""")

# -------------------------------
# Title
# -------------------------------
st.title("🖼️ Image Classification using CNN")
st.write("Classify images using a Deep Learning model trained on the CIFAR-10 dataset.")

st.divider()

# -------------------------------
# File Upload
# -------------------------------
uploaded_file = st.file_uploader(
    "📤 Upload an Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Uploaded Image")
        st.image(image, use_container_width=True)

    # Save uploaded image temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        image.save(tmp_file.name)
        temp_path = tmp_file.name

    with col2:

        st.subheader("Prediction")

        if st.button("🔍 Predict", use_container_width=True):

            with st.spinner("Predicting..."):

                predicted_class, confidence, top3 = predict_image(temp_path)

            confidence = float(confidence)

            st.success("Prediction Completed Successfully!")

            st.markdown("## 🤖 Prediction")

            st.info(predicted_class)

            st.metric(
                label="Confidence",
                value=f"{confidence:.2f}%"
            )

            # Main Prediction Progress
            st.progress(max(0.0, min(confidence / 100.0, 1.0)))

            st.divider()

            st.subheader("🏆 Top 3 Predictions")

            medals = ["🥇", "🥈", "🥉"]

            for medal, item in zip(medals, top3):

                item_confidence = float(item["confidence"])

                st.write(
                    f"{medal} **{item['class']}** — {item_confidence:.2f}%"
                )

                st.progress(max(0.0, min(item_confidence / 100.0, 1.0)))

    os.remove(temp_path)

st.divider()

st.caption("Developed using Python, TensorFlow, Keras and Streamlit.")