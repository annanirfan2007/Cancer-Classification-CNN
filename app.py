
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ==================================
# PAGE CONFIGURATION
# ==================================

st.set_page_config(
    page_title="Cancer Image Classification",
    page_icon="🔬",
    layout="wide"
)

# ==================================
# LOAD CNN MODEL
# ==================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "cancer_cnn_model.keras"
    )

model = load_model()

# Class order from Kaggle training
class_names = [
    'benign',
    'malignant',
    'no disease'
]

# ==================================
# HEADER
# ==================================

st.title("🔬 Cancer Image Classification")
st.subheader("CNN Based Image Classification System")

st.write(
    "Upload an image to get a prediction "
    "from the trained CNN model."
)

st.divider()

# ==================================
# IMAGE UPLOAD
# ==================================

uploaded_file = st.file_uploader(
    "Upload Skin Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded Image")
        st.image(
            image,
            caption="Input Image",
            use_container_width=True
        )

    # ==================================
    # IMAGE PREPROCESSING
    # ==================================

    resized_image = image.resize((128, 128))

    image_array = np.array(
        resized_image,
        dtype=np.float32
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # ==================================
    # MODEL PREDICTION
    # ==================================

    if st.button("Predict Image", type="primary"):

        with st.spinner("Analyzing Image..."):

            predictions = model.predict(
                image_array,
                verbose=0
            )

            predicted_index = np.argmax(
                predictions[0]
            )

            predicted_class = class_names[
                predicted_index
            ]

            confidence = float(
                predictions[0][predicted_index]
            ) * 100

        # ==================================
        # DISPLAY RESULTS
        # ==================================

        with col2:

            st.subheader("Prediction Result")

            st.success(
                f"Predicted Class: {predicted_class.upper()}"
            )

            st.metric(
                "Model Confidence",
                f"{confidence:.2f}%"
            )

        # ==================================
        # ALL CLASS PROBABILITIES
        # ==================================

        st.divider()

        st.subheader("Class Prediction Probabilities")

        for i, class_name in enumerate(class_names):

            probability = float(
                predictions[0][i]
            ) * 100

            st.write(
                f"**{class_name.upper()}**"
            )

            st.progress(
                probability / 100
            )

            st.write(
                f"{probability:.2f}%"
            )

        # ==================================
        # DISCLAIMER
        # ==================================

        st.warning(
            "This application is an experimental "
            "machine learning project. It is not "
            "a medical diagnosis tool. Please consult "
            "a qualified dermatologist for medical advice."
        )

# ==================================
# FOOTER
# ==================================

st.divider()

st.caption(
    "Developed using Python, TensorFlow, CNN and Streamlit"
)
