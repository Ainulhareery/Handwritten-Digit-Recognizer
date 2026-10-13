import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

from PIL import Image, ImageOps


# -----------------------------
# 1. Application configuration
# -----------------------------
st.set_page_config(
    page_title="Handwritten Digit Recognizer",
    page_icon="🔢",
    layout="centered",
)

st.title("Handwritten Digit Recognizer")

st.caption(
    "Preprocessing version: "
    + (
        "UPDATED"
        if "dark_tail" in inspect.getsource(preprocess_image)
        else "OLD"
    )
)

st.write(
    "Upload an image containing a single handwritten digit "
    "from 0 to 9, and the trained CNN will predict the digit."
)

MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "final_digit_model.keras"
)


# -----------------------------
# 2. Load the trained model
# -----------------------------
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
    )


# -----------------------------
# 3. Preprocess the image
# -----------------------------
def preprocess_image(image):
    # Convert to grayscale and improve contrast
    image = ImageOps.grayscale(image)
    image = ImageOps.autocontrast(image)

    # MNIST uses bright digits on a dark background.
    # Invert typical dark digits on a white background.
    if np.asarray(image).mean() > 127:
        image = ImageOps.invert(image)

    # Remove background noise and locate the digit
    image = image.point(
        lambda pixel: 255 if pixel > 40 else 0
    )

    bbox = image.getbbox()

    if bbox is None:
        raise ValueError(
            "No visible digit was detected. "
            "Upload a clearer image."
        )

    image = image.crop(bbox)

    # Add padding and place the digit on a square canvas
    width, height = image.size
    side = max(width, height) + 8

    canvas = Image.new("L", (side, side), 0)
    canvas.paste(
        image,
        ((side - width) // 2, (side - height) // 2),
    )

    # MNIST-style images use a 28x28 canvas.
    # Resize the digit to fit within a 20x20 area.
    canvas.thumbnail((20, 20), Image.Resampling.LANCZOS)

    final_image = Image.new("L", (28, 28), 0)
    final_image.paste(
        canvas,
        (
            (28 - canvas.width) // 2,
            (28 - canvas.height) // 2,
        ),
    )

    # Normalize pixels and add batch/channel dimensions
    array = np.asarray(
        final_image,
        dtype=np.float32,
    ) / 255.0

    array = array.reshape(1, 28, 28, 1)

    return final_image, array


# -----------------------------
# 4. Upload an image
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload a digit image",
    type=["png", "jpg", "jpeg"],
    help="Use a clear image containing one digit only.",
)


# -----------------------------
# 5. Predict the digit
# -----------------------------
if uploaded_file is not None:
    try:
        original_image = Image.open(uploaded_file)

        processed_image, input_array = preprocess_image(
            original_image
        )

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original image")
            st.image(
                original_image,
                use_container_width=True,
            )

        with col2:
            st.subheader("Processed image")
            st.image(
                processed_image,
                use_container_width=True,
            )

        model = load_model()

        scores = model.predict(
            input_array,
            verbose=0,
        )[0]

        predicted_digit = int(np.argmax(scores))
        top_score = float(scores[predicted_digit])

        st.divider()
        st.subheader("Prediction")

        st.metric(
            label="Predicted digit",
            value=str(predicted_digit),
        )

        st.write(
            f"Highest model score: **{top_score:.2%}**"
        )

        st.caption(
            "This score is the model's softmax output, "
            "not a calibrated probability or a guarantee "
            "that the prediction is correct."
        )

        results = pd.DataFrame(
            {
                "Digit": list(range(10)),
                "Model score": scores,
            }
        ).set_index("Digit")

        st.subheader("Scores for all digits")
        st.bar_chart(results)

    except (ValueError, OSError) as error:
        st.error(f"Could not process this image: {error}")

    except Exception as error:
        st.error(
            "The prediction could not be completed. "
            "Check the model file and application "
            "dependencies in the deployment environment."
        )
        st.exception(error)

else:
    st.info(
        "Upload an image above to begin. "
        "The application recognizes one digit at a time."
    )

st.divider()
st.caption(
    "Model benchmark: 99.57% accuracy on the MNIST test set. "
    "Real-world image performance may differ."
)
