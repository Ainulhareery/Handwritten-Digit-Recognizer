
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps


# ---------------------------------
# 1. Set up project imports
# ---------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import preprocess_image
from src.inference import predict_digit


# ---------------------------------
# 2. Streamlit configuration
# ---------------------------------
st.set_page_config(
    page_title="Handwritten Digit Recognizer",
    page_icon="🔢",
    layout="centered",
)

st.title("Handwritten Digit Recognizer")

st.write(
    "Upload an image containing one handwritten digit "
    "from 0 to 9, and the trained CNN will predict it."
)

# Temporary verification
st.caption(
    f"Preprocessing module: {preprocess_image.__module__}"
)


# ---------------------------------
# 3. Upload image
# ---------------------------------
uploaded_file = st.file_uploader(
    "Upload a digit image",
    type=["png", "jpg", "jpeg"],
    help="Upload a clear image containing one digit.",
)


# ---------------------------------
# 4. Process image and predict
# ---------------------------------
if uploaded_file is not None:
    try:
        
        # Load image and correct camera orientation metadata
        original_image = ImageOps.exif_transpose(
            Image.open(uploaded_file)
        ).convert("RGB")

        # Let the user correct the image orientation
        rotation = st.selectbox(
            "Rotate image to make the digit upright",
            options=[
                "No rotation",
                "90° clockwise",
                "90° counterclockwise",
                "180°",
            ],
        )

        rotation_angles = {
            "No rotation": 0,
            "90° clockwise": -90,
            "90° counterclockwise": 90,
            "180°": 180,
        }

        original_image = original_image.rotate(
            rotation_angles[rotation],
            expand=True,
        )

        # Preprocess the correctly oriented image
        processed_image, model_input = preprocess_image(
            original_image
        )


        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original image")
            st.image(
                original_image,
                width="stretch",
            )

        with col2:
            st.subheader("Processed image")
            st.image(
                processed_image,
                width="stretch",
            )

        predicted_digit, scores = predict_digit(
            model_input
        )

        highest_score = float(scores[predicted_digit])

        st.divider()
        st.subheader("Prediction")

        st.metric(
            label="Predicted digit",
            value=str(predicted_digit),
        )

        st.write(
            f"Highest model score: **{highest_score:.2%}**"
        )

        st.caption(
            "This score is the model's softmax output, "
            "not a calibrated probability or a guarantee "
            "that the prediction is correct."
        )

        results = pd.DataFrame(
            {
                "Digit": list(range(10)),
                "Model score": np.asarray(scores),
            }
        ).set_index("Digit")

        st.subheader("Scores for all digits")
        st.bar_chart(results)

    except (ValueError, OSError, TypeError) as error:
        st.error(f"Could not process this image: {error}")

    except Exception as error:
        st.error(
            "Prediction failed. Check the model and "
            "application configuration."
        )
        st.exception(error)

else:
    st.info(
        "Upload an image above to begin. "
        "The application recognizes one digit at a time."
    )


# ---------------------------------
# 5. Model information
# ---------------------------------
st.divider()

st.caption(
    "Model benchmark: 99.57% accuracy on the MNIST "
    "test set. Real-world image performance may differ."
)
