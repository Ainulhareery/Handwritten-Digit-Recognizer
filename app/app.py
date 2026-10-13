
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas


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

# Allow user to draw a digit on a canvas
st.subheader("Draw a Number")

st.write(
    "Use your mouse or finger to draw any number "
    "from 0 to 9 in the white box below."
)


def preprocess_drawing(image_data):
    """Convert a canvas drawing to a 28x28 MNIST input."""

    pixels = np.asarray(image_data)

    if pixels.ndim != 3 or pixels.shape[2] not in (3, 4):
        raise ValueError("Invalid drawing image.")

    pixels = pixels.astype(np.float32)

    if not np.isfinite(pixels).all():
        raise ValueError("Invalid pixel values.")

    # Support both 0-1 and 0-255 canvas data
    if pixels.max() <= 1.0:
        pixels *= 255.0

    pixels = np.clip(pixels, 0, 255)
    rgb = pixels[:, :, :3]

    # Handle transparent backgrounds without PIL compositing
    if pixels.shape[2] == 4:
        alpha = pixels[:, :, 3:4] / 255.0
        rgb = rgb * alpha + 255.0 * (1.0 - alpha)

    # Convert dark strokes to bright strokes
    gray = rgb.mean(axis=2)
    ink = np.clip(255.0 - gray, 0, 255).astype(np.uint8)

    ink[ink < 35] = 0

    digit = Image.fromarray(ink)

    bbox = digit.getbbox()

    if bbox is None:
        raise ValueError("Please draw a digit first.")

    # Crop and resize without distorting the shape
    digit = digit.crop(bbox)
    digit.thumbnail((20, 20), Image.Resampling.LANCZOS)

    # Center on a 28x28 black background
    processed = Image.new("L", (28, 28), 0)

    processed.paste(
        digit,
        (
            (28 - digit.width) // 2,
            (28 - digit.height) // 2,
        ),
    )

    array = np.asarray(processed, dtype=np.float32) / 255.0
    model_input = array.reshape(1, 28, 28, 1)

    return processed, model_input


if "Draw a Number"  not in st.session_state:
    st.session_state.canvas_key = 0

# Display the drawing canvas
canvas = st_canvas(
    fill_color="rgba(255, 255, 255, 0)",
    stroke_width=15,
    stroke_color="#000000",
    background_color="#FFFFFF",
    update_streamlit=True,
    return_image_data=True,
    width=300,
    height=300,
    drawing_mode="freedraw",
    key=f"canvas_{st.session_state.canvas_key}",
)

# Recognize the digit when the user clicks the "Predict" button
if st.button(
     "Recognize My Drawing",
    type="primary",
    icon="✨",
    key="draw_predict",
):
    try:
        if canvas.image_data is None:
            raise ValueError(
                "Please draw a digit on the canvas."
            )

        else:
            final_image, model_input = preprocess_drawing(
                canvas.image_data
            )

            predicted_digit, scores = predict_digit(
                model_input
            )

            st.success(
                f"The model predicts your drawing as: **{predicted_digit}**"
            )

            st.image(
                final_image,
                caption="Processed drawing",
                width=100,
            )

            st.write(
                f"Highest model score: **{float(scores[predicted_digit]):.2%}**"
            )

            st.caption(
                "This score is the model's softmax output, "
                "not a calibrated probability or a guarantee "
                "that the prediction is correct."
            )

    except ValueError as error:
        st.warning(
            str(error)
        )

    except Exception as error:
        st.error(
            "Could not process the drawing. "
            "Please try again."
        )
        st.exception(error)

# Clear the canvas when the user clicks the "Clear" button
if st.button(
    "Clear Drawing",
    type="secondary",
    icon="🧽",
    key="clear_drawing",
):
    st.session_state.canvas_key += 1
    st.rerun()

st.divider()
st.subheader("Or Upload an Image")


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
