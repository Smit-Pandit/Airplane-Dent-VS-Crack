import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.applications.vgg16 import preprocess_input
import joblib

IMG_SIZE = (224, 224)
MODEL_PATH = "model/MODEL_api.pkl"

DENT_THRESHOLD = 0.80
CRACK_THRESHOLD = 0.20

st.set_page_config(
    page_title="✈️ Airplane Dent vs Crack Detection",
    page_icon="✈️",
    layout="centered"
)
model = joblib.load(MODEL_PATH)


st.title("✈️ Airplane Dent vs Crack Detection")

st.markdown("""
Upload an **aircraft surface image** to classify whether it contains a **dent** or a **crack**.

> **Note:** This model is trained only on aircraft surface images. Predictions for unrelated images may be unreliable.
""")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    img = image.resize(IMG_SIZE)
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    with st.spinner("Analyzing image..."):
        prediction = model.predict(img, verbose=0)

    probability = float(prediction[0][0])

    dent_probability = probability
    crack_probability = 1 - probability

    st.subheader("Prediction")

    if probability >= DENT_THRESHOLD:

        st.success("🟢 **Dent Detected**")

        st.metric(
            "Confidence",
            f"{dent_probability*100:.2f}%"
        )

        st.progress(dent_probability)

    elif probability <= CRACK_THRESHOLD:

        st.error("🔴 **Crack Detected**")

        st.metric(
            "Confidence",
            f"{crack_probability*100:.2f}%"
        )

        st.progress(crack_probability)

    else:

        st.warning(
            """
⚠️ Unable to confidently classify this image.

Possible reasons:
- The image is not an aircraft surface.
- The image quality is poor.
- The damage is ambiguous.
- Confidence is below the acceptance threshold.
"""
        )

        st.metric(
            "Highest Confidence",
            f"{max(dent_probability, crack_probability)*100:.2f}%"
        )

    st.divider()

    st.subheader("Prediction Scores")

    st.write(f"🟢 Dent : **{dent_probability*100:.2f}%**")

    st.progress(dent_probability)

    st.write(f"🔴 Crack : **{crack_probability*100:.2f}%**")

    st.progress(crack_probability)

    st.divider()

    st.caption(
        "This AI model is intended for educational and research purposes only."
    )