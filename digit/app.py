import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from streamlit_drawable_canvas import st_canvas

# -------------------
# 1. Load Trained Model
# -------------------
model = tf.keras.models.load_model("digit_model.h5")

# -------------------
# 2. Streamlit UI
# -------------------
st.set_page_config(page_title="Digit Recognition", layout="wide")
st.title("🧠 Neural Network Live Digit Recognition")

st.markdown("Draw a digit (0–9) in the box below:")

# -------------------
# 3. Drawing Canvas
# -------------------
canvas_result = st_canvas(
    fill_color="black",
    stroke_width=12,
    stroke_color="white",
    background_color="black",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas"
)

# -------------------
# 4. Prediction Logic
# -------------------
if canvas_result.image_data is not None:
    img = canvas_result.image_data

    if np.sum(img) > 0:  # Only if something is drawn
        # Convert to grayscale & resize using TensorFlow
        gray = img[:, :, 0]  # take single channel
        gray = tf.image.resize(gray[..., np.newaxis], (28, 28)).numpy()
        gray = gray.astype("float32") / 255.0
        gray = gray.reshape(1, 28, 28)

        # Predict
        preds = model.predict(gray)
        pred_class = np.argmax(preds)

        st.subheader(f"✅ Predicted Digit: **{pred_class}**")

        # -------------------
        # 5. Probability Bar Chart
        # -------------------
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.bar(range(10), preds[0])
        ax.set_xticks(range(10))
        ax.set_xlabel("Digit")
        ax.set_ylabel("Probability")
        ax.set_title("Prediction Confidence")
        st.pyplot(fig)

        # -------------------
        # 6. Visualize Activations
        # -------------------
        st.subheader("🔍 Layer Activations")

        # Create sub-model to get activations
        layer_outputs = [layer.output for layer in model.layers if "dense" in layer.name]
        activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)
        activations = activation_model.predict(gray)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.axis("off")
        ypos = 1.0

        # Plot each layer's activations
        for idx, act in enumerate(activations[:-1]):
            values = act[0]
            x = np.linspace(0.1, 0.9, len(values))
            ax.scatter(x, [ypos] * len(values), s=100, c=values, cmap="viridis", edgecolors="black")
            ax.text(0.0, ypos, f"Layer {idx+1}", ha="right")
            ypos -= 0.15

        # Final output layer
        output_act = activations[-1][0]
        x = np.linspace(0.1, 0.9, 10)
        ax.scatter(x, [ypos]*10, s=200, c=output_act, cmap="plasma", edgecolors="black")
        for i, d in enumerate(range(10)):
            ax.text(x[i], ypos-0.05, str(d), ha="center")

        st.pyplot(fig)

# -------------------
# 7. Clear Button
# -------------------
if st.button("🧹 Clear Canvas"):
    st.experimental_rerun()
