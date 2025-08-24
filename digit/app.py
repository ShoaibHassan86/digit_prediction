import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.utils import to_categorical
from streamlit_drawable_canvas import st_canvas

# -------------------
# 1. Load/Train Model
# -------------------
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# ✅ Functional API
inputs = Input(shape=(28, 28))
x = Flatten()(inputs)
x = Dense(25, activation="relu")(x)
x = Dense(25, activation="relu")(x)
outputs = Dense(10, activation="softmax")(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(x_train, y_train, epochs=1, verbose=0)

# ✅ Activation model (layer-wise outputs)
layer_outputs = [layer.output for layer in model.layers if isinstance(layer, Dense) or isinstance(layer, Flatten)]
activation_model = Model(inputs=model.input, outputs=layer_outputs)

# -------------------
# 2. Streamlit UI
# -------------------
st.title("🧠 Neural Network Live Digit Recognition")
st.write("Draw a digit (0–9) and watch how it flows through the layers.")

canvas_result = st_canvas(
    fill_color="black",
    stroke_width=12,
    stroke_color="white",
    background_color="black",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)

# -------------------
# 3. Helper for connections
# -------------------
def draw_layer_connections(ax, activations, weights, ypos, next_ypos):
    """Draw nodes & connections between two layers"""
    nodes = activations.shape[-1]
    next_nodes = weights.shape[1]

    x = np.linspace(0.1, 0.9, nodes)
    next_x = np.linspace(0.1, 0.9, next_nodes)

    # connections
    for i in range(nodes):
        for j in range(next_nodes):
            w = weights[i, j]
            color = "purple" if w > 0 else "green"
            alpha = min(1.0, abs(w) / np.max(np.abs(weights)))
            ax.plot([x[i], next_x[j]], [ypos, next_ypos], color=color, alpha=alpha, linewidth=0.5)

    # nodes
    ax.scatter(x, [ypos]*nodes, s=100, c=activations, cmap="viridis", edgecolors="black")
    return next_x

# -------------------
# 4. Run prediction if drawn
# -------------------
if canvas_result.image_data is not None:
    img = canvas_result.image_data[:, :, 0]
    img = cv2.resize(img, (28, 28)) / 255.0
    img = np.expand_dims(img, axis=(0,))

    activations = activation_model.predict(img)
    prediction = np.argmax(activations[-1])

    st.subheader(f"🎯 Predicted Digit: {prediction}")

    # -------------------
    # 5. Visualization
    # -------------------
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.axis("off")

    ypos = 0.9

    # Flatten layer (first activation)
    flatten_act = activations[0][0]
    ax.scatter(np.linspace(0.1, 0.9, len(flatten_act)), [ypos]*len(flatten_act),
               s=50, c=flatten_act, cmap="gray", edgecolors="black")

    prev_act = flatten_act
    prev_ypos = ypos

    # Loop over Dense layers and draw connections
    dense_layer_index = 0  # index inside model.layers to fetch weights

    for act in activations[1:]:  # skip flatten layer
        # Find the matching Dense layer in the model
        while not isinstance(model.layers[dense_layer_index], Dense):
            dense_layer_index += 1
        dense_layer = model.layers[dense_layer_index]
        weights, bias = dense_layer.get_weights()

        ypos -= 0.25
        draw_layer_connections(ax, act[0], weights, prev_ypos, ypos)

        prev_ypos = ypos
        prev_act = act[0]
        dense_layer_index += 1

    # Output layer (last one)
    output_act = activations[-1][0]
    ypos -= 0.3
    x = np.linspace(0.1, 0.9, 10)
    ax.scatter(x, [ypos]*10, s=200, c=output_act, cmap="plasma", edgecolors="black")
    for i, d in enumerate(range(10)):
        ax.text(x[i], ypos-0.05, str(d), ha="center")

    st.pyplot(fig)
