from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

model = tf.keras.models.load_model("mobilenet_aug.keras")

IMG_SIZE = 224

def predict_image(image_path):
    img = Image.open(image_path).resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]

    if prediction > 0.5:
        return "Dog", prediction
    else:
        return "Cat", 1 - prediction

@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
        file = request.files["file"]
        if file:
            filepath = os.path.join("static", file.filename)
            file.save(filepath)

            label, confidence = predict_image(filepath)
            result = f"{label} ({confidence*100:.2f}%)"

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
