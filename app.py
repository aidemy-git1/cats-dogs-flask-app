from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

MODEL_PATH = "mobilenet_aug.keras"   # ←ここ重要（.kerasにする）
IMG_SIZE = 224
model = None

def get_model():
    global model
    if model is None:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return model

def predict_image(image_path):
    img = Image.open(image_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = get_model().predict(img_array, verbose=0)[0][0]
    if pred > 0.5:
        return "Dog", float(pred)
    else:
        return "Cat", float(1 - pred)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None

    if request.method == "POST":
        file = request.files.get("file")
        if file and file.filename:
            os.makedirs("static", exist_ok=True)

            filename = secure_filename(file.filename)
            filepath = os.path.join("static", filename)
            file.save(filepath)

            try:
                label, confidence = predict_image(filepath)
                result = f"{label} ({confidence*100:.2f}%)"
            except Exception as e:
                error = f"推論でエラー: {e}"
                print(error)

    return render_template("index.html", result=result, error=error)

if __name__ == "__main__":
    app.run(debug=True)
