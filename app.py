from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
import os
from keras.layers import TFSMLayer

app = Flask(__name__)

IMG_SIZE = 224
model = None  # ← ここ重要


def get_model():
    global model
    if model is None:
        # SavedModelを推論用Layerとして読み込む
        model = TFSMLayer("mobilenet_aug_savedmodel", call_endpoint="serve")
    return model


def predict_image(image_path):
    img = Image.open(image_path).resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = get_model()(img_array)          # Tensorが返る
    prediction = float(pred.numpy()[0][0]) # 数値に変換

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
    app.run()