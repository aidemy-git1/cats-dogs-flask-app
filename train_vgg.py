import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# パラメータ（MobileNetと合わせる）
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 5

train_dir = 'data/train'
valid_dir = 'data/valid'

# ★水増しなし（比較用）
train_datagen = ImageDataGenerator(rescale=1./255)
valid_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=True
)

valid_generator = valid_datagen.flow_from_directory(
    valid_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

# ベースモデル：VGG16
base_model = VGG16(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)

# ★比較のため凍結（まずはこれでOK）
base_model.trainable = False

# モデル構築
model = models.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=EPOCHS
)

# 保存（わかりやすく）
model.save("vgg_noaug.keras")
print("モデル保存完了！: vgg_noaug.keras")

# 最終値を必ず表示（提出用）
print("BEST val_accuracy:", max(history.history["val_accuracy"]))
print("LAST val_accuracy:", history.history["val_accuracy"][-1])
print("LAST val_loss:", history.history["val_loss"][-1])

# グラフ（任意）
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='valid')
plt.legend()
plt.show()