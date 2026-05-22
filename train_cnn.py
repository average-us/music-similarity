# RUN ON KAGGLE

import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image_dataset_from_directory

# 1. Configuration (Paths match your working directories)
train_path = "/kaggle/input/datasets/rogergucui/split-sliced-spectrograms/train"
val_path = "/kaggle/input/datasets/rogergucui/split-sliced-spectrograms/val"
model_save_path = "/kaggle/working/models/genre_cnn_split.keras"

image_height = 128
image_width = 128
batch_size = 32

# 2. Load Datasets
train_dataset = image_dataset_from_directory(
    train_path, shuffle=True, image_size=(image_height, image_width), batch_size=batch_size, color_mode="rgb"
)
validation_dataset = image_dataset_from_directory(
    val_path, shuffle=False, image_size=(image_height, image_width), batch_size=batch_size, color_mode="rgb"
)

num_classes = len(train_dataset.class_names)
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)


# 3. Build Base Architecture
def build_base_network():
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(image_height, image_width, 3), include_top=False, weights="imagenet"
    )
    # Start with the base model frozen
    base_model.trainable = False

    model = models.Sequential([
        layers.Input(shape=(image_height, image_width, 3)),
        layers.Lambda(lambda x: tf.keras.applications.mobilenet_v2.preprocess_input(x)),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax")
    ])
    return model, base_model

model, base_model = build_base_network()

print("\nTraining...")
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Early stopping checks that validation loss continues to drop smoothly
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True
)

# Train for 15 epochs to stabilize the top dense layers
history_phase1 = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=15,
    callbacks=[early_stopping]
)

# 4. Save and Export
os.makedirs("/kaggle/working/models", exist_ok=True)
model.save(model_save_path)
print("\nModel successfully saved!")
