# RAN ON KAGGLE

import os
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

MODEL_PATH = "/kaggle/working/models/genre_cnn_real_final.keras"
IMAGE_SIZE = (128, 128)
SR = 22050
SEGMENT_DURATION = 3.0
SAMPLES_PER_SEGMENT = int(SR * SEGMENT_DURATION)
EXPECTED_SLICES = 10

CLASS_NAMES = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock"
]
num_classes = len(CLASS_NAMES)

INPUT_SONG = "/kaggle/input/datasets/rogergucui/more-test-audio/mrclaps-this-heavy-metal-492569.mp3"


def process_audio_to_slices(audio_path):
    y, sr = librosa.load(audio_path, sr=SR, duration=30.0)
    slice_tensors = []
    
    for s in range(EXPECTED_SLICES):
        start_sample = s * SAMPLES_PER_SEGMENT
        end_sample = start_sample + SAMPLES_PER_SEGMENT
        y_chunk = y[start_sample:end_sample]
        
        if len(y_chunk) < SAMPLES_PER_SEGMENT:
            y_chunk = np.pad(y_chunk, (0, SAMPLES_PER_SEGMENT - len(y_chunk)), mode="constant")
            
        spectrogram = librosa.feature.melspectrogram(y=y_chunk, sr=sr, n_mels=128)
        spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
        
        if spectrogram_db.shape[1] > 128:
            spectrogram_db = spectrogram_db[:, :128]
        elif spectrogram_db.shape[1] < 128:
            spectrogram_db = np.pad(spectrogram_db, ((0, 0), (0, 128 - spectrogram_db.shape[1])), mode="edge")
            
        spectrogram_db = np.flipud(spectrogram_db)
        db_min, db_max = spectrogram_db.min(), spectrogram_db.max()
        if db_max - db_min > 0:
            pixel_matrix = 255.0 * (spectrogram_db - db_min) / (db_max - db_min + 1e-6)
        else:
            pixel_matrix = np.zeros_like(spectrogram_db)
            
        tensor = tf.convert_to_tensor(pixel_matrix, dtype=tf.float32)
        tensor = tf.expand_dims(tensor, axis=-1)
        tensor = tf.image.grayscale_to_rgb(tensor)
        slice_tensors.append(tensor)
        
    return tf.stack(slice_tensors)


def classify_song_on_kaggle(song_path):
    print(f"Slicing and parsing audio tracks: {song_path}...")
    try:
        input_batch = process_audio_to_slices(song_path)
        
        print("Extracting weights from the trained model...")
        custom_mapping = {"<lambda>": lambda x: x}
        raw_loaded_model = tf.keras.models.load_model(MODEL_PATH, compile=False, custom_objects=custom_mapping)
        
        print("Reconstructing clean network topology...")
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3), include_top=False, weights=None
        )
        
        clean_model = models.Sequential([
            layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
            layers.Lambda(lambda x: tf.keras.applications.mobilenet_v2.preprocess_input(x)),
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax")
        ])
        
        clean_model.set_weights(raw_loaded_model.get_weights())
        
        print("Running batch inferencing via sliced voting routine...")
        predictions = clean_model.predict(input_batch)
        
        averaged_predictions = np.mean(predictions, axis=0)
        
        print("\nAveraged Song Metrics:")
        for genre, pct in zip(CLASS_NAMES, averaged_predictions):
            print(f" - {genre}: {pct*100:.2f}%")
            
        best_class_idx = np.argmax(averaged_predictions)
        predicted_genre = CLASS_NAMES[best_class_idx]
        confidence = averaged_predictions[best_class_idx] * 100
        
        print("\n" + "=" * 40)
        print(f"VOTED GENRE MATCH : {predicted_genre.upper()}")
        print(f"AGGREGATED CONFIDENCE : {confidence:.2f}%")
        print("=" * 40)
        
    except Exception as e:
        print(f"An error occurred: {e}")

classify_song_on_kaggle(INPUT_SONG)
