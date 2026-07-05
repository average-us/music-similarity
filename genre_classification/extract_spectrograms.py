import os
import librosa
import numpy as np
from PIL import Image

# Configuration
source_dataset_path = "data/genres_original"
output_dir = "sliced_spectrograms"

SR = 22050
SEGMENT_DURATION = 3.0  # 3-second slices
SAMPLES_PER_SEGMENT = int(SR * SEGMENT_DURATION)  # 66,150 samples per slice
EXPECTED_SLICES = 10  # 30 seconds / 3 seconds = 10 slices per track

for genre in os.listdir(source_dataset_path):
    genre_path = os.path.join(source_dataset_path, genre)
    if not os.path.isdir(genre_path):
        continue
    print(f"Slicing genre: {genre}")

    for file_name in os.listdir(genre_path):
        if not file_name.endswith(".wav"):
            continue
        file_path = os.path.join(genre_path, file_name)

        try:
            # Load the full 30 seconds of audio
            y, sr = librosa.load(file_path, sr=SR, duration=30.0)

            # Loop through and cut 10 separate slices
            for s in range(EXPECTED_SLICES):
                start_sample = s * SAMPLES_PER_SEGMENT
                end_sample = start_sample + SAMPLES_PER_SEGMENT

                # Extract the 3-second audio array chunk
                y_chunk = y[start_sample:end_sample]

                # Ensure exact sample count (pad with silence if short at the absolute end)
                if len(y_chunk) < SAMPLES_PER_SEGMENT:
                    y_chunk = np.pad(
                        y_chunk,
                        (0, SAMPLES_PER_SEGMENT - len(y_chunk)),
                        mode="constant",
                    )

                # Generate numerical Mel-spectrogram (Fixed 128 height bins)
                spectrogram = librosa.feature.melspectrogram(
                    y=y_chunk, sr=sr, n_mels=128
                )
                spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)

                # Crop or pad the time axis to guarantee exactly 128 bins wide
                if spectrogram_db.shape[1] > 128:
                    spectrogram_db = spectrogram_db[:, :128]
                elif spectrogram_db.shape[1] < 128:
                    spectrogram_db = np.pad(
                        spectrogram_db,
                        ((0, 0), (0, 128 - spectrogram_db.shape[1])),
                        mode="edge",
                    )

                # Flip frequencies upright and scale strictly to 0-255 pixel bytes
                spectrogram_db = np.flipud(spectrogram_db)
                db_min, db_max = spectrogram_db.min(), spectrogram_db.max()
                if db_max - db_min > 0:
                    pixel_matrix = (
                        255.0
                        * (spectrogram_db - db_min)
                        / (db_max - db_min)
                    )
                else:
                    pixel_matrix = np.zeros_like(spectrogram_db)

                # Save directly as an uncompressed 128x128 grayscale image file
                img = Image.fromarray(pixel_matrix.astype(np.uint8), mode="L")

                output_folder = os.path.join(output_dir, genre)
                os.makedirs(output_folder, exist_ok=True)

                # Unique naming structure: e.g., rock.00000_slice0.png, rock.00000_slice1.png
                output_file = file_name.replace(".wav", f"_slice{s}.png")
                img.save(os.path.join(output_folder, output_file))

        except Exception as e:
            print(f"Skipping corrupt track {file_name}: {e}")

print("\nSuccessfully expanded your dataset into 3-second slices!")
