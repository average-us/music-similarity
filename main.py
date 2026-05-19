import os
import librosa
import numpy

dataset_path = "data/genres_original"

features = []
labels = []

for genre in os.listdir(dataset_path):

    genre_path = os.path.join(dataset_path, genre)

    if not os.path.isdir(genre_path):
        continue

    print(f"currently processing {genre}")

    for file_name in os.listdir(genre_path):

        file_path = os.path.join(genre_path, file_name)

        try:
            y, sr = librosa.load(file_path)

            mfccs = librosa.feature.mfcc(
                y=y,
                sr=sr,
                n_mfcc=13
            )

            mfccs_scaled = numpy.mean(mfccs.T, axis=0)

            features.append(mfccs_scaled)

            labels.append(genre)

        except Exception as e:
            print(f"error with {file_path}")
            print(e)

print("done")
print("number of songs processed:", len(features))