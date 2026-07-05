import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

dataset_path = "/kaggle/input/datasets/andradaolteanu/gtzan-dataset-music-genre-classification/Data/genres_original"

features = []
labels = []


def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=30)
    y, _ = librosa.effects.trim(y)

    segment_duration = 5
    segment_length = sr * segment_duration

    all_features = []

    for start in range(0, len(y), segment_length):

        segment = y[start:start + segment_length]

        if len(segment) < segment_length:
            continue

        features = []

        mfccs = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=40)
        features.extend(np.mean(mfccs.T, axis=0))
        features.extend(np.std(mfccs.T, axis=0))

        spectral_centroids = librosa.feature.spectral_centroid(y=segment, sr=sr)
        features.append(np.mean(spectral_centroids))
        features.append(np.std(spectral_centroids))

        zero_crossing_rate = librosa.feature.zero_crossing_rate(segment)
        features.append(np.mean(zero_crossing_rate))
        features.append(np.std(zero_crossing_rate))

        chroma = librosa.feature.chroma_stft(y=segment, sr=sr)
        features.extend(np.mean(chroma.T, axis=0))
        features.extend(np.std(chroma.T, axis=0))

        rms = librosa.feature.rms(y=segment)
        features.append(np.mean(rms))
        features.append(np.std(rms))

        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=segment, sr=sr)
        features.append(np.mean(spectral_bandwidth))
        features.append(np.std(spectral_bandwidth))

        all_features.append(features)

    return all_features

for genre in os.listdir(dataset_path):
    genre_path = os.path.join(dataset_path, genre)
    if not os.path.isdir(genre_path):
        continue
        
    print(f"currently processing {genre}")
    for file_name in os.listdir(genre_path):
        file_path = os.path.join(genre_path, file_name)
        try:
            song_features = extract_features(file_path)
            
            for segment in song_features:
                features.append(segment)
                labels.append(genre)
                
        except Exception as e:
            print(f"error with {file_path}")
            print(e)

print("done")

X_train, X_test, y_train, y_test = train_test_split(
    features,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = SVC(kernel='rbf', C=10, gamma='scale')

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("accuracy:", accuracy)