import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import os
import librosa
import numpy as np

DATASET_PATH = "data/genres_original"
CACHE_FILE = "data/gtzan_extracted_features.csv"

def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=22050, mono=True, duration=30.0)

        mfccs = librosa.feature.mfcc(y=y,sr=sr, n_mfcc=20)
        mfccs_mean = np.mean(mfccs, axis=1)

        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        centroid_mean = np.mean(centroid)
        
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)

        feature_vector = np.concatenate((mfccs_mean, centroid_mean, chroma_mean))
        return feature_vector
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

if os.path.exists(CACHE_FILE):
    df = pd.read_csv(CACHE_FILE)
else:
    features = []
    names = []

    for genre in os.listdir(DATASET_PATH):
        genre_path = os.path.join(DATASET_PATH, genre)

        if not os.path.isdir(genre_path):
                continue

        for file_name in os.listdir(genre_path):
            if file_name.endswith('.wav'):
                file_path = os.path.join(genre_path, file_name)

                vector = extract_features(file_path)

                if vector is not None:
                        features.append(vector)
                        names.append(file_name)


    X = np.array(features)

    mfcc_cols = [f'mfcc_{i}' for i in range(20)]
    centroid_cols = ['spectral_centroid']
    chroma_cols = [f'chroma_{i}' for i in range(12)]
    all_cols = mfcc_cols + centroid_cols + chroma_cols

    df = pd.DataFrame(X, columns=all_cols)

    df.insert(0, 'song_title', names)


#--------------------------------------------------------#

numeric_features = df.drop(columns=['song_title', 'file_path'])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(numeric_features)

similarity_matrix = cosine_similarity(X_scaled)

similarity_df = pd.DataFrame(similarity_matrix, index=df['song_title'], columns=df['song_title'])

query_song = df['song_title'].iloc[0]

print(f"\nQuery Song: {query_song}")
print("Top 5 Most Similar Songs:")

recommendations = similarity_df[query_song].sort_values(ascending=False).iloc[1:6]

for rank, (song, score) in enumerate(recommendations.items(), start=1):
    print(f"{rank}. {song} (Score: {score:.4f})")
