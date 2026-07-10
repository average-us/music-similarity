import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
import joblib
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
REDIRECT_URI = os.getenv("REDIRECT_URI")

sp = spotipy.Spotify(
    auth_manager=SpotifyOAuth(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        redirect_uri=REDIRECT_URI,
        scope="user-read-currently-playing user-read-playback-state"
    )
)

model = CatBoostClassifier()
model.load_model("models/skip_model.cbm")

feature_columns = joblib.load("models/features.pkl")
playlist = pd.read_csv("data/playlist.csv")

current_playback = sp.current_playback()

track = current_playback["item"]


current_context = {
    "previous_track_id": track["id"],
    "previous_artist_id": track["artists"][0]["id"],
    "hour": datetime.now().hour,
    "weekday": datetime.now().weekday()
}

hour_sin = np.sin(2 * np.pi * current_context["hour"] / 24)
hour_cos = np.cos(2 * np.pi * current_context["hour"] / 24)

weekday_sin = np.sin(2 * np.pi * current_context["weekday"] / 7)
weekday_cos = np.cos(2 * np.pi * current_context["weekday"] / 7)

rows = []

for _, song in playlist.iterrows():
    rows.append({
        "track_id": song["track_id"],
        "artist_id": song["artist_id"],
        "previous_track_id": current_context["previous_track_id"],
        "previous_artist_id": current_context["previous_artist_id"],
        "explicit": song["explicit"],
        "duration": song["duration"],
        "same_artist": int(song["artist_id"] == current_context["previous_artist_id"]),
        "same_track": 0,  # never recommend current song again
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "weekday_sin": weekday_sin,
        "weekday_cos": weekday_sin
    })

candidate_df = pd.DataFrame(rows)

candidate_df = candidate_df[feature_columns]

skip_probs = model.predict_proba(candidate_df)[:, 1]

candidate_df["skip_probability"] = skip_probs
candidate_df = candidate_df.sort_values("skip_probability")

print("\nTop 10 recommended songs:\n")

top10 = candidate_df.head(10)

for i, row in top10.iterrows():
    song = playlist.iloc[i]
    print(
        f"{sp.track(song['track_id'])['name']} | "
        f"skip_prob={row['skip_probability']:.3f}"
    )

best = playlist.iloc[candidate_df.index[0]]

print("\nBEST RECOMMENDATION:")
print(sp.track(best["track_id"])["name"])