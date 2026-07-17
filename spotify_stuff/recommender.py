import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
import joblib
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import os
from dotenv import load_dotenv
from datetime import datetime
import time

load_dotenv()

CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
REDIRECT_URI = os.getenv("REDIRECT_URI")

last_track_id = None
selected_track_id = None

listened_tracks = []
listened_artists = []

sp = spotipy.Spotify(
    auth_manager=SpotifyOAuth(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        redirect_uri=REDIRECT_URI,
        scope="user-read-currently-playing user-read-playback-state user-modify-playback-state"
    )
)

model = CatBoostClassifier()
model.load_model("models/skip_model.cbm")

feature_columns = joblib.load("models/features.pkl")
playlist = pd.read_csv("data/playlist.csv")

while True:
    current_playback = sp.current_playback()

    track = current_playback["item"]

    current_track_id = track["id"]

    if (current_track_id != last_track_id) and (last_track_id is not None):

        listened_tracks.append(last_track_id)
        listened_artists.append(sp.track(last_track_id)["artists"][0]["id"])

        current_context = {
            "previous_track_id": last_track_id,
            "previous_artist_id": sp.track(last_track_id)["artists"][0]["id"],
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
                "hour_sin": hour_sin,
                "hour_cos": hour_cos,
                "weekday_sin": weekday_sin,
                "weekday_cos": weekday_cos
            })

        candidate_df = pd.DataFrame(rows)

        candidate_df = candidate_df[feature_columns]

        skip_probs = model.predict_proba(candidate_df)[:, 1]

        candidate_df["skip_probability"] = skip_probs
        candidate_df = candidate_df.sort_values("skip_probability")

        for i, row in candidate_df.iterrows():
            song = playlist.iloc[i]

            if song["track_id"] not in listened_tracks and (song["artist_id"] not in listened_artists[-5:]):
                selected_track_id = song["track_id"]
                break

            continue

        sp.add_to_queue(uri=f"spotify:track:{selected_track_id}")
    
    last_track_id = current_track_id

    time.sleep(5)