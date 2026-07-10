import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import os
from dotenv import load_dotenv

load_dotenv()

CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
REDIRECT_URI = os.getenv("REDIRECT_URI")

PLAYLIST_ID = "00ALUNyrNrJiCrbz1EkqFp"

sp = spotipy.Spotify(
    auth_manager=SpotifyOAuth(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        redirect_uri=REDIRECT_URI,
        scope="playlist-read-private playlist-read-collaborative"
    )
)

rows = []

results = sp.playlist_items(
    PLAYLIST_ID,
    additional_types=["track"]
)

while True:

    for item in results["items"]:

        track = item["item"]

        if track is None:
            continue

        rows.append({

            "track_id": track["id"],

            "artist_id": track["artists"][0]["id"],

            "duration": track["duration_ms"],

            "explicit": int(track["explicit"]),
        })

    if results["next"]:
        results = sp.next(results)
    else:
        break

playlist = pd.DataFrame(rows)

playlist.to_csv(
    "data/playlist.csv",
    index=False
)

print(playlist.head())

print(f"\nDownloaded {len(playlist)} songs.")