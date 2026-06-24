import spotipy
from spotipy.oauth2 import SpotifyOAuth
from dotenv import load_dotenv
import os
import time
from datetime import datetime
import csv

load_dotenv()

CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
REDIRECT_URI = os.getenv("REDIRECT_URI")

last_track_id = None
last_track_name = None
last_progress_ms = None
last_duration_ms = None
last_artist_id = None
last_artist_name = None


sp = spotipy.Spotify(
    auth_manager=SpotifyOAuth(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        redirect_uri=REDIRECT_URI,
        scope="user-read-currently-playing user-read-playback-state"
    )
)

while True:
    current = sp.current_playback()
    track = current["item"]

    current_track_id = track["id"]

    if (current_track_id != last_track_id) and (last_track_id != None):
        
        completion_percent = round((last_progress_ms/last_duration_ms) * 100)
        skipped = int(completion_percent < 80)
        now = datetime.now()
        hour = now.hour
        weekday = now.weekday()

        with open("data/listening_log.csv", "a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)

            writer.writerow([last_track_id, last_track_name, last_artist_id, last_artist_name, last_duration_ms, hour, weekday, completion_percent, skipped])
    


    last_track_id = current_track_id
    last_track_name = track["name"]
    last_progress_ms = current["progress_ms"]
    last_duration_ms = track["duration_ms"]
    last_artist_id = track["artists"][0]["id"]
    last_artist_name = track["artists"][0]["name"]

    time.sleep(5)