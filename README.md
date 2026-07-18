# music-projects

Variety of projects including genre classification, music similarity, and a personal spotify song recommender (WIP).

# Spotify Recommender

NOTE: THIS IS A PROJECT PURELY FOR PERSONAL USE/RESEARCH AND IS NOT DISTRIBUTED


The Spotify recommender uses the Spotify API through the Spotipy library in order to access the user's current listening context in order to make a prediction of a song that the user is least likely to skip.


The model is trained off of the user's personal listening data and makes its predictions off of many feautres including the duration of the song, the time of day, day of week, and whether the song is explicit or not. The model uses a CatBoost decision tree model to make the prediction.


The model works in real time, polling the user's playback in order to detect song change and makes live predictions off of current listening context, adding its prediction into the Spotify queue automatically.


Features:
* The recommender never plays a song that it has played before in the current listening session.
* The recommender never plays a song by an artist if that artist has been listened to in the last 5 songs in the listening session.

