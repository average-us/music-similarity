import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
import joblib

df = pd.read_csv("data/listening_log.csv")

df["previous_track_id"] = df["track_id"].shift(1)
df["previous_artist_id"] = df["artist_id"].shift(1)

df = df.dropna().reset_index(drop=True)

df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)

feature_columns = [
    "track_id",
    "artist_id",
    "previous_track_id",
    "previous_artist_id",
    "explicit",
    "duration",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
]

target = "skipped"

X = df[feature_columns]
y = df[target]

split_idx = int(len(df) * 0.8)

X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

cat_features = [
    "track_id",
    "artist_id",
    "previous_track_id",
    "previous_artist_id",
]

model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    loss_function="Logloss",
    verbose=100
)

model.fit(
    X_train,
    y_train,
    cat_features=cat_features,
    eval_set=(X_test, y_test),
    use_best_model=True
)

model.save_model("models/skip_model.cbm")

joblib.dump(feature_columns, "models/features.pkl")

print("\nModel saved to models/skip_model.cbm")