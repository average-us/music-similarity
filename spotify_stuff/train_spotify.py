import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

df = pd.read_csv("data/listening_log.csv")

df["previous_track_id"] = df["track_id"].shift(1)
df["previous_artist_id"] = df["artist_id"].shift(1)

df = df.dropna().reset_index(drop=True)

categorical_columns = [
    "track_id",
    "artist_id",
    "previous_track_id",
    "previous_artist_id",
]

encoders = {}

for column in categorical_columns:
    encoder = LabelEncoder()
    df[column] = encoder.fit_transform(df[column].astype(str))
    encoders[column] = encoder

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

X = df[feature_columns]
y = df["skipped"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
)

model.fit(X_train, y_train)


predictions = model.predict(X_test)

print("\nAccuracy:")
print(f"{accuracy_score(y_test, predictions):.3f}")

print("\nClassification Report:")
print(classification_report(y_test, predictions))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

importance = pd.DataFrame({
    "Feature": feature_columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importances:")
print(importance)

example = X.iloc[[0]]

prediction = model.predict(example)[0]
probability = model.predict_proba(example)[0]

print("\nExample Prediction")
print("------------------")
print("Prediction:", "Skip" if prediction else "Don't Skip")
print("Probabilities:", probability)