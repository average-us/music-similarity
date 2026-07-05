import os
import random
import shutil

source_dir = "sliced_spectrograms"
output_dir = "split_sliced_spectrograms"
train_prop = 0.8

for split in ["train", "val"]:
    for genre in os.listdir(source_dir):
        if os.path.isdir(os.path.join(source_dir, genre)):
            os.makedirs(os.path.join(output_dir, split, genre), exist_ok=True)

random.seed(42)
for genre in os.listdir(source_dir):
    genre_path = os.path.join(source_dir, genre)
    if not os.path.isdir(genre_path):
        continue

    files = [
        f
        for f in os.listdir(genre_path)
        if os.path.isfile(os.path.join(genre_path, f))
    ]
    random.shuffle(files)

    split_idx = int(len(files) * train_prop)
    train_files = files[:split_idx]
    val_files = files[split_idx:]

    for f in train_files:
        shutil.copy(
            os.path.join(genre_path, f),
            os.path.join(output_dir, "train", genre, f),
        )
    for f in val_files:
        shutil.copy(
            os.path.join(genre_path, f),
            os.path.join(output_dir, "val", genre, f),
        )

print("Split finished! Zip up the 'split_sliced_spectrograms' folder.")
