import os
import random
import shutil

SOURCE_ROOT = '/Users/djf2/Music/MP3'
DEST_FOLDER = '/Users/djf2/Music/untitled folder'
ALLOWED_EXTENSIONS = ('.mp3', '.m4a', '.flac', '.wav', '.aiff')
MAX_FILE_SIZE = 80 * 1024 * 1024  # 80 MB

os.makedirs(DEST_FOLDER, exist_ok=True)

# Keep track of filenames already copied
copied_basenames = set()

for root, dirs, files in os.walk(SOURCE_ROOT):
    # Filter for valid audio files and size limit
    valid_files = [
        f for f in files
        if f.lower().endswith(ALLOWED_EXTENSIONS)
        and os.path.getsize(os.path.join(root, f)) <= MAX_FILE_SIZE
    ]

    if not valid_files:
        continue

    selected = []
    attempts = 0
    max_attempts = len(valid_files) * 2

    while len(selected) < min(2, len(valid_files)) and attempts < max_attempts:
        candidate = random.choice(valid_files)
        base_name = os.path.basename(candidate)

        if base_name not in copied_basenames:
            selected.append(candidate)
            copied_basenames.add(base_name)

        attempts += 1

    for filename in selected:
        source_path = os.path.join(root, filename)
        dest_path = os.path.join(DEST_FOLDER, os.path.basename(filename))
        shutil.copy2(source_path, dest_path)
        print(f"Copied: {source_path} → {dest_path}")
