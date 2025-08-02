from pathlib import Path
from demucs.pretrained import get_model

def extract_stems(input_path, output_root="/Volumes/Dataset", model_name="htdemucs_6s"):
    """
    Extract stems from a single audio file or all files in a folder.
    Prompts for file/folder if run as main script.
    """
    import os
    os.chdir(os.path.expanduser("~"))
    import json
    import torchaudio
    import subprocess
    import soundfile as sf
    import torch
    import shutil
    from demucs.apply import apply_model

    def get_genre_from_metadata(file_path):
        ffprobe_bin = "/Users/djf2/Desktop/AppsBinsLibs/Binary/ffprobe"
        cmd = [
            ffprobe_bin, "-v", "error",
            "-show_entries", "format_tags=genre",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(file_path)
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            genre = result.stdout.strip()
            return genre if genre else "Unknown"
        except subprocess.CalledProcessError:
            return "Unknown"

    def sanitize_genre(raw_genre):
        primary = raw_genre.split("/")[0].split(",")[0].strip()
        return primary.title().replace(" ", "_")

    def get_key_and_bpm(file_path):
        ffprobe_bin = "/Users/djf2/Desktop/AppsBinsLibs/Binary/ffprobe"
        ext = file_path.suffix.lower()

        if ext == ".wav":
            tags = ["INAM", "ICMT", "key", "tempo"]
        elif ext == ".mp3":
            tags = ["TKEY", "TBPM"]
        elif ext == ".m4a":
            tags = ["key", "tempo"]
        elif ext == ".flac":
            tags = ["key", "bpm"]
        else:
            tags = ["key", "tempo"]

        cmd = [
            ffprobe_bin, "-v", "error",
            "-show_entries", "format_tags=" + ",".join(tags),
            "-of", "default=noprint_wrappers=1:nokey=0",
            str(file_path)
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            key = "Unknown"
            bpm = "Unknown"
            for line in result.stdout.strip().splitlines():
                tag, val = line.strip().split("=", 1)
                tag_lower = tag.lower()
                val = val.strip()
                if any(k in tag_lower for k in ["key", "tkey", "inam"]) and key == "Unknown":
                    key = val
                elif any(b in tag_lower for b in ["bpm", "tempo", "tbpm", "icmt"]) and bpm == "Unknown":
                    bpm = val
            return key, bpm
        except subprocess.CalledProcessError:
            return "Unknown", "Unknown"

    def process_single_file(file_path, output_root, model):
        print(f"▶️ extract_stems() START — {file_path.name}")
        temp_wav = file_path.with_suffix(".temp.wav")
        ffmpeg_bin = "/Users/djf2/Desktop/AppsBinsLibs/Binary/ffmpeg"
        ffmpeg_cmd = [ffmpeg_bin, "-y", "-i", str(file_path), "-ar", "44100", "-ac", "2", str(temp_wav)]
        subprocess.run(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        waveform, sr = torchaudio.load(str(temp_wav))
        if waveform.shape[0] > 2:
            waveform = waveform[:2]
        elif waveform.shape[0] == 1:
            waveform = torch.cat([waveform, waveform], dim=0)

        with torch.no_grad():
            stems = apply_model(model, waveform.unsqueeze(0), split=True, overlap=0.25)[0]

        stem_names = model.sources
        raw_genre = get_genre_from_metadata(file_path)
        primary_genre = sanitize_genre(raw_genre)
        key, bpm = get_key_and_bpm(file_path)
        track_name = file_path.stem
        track_folder = Path(output_root) / primary_genre / track_name
        track_folder.mkdir(parents=True, exist_ok=True)

        for name, audio in zip(stem_names, stems):
            stem_path = track_folder / f"{name}.wav"
            sf.write(stem_path, audio.cpu().T.numpy(), sr)

        original_copy_path = track_folder / file_path.name
        if not original_copy_path.exists():
            shutil.move(str(file_path), str(original_copy_path))

        json_path = track_folder / f"{track_name}.json"
        if not json_path.exists():
            track_data = {
                "track_id": track_name,
                "genre": raw_genre,
                "primary_genre": primary_genre,
                "key": key,
                "bpm": bpm,
                "original_filename": file_path.name,
                "source_audio": str(original_copy_path),
                "stems": {name: str(track_folder / f"{name}.wav") for name in stem_names}
            }
            with open(json_path, "w") as f:
                json.dump(track_data, f, indent=2)

        if temp_wav.exists():
            temp_wav.unlink()

        print(f"✅ extract_stems() COMPLETE — {file_path.name}")
        print(f"::TRACK_FOLDER::{track_folder}")
        print(f"::SOURCE_FILE::{original_copy_path}")
        return track_folder

    # Main logic
    input_path = Path(input_path)
    output_root = Path(output_root)
    model = get_model(model_name)
    model.cpu()
    model.eval()

    if input_path.is_file():
        return process_single_file(input_path, output_root, model)
    elif input_path.is_dir():
        audio_exts = [".mp3", ".wav", ".flac", ".m4a"]
        files = [f for f in input_path.rglob("*") if f.suffix.lower() in audio_exts]
        for i, file in enumerate(files, 1):
            print(f"   📀 {i}/{len(files)} — {file.name}")
            process_single_file(file, output_root, model)
        print("✅ Folder processing complete.")
    else:
        raise ValueError(f"Invalid path: {input_path}")


# Interactive run mode
if __name__ == "__main__":
    user_input = input("Enter the full path to a file or folder: ").strip()
    output_dir = "/Volumes/Dataset"
    extract_stems(user_input, output_dir)
