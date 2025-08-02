import os
import json
import argparse
import subprocess
import numpy as np
from pathlib import Path

def load_audio_ffmpeg(filepath, target_sr=22050):
    command = [
        "ffmpeg", "-i", filepath, "-ac", "1", "-ar", str(target_sr),
        "-f", "f32le", "-"
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg error:\n{result.stderr.decode()}")
    audio = np.frombuffer(result.stdout, dtype=np.float32)
    return target_sr, audio

def extract_bpm_and_key_from_tags(source_file):
    ffprobe_path = "/Users/djf2/Desktop/AppsBinsLibs/Binary/ffprobe"
    cmd = [
        ffprobe_path, "-v", "quiet", "-print_format", "json",
        "-show_entries", "format_tags", source_file
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return None, None
    try:
        tags = json.loads(result.stdout).get("format", {}).get("tags", {})
        bpm = (
            tags.get("TBPM") or tags.get("BPM") or
            tags.get("bpm") or tags.get("tempo")
        )
        key = (
            tags.get("TKEY") or tags.get("KEY") or
            tags.get("initialkey") or tags.get("key")
        )
        return float(bpm) if bpm else None, key.strip() if key else None
    except Exception:
        return None, None

def compute_onset_times_fixed_grid(audio, sr, frame_size=1102, hop_size=1102):
    num_frames = (len(audio) - frame_size) // hop_size
    times = [(i * hop_size) / sr for i in range(num_frames)]
    return times

def run_stage1(folder_path):
    json_path = None
    for file in os.listdir(folder_path):
        if file.lower().endswith(".json"):
            json_path = os.path.join(folder_path, file)
            break

    if not json_path:
        print("❌ Could not locate the JSON file.")
        return

    with open(json_path, "r") as f:
        data = json.load(f)

    bpm_val = data.get("bpm")
    key_val = data.get("key")
    source_file = data.get("source_audio") or data.get("source", "")

    if source_file and os.path.isfile(source_file):
        bpm, key = extract_bpm_and_key_from_tags(source_file)

        if (not isinstance(bpm_val, (int, float)) and str(bpm_val).lower() in ("", "unknown")) and bpm:
            data["bpm"] = round(bpm, 2)
            print(f"🎼 BPM extracted from tags: {bpm}")
        else:
            print("⚠️ No BPM tag found or already present.")

        if (not key_val or key_val.lower() == "unknown") and key:
            data["key"] = key
            print(f"🎹 Key extracted from tags: {key}")
        else:
            print("⚠️ No KEY tag found or already present.")
    else:
        print("⚠️ Source file not found for BPM/key extraction.")

    if "stems" not in data:
        print("❌ No 'stems' field found in JSON.")
        return

    # Check source file vs stem durations
    try:
        sr_src, audio_src = load_audio_ffmpeg(source_file)
    except Exception as e:
        print(f"❌ Could not load source audio: {e}")
        return

    source_length = len(audio_src)
    consistent_stems = []
    for stem_name, stem_path in data["stems"].items():
        if not os.path.isfile(stem_path):
            print(f"⚠️ File not found for stem: {stem_name}")
            continue
        try:
            sr_stem, audio_stem = load_audio_ffmpeg(stem_path)
            if abs(len(audio_stem) - source_length) > sr_src:  # allow ~1 sec difference
                print(f"⚠️ Stem {stem_name} length mismatch with source audio.")
            else:
                consistent_stems.append(stem_name)
        except Exception as e:
            print(f"⚠️ Could not load stem {stem_name}: {e}")

    if not consistent_stems:
        print("❌ No stems matched source duration — skipping onset computation.")
        return

    # Compute onsets from source audio directly
    print(f"📡 Detecting fixed-grid onsets from source file: {Path(source_file).name}")
    try:
        times = compute_onset_times_fixed_grid(audio_src, sr_src)
        if not times:
            print("❌ Onset extraction failed — empty result")
            return
        data["onsets"] = times
    except Exception as e:
        print(f"❌ Failed to compute onsets: {e}")
        return

    keys_to_remove = [k for k in data if k.startswith("onsets_") and k.endswith("_raw")]
    for key in keys_to_remove:
        print(f"🧹 Removing old field: {key}")
        del data[key]

    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)

    print("✅ Onset detection complete. 'onsets', 'bpm', and 'key' fields updated.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1: Detect fixed-grid onsets aligned to energy envelope resolution")
    parser.add_argument("folder", type=str, help="Path to the track folder")
    args = parser.parse_args()

    if os.path.isdir(args.folder):
        run_stage1(args.folder)
    else:
        print("❌ Invalid folder path.")
