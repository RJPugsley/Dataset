import os
import json
import subprocess
import numpy as np
from pathlib import Path
from typing import Dict

FFMPEG_BIN = "/Users/djf2/Desktop/AppsBinsLibs/Binary/ffmpeg"


def load_audio_ffmpeg(filepath: str, target_sr=22050) -> tuple:
    command = [FFMPEG_BIN, "-i", filepath, "-ac", "1", "-ar", str(target_sr), "-f", "f32le", "-"]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg error for {filepath}: {result.stderr.decode()}")
    audio = np.frombuffer(result.stdout, dtype=np.float32)
    return target_sr, audio


def compute_energy_envelope(audio: np.ndarray, sr: int, frame_ms=50) -> np.ndarray:
    frame_len = int(sr * frame_ms / 1000)
    if len(audio) < frame_len:
        return np.array([0.0])
    hop_len = frame_len
    num_frames = (len(audio) - frame_len) // hop_len
    envelope = [
        np.sqrt(np.mean(audio[i * hop_len: i * hop_len + frame_len] ** 2))
        for i in range(num_frames)
    ]
    envelope = np.array(envelope)
    return envelope / (np.max(envelope) + 1e-9)


def analyze_energy_envelopes(folder_path: str) -> None:
    folder = Path(folder_path)
    if not folder.is_dir():
        print("❌ Invalid folder path.")
        return

    json_path = next((f for f in folder.glob("*.json")), None)
    if not json_path or not json_path.exists():
        print("❌ Could not locate the JSON file.")
        return

    with open(json_path, "r") as f:
        data = json.load(f)

    if "stems" not in data:
        print("❌ No 'stems' field found in JSON.")
        return

    stem_results: Dict[str, list] = {}
    for stem_name, stem_path in data["stems"].items():
        if not os.path.isfile(stem_path):
            print(f"⚠️ File not found for stem: {stem_name}")
            continue
        print(f"🎵 Processing stem: {stem_name}")
        try:
            sr, audio = load_audio_ffmpeg(stem_path)
            envelope = compute_energy_envelope(audio, sr)
            stem_results[stem_name] = envelope.tolist()
            print(f"    ✅ Envelope length: {len(envelope)}")
        except Exception as e:
            print(f"❌ Failed to process {stem_name}: {e}")

    data["energy_envelopes_by_stem"] = stem_results
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)

    print("✅ Energy envelope analysis complete for all stems.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run energy envelope analysis on stem files in a folder")
    parser.add_argument("folder", type=str, help="Path to the track folder")
    args = parser.parse_args()
    analyze_energy_envelopes(args.folder)
