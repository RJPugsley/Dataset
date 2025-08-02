""
import os
import json
import subprocess
import numpy as np
import librosa
import scipy.signal as signal
from scipy.spatial.distance import cosine

FFMPEG_BIN = "/Users/djf2/Desktop/AppsBinsLibs/Binary/ffmpeg"

MAJOR_PROFILE = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
MINOR_PROFILE = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

def rotate(template, n):
    return template[-n:] + template[:-n]

def detect_key_from_chroma(chroma_mean):
    scores = {}
    for i in range(12):
        major = rotate(MAJOR_PROFILE, i)
        minor = rotate(MINOR_PROFILE, i)
        scores[f"{(i or 12)}B"] = 1 - cosine(chroma_mean, major)
        scores[f"{(i or 12)}A"] = 1 - cosine(chroma_mean, minor)
    best = max(scores, key=scores.get)
    return best, scores[best]

def load_audio_ffmpeg(path, target_sr=22050):
    command = [FFMPEG_BIN, "-i", path, "-ac", "1", "-ar", str(target_sr), "-f", "f32le", "-"]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg error: {result.stderr.decode()}")
    audio = np.frombuffer(result.stdout, dtype=np.float32)
    return target_sr, audio

def analyze_harmonic(audio, sr):
    audio = audio.astype(np.float32)
    audio /= np.max(np.abs(audio)) + 1e-9
    energy = np.mean(audio**2)
    energy_variance = float(np.std(audio))
    sos = signal.butter(4, 300 / (sr / 2), btype='low', output='sos')
    low_energy = signal.sosfilt(sos, audio)
    low_freq_ratio = np.sum(low_energy**2) / (np.sum(audio**2) + 1e-9)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    chroma_mean = np.mean(chroma, axis=1).tolist()
    chroma_std = np.std(chroma, axis=1).tolist()
    tonnetz = librosa.feature.tonnetz(y=audio, sr=sr)
    tonnetz_mean = np.mean(tonnetz, axis=1).tolist()

    tonal_brightness = float(np.mean(chroma_mean[7:12]) - np.mean(chroma_mean[0:5]))
    tonal_variability = float(np.std(chroma_mean))
    harmony_complexity = float(np.mean(np.abs(tonnetz_mean)))

    return {
        "avg_energy": float(energy),
        "energy_variance": float(energy_variance),
        "low_freq_ratio": float(low_freq_ratio),
        "chroma_mean": chroma_mean,
        "chroma_std": chroma_std,
        "tonnetz_mean": tonnetz_mean,
        "tonal_brightness": tonal_brightness,
        "tonal_variability": tonal_variability,
        "harmony_complexity": harmony_complexity
    }

def harmonic_analysis(folder):
    print(f"📁 Starting harmonic analysis for: {folder}")
    json_path = next((os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".json")), None)
    if not json_path:
        print("❌ JSON file not found.")
        return

    with open(json_path, "r") as f:
        data = json.load(f)

    source_path = data.get("source_audio")
    if not source_path or not os.path.isfile(source_path):
        print("❌ Source audio not found.")
        return

    print(f"🎧 Analyzing source mix: {os.path.basename(source_path)}")
    sr, audio = load_audio_ffmpeg(source_path)
    features = analyze_harmonic(audio, sr)

    data["harmonic_analysis"] = {
        "source_mix": features
    }

    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)

    print("✅ Harmonic analysis of source mix complete.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="Path to track folder containing the JSON and source_audio")
    args = parser.parse_args()
    harmonic_analysis(args.folder)
