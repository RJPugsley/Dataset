import os
import json
import numpy as np
import subprocess
import argparse
from scipy.signal import find_peaks
from scipy.stats import skew, variation

def final_energy_analysis(folder):
    print(f"\n📁 Analyzing folder: {folder}")

    # Locate JSON and source audio
    json_path = next((os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".json")), None)
    if not json_path:
        print("❌ JSON file not found.")
        return

    folder_name = os.path.basename(folder).lower()
    source_path = next(
        (os.path.join(folder, folder_name + ext) for ext in [".wav", ".mp3", ".flac", ".m4a"]
         if os.path.exists(os.path.join(folder, folder_name + ext))),
        None
    )
    if not source_path:
        print("❌ Source audio file not found.")
        return

    with open(json_path, "r") as f:
        data = json.load(f)

    sr, audio = load_audio_ffmpeg(source_path)
    envelope = compute_energy_envelope(audio, sr)

    avg_energy = float(np.mean(envelope))
    energy_std = float(np.std(envelope))
    peak_energy = float(np.max(envelope))
    skewness = float(skew(envelope))
    peaks, _ = find_peaks(envelope, height=avg_energy * 1.5, distance=10)
    drop_count = int(len(peaks))
    ramp = np.linspace(0, 1, len(envelope))
    norm_env = (envelope - np.min(envelope)) / (np.max(envelope) - np.min(envelope) + 1e-9)
    ramp_up_score = float(np.corrcoef(norm_env, ramp)[0, 1])
    silence_thresh = 0.05 * np.max(envelope)
    break_silence_ratio = float(np.sum(envelope < silence_thresh) / len(envelope))

    bpm = data.get("bpm")
    onsets = data.get("onsets")

    features = {
        "avg_energy": avg_energy,
        "energy_std": energy_std,
        "peak_energy": peak_energy,
        "energy_skewness": skewness,
        "drop_count": drop_count,
        "ramp_up_score": ramp_up_score,
        "break_silence_ratio": break_silence_ratio,
        "frame_ms": 50,
        "sampling_rate": sr
    }

    features.update(advanced_energy_features(envelope, sr, 50, bpm=bpm, onsets=onsets))

    data["energy_envelope_analysis"] = features
    data["energy_envelopes"] = { "mix": ",".join([f"{v:.6f}" for v in envelope]) }

    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)

    print("✅ Full energy envelope + genre feature analysis complete.")


# --- Supporting Functions ---

def load_audio_ffmpeg(wav_path, target_sr=22050):
    print(f"🔊 Loading full mix: {wav_path}")
    command = ["ffmpeg", "-i", wav_path, "-ac", "1", "-ar", str(target_sr), "-f", "f32le", "-"]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg error: {result.stderr.decode()}")
    audio = np.frombuffer(result.stdout, dtype=np.float32)
    return target_sr, audio

def compute_energy_envelope(audio, sr, frame_ms=50):
    frame_len = int(sr * frame_ms / 1000)
    hop_len = frame_len
    envelope = []
    for i in range(0, len(audio) - frame_len, hop_len):
        frame = audio[i:i+frame_len]
        rms = np.sqrt(np.mean(frame ** 2))
        envelope.append(rms)
    return np.array(envelope)

def advanced_energy_features(envelope, sr, frame_ms, bpm=None, onsets=None):
    features = {}
    norm_env = (envelope - np.min(envelope)) / (np.max(envelope) - np.min(envelope) + 1e-9)

    features["histogram"] = ",".join([str(v) for v in np.histogram(norm_env, bins=10, range=(0,1))[0]])
    features["energy_flatness"] = float(np.exp(np.mean(np.log(envelope + 1e-9))) / (np.mean(envelope) + 1e-9))
    features["energy_crest"] = float(np.max(envelope) / (np.mean(envelope) + 1e-9))
    features["energy_variation"] = float(variation(envelope))

    autocorr = np.correlate(norm_env, norm_env, mode='full')[len(norm_env):]
    autocorr[0] = 0
    features["energy_repetition_score"] = float(np.max(autocorr))

    # ✅ Convert BPM safely
    try:
        bpm = float(bpm)
        if bpm <= 0:
            raise ValueError
    except (ValueError, TypeError):
        bpm = None

    if bpm:
        seconds_per_beat = 60.0 / bpm
        frames_per_beat = int((seconds_per_beat * 1000) / frame_ms)
        beat_means = [np.mean(norm_env[i:i+frames_per_beat]) for i in range(0, len(norm_env), frames_per_beat)]
        features["beat_energy_mean"] = float(np.mean(beat_means))
        features["beat_energy_std"] = float(np.std(beat_means))

    if onsets and len(onsets) == len(envelope):
        features["onset_energy_correlation"] = float(np.corrcoef(onsets, envelope)[0, 1])

    return features


# --- Command Line Entry ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="Path to track folder")
    args = parser.parse_args()
    final_energy_analysis(args.folder)
