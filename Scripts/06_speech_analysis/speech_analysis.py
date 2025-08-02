import os
import json
import numpy as np
import subprocess
import scipy.signal as signal
import argparse
from pathlib import Path
from scipy.stats import entropy
import librosa

def speech_analysis(folder):
    print(f"📁 Starting vocal analysis for: {folder}")
    json_path = None
    vocal_path = None
    for file in os.listdir(folder):
        if file.endswith(".json"):
            json_path = os.path.join(folder, file)
        if "vocals" in file.lower() and file.endswith(".wav"):
            vocal_path = os.path.join(folder, file)

    if not json_path or not vocal_path:
        print("❌ Required files not found.")
        return

    with open(json_path, "r") as f:
        data = json.load(f)

    # ✅ PATCHED: safe parsing of envelope and onsets
    raw_env = data.get("energy_envelopes_by_stem", {}).get("vocals", [])
    envelope = [float(x) for x in raw_env] if isinstance(raw_env, list) else [float(x) for x in raw_env.split(",") if x]

    raw_onsets = data.get("onsets", [])
    onsets = [float(x) for x in raw_onsets] if isinstance(raw_onsets, list) else [float(x) for x in raw_onsets.split(",") if x]

    bpm = float(data.get("bpm", 0))
    rhythm = data.get("rhythm_pattern", {})
    kick_onsets = extract_kick_onsets(rhythm)

    sr, audio = load_audio_ffmpeg(vocal_path)
    features = analyze_vocals(audio, sr, onsets, envelope, bpm, kick_onsets, rhythm, data.get("track_length"))

    print("📝 Writing vocals_analysis to JSON...")
    data["vocals_analysis"] = features

    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)
    print("✅ Vocal analysis complete.")

# ----------------- Helper functions -----------------

def extract_kick_onsets(rhythm_pattern_json):
    labels = rhythm_pattern_json.get("labels", [])
    beats = rhythm_pattern_json.get("beats", [])
    return [beat for label, beat in zip(labels, beats) if isinstance(label, list) and "kick" in label]

def load_audio_ffmpeg(wav_path, target_sr=22050):
    print(f"🔊 Loading {wav_path} with FFmpeg...")
    command = ["ffmpeg", "-i", wav_path, "-ac", "1", "-ar", str(target_sr), "-f", "f32le", "-"]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg error: {result.stderr.decode()}")
    audio = np.frombuffer(result.stdout, dtype=np.float32)
    print("✅ Audio loaded.")
    return target_sr, audio

def vocal_energy_entropy(envelope):
    norm_env = np.array(envelope) + 1e-9
    norm_env /= np.sum(norm_env)
    return float(entropy(norm_env))

def avg_phrase_length(envelope, sr=20):
    threshold = 0.1 * np.max(envelope)
    active_frames = [i for i, val in enumerate(envelope) if val >= threshold]
    if not active_frames:
        return 0.0
    gaps = np.diff(active_frames)
    gap_durations = [g for g in gaps if g > 1]
    if not gap_durations:
        return len(active_frames) / sr
    return float(np.mean(gap_durations)) / sr

def compute_pitch_features(audio, sr):
    f0, voiced_flag, voiced_probs = librosa.pyin(audio, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    voiced = f0[~np.isnan(f0)]
    if len(voiced) < 2:
        return 0.0, 0.0, 0.0
    pitch_stability = float(np.std(voiced))
    avg_pitch_gap = float(np.mean(np.abs(np.diff(voiced))))
    vibrato_strength = float(np.std(signal.detrend(voiced)))
    return pitch_stability, avg_pitch_gap, vibrato_strength

def compute_spectral_flatness(audio, sr, frame_length=2048, hop_length=512):
    S = librosa.stft(audio, n_fft=frame_length, hop_length=hop_length)
    flatness = librosa.feature.spectral_flatness(S=np.abs(S))
    return float(np.mean(flatness))

def compute_mfcc_dynamics(audio, sr, n_mfcc=13):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    delta_mfcc = librosa.feature.delta(mfcc)
    return float(np.std(delta_mfcc))

def formant_band_energy(audio, sr):
    S = librosa.stft(audio)
    freqs = librosa.fft_frequencies(sr=sr)
    mag = np.abs(S)
    total_energy = np.sum(mag)
    mask = (freqs >= 300) & (freqs <= 3000)
    formant_energy = np.sum(mag[mask, :])
    return float(formant_energy / (total_energy + 1e-9))

def voiced_segment_stats(envelope, sr=20):
    threshold = 0.1 * np.max(envelope)
    voiced = np.array(envelope) >= threshold
    segments = []
    current_len = 0
    silence_frames = 0
    for val in voiced:
        if val:
            if current_len > 0:
                segments.append(current_len)
            current_len = 1
        else:
            silence_frames += 1
            current_len += 1
    if current_len > 0:
        segments.append(current_len)
    voiced_durations = np.array(segments) / sr
    silence_ratio = silence_frames / len(envelope) if envelope else 0
    return len(segments), float(np.mean(voiced_durations)) if len(voiced_durations) else 0.0, silence_ratio

def rhythmic_alignment_score(onsets, beat_grid):
    if not beat_grid or not onsets:
        return None
    aligned = sum(1 for o in onsets if min(abs(o - b) for b in beat_grid) <= 0.05)
    return float(aligned / len(onsets))

def pitch_range_estimate(audio, sr):
    f0, _, _ = librosa.pyin(audio, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    voiced = f0[~np.isnan(f0)]
    if len(voiced) < 2:
        return 0.0
    return float(np.max(voiced) - np.min(voiced))

def analyze_vocals(audio, sr, onsets, envelope, bpm, kick_onsets, rhythm_pattern, track_length):
    print("🔍 Analyzing vocal stem...")
    duration = len(audio) / sr
    audio = audio.astype(np.float32)
    audio /= np.max(np.abs(audio)) + 1e-9
    vocal_density = len(onsets) / duration if duration > 0 else 0.0

    beat_grid = [b["time"] if isinstance(b, dict) and "time" in b else b for b in rhythm_pattern.get("beats", [])]
    labels = rhythm_pattern.get("labels", [])
    loop_start = rhythm_pattern.get("start_time", 0.0)

    vocal_quant_score = rhythmic_alignment_score(onsets, beat_grid)
    kick_beats = [b for b, l in zip(beat_grid, labels) if isinstance(l, list) and "kick" in l]
    vocal_on_kick_count = sum(1 for o in onsets if any(abs(o - k) <= 0.05 for k in kick_beats))
    vocal_on_kick_ratio = vocal_on_kick_count / len(onsets) if onsets else 0.0

    env_sr = 20
    threshold = 0.1 * np.max(envelope)
    energy_kick_hits = sum(1 for k in kick_beats if int(k * env_sr) < len(envelope) and envelope[int(k * env_sr)] >= threshold)
    vocal_energy_on_kick_ratio = energy_kick_hits / len(kick_beats) if kick_beats else 0.0

    entry_delay = onsets[0] - loop_start if onsets else None
    vocal_energy_sustain_ratio = float(np.sum(np.array(envelope) >= threshold) / len(envelope)) if envelope else 0.0
    vocal_energy_mean = float(np.mean(envelope)) if envelope else 0.0
    vocal_energy_std = float(np.std(envelope)) if envelope else 0.0

    bar_length = 60.0 / bpm * 4 if bpm > 0 else 1.0
    total_bars = duration / bar_length if bar_length > 0 else 1.0
    vocal_density_per_bar = len(onsets) / total_bars if total_bars > 0 else 0.0

    energy_entropy = vocal_energy_entropy(envelope)
    phrase_len = avg_phrase_length(envelope)
    pitch_range = pitch_range_estimate(audio, sr)
    vocal_peak_count = int(np.sum(np.array(envelope) > 0.8 * np.max(envelope)))
    chopped_score = float(vocal_density * energy_entropy)

    pitch_stability, avg_pitch_gap, vibrato_strength = compute_pitch_features(audio, sr)
    flatness = compute_spectral_flatness(audio, sr)
    mfcc_delta_std = compute_mfcc_dynamics(audio, sr)
    formant_ratio = formant_band_energy(audio, sr)
    voiced_segments, avg_voiced_duration, silence_ratio = voiced_segment_stats(envelope, sr=20)

    result = {
        "vocal_density": vocal_density,
        "vocal_density_per_bar": vocal_density_per_bar,
        "vocal_energy_mean": vocal_energy_mean,
        "vocal_energy_std": vocal_energy_std,
        "vocal_energy_sustain_ratio": vocal_energy_sustain_ratio,
        "vocal_on_kick_ratio": vocal_on_kick_ratio,
        "vocal_energy_on_kick_ratio": vocal_energy_on_kick_ratio,
        "vocal_entry_delay_from_loop": entry_delay,
        "vocal_quantization_score": vocal_quant_score,
        "vocal_energy_entropy": energy_entropy,
        "avg_phrase_length": phrase_len,
        "pitch_range": pitch_range,
        "vocal_peak_count": vocal_peak_count,
        "chopped_phrasing_score": chopped_score,
        "pitch_stability": pitch_stability,
        "avg_pitch_gap": avg_pitch_gap,
        "vibrato_strength": vibrato_strength,
        "spectral_flatness": flatness,
        "mfcc_delta_std": mfcc_delta_std,
        "formant_band_energy_ratio": formant_ratio,
        "voiced_segments": voiced_segments,
        "avg_voiced_duration": avg_voiced_duration,
        "silence_to_voice_ratio": silence_ratio,
        "voiced_segments_per_second": round(voiced_segments / track_length, 4) if track_length else None,
        "vocal_breathiness": round(flatness * silence_ratio, 4) if flatness is not None and silence_ratio is not None else None,
        "vocal_power": round(vocal_energy_mean * pitch_range, 4) if vocal_energy_mean is not None and pitch_range is not None else None
    }

    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="Path to track folder")
    args = parser.parse_args()
    speech_analysis(args.folder)
