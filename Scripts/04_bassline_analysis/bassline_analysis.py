""
import os
import json
import numpy as np
import librosa
import soundfile as sf

def bassline_analysis(folder):
    """
    Analyze bassline features from stems and write results to the track's JSON file.
    """
    json_name = os.path.basename(folder) + ".json"
    json_path = os.path.join(folder, json_name)
    bass_wav = os.path.join(folder, "bass.wav")

    with open(json_path, "r") as f:
        data = json.load(f)

    bpm_raw = data.get("bpm", 0)
    try:
        bpm = float(bpm_raw)
        if bpm <= 0:
            raise ValueError
    except (ValueError, TypeError):
        print("❌ Invalid BPM value.")
        bpm = None

    onset_raw = data.get("onsets", "")
    onsets = list(map(float, onset_raw.split(","))) if isinstance(onset_raw, str) else []

    bass_onsets = onsets
    kick_onsets = onsets

    bass_energy_raw = data.get("energy_envelopes_by_stem", {}).get("bass", "")
    bass_energy = list(map(float, bass_energy_raw.split(","))) if isinstance(bass_energy_raw, str) else []

    features = {}

    if bass_energy and bass_onsets:
        threshold = 0.05 * max(bass_energy)
        for i, e in enumerate(bass_energy):
            if e > threshold and i < len(bass_onsets):
                features["bass_first_entry_time"] = round(bass_onsets[i], 3)
                break
        else:
            features["bass_first_entry_time"] = None
    else:
        features["bass_first_entry_time"] = None

    max_time = data.get("track_length", 300.0)
    raw_time = features["bass_first_entry_time"]
    if raw_time is not None and max_time > 0:
        norm = raw_time / max_time
        features["bass_first_entry_time_norm"] = round(min(max(norm, 0), 1), 4)
    else:
        features["bass_first_entry_time_norm"] = None

    if bass_energy:
        energy_arr = np.array(bass_energy)
        features["bass_energy_mean"] = float(np.mean(energy_arr))
        features["bass_energy_std"] = float(np.std(energy_arr))
        features["bass_energy_sustain_ratio"] = float(np.sum(energy_arr > 0.7 * np.max(energy_arr)) / len(energy_arr))
        features["bass_energy_dynamics"] = features["bass_energy_std"]
    else:
        features.update({
            "bass_energy_mean": None,
            "bass_energy_std": None,
            "bass_energy_sustain_ratio": None,
            "bass_energy_dynamics": None,
        })

    if bass_onsets and len(bass_onsets) > 1:
        duration = max(bass_onsets) - min(bass_onsets)
        features["bass_onset_density"] = len(bass_onsets) / duration if duration > 0 else 0
        inter_onsets = np.diff(sorted(bass_onsets))
        features["bass_onset_variance"] = float(np.var(inter_onsets))
    else:
        features["bass_onset_density"] = 0
        features["bass_onset_variance"] = None

    if bpm and bass_onsets:
        interval = 60.0 / bpm / 4
        distances = [min(abs(o - n * interval) for n in range(int(max(bass_onsets) / interval) + 1)) for o in bass_onsets]
        mean_dist = float(np.mean(distances))
        features["bass_quantization_score"] = mean_dist
    else:
        features["bass_quantization_score"] = None

    if bass_onsets and kick_onsets:
        distances = []
        hits = 0
        for b in bass_onsets:
            closest_k = min(kick_onsets, key=lambda k: abs(k - b))
            dist = abs(b - closest_k)
            distances.append(dist)
            if dist <= 0.04:
                hits += 1
        features["bass_syncopation_score"] = float(np.mean(distances)) if distances else None
        features["bass_on_kick_ratio"] = round(hits / len(bass_onsets), 3) if bass_onsets else 0
    else:
        features["bass_syncopation_score"] = None
        features["bass_on_kick_ratio"] = None

    if os.path.isfile(bass_wav):
        y, sr = sf.read(bass_wav)
        if y.ndim > 1:
            y = y.mean(axis=1)

        fft = np.abs(np.fft.rfft(y))
        freqs = np.fft.rfftfreq(len(y), 1/sr)
        bass_energy_total = np.sum(fft)
        bass_low_energy = np.sum(fft[freqs < 100]) if bass_energy_total > 0 else 0
        features["bass_low_freq_ratio"] = bass_low_energy / bass_energy_total if bass_energy_total > 0 else 0

        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_vals = pitches[magnitudes > np.median(magnitudes)]
        if len(pitch_vals) > 0:
            mean_pitch = np.mean(pitch_vals)
            std_pitch = np.std(pitch_vals)
            features["bass_pitch_stability"] = std_pitch / mean_pitch if mean_pitch != 0 else 0
        else:
            features["bass_pitch_stability"] = None
    else:
        features["bass_low_freq_ratio"] = None
        features["bass_pitch_stability"] = None

    # ✅ Composite features
    if features["bass_pitch_stability"] is not None and features["bass_energy_sustain_ratio"] is not None:
        features["bass_smoothness"] = round(features["bass_pitch_stability"] * features["bass_energy_sustain_ratio"], 4)
    else:
        features["bass_smoothness"] = None

    if features["bass_onset_density"] and features["bass_quantization_score"]:
        features["bass_groove_score"] = round(features["bass_onset_density"] * features["bass_quantization_score"], 4)
    else:
        features["bass_groove_score"] = None

    data["bassline_analysis"] = features
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2, separators=(',', ': '))

    print("✅ Bassline analysis complete. Features added to JSON.")
    return features

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python bassline_analysis.py <folder_path>")
    else:
        bassline_analysis(sys.argv[1])
