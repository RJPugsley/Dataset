#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bassline analysis (controller-friendly)

- Robust JSON parsing (arrays or comma-strings)
- Safer numeric conversions (ignore blanks)
- Uses track_length_sec or track_length (fallbacks)
- Exposes `run_stage_bassline(folder_path)` for control scripts
- CLI remains: `python bassline_analysis.py <folder>`
"""
import os
import json
from typing import List, Optional, Union
import numpy as np
import librosa
import soundfile as sf

def _to_float_list(x: Union[str, List[float], List[str], None]) -> List[float]:
    if x is None:
        return []
    if isinstance(x, list):
        out = []
        for v in x:
            try:
                out.append(float(v))
            except (TypeError, ValueError):
                continue
        return out
    if isinstance(x, str):
        toks = [t.strip() for t in x.split(',') if t.strip() != '']
        out = []
        for t in toks:
            try:
                out.append(float(t))
            except ValueError:
                continue
        return out
    return []

def _read_json(json_path: str) -> dict:
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def _write_json(json_path: str, data: dict) -> None:
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def bassline_analysis(folder: str):
    folder = os.path.abspath(folder)
    json_name = os.path.basename(folder) + ".json"
    json_path = os.path.join(folder, json_name)
    bass_wav = os.path.join(folder, "bass.wav")

    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"JSON not found: {json_path}")

    data = _read_json(json_path)

    # --- BPM ---
    bpm = None
    bpm_raw = data.get("bpm", None)
    try:
        bpm = float(bpm_raw) if bpm_raw is not None else None
        if bpm is not None and bpm <= 0:
            bpm = None
    except (TypeError, ValueError):
        bpm = None

    # --- Onsets --- (list or string). If missing, leave empty.
    onsets = _to_float_list(data.get("onsets", None))

    # For now we don't have kick-specific onsets; use the global onsets as proxy
    bass_onsets = onsets
    kick_onsets = onsets

    # --- Bass energy envelope ---
    bass_energy_raw = None
    if isinstance(data.get("energy_envelopes_by_stem"), dict):
        bass_energy_raw = data["energy_envelopes_by_stem"].get("bass", None)
    bass_energy = _to_float_list(bass_energy_raw)

    # --- Track length ---
    track_len = None
    # Prefer explicit seconds field if available
    for k in ("track_length_sec", "track_length"):
        if k in data:
            try:
                track_len = float(data[k])
                break
            except (TypeError, ValueError):
                pass
    # Fallback: from onsets
    if track_len is None and onsets:
        track_len = max(onsets)
    # Last resort: length from energy envelope sampling count is unknown, so skip

    features = {}

    # First entry time: threshold-crossing index aligned to bass_onsets index if possible
    if bass_energy and bass_onsets:
        threshold = 0.05 * max(bass_energy)
        entry_index = None
        # If energy is frame-based and onsets are time-based, we can't align by index;
        # pick first non-trivial onset as a coarse proxy.
        for i, e in enumerate(bass_energy):
            if e > threshold:
                entry_index = i
                break
        if entry_index is not None and entry_index < len(bass_onsets):
            features["bass_first_entry_time"] = round(float(bass_onsets[entry_index]), 3)
        else:
            # fallback: earliest onset
            features["bass_first_entry_time"] = round(float(min(bass_onsets)), 3)
    elif bass_onsets:
        features["bass_first_entry_time"] = round(float(min(bass_onsets)), 3)
    else:
        features["bass_first_entry_time"] = None

    # Normalized entry time
    if features["bass_first_entry_time"] is not None and track_len and track_len > 0:
        norm = features["bass_first_entry_time"] / track_len
        features["bass_first_entry_time_norm"] = round(max(0.0, min(1.0, norm)), 4)
    else:
        features["bass_first_entry_time_norm"] = None

    # Energy stats
    if bass_energy:
        energy_arr = np.asarray(bass_energy, dtype=float)
        features["bass_energy_mean"] = float(np.mean(energy_arr))
        features["bass_energy_std"] = float(np.std(energy_arr))
        peak = float(np.max(energy_arr))
        features["bass_energy_sustain_ratio"] = float(np.sum(energy_arr > 0.7 * peak) / len(energy_arr)) if peak > 0 else 0.0
        features["bass_energy_dynamics"] = features["bass_energy_std"]
    else:
        features.update({
            "bass_energy_mean": None,
            "bass_energy_std": None,
            "bass_energy_sustain_ratio": None,
            "bass_energy_dynamics": None,
        })

    # Onset density & variance
    if bass_onsets and len(bass_onsets) > 1:
        duration = max(bass_onsets) - min(bass_onsets)
        features["bass_onset_density"] = float(len(bass_onsets) / duration) if duration > 0 else 0.0
        inter_onsets = np.diff(sorted(bass_onsets))
        features["bass_onset_variance"] = float(np.var(inter_onsets)) if len(inter_onsets) > 0 else 0.0
    else:
        features["bass_onset_density"] = 0.0
        features["bass_onset_variance"] = None

    # Quantization to 1/16 grid at the given BPM
    if bpm and bass_onsets:
        interval = 60.0 / bpm / 4.0  # quarter->sixteenth
        if interval > 0:
            max_t = max(bass_onsets)
            n_steps = int(max_t / interval) + 1
            distances = []
            grid_times = [n * interval for n in range(n_steps)]
            for o in bass_onsets:
                distances.append(min(abs(o - gt) for gt in grid_times))
            features["bass_quantization_score"] = float(np.mean(distances)) if distances else None
        else:
            features["bass_quantization_score"] = None
    else:
        features["bass_quantization_score"] = None

    # Sync with kicks (proxy = global onsets)
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
        features["bass_on_kick_ratio"] = round(hits / len(bass_onsets), 3) if bass_onsets else 0.0
    else:
        features["bass_syncopation_score"] = None
        features["bass_on_kick_ratio"] = None

    # Spectral/pitch stats from bass stem if present
    if os.path.isfile(bass_wav):
        try:
            y, sr = sf.read(bass_wav, always_2d=False)
            if isinstance(y, np.ndarray) and y.ndim > 1:
                y = y.mean(axis=1)
            if not isinstance(y, np.ndarray):
                y = np.asarray(y, dtype=float)

            # FFT-based low freq ratio
            if len(y) > 0:
                fft = np.abs(np.fft.rfft(y))
                freqs = np.fft.rfftfreq(len(y), 1.0 / sr)
                total = float(np.sum(fft))
                low = float(np.sum(fft[freqs < 100.0])) if total > 0 else 0.0
                features["bass_low_freq_ratio"] = (low / total) if total > 0 else 0.0
            else:
                features["bass_low_freq_ratio"] = None

            # Pitch stability via piptrack
            pitches, mags = librosa.piptrack(y=y, sr=sr)
            mask = mags > np.median(mags)
            pitch_vals = pitches[mask]
            if pitch_vals.size > 0:
                mean_pitch = float(np.mean(pitch_vals))
                std_pitch = float(np.std(pitch_vals))
                features["bass_pitch_stability"] = (std_pitch / mean_pitch) if mean_pitch != 0.0 else None
            else:
                features["bass_pitch_stability"] = None
        except Exception as e:
            print(f"⚠️  Bass spectral analysis skipped: {e}")
            features.setdefault("bass_low_freq_ratio", None)
            features.setdefault("bass_pitch_stability", None)
    else:
        features["bass_low_freq_ratio"] = None
        features["bass_pitch_stability"] = None

    # Composite
    if (features.get("bass_pitch_stability") is not None and
        features.get("bass_energy_sustain_ratio") is not None):
        features["bass_smoothness"] = round(features["bass_pitch_stability"] * features["bass_energy_sustain_ratio"], 4)
    else:
        features["bass_smoothness"] = None

    if (features.get("bass_onset_density") is not None and
        features.get("bass_quantization_score") is not None):
        features["bass_groove_score"] = round(features["bass_onset_density"] * features["bass_quantization_score"], 4)
    else:
        features["bass_groove_score"] = None

    data["bassline_analysis"] = features
    _write_json(json_path, data)

    print("✅ Bassline analysis complete. Features added to JSON.")
    return features

# Controller-compatible entrypoint
def run_stage_bassline(folder_path: str):
    return bassline_analysis(folder_path)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python bassline_analysis.py <folder_path>")
        raise SystemExit(2)
    sys.exit(0 if bassline_analysis(sys.argv[1]) is not None else 1)
