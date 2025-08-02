import os
import sys
import json
import numpy as np
from pathlib import Path
from scipy.ndimage import uniform_filter1d

def parse_onsets(onset_data):
    if isinstance(onset_data, list):
        return [float(x) for x in onset_data if x is not None]
    return [float(x) for x in onset_data.strip().split(",") if x]

def resolve_json_path(folder_path):
    path = Path(folder_path)
    json_file = next((f for f in path.glob("*.json")), None)
    return json_file if json_file else None

def detect_rhythm(onsets, energy, bpm, resolution_hz):
    beat_interval = 60.0 / bpm
    for t in onsets:
        idx = int(t * resolution_hz)
        if idx < len(energy) and energy[idx] >= 0.75:
            anchor = t
            break
    else:
        return "broken"

    beat_grid = [anchor + i * beat_interval for i in range(8)]
    match_window = 0.2
    matches = sum(
        1 for target in beat_grid for t in onsets
        if abs(t - target) <= match_window and energy[int(t * resolution_hz)] >= 0.75
    )
    return "4x4" if matches >= 6 else "broken"

def run_stage2(folder_path, output_key="first_downbeat", resolution_hz=20, min_score=2, high_conf_score=5):
    json_path = resolve_json_path(folder_path)
    if not json_path:
        print("❌ Could not find JSON file in folder.")
        return

    with open(json_path, "r") as f:
        data = json.load(f)

    bpm = float(data.get("bpm", 0))
    if bpm == 0:
        raise ValueError("BPM missing or invalid in JSON.")

    beat_interval = 60.0 / bpm
    onsets = parse_onsets(data["onsets"])

    # ✅ FIXED: safe parsing of energy field
    raw_energy = data["energy_envelopes_by_stem"].get("drums", [])
    energy = (
        np.array([float(x) for x in raw_energy])
        if isinstance(raw_energy, list)
        else np.array([float(x) for x in raw_energy.split(",") if x])
    )

    norm_energy = (energy - np.min(energy)) / (np.max(energy) - np.min(energy) + 1e-6)
    smoothed_energy = uniform_filter1d(norm_energy, size=3)
    track_mean = np.mean(smoothed_energy)

    beat_times = [t for t in onsets if t < len(smoothed_energy) / resolution_hz]

    raw_candidates = []
    for t in beat_times:
        idx = int(t * resolution_hz)
        if idx <= 0 or idx >= len(smoothed_energy) - 1:
            continue
        if not (smoothed_energy[idx] > smoothed_energy[idx - 1] and smoothed_energy[idx] > smoothed_energy[idx + 1]):
            continue

        window_times = [t + i * beat_interval for i in range(-4, 5)]
        window_idx = [int(w * resolution_hz) for w in window_times if 0 <= int(w * resolution_hz) < len(smoothed_energy)]
        window_energy = [smoothed_energy[i] for i in window_idx]

        if len(window_energy) < 5:
            continue

        B0 = smoothed_energy[idx]
        pre = window_energy[:4]
        post = window_energy[5:]

        if B0 > 1.5 * track_mean and (len(pre) == 0 or len([e for e in pre if e < 0.5 * B0]) >= 2):
            raw_candidates.append((t, idx, B0, pre, post))

    rhythm_type = detect_rhythm(onsets, smoothed_energy, bpm, resolution_hz)

    candidates = []
    for t, idx, B0, pre, post in raw_candidates:
        score = 0
        if B0 > 1.5 * track_mean:
            score += 1
        if len(pre) == 0 or len([e for e in pre if e < 0.5 * B0]) >= 2:
            score += 2
        if len([e for e in pre if e < 0.5 * B0]) >= 2:
            score += 1
        if rhythm_type == "4x4":
            if len([e for e in post if e >= 0.9 * B0]) >= 3:
                score += 1
        else:
            if sum(1 for e in post if e >= 0.7 * B0) >= 2:
                score += 1
            if any(e >= 0.9 * B0 for e in post):
                score += 1
        if score >= min_score:
            candidates.append((t, score, rhythm_type))

    candidates.sort(key=lambda x: (-x[1], x[0]))

    if not candidates:
        print("❌ No valid downbeat candidates found.")
        return

    if rhythm_type == "broken":
        downbeat = next((t for t, s, _ in sorted(candidates, key=lambda x: x[0]) if s >= high_conf_score), candidates[0][0])
    else:
        downbeat = candidates[0][0]

    data[output_key] = round(downbeat, 3)
    track_length = data.get("track_length")
    if track_length and track_length > 0:
        data[output_key + "_normalized"] = round(downbeat / track_length, 6)

    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"🎯 Stage 2 complete. First downbeat = {downbeat:.3f}s written to JSON under '{output_key}'")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python stage2_downbeat_detection.py <folder_path>")
        sys.exit(1)
    run_stage2(sys.argv[1])
