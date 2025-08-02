import os
import sys
import json
import csv
import numpy as np
import soundfile as sf
import subprocess
from pathlib import Path
from scipy.signal import butter, sosfilt
from scipy.ndimage import maximum_filter1d

FFMPEG_BIN = "/Users/djf2/Desktop/AppsBinsLibs/Binary/ffmpeg"

def load_and_filter_kick(drum_path, sr=44100):
    temp_wav = "_temp_kick.wav"
    subprocess.run([FFMPEG_BIN, "-i", drum_path, "-ac", "1", "-ar", str(sr), "-y", temp_wav],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    y, sr = sf.read(temp_wav)
    os.remove(temp_wav)
    sos = butter(4, [30, 180], btype='bandpass', fs=sr, output='sos')
    return sosfilt(sos, y), sr

def compute_energy_and_onsets(y, sr, frame_size=1024, hop_size=512):
    energy = np.array([
        np.sum(np.square(y[i:i + frame_size]))
        for i in range(0, len(y) - frame_size, hop_size)
    ])
    energy /= np.max(energy) + 1e-8
    peak_mask = energy == maximum_filter1d(energy, size=5)
    peak_idxs = np.where(peak_mask & (energy > 0.05))[0]
    onset_times = peak_idxs * hop_size / sr
    return energy, onset_times

def filter_onsets_by_energy_mean(onsets, energy, sr, hop_size=512, threshold=0.7):
    times = np.arange(len(energy)) * hop_size / sr
    peak_energies = [(onset, energy[np.argmin(np.abs(times - onset))]) for onset in onsets]
    mean_energy = np.mean([e for _, e in peak_energies])
    return [(o, e) for o, e in peak_energies if e >= mean_energy * threshold], mean_energy

def compute_kick_pattern(onsets, start_time, bpm):
    beat_dur = 60.0 / bpm
    slot_dur = beat_dur / 4
    return [1 if any(abs(o - (start_time + i * slot_dur)) <= 0.05 for o in onsets) else 0 for i in range(64)]

def compute_spacing_std(pattern):
    idx = [i for i, v in enumerate(pattern) if v == 1]
    return np.std(np.diff(idx)) if len(idx) > 1 else 10.0

def compute_fuzzy_anchor_score(index, onset_time, pattern, all_onsets, pattern_csv_path, bpm, tolerance=0.08):
    with open(pattern_csv_path) as f:
        pattern_rows = [(float(r[0]), list(map(int, r[1:]))) for r in csv.reader(f)]

    loop_dur = 16 * (60.0 / bpm)
    match_count, n = 0, 1
    while True:
        expected_time = onset_time + n * loop_dur
        if expected_time > all_onsets[-1]:
            break
        actual_time, cand_pattern = min(pattern_rows, key=lambda x: abs(x[0] - expected_time))
        if abs(actual_time - expected_time) <= tolerance:
            diff = sum(abs(a - b) for a, b in zip(pattern, cand_pattern))
            if diff <= 10:
                match_count += 1
                print(f"Fuzzy match at {actual_time:.3f}s (diff={diff})")
        n += 1
    return match_count

def score_loop_candidates(onset_energy_list, kick_patterns, bpm, raw_onsets, pattern_csv_path):
    max_e = max(e for _, e in onset_energy_list)
    last_t = onset_energy_list[-1][0]
    results = []

    for i, (t, e) in enumerate(onset_energy_list):
        p = kick_patterns[i]
        anchor = compute_fuzzy_anchor_score(i, t, p, raw_onsets, pattern_csv_path, bpm)
        score = (
            0.40 * anchor +
            0.20 * (1.0 - compute_spacing_std(p) / 10.0) +
            0.15 * (e / max_e) +
            0.15 * (1.0 if p[0] == 1 else 0.0) +
            0.10 * (1.0 - (t / last_t))
        )
        results.append({"index": i, "onset_time": t, "score": score, "anchor_score": anchor})
        print(f"Candidate at {t:.3f}s | Score: {score:.4f} | Anchor: {anchor}\n")

    return sorted(results, key=lambda x: -x["score"])

def run_stage3(folder_path):
    folder = Path(folder_path)
    drum_path = folder / "drums.wav"
    json_path = folder / f"{folder.name}.json"

    if not drum_path.exists() or not json_path.exists():
        print("❌ Missing drums.wav or JSON file.")
        return

    with open(json_path) as f:
        metadata = json.load(f)

    bpm = float(metadata.get("bpm", 0))
    if bpm <= 0:
        print("❌ Invalid or missing BPM in JSON.")
        return

    y_kick, sr = load_and_filter_kick(drum_path)
    energy_env, raw_onsets = compute_energy_and_onsets(y_kick, sr)
    onset_energy_list, _ = filter_onsets_by_energy_mean(raw_onsets, energy_env, sr)
    kick_patterns = [compute_kick_pattern(raw_onsets, t, bpm) for t, _ in onset_energy_list]

    csv_path = folder / "kick_patterns.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        for t, p in zip(raw_onsets, [compute_kick_pattern(raw_onsets, t, bpm) for t in raw_onsets]):
            writer.writerow([f"{t:.5f}"] + p)

    results = score_loop_candidates(onset_energy_list, kick_patterns, bpm, raw_onsets, csv_path)
    best_loop = results[0]
    loop_start = round(best_loop["onset_time"], 5)

    with open(json_path, "r") as f:
        data = json.load(f)
    data["loop_start"] = loop_start

    if data.get("track_length", 0) > 0:
        data["loop_start_normalized"] = round(loop_start / data["track_length"], 6)

    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"🎯 Loop start = {loop_start}s written to JSON")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python stage3_define_loop.py /path/to/folder")
        sys.exit(1)
    run_stage3(sys.argv[1])
