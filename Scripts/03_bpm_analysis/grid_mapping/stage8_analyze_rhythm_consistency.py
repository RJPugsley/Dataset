import csv
import json
import numpy as np
import soundfile as sf
from pathlib import Path
import scipy.fftpack
import argparse
from decimal import Decimal, getcontext

getcontext().prec = 10

LABEL_BANDS = {
    "sub": (20, 80),
    "kick": (30, 180),
    "snare": (100, 450),
    "snare_roll": (100, 450),
    "clap": (1000, 3000),
    "rim": (1500, 4000),
    "tom": (80, 180),
    "hihat": (5000, 12000),
    "ride": (4000, 10000),
    "crash": (2000, 8000),
    "stab": (300, 900),
    "bell": (700, 2000),
    "woodblock": (800, 1500),
    "glitch": (2000, 12000),
    "fx": (8500, 14000),
    "vocal": (200, 2000),
    "bass_click": (100, 300)
}

LABELS = list(LABEL_BANDS.keys())

def classify_frame_multi(frame, sr, threshold=0.2):
    spectrum = np.abs(scipy.fftpack.fft(frame))[:len(frame)//2]
    freqs = np.fft.rfftfreq(len(frame)*2, d=1/sr)[:len(spectrum)]
    energy_total = np.sum(spectrum)
    labels = [
        label for label, (low, high) in LABEL_BANDS.items()
        if energy_total > 0 and (np.sum(spectrum[(freqs >= low) & (freqs <= high)]) / energy_total) > threshold
    ]
    return labels if labels else ["none"]

def block_match_score(ref_block, test_block):
    match, total = 0, 0
    for ref_slot, test_slot in zip(ref_block, test_block):
        ref_set, test_set = set(ref_slot), set(test_slot)
        match += len(ref_set & test_set)
        total += len(ref_set | test_set)
    return round(match / total, 4) if total > 0 else 0.0

def compute_instrument_consistency(labels):
    consistency = {label: [] for label in LABELS}
    for i in range(0, len(labels) - 63, 64):
        block = labels[i:i+64]
        for label in LABELS:
            active = sum(1 for slot in block if label in slot)
            consistency[label].append(active / 64)
    avg_consistency = {k: round(np.mean(v), 4) for k, v in consistency.items()}
    return consistency, avg_consistency

def compute_grid_structure_ratios(labels):
    quant_slots = {0, 4, 8, 12}
    quantized_hits = 0
    syncopated_hits = 0
    for i, slot in enumerate(labels):
        if i % 16 in quant_slots:
            quantized_hits += len(slot)
        else:
            syncopated_hits += len(slot)
    total_hits = quantized_hits + syncopated_hits or 1
    return {
        "quantized_hit_ratio": round(quantized_hits / total_hits, 4),
        "syncopated_hit_ratio": round(syncopated_hits / total_hits, 4)
    }

def compute_offset_histograms(labels):
    kick_hist = [0] * 16
    snare_hist = [0] * 16
    for i, slot in enumerate(labels):
        pos = i % 16
        if "kick" in slot:
            kick_hist[pos] += 1
        if "snare" in slot:
            snare_hist[pos] += 1
    total_kicks = sum(kick_hist) or 1
    total_snares = sum(snare_hist) or 1
    kick_hist = [round(x / total_kicks, 4) for x in kick_hist]
    snare_hist = [round(x / total_snares, 4) for x in snare_hist]
    return kick_hist, snare_hist

def run_stage8(folder_path, threshold=0.2, drift_sec=0.025):
    folder = Path(folder_path)
    drums_path = folder / "drums.wav"
    json_path = next((f for f in folder.glob("*.json") if f.is_file()), None)

    if not drums_path.exists() or not json_path:
        print("❌ Missing drums.wav or JSON file.")
        return

    with open(json_path) as f:
        data = json.load(f)

    bpm = data.get("bpm")
    first_downbeat = data.get("first_downbeat")
    if bpm is None or first_downbeat is None:
        print("❌ Missing BPM or first_downbeat.")
        return

    try:
        bpm = float(bpm)
        if bpm <= 0:
            raise ValueError
    except (ValueError, TypeError):
        print("❌ Invalid BPM value. Must be a positive number.")
        return

    y, sr = sf.read(drums_path)
    if y.ndim > 1:
        y = y.mean(axis=1)

    slot_dur = 60 / (bpm * 4)
    track_len = len(y) / sr
    beat_grid = [round(first_downbeat + i * slot_dur, 5) for i in range(int((track_len - first_downbeat) // slot_dur))]

    frame_len = int(0.03 * sr)
    half = frame_len // 2
    drift_range = int(sr * drift_sec)
    step = int(sr * 0.005)

    labels = []
    for t in beat_grid:
        best_score, best_labels = 0, ["none"]
        for offset in range(-drift_range, drift_range + 1, step):
            idx = int(t * sr) + offset
            frame = y[max(0, idx - half):min(len(y), idx + half)]
            if len(frame) < frame_len:
                frame = np.pad(frame, (0, frame_len - len(frame)))
            label_set = classify_frame_multi(frame, sr, threshold)
            if len(label_set) > best_score:
                best_score = len(label_set)
                best_labels = label_set
        labels.append(best_labels)

    base_loop = labels[:64]
    scores = [block_match_score(base_loop, labels[i:i+64]) for i in range(0, len(labels) - 63, 64)]
    loop_repeat_score = round(np.mean(scores), 4)

    consistency_blocks, consistency_avg = compute_instrument_consistency(labels)
    presence_map = {label: sum(1 for slot in labels if label in slot) for label in LABELS}
    kick_hist, snare_hist = compute_offset_histograms(labels)
    grid_ratios = compute_grid_structure_ratios(labels)

    result = {
        "loop_repeat_score": loop_repeat_score,
        **grid_ratios,
        "instrument_consistency_blocks": consistency_blocks,
        "instrument_consistency_avg": consistency_avg,
        "instrument_presence_map": presence_map,
        "kick_offset_histogram": kick_hist,
        "snare_offset_histogram": snare_hist
    }

    data["rhythm_analysis"] = result

    # 🔁 Overwrite normalized values and force decimal (not scientific) formatting
    track_duration = len(y) / sr
    loop_start = data.get("loop_start")
    first_downbeat = data.get("first_downbeat")

    if isinstance(loop_start, (int, float)):
        normalized = loop_start / track_duration
        formatted = Decimal(f"{normalized:.6f}")
        data["loop_start_normalized"] = float(formatted)

    if isinstance(first_downbeat, (int, float)):
        normalized = first_downbeat / track_duration
        formatted = Decimal(f"{normalized:.6f}")
        data["first_downbeat_normalized"] = float(formatted)

    with open(json_path, "w") as f:
        json.dump(data, f, indent=2, separators=(',', ': '))

    print("✅ Rhythm analysis complete. Results written to JSON.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 8: Analyze rhythm consistency and loop structure")
    parser.add_argument("folder", type=str, help="Path to folder with JSON and drums.wav")
    args = parser.parse_args()
    run_stage8(args.folder)
