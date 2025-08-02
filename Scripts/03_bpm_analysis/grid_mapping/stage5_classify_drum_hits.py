import json
import numpy as np
import soundfile as sf
from pathlib import Path
import scipy.fftpack
from scipy.stats import entropy
import argparse

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
    "bass_click": (100, 300),
}

def classify_frame_multi(frame, sr, threshold=0.2):
    spectrum = np.abs(scipy.fftpack.fft(frame))[:len(frame)//2]
    freqs = np.fft.rfftfreq(len(frame)*2, d=1/sr)[:len(spectrum)]
    energy_total = np.sum(spectrum)
    labels = [
        label for label, (low, high) in LABEL_BANDS.items()
        if energy_total > 0 and (np.sum(spectrum[(freqs >= low) & (freqs <= high)]) / energy_total) > threshold
    ]
    return labels or ["none"]

def compute_density_and_entropy(labels):
    total_steps = len(labels)
    density = {}
    entropy_vals = {}
    for inst in LABEL_BANDS:
        sequence = [1 if inst in step else 0 for step in labels]
        count = sum(sequence)
        density[f"{inst}_density"] = count / total_steps
        dist = np.array(sequence) + 1e-9
        entropy_vals[f"{inst}_entropy"] = float(entropy(dist, base=2))
    return density, entropy_vals

def compute_downbeat_alignment(labels):
    downbeat_steps = {0, 16, 32, 48}
    return {
        "kick_downbeat_ratio": sum(1 for i in downbeat_steps if "kick" in labels[i]) / 4.0,
        "snare_downbeat_ratio": sum(1 for i in downbeat_steps if "snare" in labels[i]) / 4.0
    }

def compute_swing_ratio(labels):
    ratios = [
        len(labels[odd]) / (len(labels[even]) + len(labels[odd]))
        for even, odd in zip(range(0, 63, 2), range(1, 64, 2))
        if len(labels[even]) + len(labels[odd]) > 0
    ]
    return {"swing_ratio": round(np.mean(ratios), 4) if ratios else 0.0}

def run_stage5(folder_path, threshold=0.2):
    folder = Path(folder_path)
    drums_path = folder / "drums.wav"
    json_path = next((f for f in folder.glob("*.json") if not f.name.startswith(".")), None)

    if not drums_path.exists() or not json_path:
        print("❌ Missing drums.wav or JSON file.")
        return

    y, sr = sf.read(drums_path)
    if y.ndim > 1:
        y = y.mean(axis=1)

    with open(json_path) as f:
        data = json.load(f)

    grid_raw = data.get("beat_grid", "")
    grid = list(map(float, grid_raw.split(","))) if isinstance(grid_raw, str) else grid_raw

    if not grid:
        print("❌ No beat grid found. Run Stage 4 first.")
        return

    print(f"\n🥁 Classifying percussive content per 1/16 beat (multi-label, threshold={threshold})...")
    frame_len = int(0.03 * sr)
    half = frame_len // 2
    labels = []

    for t in grid:
        idx = int(t * sr)
        start = max(0, idx - half)
        end = min(len(y), idx + half)
        frame = y[start:end]
        if len(frame) < frame_len:
            frame = np.pad(frame, (0, frame_len - len(frame)))
        labels.append(classify_frame_multi(frame, sr, threshold))

    data.setdefault("rhythm_pattern", {})["labels"] = labels

    density, entropy_vals = compute_density_and_entropy(labels)
    downbeat_metrics = compute_downbeat_alignment(labels)
    swing_metrics = compute_swing_ratio(labels)

    data.setdefault("rhythm_analysis", {}).update(density)
    data["rhythm_analysis"].update(entropy_vals)
    data["rhythm_analysis"].update(downbeat_metrics)
    data["rhythm_analysis"].update(swing_metrics)

    with open(json_path, "w") as f:
        json.dump(data, f, indent=2, separators=(',', ': '))

    print(f"✅ Multi-label drum classification complete. Labels and rhythm features written to {json_path.name}")
    return labels

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 5: Multi-label percussive classification")
    parser.add_argument("folder", type=str, help="Path to folder with drums.wav and beat grid JSON")
    parser.add_argument("--threshold", type=float, default=0.2, help="Minimum band energy ratio per label")
    args = parser.parse_args()
    run_stage5(args.folder, threshold=args.threshold)
