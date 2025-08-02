import json
import argparse
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.stats import entropy

def flatten_multilabel(labels):
    return [label for step in labels for label in step if label != "none"]

def compute_density(labels):
    active = sum(1 for step in labels if any(l != "none" for l in step))
    return round(active / len(labels), 3)

def jaccard_similarity(set_a, set_b):
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union else 0.0

def compute_repetition(labels, bars=4):
    if len(labels) % bars != 0:
        print("⚠️ Label count not divisible by bar count.")
        return 0.0
    bar_len = len(labels) // bars
    chunks = [labels[i * bar_len:(i + 1) * bar_len] for i in range(bars)]
    base = [{frozenset(slot) for slot in bar} for bar in chunks]
    similarities = [
        np.mean([jaccard_similarity(a, b) for a, b in zip(base[0], bar)])
        for bar in base[1:]
    ]
    return round(np.mean(similarities), 3)

def compute_kick_snare_ratio(flat_labels):
    count = Counter(flat_labels)
    kicks = count.get("kick", 0)
    snares = count.get("snare", 0)
    if kicks == 0 and snares == 0:
        return 0.0
    elif snares == 0:
        return float('inf')
    return round(kicks / snares, 2)

def compute_dominant_label(flat_labels):
    count = Counter(flat_labels)
    return count.most_common(1)[0][0] if count else "none"

def compute_label_entropy(flat_labels):
    count = Counter(flat_labels)
    return round(entropy(list(count.values()), base=2), 3)

def compute_avg_labels_per_step(labels):
    return round(np.mean([len([l for l in step if l != "none"]) for step in labels]), 3)

def run_stage6(folder_path):
    print(f"\n🚀 Starting Stage 6: Analyze Loop Confidence")
    folder = Path(folder_path)

    json_path = next((f for f in folder.glob("*.json") if not f.name.startswith(".")), None)
    if not json_path:
        print("❌ JSON file not found.")
        return

    with open(json_path) as f:
        data = json.load(f)

    rhythm = data.get("rhythm_pattern")
    if not rhythm or "labels" not in rhythm:
        print("❌ Missing rhythm pattern or labels.")
        return

    labels = rhythm["labels"]
    bars = rhythm.get("bars", 4)
    flat_labels = flatten_multilabel(labels)

    print("📈 Calculating confidence metrics...")
    rhythm["confidence"] = {
        "density": compute_density(labels),
        "repetition": compute_repetition(labels, bars),
        "kick_snare_ratio": compute_kick_snare_ratio(flat_labels),
        "dominant_label": compute_dominant_label(flat_labels),
        "label_entropy": compute_label_entropy(flat_labels),
        "avg_labels_per_step": compute_avg_labels_per_step(labels)
    }

    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)

    for k, v in rhythm["confidence"].items():
        print(f"   • {k}: {v}")
    print("🏁 Stage 6 complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 6: Analyze loop confidence (multi-label)")
    parser.add_argument("folder", type=str, help="Path to folder with rhythm JSON")
    args = parser.parse_args()

    run_stage6(args.folder)
