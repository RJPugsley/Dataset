#!/usr/bin/env python3
# 04_scan.py — Feature extraction and style tagging using raw_data.json

import json, argparse
from pathlib import Path
import numpy as np
from scipy.signal import find_peaks

# --------------- Snapping Logic ---------------

def find_snap_point(energy_curve, target_time_s, bpm, window_s=0.5):
    if not energy_curve or bpm is None:
        return target_time_s

    frame_rate = 20.0
    window_frames = int(window_s * frame_rate)
    target_frame = int(target_time_s * frame_rate)
    lo = max(0, target_frame - window_frames)
    hi = min(len(energy_curve), target_frame + window_frames)

    segment = np.array(energy_curve[lo:hi])
    if len(segment) < 3:
        return target_time_s

    peaks, _ = find_peaks(segment, distance=1)
    if not len(peaks):
        return target_time_s

    peak_frames = peaks + lo
    peak_times = [p / frame_rate for p in peak_frames]
    snapped = min(peak_times, key=lambda x: abs(x - target_time_s))
    return snapped

# --------------- Rhythmic Stem Selection ---------------

def select_best_rhythmic_stem(raw, bpm, loop_start, frame_rate=20.0, beat_window=32, tolerance=0.08):
    beat_interval = 60.0 / bpm
    num_frames = max(len(v) for v in raw["energy_curves"].values())
    max_time = num_frames / frame_rate

    beat_times = [
        loop_start + i * beat_interval
        for i in range(-beat_window, beat_window + 1)
        if 0 <= (loop_start + i * beat_interval) <= max_time
    ]

    stem_priority = ["drums", "guitar", "piano", "other", "vocals"]
    best_stem = None
    best_score = -1

    for stem in stem_priority:
        if stem not in raw["energy_curves"]:
            continue
        curve = np.array([v for _, v in raw["energy_curves"][stem]])
        if np.mean(curve) < 0.001:
            continue
        peaks, _ = find_peaks(curve, distance=int(0.1 * frame_rate))  # 100ms
        peak_times = peaks / frame_rate

        score = 0.0
        for bt in beat_times:
            nearby = [pt for pt in peak_times if abs(pt - bt) <= tolerance]
            if nearby:
                idx = int(nearby[0] * frame_rate)
                if 0 <= idx < len(curve):
                    score += curve[idx]

        if score > best_score:
            best_score = score
            best_stem = stem

        # Only break if this is the highest-priority stem with energy
        if best_stem == stem:
            break

    return best_stem

# --------------- Style Rule Evaluation ---------------

def load_definitions(def_path):
    with open(def_path, "r") as f:
        code = compile(f.read(), def_path.name, 'exec')
        scope = {}
        exec(code, scope)
    return scope.get("STYLE_DEFINITIONS", {})

def apply_rule(rule, features):
    feature = rule["feature"]
    op = rule["op"]
    value = rule["value"]
    mode = rule.get("mode", "abs")

    feature_val = features.get(feature)
    if feature_val is None:
        return False

    if mode == "pct":
        feature_val *= 100.0

    try:
        return eval(f"{feature_val} {op} {value}")
    except Exception:
        return False

def evaluate_style(ruleset, features):
    matched = []
    for position, styles in ruleset.items():
        for name, style_def in styles.items():
            if all(apply_rule(r, features) for r in style_def["rules"]):
                matched.append((position, name))
    return matched

def extract_features(energy, start_s, end_s, sr=20.0):
    idx_start = int(start_s * sr)
    idx_end = int(end_s * sr)
    segment = energy[idx_start:idx_end]

    if not segment or len(segment) < 2:
        return {}

    seg = np.array(segment)
    slope = (seg[-1] - seg[0]) / len(seg)
    abs_slope = np.abs(slope)
    jump = seg[1] - seg[0]
    mean = float(np.mean(seg))
    std = float(np.std(seg))
    spike_count = int(np.sum((seg[1:] - seg[:-1]) > std))
    flatness = float(np.mean(np.abs(np.diff(seg))))

    return {
        "mean_energy": mean,
        "std_energy": std,
        "slope": slope,
        "abs_slope": abs_slope,
        "energy_jump": jump,
        "spike_count": spike_count,
        "flatness_score": flatness
    }

def load_jsons(folder: Path):
    meta_json = sorted(folder.glob("*.json"))[0]
    raw_json = folder / "raw_data.json"
    with open(meta_json, "r") as f: meta = json.load(f)
    with open(raw_json, "r") as f: raw = json.load(f)
    return meta_json, meta, raw

# --------------- Main Processing Function ---------------

def process_folder(folder: Path, defs_path: Path, win_beats=32):
    meta_path, meta, raw = load_jsons(folder)
    bpm = raw.get("bpm", {}).get("value")
    if not bpm:
        print(f"[!] No BPM found in {meta_path.name}")
        return

    beat_len = 60.0 / bpm
    win_len_s = beat_len * win_beats
    ruleset = load_definitions(defs_path)

    loop_start = meta.get("loop_start")
    if not isinstance(loop_start, (float, int)):
        print(f"[!] loop_start missing — skipping")
        return

    snapping_stem = select_best_rhythmic_stem(raw, bpm, loop_start)
    if not snapping_stem:
        print(f"[!] No valid rhythmic stem found — skipping")
        return

    print(f"[✓] Snapping stem: {snapping_stem}")
    out_events = []

    for target_field in ["first_downbeat", "loop_start"]:
        t0 = meta.get(target_field)
        if not isinstance(t0, (float, int)):
            continue

        base_curve = [v for _, v in raw["energy_curves"].get(snapping_stem, [])]
        snapped = find_snap_point(base_curve, t0, bpm)
        meta[target_field + "_snapped"] = snapped
        meta[target_field + "_snapping_stem"] = snapping_stem

        pre_start, pre_end = max(0, snapped - win_len_s), snapped
        post_start, post_end = snapped, snapped + win_len_s

        for stem, tv_pairs in raw["energy_curves"].items():
            curve = [v for _, v in tv_pairs]
            feats_pre = extract_features(curve, pre_start, pre_end)
            feats_post = extract_features(curve, post_start, post_end)

            pre_feats = {f"pre_{k}_{stem}": v for k, v in feats_pre.items()}
            post_feats = {f"post_{k}_{stem}": v for k, v in feats_post.items()}
            all_feats = {**pre_feats, **post_feats}

            matched = evaluate_style(ruleset, all_feats)
            for pos, label in matched:
                out_events.append({
                    "event": label,
                    "start": float(pre_start if pos == "pre" else post_start),
                    "end": float(pre_end if pos == "pre" else post_end),
                    "position": pos,
                    "stem": stem,
                    "target": target_field
                })

    meta["feature_events"] = out_events
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[✓] {meta_path.name} updated with {len(out_events)} events.")
    return out_events

def main():
    parser = argparse.ArgumentParser(description="Scan for feature events using raw_data.json")
    parser.add_argument("folder", help="Folder containing raw_data.json + track JSON")
    parser.add_argument("--defs", type=str, default="03_definitions.py", help="Path to definitions file")
    args = parser.parse_args()
    process_folder(Path(args.folder), Path(args.defs))

if __name__ == "__main__":
    main()
