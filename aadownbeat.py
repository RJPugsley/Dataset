
import os
import sys
import json
import numpy as np
import librosa
from pathlib import Path
from scipy.signal import find_peaks
import difflib
import matplotlib.pyplot as plt

HOP_SIZE = 512
SR = 44100
WINDOW_SEC = 32  # beats before/after
EXPECTED_STEMS = ["drums", "bass", "other", "vocals", "guitar", "piano"]


STYLE_DEFINITIONS = {
    "pre": {
        "silence-entry": {
            "rules": [
                {"feature": "mean_energy_drums", "op": "<", "value": 0.01},
                {"feature": "mean_energy_bass", "op": "<", "value": 0.01},
                {"feature": "mean_energy_other", "op": "<", "value": 0.01},
                {"feature": "mean_energy_vocals", "op": "<", "value": 0.01},
                {"feature": "mean_energy_guitar", "op": "<", "value": 0.01},
                {"feature": "mean_energy_piano", "op": "<", "value": 0.01}
            ]
        },
        "ramp": {
            "rules": [
                {"feature": "slope_drums", "op": ">", "value": 0.001},
                {"feature": "slope_bass", "op": ">", "value": 0.001},
                {"feature": "slope_other", "op": ">", "value": 0.001},
                {"feature": "slope_vocals", "op": ">", "value": 0.001},
                {"feature": "slope_guitar", "op": ">", "value": 0.001},
                {"feature": "slope_piano", "op": ">", "value": 0.001}
            ]
        },
        "vocal-pickup": {
            "rules": [
                {"feature": "pickup_like", "op": "==", "value": True}
            ]
        },
        "hard-start": {
            "rules": [
                {"feature": "first_downbeat_s", "op": "<", "value": 0.5}
            ]
        },
        "breath-entry": {
            "rules": [
                {"feature": "mean_energy_drums", "op": "<", "value": 0.01},
                {"feature": "energy_jump_vocals", "op": ">", "value": 0.05}
            ]
        },
        "rubato-rise": {
            "rules": [
                {"feature": "flatness_score_drums", "op": ">", "value": 0.01},
                {"feature": "flatness_score_vocals", "op": ">", "value": 0.01}
            ]
        },
        "crescendo-drop": {
            "rules": [
                {"feature": "slope_other", "op": ">", "value": 0.001},
                {"feature": "tail_energy_other", "op": "<", "value": 0.05}
            ]
        },
        "phrase-gap": {
            "rules": [
                {"feature": "vocal_valley_count", "op": ">=", "value": 2}
            ]
        }
    },
    "post": {
        "vocal-loop": {
            "rules": [
                {"feature": "post_mean_energy_vocals", "op": ">", "value": 0.05},
                {"feature": "post_spike_count_vocals", "op": ">=", "value": 2}
            ]
        },
        "instrumental-loop": {
            "rules": [
                {"feature": "post_mean_energy_vocals", "op": "<", "value": 0.02},
                {"feature": "post_mean_energy_drums", "op": ">", "value": 0.05},
                {"feature": "post_mean_energy_bass", "op": ">", "value": 0.05},
                {"feature": "post_mean_energy_other", "op": ">", "value": 0.05}
            ]
        },
        "lowpass-loop": {
            "rules": [
                {"feature": "post_mean_energy_bass", "op": "<", "value": 0.03},
                {"feature": "post_slope_other", "op": ">", "value": 0.001}
            ]
        },
        "filtered-loop": {
            "rules": [
                {"feature": "post_highpass_like", "op": "==", "value": True}
            ]
        },
        "static-loop": {
            "rules": [
                {"feature": "post_abs_slope_drums", "op": "<", "value": 0.001},
                {"feature": "post_spike_count_drums", "op": "<=", "value": 1}
            ]
        },
        "syncopated-loop": {
            "rules": [
                {"feature": "post_offgrid_like", "op": "==", "value": True}
            ]
        },
        "full-groove": {
            "rules": [
                {"feature": "post_mean_energy_drums", "op": ">", "value": 0.05},
                {"feature": "post_mean_energy_bass", "op": ">", "value": 0.05},
                {"feature": "post_mean_energy_other", "op": ">", "value": 0.05},
                {"feature": "post_abs_slope_drums", "op": "<", "value": 0.001}
            ]
        }
    },
    "crossover": {
        "fill": {
            "rules": [
                {"feature": "spike_count_drums", "op": ">=", "value": 3},
                {"feature": "spike_density_drums", "op": ">", "value": 0.08}
            ]
        },
        "dry-beat-entry": {
            "rules": [
                {"feature": "mean_energy_drums", "op": "<", "value": 0.03},
                {"feature": "energy_jump_drums", "op": ">", "value": 0.1}
            ]
        },
        "mid-energy-loop": {
            "rules": [
                {"feature": "abs_slope_drums", "op": "<", "value": 0.001},
                {"feature": "spike_count_drums", "op": "<=", "value": 2},
                {"feature": "spike_count_vocals", "op": "<=", "value": 0}
            ]
        },
        "lowpass-ramp": {
            "rules": [
                {"feature": "slope_other", "op": ">", "value": 0.001},
                {"feature": "mean_energy_bass", "op": "<", "value": 0.05}
            ]
        },
        "highpass-ramp": {
            "rules": [
                {"feature": "highpass_like", "op": "==", "value": True}
            ]
        },
        "rubato-rise": {
            "rules": [
                {"feature": "flatness_score_drums", "op": ">", "value": 0.01},
                {"feature": "flatness_score_vocals", "op": ">", "value": 0.01}
            ]
        }
    }
}



def validate_input_path(user_path):
    if os.path.exists(user_path):
        return user_path
    parent = os.path.dirname(user_path)
    target = os.path.basename(user_path)
    if not os.path.exists(parent):
        print(f"[!] Invalid parent directory: {parent}")
        sys.exit(1)
    dirs = [d for d in os.listdir(parent) if os.path.isdir(os.path.join(parent, d))]
    closest = difflib.get_close_matches(target, dirs, n=1)
    print(f"[!] Path not found: {user_path}")
    if closest:
        print(f"    💡 Did you mean:\n    {os.path.join(parent, closest[0])}")
    else:
        print("    💡 No close match found.")
    sys.exit(1)

def compute_energy_curve(wav_data):
    return np.sqrt(librosa.feature.rms(y=wav_data, frame_length=1024, hop_length=HOP_SIZE)[0])

def find_snap_point(energy_curve, target_time_s, bpm, window_s=0.5):
    times = np.arange(len(energy_curve)) * (HOP_SIZE / SR)
    target_idx = np.argmin(np.abs(times - target_time_s))
    frame_window = int(window_s * SR / HOP_SIZE)
    start = max(0, target_idx - frame_window)
    end = min(len(energy_curve), target_idx + frame_window)
    local_times = times[start:end]
    local_energies = energy_curve[start:end]
    peaks, _ = find_peaks(local_energies)
    if not len(peaks):
        return target_time_s
    deltas = np.abs(local_times[peaks] - target_time_s)
    nearest = peaks[np.argmin(deltas)]
    return local_times[nearest]

def extract_stem_energy(track_path, stem_name):
    path = track_path / f"{stem_name}.wav"
    if not path.exists():
        return None
    y, sr = librosa.load(path, sr=SR, mono=True)
    return compute_energy_curve(y)

def extract_bpm(json_path):
    with open(json_path) as f:
        data = json.load(f)
    bpm = data.get("bpm")
    return float(bpm) if bpm else None

def extract_features(window_curves):
    feats = {}
    for stem, curve in window_curves.items():
        arr = np.array(curve)
        feats[f"mean_energy_{stem}"] = float(np.mean(arr))
        feats[f"slope_{stem}"] = float(np.polyfit(np.arange(len(arr)), arr, 1)[0])
        feats[f"spike_count_{stem}"] = int(len(find_peaks(arr, height=np.mean(arr) + 0.05)[0]))
        feats[f"spike_density_{stem}"] = float(feats[f"spike_count_{stem}"] / len(arr))

    # Extra features for extended style definitions
    for stem, arr in window_curves.items():
        arr_np = np.array(arr)
        if len(arr_np) > 2:
            feats[f"energy_jump_{stem}"] = float(arr_np[-1] - arr_np[0])
            feats[f"abs_slope_{stem}"] = float(np.abs(np.polyfit(np.arange(len(arr_np)), arr_np, 1)[0]))
            feats[f"flatness_score_{stem}"] = float(np.std(np.diff(arr_np)))
            feats[f"tail_energy_{stem}"] = float(np.mean(arr_np[-5:])) if len(arr_np) >= 5 else float(np.mean(arr_np))

    # Vocal phrasing features
    if "vocals" in window_curves:
        v = np.array(window_curves["vocals"])
        valleys = np.where((v[1:-1] < v[:-2]) & (v[1:-1] < v[2:]) & (v[1:-1] < 0.05))[0]
        feats["vocal_valley_count"] = int(len(valleys))

    # Special heuristics

    # Improved pickup detection based on onset shape
    if "vocals" in window_curves:
        v = np.array(window_curves["vocals"])
        if len(v) > 4 and v[0] < 0.02 and v[1] > (np.mean(v) + 0.05):
            feats["pickup_like"] = True
        else:
            feats["pickup_like"] = False
    else:
        feats["pickup_like"] = False

    # Highpass-like heuristic
    feats["highpass_like"] = feats.get("slope_bass", 0) < -0.001 and feats.get("mean_energy_bass", 0) < 0.02

    if "drums" in window_curves:
        y = np.array(window_curves["drums"])
        onsets = np.where(
            (y[1:-1] > y[:-2]) & (y[1:-1] > y[2:]) & (y[1:-1] > np.mean(y) + 0.05)
        )[0]
        beat_spacing = len(y) / 32  # assuming 32 beats in window
        offgrid_hits = [
            o for o in onsets if np.abs((o % beat_spacing) - beat_spacing / 2) < beat_spacing * 0.2
        ]
        feats["offgrid_like"] = len(offgrid_hits) >= 3
    else:
        feats["offgrid_like"] = False


    return feats



STYLE_DEFINITIONS = {
    "pre": {
        "silence-entry": {
            "rules": [
                {"feature": "mean_energy_drums", "op": "<", "value": 0.01},
                {"feature": "mean_energy_bass", "op": "<", "value": 0.01},
                {"feature": "mean_energy_other", "op": "<", "value": 0.01},
                {"feature": "mean_energy_vocals", "op": "<", "value": 0.01},
                {"feature": "mean_energy_guitar", "op": "<", "value": 0.01},
                {"feature": "mean_energy_piano", "op": "<", "value": 0.01}
            ]
        },
        "ramp": {
            "rules": [
                {"feature": "slope_drums", "op": ">", "value": 0.001},
                {"feature": "slope_bass", "op": ">", "value": 0.001},
                {"feature": "slope_other", "op": ">", "value": 0.001},
                {"feature": "slope_vocals", "op": ">", "value": 0.001},
                {"feature": "slope_guitar", "op": ">", "value": 0.001},
                {"feature": "slope_piano", "op": ">", "value": 0.001}
            ]
        },
        "pickup-vocal": {
            "rules": [
                {"feature": "pickup_like", "op": "==", "value": True}
            ]
        },
        "hard-start": {
            "rules": [
                {"feature": "first_downbeat_s", "op": "<", "value": 0.5}
            ]
        },
        "breath-entry": {
            "rules": [
                {"feature": "mean_energy_drums", "op": "<", "value": 0.01},
                {"feature": "energy_jump_vocals", "op": ">", "value": 0.05}
            ]
        },
        "rubato-rise": {
            "rules": [
                {"feature": "flatness_score_drums", "op": ">", "value": 0.01},
                {"feature": "flatness_score_vocals", "op": ">", "value": 0.01}
            ]
        },
        "crescendo-drop": {
            "rules": [
                {"feature": "slope_other", "op": ">", "value": 0.001},
                {"feature": "tail_energy_other", "op": "<", "value": 0.05}
            ]
        }
    },
    "post": {
        "full-groove": {
            "rules": [
            {"feature": "post_mean_energy_drums", "op": ">", "value": 0.05},
            {"feature": "post_mean_energy_bass", "op": ">", "value": 0.05},
            {"feature": "post_mean_energy_other", "op": ">", "value": 0.01},
            {"feature": "post_abs_slope_drums", "op": "<", "value": 0.001}
            ]
        },
        "vocal-loop": {
            "rules": [
                {"feature": "post_mean_energy_vocals", "op": ">", "value": 0.05},
                {"feature": "post_spike_count_vocals", "op": ">=", "value": 2}
            ]
        },
        "instrumental-loop": {
            "rules": [
                {"feature": "post_mean_energy_vocals", "op": "<", "value": 0.02},
                {"feature": "post_mean_energy_drums", "op": ">", "value": 0.05},
                {"feature": "post_mean_energy_bass", "op": ">", "value": 0.05},
                {"feature": "post_mean_energy_other", "op": ">", "value": 0.05}
            ]
        },
        "lowpass-loop": {
            "rules": [
                {"feature": "post_mean_energy_bass", "op": "<", "value": 0.03},
                {"feature": "post_slope_other", "op": ">", "value": 0.001}
            ]
        },
        "filtered-loop": {
            "rules": [
                {"feature": "post_highpass_like", "op": "==", "value": True}
            ]
        },
        "static-loop": {
            "rules": [
                {"feature": "post_abs_slope_drums", "op": "<", "value": 0.001},
                {"feature": "post_spike_count_drums", "op": "<=", "value": 1}
            ]
        },
        "syncopated-loop": {
            "rules": [
                {"feature": "post_offgrid_like", "op": "==", "value": True}
            ]
        }
    },
    "crossover": {
        "fill": {
            "rules": [
                {"feature": "spike_count_drums", "op": ">=", "value": 3},
                {"feature": "spike_density_drums", "op": ">", "value": 0.08}
            ]
        },
        "dry-beat-entry": {
        "rules": [
            {"feature": "mean_energy_drums", "op": "<", "value": 0.03},
            {"feature": "energy_jump_drums", "op": ">", "value": 0.1},
            {"feature": "mean_energy_vocals", "op": "<", "value": 0.05}
        ]
    },
        "mid-energy-loop": {
            "rules": [
                {"feature": "abs_slope_drums", "op": "<", "value": 0.001},
                {"feature": "spike_count_drums", "op": "<=", "value": 2},
                {"feature": "spike_count_vocals", "op": "<=", "value": 0}
            ]
        },
        "lowpass-ramp": {
            "rules": [
                {"feature": "slope_other", "op": ">", "value": 0.001},
                {"feature": "mean_energy_bass", "op": "<", "value": 0.05}
            ]
        },
        "highpass-ramp": {
            "rules": [
                {"feature": "highpass_like", "op": "==", "value": True}
            ]
        },
        "rubato-rise": {
            "rules": [
                {"feature": "flatness_score_drums", "op": ">", "value": 0.01},
                {"feature": "flatness_score_vocals", "op": ">", "value": 0.01}
            ]
        }
    }
}



def evaluate_styles(feats, group):
    matched = []
    for style, ruleset in STYLE_DEFINITIONS.get(group, {}).items():
        if all(eval_rule(feats, r) for r in ruleset["rules"]):
            matched.append(style)
    return matched

def eval_rule(feats, rule):
    f = rule["feature"]
    op = rule["op"]
    val = rule["value"]
    if f not in feats:
        return False
    x = feats[f]
    return eval(f"x {op} val") if op in [">", "<", "==", ">=", "<="] else False

def process_track(track_path):
    json_path = next(track_path.glob("*.json"), None)
    if not json_path:
        print(f"[!] No JSON found in {track_path}")
        return

    with open(json_path) as f:
        data = json.load(f)

    bpm = extract_bpm(json_path)
    if not bpm:
        print(f"[!] BPM missing in {json_path.name}")
        return

    stem_curves = {}
    for stem in EXPECTED_STEMS:
        curve = extract_stem_energy(track_path, stem)
        if curve is not None:
            stem_curves[stem] = curve

    if not stem_curves:
        print(f"[!] No valid stems found.")
        return

    # Warn if any expected stems are missing
    required_stems = ["drums", "bass", "other", "vocals", "guitar", "piano"]
    for rs in required_stems:
        if rs not in stem_curves:
            print(f"[!] Missing expected stem: {rs}. Some style rules may fail.")

    base_curve = stem_curves.get("drums", next(iter(stem_curves.values())))
    hop_s = HOP_SIZE / SR
    beat_frames = int(round((60 / bpm) / hop_s))

    for target_field in ["first_downbeat", "loop_start"]:
        manual_time = data.get(target_field)
        if manual_time is None:
            continue
        snap_s = find_snap_point(base_curve, manual_time, bpm)
        data[target_field + "_snapped"] = snap_s
        snap_frame = int(round(snap_s / hop_s))
        window_frames = 32 * beat_frames
        start_frame = max(0, snap_frame - window_frames)
        end_frame = snap_frame
        window = {
            stem: curve[start_frame:end_frame].tolist()
            for stem, curve in stem_curves.items()
        }
        feats = extract_features(window)
        data.setdefault("features", {})[target_field] = feats
        if target_field == "first_downbeat":
            matches = list(set(evaluate_styles(feats, "pre") + evaluate_styles(feats, "crossover")))
        else:
            matches = evaluate_styles(feats, "crossover")

        # Also extract post window (32 beats after downbeat)
        post_start_frame = snap_frame
        post_end_frame = min(len(base_curve), snap_frame + window_frames)
        post_window = {
            stem: curve[post_start_frame:post_end_frame].tolist()
            for stem, curve in stem_curves.items()
        }
        post_feats = extract_features(post_window)
        # Prefix post_feats keys to distinguish them
        post_feats = {f"post_{k}": v for k, v in post_feats.items()}
        data["features"][target_field].update(post_feats)
        post_matches = evaluate_styles(post_feats, "post")
        if target_field == "loop_start":
            data["loop_post_style"] = post_matches


        print(f"[+] Processing {target_field} at {manual_time:.2f}s → snapped to {snap_s:.2f}s")
        print("    ↪ Extracted pre-window features:")
        for k, v in feats.items():
            print(f"      {k}: {v:.4f}" if isinstance(v, float) else f"      {k}: {v}")
        print(f"    ↪ Matched pre-window styles: {matches}")

        print("    ↪ Extracted post-window features:")
        for k, v in post_feats.items():
            print(f"      {k}: {v:.4f}" if isinstance(v, float) else f"      {k}: {v}")
        print(f"    ↪ Matched post-window loop styles: {post_matches}")

        if target_field == "first_downbeat":
            data["downbeat_style"] = matches
        elif target_field == "loop_start":
            data["loop_intro_style"] = matches

    with open(json_path, "w") as f:
        json.dump(data, f, indent=4)
        
    print("\n==== SUMMARY ====")
    print(f"First Downbeat Pre  → {data.get('downbeat_style', [])}")
    print(f"First Downbeat Post → {evaluate_styles({k: v for k, v in data['features'].get('first_downbeat', {}).items() if k.startswith('post_')}, 'post')}")
    print(f"Loop Start Pre      → {data.get('loop_intro_style', [])}")
    print(f"Loop Start Post     → {data.get('loop_post_style', [])}")
    data["track_path"] = str(json_path.parent)
    data["energy_curves"] = stem_curves
    
    return data

def save_overlay_stem_plots(data, dpi=400):
    # Save in the same folder as the JSON
    output_dir = Path(data["track_path"])
    output_dir.mkdir(parents=True, exist_ok=True)

    stems = ["drums", "bass", "vocals", "guitar", "piano", "other"]
    curves = data.get("energy_curves", {})
    frame_rate = data.get("frame_rate", 100)
    window_frames = data.get("window_frames", 400)

    def extract_window(arr, center_s, window_frames, frame_rate):
        center_idx = int(center_s * frame_rate)
        start = max(center_idx - window_frames, 0)
        end = center_idx + window_frames
        window = np.zeros(2 * window_frames)
        available = arr[start:end]
        window[:len(available)] = available
        return window

    def plot_overlay(center_s, title, filename):
        center_idx = int(center_s * frame_rate)
        start = max(center_idx - window_frames, 0)
        end = center_idx + window_frames

        x = np.arange(start, end) / frame_rate
        x = x - center_s  # shift x-axis so center_s is at t=0

        plt.figure(figsize=(20, 6))
        for stem in stems:
            if stem in curves:
                arr = curves[stem]
                if len(arr) > start:
                    y = arr[start:end]
                    x_trimmed = x[:len(y)]  # truncate x to match y if y is short
                    plt.plot(x_trimmed, y, label=stem)

        plt.axvline(x=0, color='black', linestyle='--', linewidth=1, label="t=0")
        plt.title(title)
        plt.xlabel("Time (s)")
        plt.ylabel("Energy")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=dpi)
        plt.close()

    # Plot each key window
    for label in ["first_downbeat", "loop_start"]:
        if label in data["features"]:
            center = data.get(f"{label}_snapped")
            plot_overlay(center, f"{label.replace('_', ' ').title()} – Pre/Post", f"{label}_overlay.png")
            
def main():
    if len(sys.argv) < 2:
        print("[!] Usage: python script.py <track-folder>")
        sys.exit(1)
    
    base = Path(validate_input_path(sys.argv[1]))
    if base.is_file():
        base = base.parent

    if base.is_dir():
        data = process_track(base)  # ← make sure process_track() returns `data`
        save_overlay_stem_plots(data, dpi=400)
        
if __name__ == "__main__":
    main()
