# stage5_classify_drum_hits.py — general, confidence-weighted classification
import json, argparse
import numpy as np
import soundfile as sf
from pathlib import Path
from scipy.stats import entropy

# ===== Band map (general, no genre logic) =====
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

# Preferred UI order for common kit parts; extras appended after
DEFAULT_ORDER = [
    "kick","snare","clap","hihat","ride","crash","rim","tom",
    "percussion","sub","bass_click","stab","woodblock","bell","glitch","fx","vocal"
]

# ===== Tunables (general) =====
WINDOW_PAD = 0.45      # fraction of tick_duration on each side of the slot center
MIN_LABEL = 0.20       # include in human-readable labels list if conf >= MIN_LABEL
CENTER_HANN = True     # reduce spectral leakage
SR_TARGET = None       # keep native SR if None

def _pick_audio_file(folder: Path):
    for name in ["drums.wav","drum.wav","drums.flac","mix.wav","mix.flac","source.wav","source.flac"]:
        p = folder / name
        if p.exists(): return p
    for f in folder.iterdir():
        if f.suffix.lower() in [".wav",".flac",".mp3",".m4a"]:
            return f
    return None

def _band_ratio(spec, freqs, lo, hi):
    sel = (freqs >= lo) & (freqs <= hi)
    band = spec[sel]
    total = spec.sum() + 1e-12
    return float(band.sum() / total)

def classify_frame_confidences(frame, sr):
    """Return per-label confidences in [0,1] using correct rFFT + rfftfreq."""
    n = len(frame)
    if CENTER_HANN:
        frame = frame * np.hanning(n)
    spec = np.abs(np.fft.rfft(frame))
    freqs = np.fft.rfftfreq(n, d=1.0/sr)
    confs = {}
    for lab, (lo, hi) in LABEL_BANDS.items():
        confs[lab] = _band_ratio(spec, freqs, lo, hi)
    return confs

def compute_weighted_metrics(conf_tracks, slot_roles, bar_len_ticks):
    total_slots = len(next(iter(conf_tracks.values()))) if conf_tracks else 0
    even_idx = list(range(0, total_slots, 2))
    odd_idx  = list(range(1, total_slots, 2))
    downbeats = (slot_roles or {}).get("downbeats") or list(range(0, total_slots, bar_len_ticks or 16))

    metrics, histograms = {}, {}

    def _mean_at(idx_list, arr):
        if not idx_list: return 0.0
        return float(np.mean([arr[i] for i in idx_list]))

    for lab, arr in conf_tracks.items():
        arr = np.asarray(arr, float)
        overall = float(np.mean(arr)) if len(arr) else 0.0
        metrics[f"{lab}_density_w"] = overall

        p_hit = np.clip(overall, 1e-9, 1 - 1e-9)
        metrics[f"{lab}_entropy"] = float(entropy([p_hit, 1 - p_hit], base=2))

        db = _mean_at(downbeats, arr)
        metrics[f"{lab}_downbeat_emphasis"] = float(db / (overall + 1e-12))

        even_m = _mean_at(even_idx, arr)
        odd_m  = _mean_at(odd_idx, arr)
        metrics[f"{lab}_swing_ratio"] = float(odd_m / (even_m + odd_m + 1e-12))

        if total_slots % 4 == 0:
            beats = total_slots // 4
            histograms[lab] = [float(np.mean(arr[b*4:(b+1)*4])) for b in range(beats)]

    return metrics, histograms

def _sync_instrument_order(instrument_tracks):
    """Return a stable order: DEFAULT_ORDER first (if present), then any extras (sorted)."""
    keys_present = list(instrument_tracks.keys())
    seen = set()
    ordered = []
    for k in DEFAULT_ORDER:
        if k in instrument_tracks and k not in seen:
            ordered.append(k); seen.add(k)
    for k in sorted(keys_present):
        if k not in seen:
            ordered.append(k); seen.add(k)
    return ordered

def run_stage5(folder_path, label_threshold=MIN_LABEL):
    folder = Path(folder_path)
    json_path = next((f for f in folder.glob("*.json") if not f.name.startswith(".")), None)
    if not json_path:
        print("❌ JSON file not found."); return

    with open(json_path) as f:
        data = json.load(f)

    rp = data.get("rhythm_pattern") or {}
    grid = data.get("beat_grid") or []
    slots = rp.get("slots") or []
    tick_duration = float(rp.get("tick_duration") or 0.0)
    if not grid or not slots or not tick_duration:
        print("❌ Missing rhythm pattern fields from Stage 4 (beat_grid/slots/tick_duration)."); return

    audio_file = _pick_audio_file(folder)
    if not audio_file:
        print("❌ No audio file found (drums.wav preferred)."); return

    print(f"\n🥁 Stage 5 (general, confidence-weighted)")
    print(f"🎧 Audio: {audio_file.name}")

    y, sr = sf.read(str(audio_file), always_2d=False)
    if y.ndim == 2: y = y.mean(axis=1)
    if SR_TARGET and sr != SR_TARGET:
        import librosa
        y = librosa.resample(y, orig_sr=sr, target_sr=SR_TARGET); sr = SR_TARGET

    # Prepare confidence tracks: one float per slot per label.
    total_slots = len(grid)
    instrument_tracks = data.get("instrument_tracks") or {}
    # Keep existing non-band tracks; (re)initialize band tracks as floats
    for lab in LABEL_BANDS.keys():
        instrument_tracks[lab] = [0.0] * total_slots

    # Slot-centered analysis window
    half_win = int(round((tick_duration * WINDOW_PAD) * sr))
    frame_len = max(256, 2 * half_win)
    if frame_len % 2 == 1: frame_len += 1
    half = frame_len // 2

    labels_per_slot = []
    confs_per_slot = []

    for i, t in enumerate(grid):
        center = int(round(t * sr))
        start  = max(0, center - half)
        end    = min(len(y), center + half)
        frame  = y[start:end]
        if len(frame) < frame_len:
            frame = np.pad(frame, (0, frame_len - len(frame)))

        confs = classify_frame_confidences(frame, sr)
        # Write confidences into instrument_tracks
        for lab, c in confs.items():
            instrument_tracks[lab][i] = float(c)

        # Human-readable labels list (optional)
        labels = [lab for lab, c in confs.items() if c >= label_threshold] or ["none"]
        labels_per_slot.append(labels)
        confs_per_slot.append({k: float(v) for k, v in confs.items()})

    # Persist core results
    data["instrument_tracks"] = instrument_tracks

    # NEW: sync instrument_order with tracks actually present
    data["instrument_order"] = _sync_instrument_order(instrument_tracks)

    # Write both 'labels' (preferred) and 'drum_labels' (compat mirror)
    label_objects = [
        {"slot": int(i), "labels": labels_per_slot[i],
         "conf": {k: round(v, 4) for k, v in confs_per_slot[i].items()}}
        for i in range(total_slots)
    ]
    rp["labels"] = label_objects
    rp["drum_labels"] = label_objects  # backward-compat mirror

    # Metrics + histograms from confidences (general; no genre assumptions)
    slot_roles = (rp.get("slot_roles") or {})
    bar_len = int(rp.get("bar_length_ticks") or 16)
    metrics, histos = compute_weighted_metrics(instrument_tracks, slot_roles, bar_len)

    data.setdefault("rhythm_analysis", {}).update(metrics)
    data.setdefault("rhythm_histograms", {}).update(histos)

    data["rhythm_pattern"] = rp  # reattach

    with open(json_path, "w") as f:
        json.dump(data, f, indent=2, separators=(",", ": "))

    print(f"✅ Classified {total_slots} slots (confidence-weighted).")
    print(f"   Updated rhythm_pattern.labels (+ drum_labels mirror), instrument_tracks, instrument_order, and metrics in {json_path.name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 5: General, confidence-weighted drum classification")
    parser.add_argument("folder", type=str, help="Path to folder with JSON + audio (drums.wav preferred)")
    parser.add_argument("--label-threshold", type=float, default=MIN_LABEL, help="Threshold to include labels (for readability only)")
    args = parser.parse_args()
    run_stage5(args.folder, label_threshold=args.label_threshold)
