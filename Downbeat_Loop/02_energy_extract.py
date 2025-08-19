
#!/usr/bin/env python3
# 02_energy_extract.py — collect true energy curves (+optional RMS) from all stems and mix

import json, argparse, pprint
from pathlib import Path
from typing import Dict, Optional, List
import numpy as np
import librosa
import difflib
from typing import Union

SR = 44100
FRAME_LENGTH = 1024
HOP_SIZE = 512
HOP_S = HOP_SIZE / SR

EXPECTED_STEMS = ["drums", "bass", "other", "vocals", "guitar", "piano"]
AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".aac", ".aiff", ".aif", ".ogg"}

def validate_input_path(user_path: str) -> Path:
    p = Path(user_path)
    if p.exists(): return p
    parent, target = p.parent, p.name
    if not parent.exists():
        raise FileNotFoundError(f"Invalid parent directory: {parent}")
    dirs = [d.name for d in parent.iterdir() if d.is_dir()]
    closest = difflib.get_close_matches(target, dirs, n=1)
    suggestion = f" Did you mean: {parent/closest[0]}" if closest else ""
    raise FileNotFoundError(f"Path not found: {user_path}.{suggestion}")

def load_json(track_dir: Path) -> Dict:
    js = sorted(track_dir.glob("*.json"))
    if not js: raise FileNotFoundError(f"No JSON found in {track_dir}")
    with open(js[0], "r") as f: return json.load(f)

def load_bpm_and_beats(track_dir: Path) -> Dict:
    meta = load_json(track_dir)
    bpm_raw = meta.get("bpm")

    if isinstance(bpm_raw, dict):
        bpm_value = float(bpm_raw.get("value")) if bpm_raw.get("value") is not None else None
        bpm_source = bpm_raw.get("source", "unknown")
    else:
        bpm_value = float(bpm_raw) if bpm_raw is not None else None
        bpm_source = "flat"  # indicates older format

    beats_blk = meta.get("beats", {})
    return {
        "bpm_value": bpm_value,
        "bpm_source": bpm_source,
        "beats_used": [float(t) for t in (beats_blk.get("used") or [])],
    }


def detect_source_mix(track_dir: Path) -> Optional[Path]:
    stem_names = {f"{s}.wav" for s in EXPECTED_STEMS}
    candidates = [p for p in track_dir.iterdir() if p.suffix.lower() in AUDIO_EXTS]
    mix_candidates = [p for p in candidates if p.name not in stem_names]
    if not mix_candidates: return None
    mix_candidates.sort(key=lambda p: p.stat().st_size, reverse=True)
    return mix_candidates[0]

def load_mono(path: Path) -> np.ndarray:
    y, _ = librosa.load(path, sr=SR, mono=True)
    return y

def energy_curve_true(y: np.ndarray) -> np.ndarray:
    frames = librosa.util.frame(y, frame_length=FRAME_LENGTH, hop_length=HOP_SIZE)
    E = np.sum(frames * frames, axis=0)
    return E.astype(np.float32)

def energy_curve_rms(y: np.ndarray) -> np.ndarray:
    rms = librosa.feature.rms(y=y, frame_length=FRAME_LENGTH, hop_length=HOP_SIZE, center=False)[0]
    return rms.astype(np.float32)

def to_time_value_pairs(curve: np.ndarray) -> List[List[float]]:
    return [[round(i * HOP_S, 6), float(v)] for i, v in enumerate(curve)]

def load_all_streams(track_dir: Path) -> Dict[str, np.ndarray]:
    curves = {}
    for stem in EXPECTED_STEMS:
        wav = track_dir / f"{stem}.wav"
        if wav.exists():
            y = load_mono(wav)
            curves[stem] = y
    src = detect_source_mix(track_dir)
    if src is not None:
        curves["source"] = load_mono(src)
    return curves

def collect_raw_data(track_dir: Union[str, Path], also_rms: bool = False) -> Dict:
    track_dir = validate_input_path(str(track_dir))
    track_dir = Path(track_dir)

    meta = load_bpm_and_beats(track_dir)
    waves = load_all_streams(track_dir)
    if not waves:
        raise RuntimeError("No stems and no source mix found — nothing to analyze.")

    energy_curves = {}
    rms_curves = {} if also_rms else None

    for name, y in waves.items():
        E = energy_curve_true(y)
        energy_curves[name] = to_time_value_pairs(E)
        if also_rms:
            R = energy_curve_rms(y)
            rms_curves[name] = to_time_value_pairs(R)

    num_frames = {name: len(vals) for name, vals in energy_curves.items()}

    out = {
        "track_dir": str(track_dir),
        "json_path": str(next(track_dir.glob("*.json"), "")),
        "bpm": {
            "value": meta["bpm_value"],
            "source": meta["bpm_source"],
        },
        "beats_used": meta["beats_used"],
        "sr": SR,
        "frame_length": FRAME_LENGTH,
        "hop_size": HOP_SIZE,
        "hop_s": HOP_S,
        "expected_stems": EXPECTED_STEMS,
        "energy_curves": energy_curves,
        "num_frames": num_frames,
    }
    if also_rms:
        out["rms_curves"] = rms_curves

    # Save
    out_path = track_dir / "raw_data.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[✓] Saved energy data to {out_path}")

    return out  # ✅ makes this pipeline-compatible

def main():
    ap = argparse.ArgumentParser(description="Collect true energy curves (and optionally RMS) for stems + source.")
    ap.add_argument("track_dir", help="Folder containing stems + JSON + source mix")
    ap.add_argument("--also-rms", action="store_true", help="Include RMS curves")
    args = ap.parse_args()

    data = collect_raw_data(args.track_dir, also_rms=args.also_rms)

    # Compact preview
    preview = {k: v for k, v in data.items() if k not in ("energy_curves", "rms_curves")}
    preview["energy_curves"] = {k: f"{len(v)} frames" for k, v in data["energy_curves"].items()}
    if "rms_curves" in data:
        preview["rms_curves"] = {k: f"{len(v)} frames" for k, v in data["rms_curves"].items()}
    pprint.pprint(preview)

if __name__ == "__main__":
    main()
