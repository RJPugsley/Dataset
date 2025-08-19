#!/usr/bin/env python3
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import importlib.util

# --- Absolute paths to Stage 2 & 3 (same folder as rhythm_utils.py) ---
STAGE_DIR = Path("/Users/djf2/Desktop/Dataset/Scripts/03_bpm_analysis/grid_mapping")
S2_PATH = str(STAGE_DIR / "stage2_downbeat_detection.py")
S3_PATH = str(STAGE_DIR / "stage3_define_loop.py")

# Make sure rhythm_utils.py (and friends) are importable for Stage 2/3
if str(STAGE_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE_DIR))

# --- Remaining pipeline scripts (full paths) ---
PIPELINE_SCRIPTS = [
    "/Users/djf2/Desktop/Dataset/Scripts/03_bpm_analysis/grid_mapping/stage4_generate_beatgrid.py",
    "/Users/djf2/Desktop/Dataset/Scripts/03_bpm_analysis/grid_mapping/stage5_classify_drum_hits.py",
    "/Users/djf2/Desktop/Dataset/Scripts/03_bpm_analysis/grid_mapping/stage6_analyze_confidence.py",
    "/Users/djf2/Desktop/Dataset/Scripts/03_bpm_analysis/grid_mapping/stage8_analyze_rhythm_consistency.py",
    "/Users/djf2/Desktop/Dataset/Scripts/04_bassline_analysis/bassline_analysis.py",
    "/Users/djf2/Desktop/Dataset/Scripts/05_harmonic_analysis/harmonic_analysis.py",
    "/Users/djf2/Desktop/Dataset/Scripts/06_speech_analysis/speech_analysis.py",
    "/Users/djf2/Desktop/Dataset/Scripts/07_final_energy_analysis/final_energy_analysis.py",
]

def _load_module(path: str, name: str):
    print(f"📦 Loading module: {name} from {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    print(f"✅ Loaded module: {name}")
    return mod

def _run_subprocess(script: str, folder: Path) -> bool:
    print(f"\n▶️ Running {script}...")
    result = subprocess.run([sys.executable, script, str(folder)], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("⚠️ STDERR:", result.stderr)
    if result.returncode != 0:
        print(f"❌ {script} exited with code {result.returncode}")
    else:
        print(f"✅ Completed {script}")
    return result.returncode == 0

def _recompute_non_detection(s2, s3, folder: Path) -> None:
    print("🔄 Starting non-detection recompute...")
    # 1) Load JSON via Stage 2’s own loader (keeps behavior identical)
    print("📂 Loading JSON data...")
    json_path, meta = s2._load_json(folder)
    print(f"✅ JSON loaded: {json_path}")

    # 2) Beat-tracker from manual first_downbeat + bpm
    print("🎯 Recomputing beat-tracker from manual edits...")
    bpm = meta.get("bpm", None)
    t0  = meta.get("first_downbeat", None)
    changed = False

    if isinstance(bpm, (int, float)) and bpm > 0 and isinstance(t0, (int, float)):
        print(f"ℹ️ Using bpm={bpm}, first_downbeat={t0}")
        stems = s2._resolve_stems(folder, meta)
        print(f"📂 Stems resolved: {stems}")
        rf = s2.RhythmFeatures.from_files(stems, params=s2.RhythmParams())
        inst_times, inst_bpm = s2._instant_bpm_curve(rf, float(bpm), float(t0), beats=96, search_frac=0.18)
        bt = meta.get("beat_tracker") or {}
        bt.update({
            "method": "snap-to-onset",
            "bpm_used": float(bpm),
            "start_time": float(round(float(t0), 6)),
            "instant_times": inst_times,
            "instant_bpm": inst_bpm
        })
        meta["beat_tracker"] = bt
        meta["first_downbeat_bpm_used"] = float(bpm)
        changed = True
        try:
            import numpy as np
            print(f"✅ Beat-tracker updated: {len(inst_bpm)} points (median {np.median(inst_bpm):.2f} BPM)")
        except Exception:
            print(f"✅ Beat-tracker updated: {len(inst_bpm)} points")
    else:
        print("ℹ️ Skipping beat-tracker recompute (missing bpm or first_downbeat).")

    # 3) Loop-derived fields per Stage 3 naming
    print("🎯 Recomputing loop normalization...")
    loop_start = meta.get("loop_start", None)
    if isinstance(loop_start, (int, float)):
        print(f"ℹ️ Using loop_start={loop_start}")
        tl = meta.get("track_length", None)
        if isinstance(tl, (int, float)) and tl > 0:
            meta["loop_start_normalized"] = round(float(loop_start) / float(tl), 6)
            changed = True
            print(f"✅ loop_start_normalized = {meta['loop_start_normalized']}")
        if isinstance(bpm, (int, float)) and bpm > 0:
            meta["loop_start_bpm_used"] = float(bpm)
            changed = True
            print(f"✅ loop_start_bpm_used = {bpm}")
        if "loop_start_method" not in meta:
            meta["loop_start_method"] = "recurrence-16beat"
            print(f"ℹ️ Added loop_start_method = recurrence-16beat")
    else:
        print("ℹ️ Skipping loop normalization recompute (missing loop_start).")

    if changed:
        with open(json_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"💾 JSON updated at {json_path}")
    else:
        print("ℹ️ No recompute changes were necessary.")

def main(folder_arg: str):
    folder = Path(folder_arg).resolve()
    if not folder.is_dir():
        print(f"❌ Invalid folder: {folder}")
        return

    if not STAGE_DIR.exists():
        print(f"❌ Stage dir not found: {STAGE_DIR}")
        return

    # Load Stage 2 & 3 modules from file paths (keeps them untouched)
    s2 = _load_module(S2_PATH, "stage2_downbeat_detection")
    s3 = _load_module(S3_PATH, "stage3_define_loop")

    # 0) Recompute non-detection pieces using Stage 2/3 logic
    _recompute_non_detection(s2, s3, folder)

    # 1) Run the rest of the pipeline
    print("🚀 Starting remaining pipeline scripts...")
    for script in PIPELINE_SCRIPTS:
        if not _run_subprocess(script, folder):
            print("⛔ Stopping due to failure.")
            break
    print("✅ All stages complete.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python Update_json.py /path/to/track_folder")
    else:
        main(sys.argv[1])
