#!/usr/bin/env python3
import argparse
import sys
import os
import gc
import time
import traceback
from pathlib import Path

# ---------- Thread limits (set BEFORE heavy imports) ----------
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("BLIS_NUM_THREADS", "1")
os.environ.setdefault("FFTW_NUM_THREADS", "1")

# ---------- Fix import paths for submodules ----------
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

subdirs = [
    "01_stems",
    "02_energy_envelope",
    "03_bpm_analysis/grid_mapping",
    "04_bassline_analysis",
    "05_harmonic_analysis",
    "06_speech_analysis",
    "07_final_energy_analysis"
]
for subdir in subdirs:
    sys.path.append(os.path.join(script_dir, subdir))

# ---------- Stage imports ----------
from stem_extract import extract_stems
from energy_envelope import analyze_energy_envelopes
from run_rhythm_pipeline import run_rhythm_pipeline
from bassline_analysis import bassline_analysis
from harmonic_analysis import harmonic_analysis
from speech_analysis import speech_analysis
from final_energy_analysis import final_energy_analysis

def format_time(seconds):
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}m {secs}s"

def run_pipeline(audio_file_path: str):
    audio_file = Path(audio_file_path)
    if not audio_file.exists() or not audio_file.is_file():
        print(f"❌ File not found: {audio_file}")
        return

    print(f"▶️ Processing: {audio_file.name}")
    step_times = {}
    overall_start = time.time()

    try:
        # New output root on local disk (cooler than external volumes)
        track_folder = extract_stems(
            audio_file,
            output_root=Path("/Users/djf2/Desktop/Dataset/Processed_Files")
        )
        if not track_folder:
            print("❌ Stem extraction failed.")
            return
        step_times["Extract Stems"] = time.time() - overall_start

        step_start = time.time()
        analyze_energy_envelopes(track_folder)
        step_times["Energy Envelope"] = time.time() - step_start

        step_start = time.time()
        run_rhythm_pipeline(track_folder)
        step_times["Rhythm + BPM"] = time.time() - step_start

        step_start = time.time()
        bassline_analysis(track_folder)
        step_times["Bassline Analysis"] = time.time() - step_start

        step_start = time.time()
        harmonic_analysis(track_folder)
        step_times["Harmonic Analysis"] = time.time() - step_start

        step_start = time.time()
        speech_analysis(track_folder)
        step_times["Speech/Vocals"] = time.time() - step_start

        step_start = time.time()
        final_energy_analysis(track_folder)
        step_times["Final Energy Analysis"] = time.time() - step_start

        total = time.time() - overall_start
        print(f"\n✅ Finished {audio_file.name} in {format_time(total)}")
        print("📊 Step Timing Summary:")
        for step, seconds in step_times.items():
            print(f"  • {step:<25} — {format_time(seconds)}")
        print("\n" + "-" * 50 + "\n")

        # ---------- End-of-track cleanup ----------
        try:
            del step_times, track_folder, audio_file
        except Exception:
            pass
        gc.collect()

    except Exception as e:
        print(f"❌ Error processing {audio_file.name}: {e}")
        traceback.print_exc()

def run_in_parallel(file_list):
    # Sequential on purpose: minimizes thermals
    print("📊 Forcing sequential processing (1 file at a time).")
    for f in file_list:
        run_pipeline(str(f))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full DJmate audio analysis pipeline on a file or folder.")
    parser.add_argument("input", nargs="?", help="Path to audio file or folder of audio files")
    args = parser.parse_args()

    input_path = args.input
    if not input_path:
        input_path = input("🗂️  Enter path to audio file or folder: ").strip()

    input_path = Path(os.path.expanduser(input_path)).resolve()

    if input_path.is_file():
        print(f"📄 Detected file: {input_path.name}")
        run_pipeline(str(input_path))

    elif input_path.is_dir():
        print(f"📁 Detected folder: {input_path}")
        audio_exts = [".mp3", ".wav", ".flac", ".m4a"]
        files = [f for f in input_path.iterdir() if f.suffix.lower() in audio_exts]
        if not files:
            print("❌ No valid audio files found.")
        else:
            run_in_parallel(files)
        print("✅ All files processed.")

    else:
        print(f"❌ Path not found: {input_path}")
