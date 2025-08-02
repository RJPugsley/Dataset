import subprocess
import sys
from pathlib import Path

def run_stage(script, folder):
    print(f"\n▶️ Running {script}...")
    result = subprocess.run(
        [sys.executable, script, folder],
        capture_output=True,
        text=True
    )
    print(result.stdout)
    if result.stderr:
        print("⚠️ STDERR:", result.stderr)

def main(folder):
    folder = Path(folder).resolve()
    if not folder.is_dir():
        print(f"❌ Invalid folder: {folder}")
        return

    # Define scripts in order
    scripts = [
        "stage4_generate_beatgrid.py",
        "stage5_classify_drum_hits.py",
        "stage6_analyze_confidence.py",
        "stage8_analyze_rhythm_consistency.py"
    ]

    for script in scripts:
        run_stage(script, str(folder))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_rhythm_stages.py /path/to/track_folder")
    else:
        main(sys.argv[1])
