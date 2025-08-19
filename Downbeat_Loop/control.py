# control.py
import argparse
import subprocess
import sys
from pathlib import Path
import time

FOLDER = Path(__file__).parent  # same folder as this script
RAW_SCRIPT = FOLDER / "raw_data_extract.py"
DOWNBEAT_SCRIPT = FOLDER / "aadownbeat.py"

AUDIO_EXTS = {".mp3", ".m4a", ".flac", ".wav", ".aac", ".ogg"}

def is_audio(p: Path) -> bool:
    return p.suffix.lower() in AUDIO_EXTS

def collect_files(path: Path):
    if path.is_file() and is_audio(path):
        yield path
    elif path.is_dir():
        for p in sorted(path.rglob("*")):
            if is_audio(p):
                yield p
    else:
        raise FileNotFoundError(f"Invalid path: {path}")

def run_raw_extract(audio_file: Path):
    cmd = [sys.executable, str(RAW_SCRIPT), str(audio_file)]
    subprocess.run(cmd, check=True)

def run_downbeat(track_dir: Path):
    cmd = [sys.executable, str(DOWNBEAT_SCRIPT), str(track_dir)]
    subprocess.run(cmd, check=True)

def main():
    ap = argparse.ArgumentParser(description="Run raw_data_extract then aadownbeat sequentially")
    ap.add_argument("path", help="Audio file or folder of audio files")
    args = ap.parse_args()

    src = Path(args.path).resolve()
    files = list(collect_files(src))
    if not files:
        print("[!] No audio files found")
        return

    print(f"[✓] Found {len(files)} file(s). Processing sequentially…")
    start_all = time.time()

    for i, f in enumerate(files, 1):
        print(f"\n—— {i}/{len(files)} — {f.name} ————")
        try:
            t0 = time.time()
            run_raw_extract(f)
            # Track dir = Downbeat_Loop/<TrackName>
            track_dir = FOLDER / f.stem
            run_downbeat(track_dir)
            print(f"[✓] Done {f.name} in {time.time()-t0:.1f}s")
        except Exception as e:
            print(f"[✗] Failed on {f.name}: {e}")

    print(f"\nAll done in {time.time()-start_all:.1f}s")

if __name__ == "__main__":
    main()
