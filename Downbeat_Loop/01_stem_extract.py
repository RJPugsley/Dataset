
#!/usr/bin/env python3
# 01_stem_extract.py — Demucs stem + BPM + beat grid + JSON metadata

import os, sys, json, shutil, subprocess, argparse
from pathlib import Path
import librosa

DEFAULT_MODEL = "htdemucs_6s"
OUTPUT_ROOT = Path("/Users/djf2/Desktop/Dataset/Downbeat_Loop")
HARDCODED_FFPROBE = "/Users/djf2/Desktop/AppsBinsLibs/Binary/ffprobe"

def extract_tag_bpm(audio_path: Path):
    try:
        result = subprocess.run(
            [HARDCODED_FFPROBE, "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", str(audio_path)],
            capture_output=True, text=True, check=True
        )
        data = json.loads(result.stdout)
        tags = data.get("format", {}).get("tags", {})
        bpm = tags.get("bpm") or tags.get("TBPM") or tags.get("tempo")
        return float(bpm) if bpm else None
    except Exception:
        return None

def extract_bpm_and_beats(audio_path: Path, sr_target=44100):
    y, sr = librosa.load(str(audio_path), sr=sr_target, mono=True)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()
    duration = float(len(y) / sr)
    return float(round(tempo, 2)), beat_times, duration, sr

def run_demucs(audio_path: Path, tmp_dir: Path, model: str) -> Path:
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)
    cmd = ["demucs", "-n", model, "-o", str(tmp_dir), str(audio_path)]
    subprocess.run(cmd, check=True)
    return tmp_dir / model / audio_path.stem

def organize(audio_path, stems_src_dir, bpm_value, bpm_tag, bpm_est,
             beat_times_used, beat_times_est, duration, sr, delete_original=False):
    track_name = audio_path.stem
    out_dir = OUTPUT_ROOT / track_name
    out_dir.mkdir(parents=True, exist_ok=True)

    stems_out = {}
    for wav in stems_src_dir.glob("*.wav"):
        dst = out_dir / wav.name
        shutil.move(str(wav), str(dst))
        stems_out[wav.stem.lower()] = str(dst.resolve())

    source_copy = out_dir / audio_path.name
    try:
        shutil.copy2(str(audio_path), str(source_copy))
        source_path_for_json = source_copy
    except Exception:
        source_path_for_json = audio_path
    if delete_original:
        try: audio_path.unlink()
        except Exception: pass

    delta_pct = None
    if bpm_tag and bpm_est:
        try:
            delta_pct = round(100.0 * abs(bpm_tag - bpm_est) / max(bpm_tag, bpm_est), 2)
        except ZeroDivisionError:
            delta_pct = None

    data = {
        "schema_version": "dl_raw_v2",
        "bpm": {
            "value": float(bpm_value) if bpm_value is not None else None,
            "source": "tag" if (bpm_tag is not None and bpm_value == bpm_tag) else "est",
            "tag": float(bpm_tag) if bpm_tag is not None else None,
            "est": float(bpm_est) if bpm_est is not None else None,
            "delta_pct": delta_pct
        },
        "beats": {
            "used": beat_times_used or [],
            "est": beat_times_est or [],
            "sr": int(sr)
        },
        "duration_s": float(duration),
        "original_filename": audio_path.name,
        "source_path": str(Path(source_path_for_json).resolve()),
        "stems": stems_out,
        "first_downbeat": None,
        "loop_start": None
    }

    json_path = out_dir / f"{track_name}.json"
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[✓] Output JSON: {json_path}")
    return out_dir  # ✅ allows pipeline chaining

def process_file(audio: Path, model: str = DEFAULT_MODEL, delete_original: bool = False):
    print(f"[>] {audio}")
    bpm_tag = extract_tag_bpm(audio)
    bpm_est, beat_times_est, duration, sr = extract_bpm_and_beats(audio)
    bpm_value = bpm_tag if bpm_tag is not None else bpm_est
    beat_times_used = beat_times_est

    tmp = audio.parent / "demucs_stems_tmp"
    try:
        stems_dir = run_demucs(audio, tmp, model)
        out_dir = organize(audio, stems_dir, bpm_value, bpm_tag, bpm_est,
                           beat_times_used, beat_times_est, duration, sr,
                           delete_original=delete_original)
        return out_dir
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

def main():
    ap = argparse.ArgumentParser(description="Demucs 6-stem + BPM/beat grid → JSON")
    ap.add_argument("path", help="Audio file or folder")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Demucs model")
    ap.add_argument("--delete-original", action="store_true", help="Delete input file after processing")
    args = ap.parse_args()

    p = Path(args.path)
    if p.is_file():
        out_dir = process_file(p, args.model, args.delete_original)
        print(out_dir)  # ✅ shell-friendly output
    elif p.is_dir():
        exts = ("*.mp3","*.m4a","*.flac","*.wav")
        files = [f for ext in exts for f in p.glob(f"**/{ext}")]
        if not files:
            print(f"[!] No audio files found under {p}")
        for f in files:
            out_dir = process_file(f, args.model, args.delete_original)
            print(out_dir)
    else:
        print(f"[!] Invalid path: {p}")
        sys.exit(1)

if __name__ == "__main__":
    main()
