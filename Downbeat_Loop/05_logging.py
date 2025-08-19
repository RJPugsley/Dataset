#!/usr/bin/env python3
# 04_logging.py — log feature events from JSONs to CSV

import json
import csv
import argparse
from pathlib import Path

def flatten_feature_events(json_data):
    events = json_data.get("feature_events", [])
    bpm_obj = json_data.get("bpm", {})
    bpm = bpm_obj.get("value") if isinstance(bpm_obj, dict) else bpm_obj
    source_path = json_data.get("source_path", "")
    track_name = Path(source_path).stem if source_path else "unknown"
    duration = json_data.get("duration_s", None)
    schema = json_data.get("schema_version", "unknown")

    rows = []
    for event in events:
        rows.append({
            "track_name": track_name,
            "source": source_path,
            "event": event.get("event"),
            "start": event.get("start"),
            "end": event.get("end"),
            "position": event.get("position"),
            "stem": event.get("stem"),
            "target": event.get("target", ""),  # first_downbeat or loop_start
            "bpm": bpm,
            "duration_s": duration,
            "schema_version": schema
        })
    return rows

def write_events_to_csv(all_rows, output_csv: Path, dedupe=False):
    if not all_rows:
        print("[!] No events to log.")
        return

    fieldnames = list(all_rows[0].keys())
    write_header = not output_csv.exists()

    # Load existing if dedupe is on
    existing_keys = set()
    if dedupe and output_csv.exists():
        with open(output_csv, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row["track_name"], row["event"], row["start"], row["stem"], row["target"])
                existing_keys.add(key)

    new_rows = []
    for row in all_rows:
        key = (row["track_name"], str(row["event"]), str(row["start"]), str(row["stem"]), str(row["target"]))
        if dedupe and key in existing_keys:
            continue
        new_rows.append(row)

    if not new_rows:
        print("[✓] No new rows (all duplicates skipped).")
        return

    with open(output_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(new_rows)

    print(f"[+] Logged {len(new_rows)} events to {output_csv.name}")

def process_json(json_path: Path) -> list[dict]:
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        return flatten_feature_events(data)
    except Exception as e:
        print(f"[!] Failed to process {json_path.name}: {e}")
        return []

def main():
    parser = argparse.ArgumentParser(description="Log feature events to CSV")
    parser.add_argument("path", type=str, help="Path to track JSON or folder")
    parser.add_argument("--all", action="store_true", help="Process all JSONs in folder")
    parser.add_argument("--dedupe", action="store_true", help="Skip already-logged rows")
    parser.add_argument("--out", type=str, default="feature_events.csv", help="Output CSV filename")

    args = parser.parse_args()
    path = Path(args.path)
    output_csv = path / args.out if path.is_dir() else path.parent / args.out

    all_rows = []

    if path.is_file() and path.suffix == ".json":
        all_rows.extend(process_json(path))

    elif args.all and path.is_dir():
        jsons = list(path.glob("**/*.json"))
        if not jsons:
            print("[!] No JSON files found.")
        for jp in jsons:
            all_rows.extend(process_json(jp))

    else:
        print("[!] Must specify a .json file or use --all with a folder.")
        return

    write_events_to_csv(all_rows, output_csv, dedupe=args.dedupe)

if __name__ == "__main__":
    main()
