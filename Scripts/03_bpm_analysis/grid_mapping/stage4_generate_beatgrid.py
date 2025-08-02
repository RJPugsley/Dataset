import json
from pathlib import Path

def run_stage4(folder_path):
    folder = Path(folder_path)
    json_path = next((f for f in folder.glob("*.json") if not f.name.startswith(".")), None)

    if not json_path:
        print("❌ JSON file not found.")
        return

    with open(json_path) as f:
        data = json.load(f)

    try:
        bpm = float(data.get("bpm"))
        start_time = float(data.get("loop_start"))
    except (TypeError, ValueError):
        print("❌ Missing or invalid bpm or loop_start in JSON.")
        return

    print("\n🚀 Starting Stage 4: Generate Beat Grid")
    print(f"🎯 Using BPM: {bpm:.2f}, start_time: {start_time:.3f}s")

    beat_interval = 60.0 / bpm
    grid_times = [round(start_time + i * (beat_interval / 4), 5) for i in range(64)]

    beats = [{
        "bar": i // 16 + 1,
        "step": i % 16 + 1,
        "time": t
    } for i, t in enumerate(grid_times)]

    data["rhythm_pattern"] = {
        "bpm": round(bpm, 2),
        "start_time": round(start_time, 5),
        "grid_resolution": "1/16",
        "bars": 4,
        "beats": beats
    }

    data["beat_grid"] = ",".join(f"{b['time']:.5f}" for b in beats)

    with open(json_path, "w") as f:
        json.dump(data, f, indent=2, separators=(",", ": "))

    print(f"✅ Beat grid written to {json_path.name}")
    print("🏁 Stage 4 complete.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Stage 4: Generate beat grid from loop start and BPM")
    parser.add_argument("folder", type=str, help="Path to track folder containing JSON")
    args = parser.parse_args()

    run_stage4(args.folder)
