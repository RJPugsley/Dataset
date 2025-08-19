import json
from pathlib import Path

SUBDIVISIONS_PER_BEAT = 4   # 1/16 grid
PHRASE_BEATS = 16           # 4 bars * 4 beats
DEFAULT_TRACKS = ["kick","snare","clap","hihat","percussion","bass","lead","pad","fx","vocal"]

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

    print("\n🚀 Stage 4: Generate Beat Grid + Helpers")
    print(f"🎯 BPM: {bpm:.2f}, loop_start: {start_time:.3f}s")

    # Geometry
    beat_interval = 60.0 / bpm
    tick_duration = beat_interval / SUBDIVISIONS_PER_BEAT
    total_ticks = PHRASE_BEATS * SUBDIVISIONS_PER_BEAT  # 16 * 4 = 64
    bars = PHRASE_BEATS // 4

    # Deterministic times (avoid float creep)
    slot_times = [round(start_time + i * tick_duration, 6) for i in range(total_ticks)]

    # Canonical slot descriptors
    slots = [{
        "slot": i,                             # PRIMARY index
        "bar": (i // 16) + 1,
        "step": (i % 16) + 1,                  # 1..16 within bar
        "beat_in_bar": ((i // 4) % 4) + 1,     # 1..4
        "sub_in_beat": (i % 4) + 1,            # 1..4 (1/16s within beat)
        "time": slot_times[i]
    } for i in range(total_ticks)]

    # Back-compat minimal beats list
    beats = [{"bar": s["bar"], "step": s["step"], "time": s["time"]} for s in slots]

    # Instrument canvas (ready to be filled by next stage)
    instrument_tracks = {name: [0]*total_ticks for name in DEFAULT_TRACKS}

    # Core rhythm pattern block
    data["rhythm_pattern"] = {
        "bpm": round(bpm, 2),
        "start_time": round(start_time, 6),
        "grid_resolution": "1/16",
        "tick_duration": round(tick_duration, 6),
        "bars": bars,
        "phrase_beats": PHRASE_BEATS,
        "slots": slots,
        "beats": beats
    }

    # Machine-friendly + compatibility
    data["beat_grid"] = slot_times
    data["beat_grid_csv"] = ",".join(f"{t:.6f}" for t in slot_times)

    # Instrument canvas and explicit order (stable for UIs)
    data["instrument_tracks"] = instrument_tracks
    data["instrument_order"] = list(instrument_tracks.keys())

    # === Helper layers for plotting & fast lookups ===
    total = len(slot_times)
    downbeat_idxs = [i for i in range(total) if i % 16 == 0]
    beat_idxs     = [i for i in range(total) if i % 4  == 0]
    offbeat_idxs  = [i for i in range(total) if i % 4  == 2]

    data["rhythm_pattern"]["bar_length_ticks"] = 16
    data["rhythm_pattern"]["slot_roles"] = {
        "downbeats": downbeat_idxs,
        "beats": beat_idxs,
        "offbeats": offbeat_idxs
    }

    # Integer millisecond times for crisp UI timelines
    slot_times_ms = [int(round(t * 1000)) for t in slot_times]
    data["rhythm_pattern"]["slot_times_ms"] = slot_times_ms

    # O(1) lookup: (bar, step) -> slot index (keys like "3-7")
    index_lookup = {}
    for s in slots:
        index_lookup[f"{s['bar']}-{s['step']}"] = s["slot"]
    data["rhythm_pattern"]["barstep_to_slot"] = index_lookup

    with open(json_path, "w") as f:
        json.dump(data, f, indent=2, separators=(",", ": "))

    print(f"✅ Beat grid + helpers written to {json_path.name}")
    print("🏁 Stage 4 complete.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Stage 4: Generate beat grid + instrument canvas + helpers")
    parser.add_argument("folder", type=str, help="Path to track folder containing JSON")
    args = parser.parse_args()
    run_stage4(args.folder)
