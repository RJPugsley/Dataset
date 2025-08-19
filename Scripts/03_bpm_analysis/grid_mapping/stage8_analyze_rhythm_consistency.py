# stage8_analyze_rhythm_consistency.py
# Updated to:
# - Use swing and (if present) syncopation from JSON; otherwise compute a /16-grid syncopation fallback
# - Stop inferring triplets from a /16 grid; optionally read a boolean flag from JSON (groove.uses_triplets)
# - Clarify tempo stability: prefer real beat-tracker curve if present; else store a proxy from kick IOIs and mark it as proxy
# - Keep/extend robust /16 features for rhythm-only genre classification
# - Add sectional change features and a coarse kick-pattern class
#
# Output is written under data["rhythm_pattern"]["genre_features"]

import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from typing import Any, List, Sequence, Set, Tuple
import math
import numpy as np

# -------------------------
# Helpers
# -------------------------

def _dig(d: dict, path: List[Any], default=None):
    cur = d
    for k in path:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur


def _coerce_step(step: Any) -> List[str]:
    if step is None:
        return []
    if isinstance(step, (list, tuple, set)):
        labels = [str(x).strip().lower() for x in step if x is not None]
    elif isinstance(step, str):
        labels = [step.strip().lower()]
    else:
        labels = [str(step).strip().lower()]
    return [l for l in labels if l and l != "none"]


def _coerce_labels(labels_any: Any) -> List[List[str]]:
    if not isinstance(labels_any, Sequence):
        return []
    return [_coerce_step(step) for step in labels_any]


def flatten_multilabel(labels: List[List[str]]) -> List[str]:
    out: List[str] = []
    for step in labels:
        out.extend(step)
    return out


def infer_steps_per_bar(total_steps: int, bars: int) -> int:
    if bars > 0 and total_steps % bars == 0:
        return total_steps // bars
    return 0


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    u = len(a | b)
    return (len(a & b) / u) if u else 0.0


def bar_chunks(labels: List[List[str]], bars: int) -> Tuple[List[List[List[str]]], int]:
    n = len(labels)
    if n == 0 or bars <= 0:
        return [], 0
    if n % bars != 0:
        n = (n // bars) * bars
        labels = labels[:n]
    spb = n // bars
    chunks = [labels[i*spb:(i+1)*spb] for i in range(bars)]
    return chunks, spb

# -------------------------
# Triplet (/12) overlay helpers (keep /16 storage; derive overlay on demand)
# -------------------------

def _remap_to_12(labels: List[List[str]], bars: int) -> Tuple[List[List[List[str]]], int]:
    """
    Map any per-bar grid (spb > 0) to a companion /12 grid by pooling nearest steps.
    For each bar, 4 beats × 3 triplet steps. Returns (chunks_12, spb12=12).
    """
    chunks_src, spb_src = bar_chunks(labels, bars)
    if not chunks_src or spb_src <= 0:
        return [], 0
    spb12 = 12
    out: List[List[List[str]]] = []
    ratio = spb12 / float(spb_src)
    for ch in chunks_src:
        mapped = [set() for _ in range(spb12)]
        for i, step in enumerate(ch):
            j = int(round(i * ratio))
            j = max(0, min(spb12 - 1, j))
            for lab in step:
                mapped[j].add(lab)
        out.append([sorted(list(s)) for s in mapped])
    return out, spb12


def _hat_triplet_offbeat_ratio_12(chunks12: List[List[List[str]]], spb12: int) -> float:
    """
    On /12 bars: count hat hits on the triplet off-positions (1/3, 2/3 within each beat)
    relative to on-beat positions. Ratio in [0,1].
    """
    if not chunks12 or spb12 != 12:
        return 0.0
    trip_off_idx = []
    beat_on_idx = []
    # each bar: 4 beats; indices per beat = [0,1,2], on-beat=0, off-triplets=1,2
    for b in range(4):
        base = b * 3
        beat_on_idx.append(base + 0)
        trip_off_idx.extend([base + 1, base + 2])
    off_hits = on_hits = 0
    for bar in chunks12:
        for i in trip_off_idx:
            if i < len(bar) and any("hat" in x for x in bar[i]):
                off_hits += 1
        for i in beat_on_idx:
            if i < len(bar) and any("hat" in x for x in bar[i]):
                on_hits += 1
    total = off_hits + on_hits
    return float(off_hits/total) if total > 0 else 0.0


# -------------------------
# Core metrics (grid-safe)
# -------------------------

def bar_similarity_mean(labels: List[List[str]], bars: int) -> float:
    chunks, spb = bar_chunks(labels, bars)
    if not chunks:
        return 0.0
    sims = []
    ref = [set(s) for s in chunks[0]]
    for b in range(1, len(chunks)):
        cur = [set(s) for s in chunks[b]]
        step_sims = [jaccard(a, c) for a, c in zip(ref, cur)]
        sims.append(np.mean(step_sims) if step_sims else 0.0)
    return float(np.mean(sims)) if sims else 0.0


def per_bar_activity(labels: List[List[str]], bars: int) -> List[dict]:
    chunks, spb = bar_chunks(labels, bars)
    out = []
    for i, ch in enumerate(chunks):
        active = sum(1 for s in ch if s)
        out.append({"bar": i+1, "active_steps": active, "density": round(active / spb, 3) if spb else 0.0})
    return out


def pattern_entropy(flat_labels: List[str]) -> float:
    c = Counter(flat_labels)
    total = sum(c.values())
    if total <= 0:
        return 0.0
    probs = [v/total for v in c.values() if v > 0]
    return float(-sum(p*math.log2(p) for p in probs))


def ngram4_unique_ratio(labels: List[List[str]]) -> float:
    # sliding window of 4 steps over binary onset presence
    on = [1 if len(s)>0 else 0 for s in labels]
    grams = set()
    denom = 0
    for i in range(len(on)-3):
        denom += 1
        grams.add(tuple(on[i:i+4]))
    if denom == 0:
        return 0.0
    return len(grams)/denom


def fill_rate(labels: List[List[str]], bars: int) -> float:
    # crude: fraction of steps in last bar that are active minus mean of bars 1-3 (if exist); clamp 0..1
    chunks, spb = bar_chunks(labels, bars)
    if len(chunks) < 2:
        return 0.0
    dens = [sum(1 for s in ch if s)/spb for ch in chunks]
    base = float(np.mean(dens[:-1])) if len(dens) > 1 else 0.0
    inc = max(0.0, dens[-1] - base)
    return float(min(1.0, inc))


def four_on_floor(labels: List[List[str]], spb: int) -> int:
    if spb <= 0:
        return 0
    # detect kick on each quarter (positions 0, spb/4, spb/2, 3*spb/4) in any bar
    quarters = [0, spb//4, spb//2, 3*spb//4]
    has = True
    for b in range(0, len(labels), spb):
        bar = labels[b:b+spb]
        for q in quarters:
            if q >= len(bar) or ("kick" not in set(bar[q])):
                has = False
                break
        if not has:
            break
    return int(has)


def backbeat_strength(labels: List[List[str]], spb: int) -> float:
    if spb <= 0:
        return 0.0
    # snare on beats 2 & 4 (quarters at spb/4 and 3*spb/4)
    idx = [spb//4, 3*spb//4]
    scores = []
    for b in range(0, len(labels), spb):
        bar = labels[b:b+spb]
        val = sum(1.0 if (i < len(bar) and "snare" in set(bar[i])) else 0.0 for i in idx)/2.0
        scores.append(val)
    return float(np.mean(scores)) if scores else 0.0


def downbeat_strength(labels: List[List[str]], spb: int) -> float:
    if spb <= 0:
        return 0.0
    idx = [0]
    scores = []
    for b in range(0, len(labels), spb):
        bar = labels[b:b+spb]
        val = 1.0 if (idx[0] < len(bar) and any(x in ("kick","snare") for x in bar[idx[0]])) else 0.0
        scores.append(val)
    return float(np.mean(scores)) if scores else 0.0


def hat_offbeat_ratio(labels: List[List[str]], spb: int) -> float:
    if spb <= 0:
        return 0.0
    # use label contains 'hat' (open/closed)
    offs = list(range(spb//8, spb, spb//4))  # 8th offbeats: 1& 2& 3& 4& -> positions 2,6,10,14 on spb=16
    on_idx = list(range(0, spb, spb//4))     # quarters: 1 2 3 4 -> 0,4,8,12
    off_hits = 0
    on_hits = 0
    for b in range(0, len(labels), spb):
        bar = labels[b:b+spb]
        for i in offs:
            if i < len(bar) and any("hat" in x for x in bar[i]):
                off_hits += 1
        for i in on_idx:
            if i < len(bar) and any("hat" in x for x in bar[i]):
                on_hits += 1
    total = off_hits + on_hits
    return off_hits/total if total>0 else 0.0


def sectional_change_features(labels: List[List[str]], bars: int) -> dict:
    chunks, spb = bar_chunks(labels, bars)
    if not chunks:
        return {"bar_density_var": 0.0, "bar_density_range": 0.0}
    dens = np.array([sum(1 for s in ch if s)/spb for ch in chunks], dtype=float)
    return {
        "bar_density_var": float(np.var(dens)) if len(dens)>1 else 0.0,
        "bar_density_range": float(dens.max()-dens.min()) if len(dens)>0 else 0.0,
    }


def kick_pattern_class(labels: List[List[str]], spb: int) -> str:
    if spb <= 0:
        return "unknown"
    # very coarse rules on first bar
    bar = labels[:spb]
    quarters = [0, spb//4, spb//2, 3*spb//4]
    q_kicks = sum(1 for q in quarters if q < len(bar) and "kick" in set(bar[q]))
    # 2-step-ish (DnB): kick often on 1, snare on 3 (spb/2); fewer quarter kicks
    has_snare_3 = (spb//2 < len(bar) and "snare" in set(bar[spb//2]))

    if q_kicks == 4:
        return "four_on_floor"
    if q_kicks <= 2 and has_snare_3:
        return "two_step"
    # backbeat rock/pop: kick on 1, snare 2&4, irregular extra kicks
    has_k1 = (0 < len(bar) and "kick" in set(bar[0]))
    bb = backbeat_strength(labels, spb)
    if has_k1 and bb >= 0.8:
        return "backbeat_pop"
    return "syncopated_break"


# -------------------------
# Syncopation fallback (WNBD-ish) for /16 grid
# -------------------------

def syncopation_from_grid(labels: List[List[str]], bars: int) -> float:
    if not labels or bars <= 0:
        return 0.0
    n = len(labels)
    if n % bars != 0:
        n = (n // bars) * bars
        labels = labels[:n]
    spb = n // bars  # should be 16 for your grid
    # metrical weights for 16-th grid: strong→1, weak→0 after norm
    base = np.array([4,1,2,1, 3,1,2,1, 4,1,2,1, 3,1,2,1], dtype=float)
    if len(base) != spb:
        # interpolate if grid differs
        base = np.interp(np.linspace(0, len(base)-1, spb), np.arange(len(base)), base)
    w = (base - base.min()) / (base.max() - base.min() + 1e-9)

    bar_scores = []
    for b in range(bars):
        chunk = labels[b*spb:(b+1)*spb]
        onset = np.array([1.0 if len(step)>0 else 0.0 for step in chunk])
        reward = float(np.sum((1.0 - w) * onset))  # hits on weak spots
        quarters = [0, spb//4, spb//2, 3*spb//4]
        miss = sum(1.0 for q in quarters if (q < len(chunk) and len(chunk[q])==0))
        score = (reward / max(1.0, spb)) + (miss / 4.0)
        bar_scores.append(score)
    norm = [min(1.0, s/2.0) for s in bar_scores] if bar_scores else [0.0]
    return round(float(np.mean(norm)), 3)


# -------------------------
# Tempo stability handling
# -------------------------

def tempo_stability_from_beat_curve(data: dict) -> Tuple[float, bool]:
    """If a real beat tracker curve exists, compute std/mean (coefficient of variation) of instant BPM.
       Return (value, is_proxy=False). If not found, return (None, False)."""
    curve = _dig(data, ["beat_tracker", "instant_bpm"], None)
    if isinstance(curve, Sequence) and len(curve) > 1:
        arr = np.array([x for x in curve if isinstance(x, (int,float)) and x>0], dtype=float)
        if arr.size >= 2:
            cv = float(np.std(arr) / (np.mean(arr) + 1e-9))
            return cv, False
    return None, False


def tempo_stability_proxy_from_kicks(labels: List[List[str]], bars: int) -> Tuple[float, bool]:
    """Proxy using kick inter-onset intervals (frames/steps)."""
    flat = []
    for i, step in enumerate(labels):
        if "kick" in step:
            flat.append(i)
    if len(flat) < 3:
        return None, True
    iois = np.diff(flat)
    cv = float(np.std(iois) / (np.mean(iois) + 1e-9)) if len(iois)>1 else 0.0
    return cv, True


# -------------------------
# Feature builder
# -------------------------

def build_rhythm_features(data: dict) -> dict:
    rp = data.get("rhythm_pattern") or {}
    bars = int(rp.get("bars", 4)) if rp.get("bars") is not None else 4
    labels = _coerce_labels(rp.get("labels"))
    total_steps = len(labels)
    spb = infer_steps_per_bar(total_steps, bars)

    flat = flatten_multilabel(labels)

    # Base indicators
    sim_mean = bar_similarity_mean(labels, bars)
    perbar = per_bar_activity(labels, bars)
    ent = pattern_entropy(flat)
    uniq4 = ngram4_unique_ratio(labels)
    fill = fill_rate(labels, bars)
    hat_off = hat_offbeat_ratio(labels, spb)
    fof = four_on_floor(labels, spb)
    bb = backbeat_strength(labels, spb)
    db = downbeat_strength(labels, spb)

    # Swing pass-through
    swing = _dig(rp, ["swing"], None)
    # Normalize if given as ratio (0..1) or percentage
    if isinstance(swing, (int,float)):
        swing_ratio_norm = float(max(0.0, min(1.0, swing))) if swing <= 1.5 else float(max(0.0, min(1.0, swing/100.0)))
    else:
        swing_ratio_norm = None

    # Syncopation: pass-through else fallback
    sync_any = _dig(rp, ["syncopation"], None)
    if isinstance(sync_any, (int,float)):
        sync_score = float(max(0.0, min(1.0, sync_any))) if sync_any <= 1.5 else float(max(0.0, min(1.0, sync_any/100.0)))
        sync_source = "provided"
    else:
        sync_score = syncopation_from_grid(labels, bars)
        sync_source = "grid_fallback"

    # Triplets & swing awareness (keep /16; derive /12 overlay on demand)
    # Optional upstream hints
    meter_68_conf = _dig(data, ["rhythm_confidence", "meter_68_conf"], None)
    prior_trip_flag = bool(_dig(data, ["groove", "uses_triplets"], False))
    # Derive /12 overlay and hat triplet ratio
    chunks12, spb12 = _remap_to_12(labels, bars) if spb > 0 else ([], 0)
    hat_trip_ratio_12 = _hat_triplet_offbeat_ratio_12(chunks12, spb12) if spb12 == 12 else 0.0
    swing_strength = round(hat_trip_ratio_12, 3)
    # Decide uses_triplets based on prior flag OR swing heuristics
    uses_triplets = 1 if (
        prior_trip_flag or
        (swing_ratio_norm is not None and swing_ratio_norm >= 0.60) or
        (meter_68_conf is not None and meter_68_conf >= 0.60) or
        swing_strength >= 0.60
    ) else 0

    # Tempo stability: prefer real beat-tracker curve; else proxy
    ts, proxy_flag = tempo_stability_from_beat_curve(data)
    if ts is None:
        ts, proxy_flag = tempo_stability_proxy_from_kicks(labels, bars)
    tempo_stability = ts

    # Sectional changes
    sect = sectional_change_features(labels, bars)

    # Kick pattern class (coarse)
    kclass = kick_pattern_class(labels, spb)

    # Meter heuristic from grid (optional, lightweight)
    meter = {"is_44_confidence": 1.0 if spb in (8, 16, 32) else 0.5}

    features = {
        # grid
        "bars": bars,
        "steps_per_bar": spb,
        "total_steps": total_steps,
        # rhythm shape
        "bar_similarity_mean": round(sim_mean, 3),
        "per_bar_activity": perbar,
        "pattern_entropy": round(ent, 3),
        "ngram4_unique_ratio": round(uniq4, 3),
        "fill_rate": round(fill, 3),
        # placements
        "four_on_floor": int(fof),
        "backbeat_strength": round(bb, 3),
        "downbeat_strength": round(db, 3),
        "hat_offbeat_ratio": round(hat_off, 3),
        # groove
        "swing_ratio_norm": swing_ratio_norm,
        "swing_strength": swing_strength,
        "syncopation_score_01": sync_score,
        "syncopation_source": sync_source,
        "uses_triplets": uses_triplets,
        # tempo
        "tempo_stability": tempo_stability,
        "tempo_stability_is_proxy": bool(proxy_flag) if tempo_stability is not None else None,
        # sections & patterns
        "bar_density_var": round(sect["bar_density_var"], 4),
        "bar_density_range": round(sect["bar_density_range"], 3),
        "kick_pattern_class": kclass,
        # meter
        "meter_hint": meter,
        # companion overlay diagnostics (do not replace /16)
        "triplet_overlay_hat_ratio_12": round(hat_trip_ratio_12, 3),
    }

    return features


# -------------------------
# IO
# -------------------------

def run(folder_path: str):
    folder = Path(folder_path)
    json_path = next((f for f in folder.glob("*.json") if not f.name.startswith(".")), None)
    if not json_path:
        print("❌ JSON file not found in", folder_path)
        return

    with open(json_path, "r") as f:
        data = json.load(f)

    # Build features from the data we actually have
    feats = build_rhythm_features(data)

    # Attach under rhythm_pattern.genre_features
    if "rhythm_pattern" not in data or not isinstance(data["rhythm_pattern"], dict):
        data["rhythm_pattern"] = {}
    data["rhythm_pattern"]["genre_features"] = feats

    with open(json_path, "w") as f:
        json.dump(data, f, indent=2, separators=(",", ": "))

    # Console preview
    print("\n✅ Wrote rhythm genre_features →", json_path.name)
    key_peek = {k: feats[k] for k in (
        "steps_per_bar", "bar_similarity_mean", "pattern_entropy", "ngram4_unique_ratio",
        "four_on_floor", "backbeat_strength", "swing_ratio_norm", "swing_strength", "syncopation_score_01",
        "tempo_stability", "tempo_stability_is_proxy", "kick_pattern_class", "uses_triplets", "triplet_overlay_hat_ratio_12"
    ) if k in feats}
    for k, v in key_peek.items():
        print(f"   • {k}: {v}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 8: Rhythm consistency → genre-ready rhythm features")
    parser.add_argument("folder", type=str, help="Path to folder containing the track JSON")
    args = parser.parse_args()
    run(args.folder)


def run_stage8(folder_path: str):
    """Controller-compatible entrypoint: mirrors `run(folder_path)`.
    Returns None; raises on failure so controller can catch.
    """
    return run(folder_path)
