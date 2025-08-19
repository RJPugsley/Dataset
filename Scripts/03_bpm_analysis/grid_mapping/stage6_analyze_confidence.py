# stage6_analyze_confidence.py
import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Sequence, Set, Any, Tuple, Dict, Optional
import math
import numpy as np

# ---------------- core helpers (unchanged) ----------------
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
    flat = []
    for step in labels:
        flat.extend(step)
    return flat

def compute_density(labels: List[List[str]]) -> float:
    if not labels:
        return 0.0
    active = sum(1 for step in labels if step)
    return round(active / len(labels), 3)

def jaccard_similarity(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0

def _labels_to_sets(chunk: List[List[str]]) -> List[Set[str]]:
    return [set(step) for step in chunk]

def compute_repetition(labels: List[List[str]], bars: int = 4) -> float:
    n = len(labels)
    if n == 0 or bars <= 0:
        return 0.0
    if n % bars != 0:
        n_trim = (n // bars) * bars
        if n_trim == 0:
            return 0.0
        labels = labels[:n_trim]
        n = n_trim
    bar_len = n // bars
    chunks = [labels[i * bar_len:(i + 1) * bar_len] for i in range(bars)]
    ref = _labels_to_sets(chunks[0])
    sims = []
    for b in range(1, bars):
        cur = _labels_to_sets(chunks[b])
        step_sims = [jaccard_similarity(a, c) for a, c in zip(ref, cur)]
        sims.append(float(np.mean(step_sims)) if step_sims else 0.0)
    return round(float(np.mean(sims)) if sims else 0.0, 3)

def compute_kick_snare_ratio(flat_labels: List[str]):
    c = Counter(flat_labels)
    kicks = c.get("kick", 0)
    snares = c.get("snare", 0)
    if kicks == 0 and snares == 0:
        return {"ratio": 0.0, "kicks": 0, "snares": 0}
    if snares == 0:
        return {"ratio": None, "kicks": kicks, "snares": 0}
    return {"ratio": round(kicks / snares, 3), "kicks": kicks, "snares": snares}

def compute_dominant_label(flat_labels: List[str]) -> str:
    c = Counter(flat_labels)
    return c.most_common(1)[0][0] if c else "none"

def _entropy_from_counts(counts: Sequence[int]) -> float:
    total = sum(counts)
    if total <= 0:
        return 0.0
    probs = [c/total for c in counts if c > 0]
    return -sum(p * math.log2(p) for p in probs)

def compute_label_entropy(flat_labels: List[str]) -> float:
    c = Counter(flat_labels)
    return round(_entropy_from_counts(c.values()), 3)

def compute_avg_labels_per_step(labels: List[List[str]]) -> float:
    if not labels:
        return 0.0
    return round(float(np.mean([len(step) for step in labels])), 3)

def per_label_counts_and_density(labels: List[List[str]]):
    flat = flatten_multilabel(labels)
    total_steps = len(labels) if labels else 0
    counts = Counter(flat)
    density = {k: round(v / total_steps, 3) if total_steps else 0.0 for k, v in counts.items()}
    return {"counts": dict(counts), "density_per_step": density}

def per_bar_activity(labels: List[List[str]], bars: int):
    n = len(labels)
    if n == 0 or bars <= 0:
        return []
    if n % bars != 0:
        n = (n // bars) * bars
        labels = labels[:n]
    bar_len = n // bars
    out = []
    for b in range(bars):
        chunk = labels[b * bar_len:(b + 1) * bar_len]
        active = sum(1 for s in chunk if s)
        out.append({
            "bar_index": b + 1,
            "active_steps": active,
            "density": round(active / bar_len, 3) if bar_len else 0.0
        })
    return out

def bar_to_bar_similarity(labels: List[List[str]], bars: int):
    n = len(labels)
    if n == 0 or bars <= 1:
        return []
    if n % bars != 0:
        n = (n // bars) * bars
        labels = labels[:n]
    bar_len = n // bars
    chunks = [_labels_to_sets(labels[i*bar_len:(i+1)*bar_len]) for i in range(bars)]
    sims = []
    for i in range(bars):
        row = []
        for j in range(bars):
            step_sims = [jaccard_similarity(a, b) for a, b in zip(chunks[i], chunks[j])]
            row.append(round(float(np.mean(step_sims)) if step_sims else 0.0, 3))
        sims.append(row)
    return sims

def infer_steps_per_bar(total_steps: int, bars: int) -> int:
    if bars > 0 and total_steps % bars == 0:
        return total_steps // bars
    return 0

# ---------------- JSON readers ----------------
def _dig(d: dict, path: List[str], default=None):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def find_labels_and_grid(data: dict) -> Tuple[List[List[str]], int]:
    labels = _coerce_labels(_dig(data, ["rhythm_pattern", "labels"], []))
    bars = _dig(data, ["rhythm_pattern", "bars"], None)
    if not labels:
        labels = _coerce_labels(_dig(data, ["events", "rhythm", "labels"], []))
    if bars is None:
        bars = _dig(data, ["meter", "bars"], None)
    if bars is None:
        bars = 4
    return labels, int(bars)

def find_swing_syncopation(data: dict) -> Tuple[Optional[Any], Optional[Any]]:
    swing = _dig(data, ["rhythm_pattern", "swing"], None)
    if swing is None:
        swing = _dig(data, ["groove", "swing"], None)
    sync = _dig(data, ["rhythm_pattern", "syncopation"], None)
    if sync is None:
        sync = _dig(data, ["groove", "syncopation"], None)
    return swing, sync

# ---------------- rhythm features for genre ----------------
def _split_into_bars(labels: List[List[str]], bars: int) -> Tuple[List[List[List[str]]], int]:
    n = len(labels)
    if bars <= 0 or n == 0:
        return [], 0
    if n % bars != 0:
        n = (n // bars) * bars
        labels = labels[:n]
    spb = n // bars if bars else 0
    chunks = [labels[i*spb:(i+1)*spb] for i in range(bars)]
    return chunks, spb

def _indices_with(labelset: Set[str], step_labels: List[str]) -> bool:
    return any(l in labelset for l in step_labels)

def _hash_step(step: List[str]) -> str:
    # canonical, order-independent
    return "|".join(sorted(set(step))) if step else "_"

def _lz76_complexity(seq: List[str]) -> float:
    """Lempel–Ziv complexity normalized by sequence length."""
    s = "\x1f".join(seq)  # separator unlikely in tokens
    i, n, c = 0, len(s), 1 if s else 0
    l, k, k_max = 1, 0, 1
    while True:
        if i + l > n:
            c += 1
            break
        if s[i:i+l] in s[0:i]:
            l += 1
            if i + l > n:
                c += 1
                break
        else:
            c += 1
            i += l
            l = 1
        if i == n:
            break
    return round(c / max(1, n), 3)

def _quarter_positions(spb: int) -> List[int]:
    # positions of beats 1..4 within a bar at step resolution
    return [round(k * (spb / 4)) % spb for k in range(4)]

def _offbeat_positions(spb: int) -> List[int]:
    # "ands" between quarters: 1/8 divisions offset by 1/8
    return [round((k + 0.5) * (spb / 4)) % spb for k in range(4)]

def _triplet_positions(spb: int) -> List[int]:
    # thirds inside each quarter: 1/12 steps into beat
    offs = []
    if spb % 12 != 0:  # we need a grid divisible by 12 to be meaningful
        return offs
    third = spb // 12  # one third of a beat
    for k in range(4):
        base = k * (spb // 4)
        offs.extend([(base + third) % spb, (base + 2*third) % spb])
    return sorted(list(set(offs)))

def _presence_ratio(steps: List[List[str]], idxs: List[int], want: Set[str]) -> float:
    if not steps or not idxs:
        return 0.0
    hits = 0
    total = 0
    m = len(steps)
    for i in idxs:
        total += 1
        if _indices_with(want, steps[i % m]):
            hits += 1
    return round(hits / total, 3) if total else 0.0

def _density_of(label: str, steps: List[List[str]]) -> float:
    if not steps:
        return 0.0
    c = sum(1 for s in steps if label in s)
    return round(c / len(steps), 3)

def _kick_ioi_cv(labels: List[List[str]], bars: int) -> Optional[float]:
    # IOIs in steps between 'kick' onsets across full grid
    idxs = [i for i, s in enumerate(labels) if "kick" in s]
    if len(idxs) < 3:
        return None
    diffs = np.diff(idxs)
    if len(diffs) < 2:
        return None
    mu = float(np.mean(diffs))
    sd = float(np.std(diffs))
    if mu <= 0:
        return None
    cv = sd / mu
    # soft clamp to [0,1] so it's classifier-friendly
    return round(max(0.0, min(1.0, cv)), 3)

def _meter_confidences(bar_steps: List[List[str]], spb: int) -> Tuple[float, float]:
    """
    Very light heuristic from step activity autocorr:
    - meter_44_conf: periodicity at 4 quarters (spb/4)
    - meter_68_conf: periodicity at 6 eighths (spb/6 * if divisible)
    """
    if spb <= 0:
        return 0.0, 0.0
    activity = np.array([1.0 if len(s) > 0 else 0.0 for s in bar_steps], dtype=float)
    if np.all(activity == 0):
        return 0.0, 0.0
    def score(period: int) -> float:
        if period <= 0:
            return 0.0
        # wrap correlation at that lag
        a = activity
        b = np.roll(activity, period)
        val = float(np.dot(a, b) / max(1.0, np.dot(a, a)))
        return val
    p44 = round(score(spb // 4), 3)
    p68 = round(score(spb // 6) if spb % 6 == 0 else 0.0, 3)
    # normalize to 0..1 by softmax-ish scaling
    total = math.exp(p44) + math.exp(p68)
    if total == 0:
        return 0.0, 0.0
    return round(math.exp(p44)/total, 3), round(math.exp(p68)/total, 3)

def _ngram_unique_ratio(step_tokens: List[str], n: int = 4) -> float:
    if len(step_tokens) < n:
        return 0.0
    grams = ["\x1f".join(step_tokens[i:i+n]) for i in range(len(step_tokens)-n+1)]
    uniq = len(set(grams))
    return round(uniq / len(grams), 3)

def _bar_similarity_mean(matrix: List[List[float]]) -> Optional[float]:
    if not matrix:
        return None
    vals = []
    B = len(matrix)
    for i in range(B):
        for j in range(B):
            if i != j:
                vals.append(matrix[i][j])
    return round(float(np.mean(vals)), 3) if vals else None

def _fill_rate(bars_chunks: List[List[List[str]]], spb: int) -> float:
    """Share of bars where the last two steps spike vs bar avg."""
    if not bars_chunks or spb <= 0:
        return 0.0
    spikes = 0
    for steps in bars_chunks:
        if not steps:
            continue
        per_step = [len(s) for s in steps]
        bar_avg = float(np.mean(per_step)) if per_step else 0.0
        tail = per_step[-2:] if len(per_step) >= 2 else per_step
        spike = any(t > (bar_avg * 1.5) and t >= 1 for t in tail)
        spikes += 1 if spike else 0
    return round(spikes / len(bars_chunks), 3) if bars_chunks else 0.0

def _boolean(x: bool) -> int:
    return 1 if x else 0

def build_rhythm_features(labels: List[List[str]], bars: int, bpm: Optional[float], swing_any: Any, sync_any: Any) -> Dict[str, Any]:
    bars_chunks, spb = _split_into_bars(labels, bars)
    flat = flatten_multilabel(labels)
    step_tokens = [_hash_step(s) for s in labels]

    # Defaults
    features: Dict[str, Any] = {}

    # Tempo
    features["tempo_bpm"] = round(float(bpm), 3) if bpm is not None else None
    features["tempo_stability"] = _kick_ioi_cv(labels, bars)

    # Meter & grid
    if bars_chunks and spb > 0:
        m44, m68 = _meter_confidences(bars_chunks[0], spb)
    else:
        m44, m68 = 0.0, 0.0
    features["meter_44_conf"] = m44
    features["meter_68_conf"] = m68

    # Swing & triplets (pass-through + heuristic flags)
    # If your JSON provided normalized swing already, we keep it. Otherwise None.
    features["swing_ratio_norm"] = swing_any if isinstance(swing_any, (int, float)) else None

    uses_trip = False
    if spb and spb % 12 == 0:
        trip_pos = set(_triplet_positions(spb))
        for steps in bars_chunks:
            if any(_indices_with({"hihat", "ohat", "hat", "perc"}, steps[i]) for i in trip_pos):
                uses_trip = True
                break
    features["uses_triplets"] = _boolean(uses_trip)
    # Optional: shuffle if triplets + offbeats sparse
    features["shuffle_flag"] = _boolean(uses_trip and features.get("swing_ratio_norm") not in (None, 0))

    # Four-on-the-floor / Backbeat / Downbeat
    if spb > 0 and bars_chunks:
        qpos = _quarter_positions(spb)        # beats 1..4
        opos = _offbeat_positions(spb)        # "&" of each beat
        kicks_q = []
        snares_q = []
        hats_off = []
        down_hits = 0
        total_down = 0

        for steps in bars_chunks:
            kicks_q.append(_presence_ratio(steps, qpos, {"kick"}))
            snares_q.append(_presence_ratio(steps, qpos, {"snare"}))
            hats_off.append(_presence_ratio(steps, opos, {"hihat","hat","ohat"}))
            total_down += 1
            if _indices_with({"kick","snare","hihat","hat","ohat"}, steps[0]):
                down_hits += 1

        # Four-on-the-floor: kick present on most quarter slots across bars
        features["four_on_floor"] = _boolean(np.mean(kicks_q) >= 0.8 if kicks_q else False)
        # Backbeat strength: snare on 2 & 4 vs all quarters
        if spb >= 4:
            b2 = round((spb/4)*1)
            b4 = round((spb/4)*3)
            sn2_4 = []
            sn_all = []
            for steps in bars_chunks:
                sn2_4.append(_presence_ratio(steps, [b2,b4], {"snare"}))
                sn_all.append(_presence_ratio(steps, _quarter_positions(spb), {"snare"}))
            num = float(np.mean(sn2_4)) if sn2_4 else 0.0
            den = float(np.mean(sn_all)) if sn_all else 0.0
            features["backbeat_strength"] = round(num / den, 3) if den > 0 else 0.0
        else:
            features["backbeat_strength"] = 0.0

        features["downbeat_strength"] = round(down_hits / total_down, 3) if total_down else 0.0
        features["hat_offbeat_ratio"] = float(np.mean(hats_off)) if hats_off else 0.0
    else:
        features["four_on_floor"] = 0
        features["backbeat_strength"] = 0.0
        features["downbeat_strength"] = 0.0
        features["hat_offbeat_ratio"] = 0.0

    # Densities per beat (kick/snare), averaged over bars
    if spb > 0 and bars_chunks:
        qpos = _quarter_positions(spb)
        kd = defaultdict(list)
        sd = defaultdict(list)
        for steps in bars_chunks:
            for bi, qp in enumerate(qpos, start=1):
                kd[bi].append(1.0 if "kick" in steps[qp] else 0.0)
                sd[bi].append(1.0 if "snare" in steps[qp] else 0.0)
        for bi in range(1,5):
            features[f"kick_density_b{bi}"] = round(float(np.mean(kd[bi])), 3) if kd[bi] else 0.0
            features[f"snare_density_b{bi}"] = round(float(np.mean(sd[bi])), 3) if sd[bi] else 0.0
    else:
        for bi in range(1,5):
            features[f"kick_density_b{bi}"] = 0.0
            features[f"snare_density_b{bi}"] = 0.0

    # Syncopation (passed through if present as scalar 0..1)
    features["syncopation_score_01"] = sync_any if isinstance(sync_any, (int, float)) else None

    # Complexity & repetition
    features["pattern_entropy"] = round(_entropy_from_counts(Counter(step_tokens).values()), 3)
    features["ngram4_unique_ratio"] = _ngram_unique_ratio(step_tokens, n=4)
    features["lzc_complexity"] = _lz76_complexity(step_tokens)

    bar_sim = bar_to_bar_similarity(labels, bars)
    features["bar_similarity_mean"] = _bar_similarity_mean(bar_sim)

    # Fills
    features["fill_rate"] = _fill_rate(bars_chunks, spb)

    return features

# ---------------- main ----------------
def run_stage6(folder_path: str):
    print("\n🚀 Stage 6: Analyze Loop Confidence + Rhythm Features (JSON‑first)")

    folder = Path(folder_path)
    json_path = next((f for f in folder.glob("*.json") if not f.name.startswith(".")), None)
    if not json_path:
        print("❌ JSON file not found.")
        return

    with open(json_path) as f:
        data = json.load(f)

    labels, bars = find_labels_and_grid(data)
    if not labels:
        print("❌ No rhythm labels found. Expected rhythm_pattern.labels or events.rhythm.labels.")
        return

    # context values if present
    bpm = _dig(data, ["bpm"], None) or _dig(data, ["tempo", "bpm"], None) or _dig(data, ["track", "bpm"], None)
    swing_any, sync_any = find_swing_syncopation(data)

    flat_labels = flatten_multilabel(labels)
    total_steps = len(labels)
    spb = infer_steps_per_bar(total_steps, bars)

    # ---------- confidence (kept as-is + pass-through swing/sync) ----------
    print("📈 Calculating confidence metrics...")
    confidence = {
        "density": compute_density(labels),
        "repetition": compute_repetition(labels, bars),
        "kick_snare_ratio": compute_kick_snare_ratio(flat_labels),
        "dominant_label": compute_dominant_label(flat_labels),
        "label_entropy": compute_label_entropy(flat_labels),
        "avg_labels_per_step": compute_avg_labels_per_step(labels),
        "per_label": per_label_counts_and_density(labels),
        "per_bar_activity": per_bar_activity(labels, bars),
        "bar_similarity_matrix": bar_to_bar_similarity(labels, bars),
        "grid": {
            "bars": bars,
            "total_steps": total_steps,
            "steps_per_bar": spb
        }
    }
    if swing_any is not None:
        confidence["swing"] = swing_any
    if sync_any is not None:
        confidence["syncopation"] = sync_any

    # ---------- NEW: classifier-friendly rhythm feature vector ----------
    print("🧠 Building rhythm features for classifier...")
    features = build_rhythm_features(labels, bars, bpm, swing_any, sync_any)

    # write back
    rp = data.get("rhythm_pattern", {})
    rp["confidence"] = confidence
    rp["features"] = features
    data["rhythm_pattern"] = rp
    # also mirror a flat copy at root for easy dataset building
    data["features_rhythm"] = features

    with open(json_path, "w") as f:
        json.dump(data, f, indent=2, separators=(",", ": "))

    # console preview
    show = ["density","repetition","dominant_label","label_entropy","avg_labels_per_step"]
    print("— confidence:")
    for k in show:
        print(f"   • {k}: {confidence.get(k)}")
    print("— features:")
    for k in ["tempo_bpm","tempo_stability","meter_44_conf","meter_68_conf",
              "swing_ratio_norm","uses_triplets","shuffle_flag",
              "four_on_floor","backbeat_strength","downbeat_strength",
              "hat_offbeat_ratio","syncopation_score_01",
              "pattern_entropy","ngram4_unique_ratio","lzc_complexity",
              "bar_similarity_mean","fill_rate"]:
        print(f"   • {k}: {features.get(k)}")

    print("🏁 Stage 6 complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 6: Analyze loop confidence + rhythm features (JSON‑first)")
    parser.add_argument("folder", type=str, help="Path to folder with rhythm JSON")
    args = parser.parse_args()
    run_stage6(args.folder)
