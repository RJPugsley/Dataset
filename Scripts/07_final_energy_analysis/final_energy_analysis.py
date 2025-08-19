import os
import json
import numpy as np
import subprocess
import argparse
from scipy.signal import find_peaks
from scipy.stats import skew, variation

def final_energy_analysis(folder):
    print(f"\n📁 Analyzing folder: {folder}")

    # Locate JSON and source audio
    json_path = next((os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".json")), None)
    if not json_path:
        print("❌ JSON file not found.")
        return

    folder_name = os.path.basename(folder).lower()
    source_path = next(
        (os.path.join(folder, folder_name + ext) for ext in [".wav", ".mp3", ".flac", ".m4a"]
         if os.path.exists(os.path.join(folder, folder_name + ext))),
        None
    )
    if not source_path:
        print("❌ Source audio file not found.")
        return

    with open(json_path, "r") as f:
        data = json.load(f)

    sr, audio = load_audio_ffmpeg(source_path)
    envelope = compute_energy_envelope(audio, sr)

    avg_energy = float(np.mean(envelope))
    energy_std = float(np.std(envelope))
    peak_energy = float(np.max(envelope))
    skewness = float(skew(envelope))
    peaks, _ = find_peaks(envelope, height=avg_energy * 1.5, distance=10)
    drop_count = int(len(peaks))
    ramp = np.linspace(0, 1, len(envelope))
    norm_env = (envelope - np.min(envelope)) / (np.max(envelope) - np.min(envelope) + 1e-9)
    ramp_up_score = float(np.corrcoef(norm_env, ramp)[0, 1])
    silence_thresh = 0.05 * np.max(envelope)
    break_silence_ratio = float(np.sum(envelope < silence_thresh) / len(envelope))

    bpm = data.get("bpm")
    onsets = data.get("onsets")

    features = {
        "avg_energy": avg_energy,
        "energy_std": energy_std,
        "peak_energy": peak_energy,
        "energy_skewness": skewness,
        "drop_count": drop_count,
        "ramp_up_score": ramp_up_score,
        "break_silence_ratio": break_silence_ratio,
        "frame_ms": 50,
        "sampling_rate": sr
    }

    features.update(advanced_energy_features(envelope, sr, 50, bpm=bpm, onsets=onsets))

    data["energy_envelope_analysis"] = features
    data["energy_envelopes"] = { "mix": ",".join([f"{v:.6f}" for v in envelope]) }

    # -------------------------
    # ADDITION: unified genre_features hub (write-once)
    # - Collate compact rhythm vector (if present)
    # - Mirror selected scalars from energy, bass, harmony with stable, flat keys
    # -------------------------
    def _dig_local(d, path, default=None):
        cur = d
        for k in path:
            if isinstance(cur, dict) and k in cur:
                cur = cur[k]
            else:
                return default
        return cur

    def _pick_numeric(d, allow):
        out = {}
        if isinstance(d, dict):
            for k in allow:
                v = d.get(k, None)
                if isinstance(v, (int, float)) and not (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
                    out[k] = float(v)
        return out

    # Rhythm scalars from prior stage if available
    rhythm_src = _dig_local(data, ["rhythm_pattern", "genre_features"], {}) or {}
    rhythm_allow = [
        "steps_per_bar","bar_similarity_mean","pattern_entropy","ngram4_unique_ratio",
        "fill_rate","four_on_floor","backbeat_strength","downbeat_strength",
        "hat_offbeat_ratio","swing_ratio_norm","syncopation_score_01",
        "tempo_stability","bar_density_var","bar_density_range"
    ]
    rhythm_flat = _pick_numeric(rhythm_src, rhythm_allow)
    if isinstance(rhythm_src.get("kick_pattern_class"), str):
        rhythm_flat["rhythm_kick_pattern_class"] = rhythm_src["kick_pattern_class"]
    rhythm_prefixed = {("rhythm_" + k) if not k.startswith("rhythm_") else k: v for k, v in rhythm_flat.items()}

    # Energy scalars from this script's output
    energy_src = data.get("energy_envelope_analysis", {}) or {}
    energy_allow = [
        "avg_energy","energy_std","peak_energy","energy_flatness","energy_crest",
        "energy_variation","beat_energy_mean","beat_energy_std","ramp_up_score",
        "drop_count","break_silence_ratio","energy_repetition_score"
    ]
    energy_pick = {f"energy_{k}": v for k, v in _pick_numeric(energy_src, energy_allow).items()}

    # Bass scalars (if available)
    bass_src = data.get("bassline_analysis", {}) or {}
    bass_allow = [
        "bass_energy_mean","bass_energy_std","bass_low_freq_ratio","bass_pitch_stability",
        "bass_smoothness","bass_onset_density","bass_energy_sustain_ratio","bass_groove_score"
    ]
    # ensure keys are bass_* prefixed
    bass_raw = _pick_numeric(bass_src, bass_allow)
    bass_pick = {(k if k.startswith("bass_") else "bass_" + k): v for k, v in bass_raw.items()}

    # Harmony scalars (if available)
    harm_src = _dig_local(data, ["harmonic_analysis","source_mix"], {}) or {}
    harm_allow = ["tonal_brightness","tonal_variability","harmony_complexity"]
    harm_pick = {f"harmony_{k}": v for k, v in _pick_numeric(harm_src, harm_allow).items()}
    # convenience mirrors
    try:
        bpm_val = float(data.get("bpm"))
        if bpm_val > 0:
            harm_pick["tempo_bpm"] = bpm_val
    except (TypeError, ValueError):
        pass
    if isinstance(data.get("key"), str):
        harm_pick["camelot_key"] = data["key"]

    hub = {}
    hub.update(rhythm_prefixed)
    hub.update(energy_pick)
    hub.update(bass_pick)
    hub.update(harm_pick)
    data["genre_features"] = hub
    # ------------------------- end addition -------------------------

    # -------------------------
    # ADDITION: Section-aware features (intros/outros/drops)
    # - Uses beat grid + energy envelope + kick/hat tracks
    # - Writes boundaries under data["sections"]
    # - Mirrors scalars into data["genre_features"]
    # -------------------------
    try:
        sections, section_feats = _compute_section_features(data, envelope, frame_ms=features.get("frame_ms", 50))
        if sections:
            data["sections"] = sections
        if section_feats:
            # mirror into hub with stable names
            gf = data.get("genre_features", {})
            gf.update({
                "section_drop_energy_crest": section_feats.get("drop_energy_crest"),
                "section_intro_hat_ratio": section_feats.get("intro_hat_ratio"),
                "section_break_syncopation": section_feats.get("break_syncopation"),
                "sections_confidence": section_feats.get("sections_confidence"),
            })
            data["genre_features"] = gf
    except Exception as e:
        print(f"⚠️ Section feature computation skipped: {e}")
    # ------------------------- end addition -------------------------

    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)

    print("✅ Full energy envelope + genre feature analysis complete.")


# --- Supporting Functions ---

def load_audio_ffmpeg(wav_path, target_sr=22050):
    print(f"🔊 Loading full mix: {wav_path}")
    command = ["ffmpeg", "-i", wav_path, "-ac", "1", "-ar", str(target_sr), "-f", "f32le", "-"]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg error: {result.stderr.decode()}")
    audio = np.frombuffer(result.stdout, dtype=np.float32)
    return target_sr, audio

def compute_energy_envelope(audio, sr, frame_ms=50):
    frame_len = int(sr * frame_ms / 1000)
    hop_len = frame_len
    envelope = []
    for i in range(0, len(audio) - frame_len, hop_len):
        frame = audio[i:i+frame_len]
        rms = np.sqrt(np.mean(frame ** 2))
        envelope.append(rms)
    return np.array(envelope)

def advanced_energy_features(envelope, sr, frame_ms, bpm=None, onsets=None):
    features = {}
    norm_env = (envelope - np.min(envelope)) / (np.max(envelope) - np.min(envelope) + 1e-9)

    features["histogram"] = ",".join([str(v) for v in np.histogram(norm_env, bins=10, range=(0,1))[0]])
    features["energy_flatness"] = float(np.exp(np.mean(np.log(envelope + 1e-9))) / (np.mean(envelope) + 1e-9))
    features["energy_crest"] = float(np.max(envelope) / (np.mean(envelope) + 1e-9))
    features["energy_variation"] = float(variation(envelope))

    autocorr = np.correlate(norm_env, norm_env, mode='full')[len(norm_env):]
    autocorr[0] = 0
    features["energy_repetition_score"] = float(np.max(autocorr))

    # ✅ Convert BPM safely
    try:
        bpm = float(bpm)
        if bpm <= 0:
            raise ValueError
    except (ValueError, TypeError):
        bpm = None

    if bpm:
        seconds_per_beat = 60.0 / bpm
        frames_per_beat = int((seconds_per_beat * 1000) / frame_ms)
        beat_means = [np.mean(norm_env[i:i+frames_per_beat]) for i in range(0, len(norm_env), frames_per_beat)]
        features["beat_energy_mean"] = float(np.mean(beat_means))
        features["beat_energy_std"] = float(np.std(beat_means))

    if onsets and len(onsets) == len(envelope):
        features["onset_energy_correlation"] = float(np.corrcoef(onsets, envelope)[0, 1])

    return features

# --- NEW: Sectioning helpers (self-contained; no external deps) ---

def _safe_get(obj, path, default=None):
    cur = obj
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def _beat_grid_from_json(data):
    # Prefer explicit beat_grid (absolute times in seconds)
    grid = data.get("beat_grid")
    if isinstance(grid, list) and grid:
        return [float(t) for t in grid], 4  # 4 slots per beat assumed by Stage 4
    # Fallback: derive from rhythm_pattern.slots
    slots = _safe_get(data, ["rhythm_pattern", "slots"], [])
    if isinstance(slots, list) and slots:
        times = [float(s.get("time", 0.0)) for s in slots]
        return times, 4
    # Last resort: loop_start + tick_duration if present
    rp = data.get("rhythm_pattern") or {}
    tick = rp.get("tick_duration")
    start = rp.get("start_time") or data.get("loop_start")
    phrase_beats = rp.get("phrase_beats") or 16
    if isinstance(tick, (int,float)) and isinstance(start, (int,float)):
        total_ticks = phrase_beats * 4
        return [round(start + i * float(tick), 6) for i in range(total_ticks)], 4
    return None, 4

def _align_envelope_to_slots(envelope, frame_ms, grid_times):
    # frame index for each slot time (seconds -> frame_ms grid)
    step = frame_ms / 1000.0
    idx = np.clip(np.round(np.asarray(grid_times, dtype=float) / step).astype(int), 0, len(envelope)-1)
    return envelope[idx]

def _aggregate_per_beat(per_slot_values):
    # 4 slots per beat
    n = len(per_slot_values)
    beats = n // 4
    vals = np.asarray(per_slot_values[:beats*4], dtype=float).reshape(beats, 4)
    return vals.mean(axis=1)

def _movavg(x, w):
    if len(x) < 1:
        return x
    w = max(1, int(w))
    c = np.convolve(x, np.ones(w)/w, mode="same")
    return c

def _local_maxima(x):
    # simple interior maxima
    x = np.asarray(x, dtype=float)
    if x.size < 3: return np.array([], dtype=int)
    return np.where((x[1:-1] > x[:-2]) & (x[1:-1] > x[2:]))[0] + 1

def _section_boundaries(data, beat_energy, beat_kick, frame_ms, bpm):
    beats_n = len(beat_energy)
    if beats_n < 16:
        return {}, {"sections_confidence": 0.0}

    # Smooth & z-score energy
    smooth = _movavg(beat_energy, w=9)
    mu, sd = float(np.mean(smooth)), float(np.std(smooth) + 1e-9)
    z = (smooth - mu) / sd

    # Kick density baseline (median)
    kd = np.asarray(beat_kick, dtype=float) if beat_kick is not None else None
    kd_med = float(np.median(kd)) if kd is not None and len(kd) else 0.0

    # Drop candidates: local maxima with z >= 1.0, positive slope, and strong kick
    slope = np.gradient(smooth)
    cands = _local_maxima(smooth)
    strong = []
    for i in cands:
        cond = (z[i] >= 1.0) and (slope[i] > 0)
        if kd is not None and len(kd):
            cond = cond and (kd[i] >= kd_med)
        if cond:
            strong.append(i)
    drop_idx = strong[0] if strong else (int(np.argmax(smooth)) if beats_n >= 3 else None)

    # Intro end: first beat where smooth >= 0.4*max or just before first drop
    intro_end = None
    if beats_n:
        thresh = 0.4 * float(np.max(smooth))
        for i in range(min(drop_idx or beats_n, beats_n)):
            if smooth[i] >= thresh:
                intro_end = i
                break
        if intro_end is None:
            intro_end = min(drop_idx or beats_n, beats_n//8)

    # Break detection: valley after drop where energy dips and kick is sparse
    break_start = break_end = None
    if drop_idx is not None:
        search_lo = min(beats_n-1, drop_idx + 4)
        search_hi = min(beats_n-1, drop_idx + max(16, beats_n//6))
        if search_hi > search_lo:
            seg = smooth[search_lo:search_hi+1]
            valley_rel = int(np.argmin(seg))
            valley = search_lo + valley_rel
            # conditions: 25% drop from local pre-drop max and low kick
            pre_max = float(np.max(smooth[max(0, drop_idx-8):drop_idx+1]))
            cond_energy = (smooth[valley] <= 0.75 * pre_max)
            cond_kick = True
            if kd is not None and len(kd):
                cond_kick = (kd[valley] <= 0.30 * (kd_med + 1e-9))
            if cond_energy and cond_kick:
                break_start = max(drop_idx + 2, valley - 4)
                break_end = min(beats_n - 1, valley + 4)

    # Outro: last 25% or sustained decline after last drop
    outro_start = None
    if beats_n >= 32:
        default_outro = int(beats_n * 0.75)
        if strong:
            last_drop = strong[-1]
            # find sustained negative slope
            neg = np.where(slope[last_drop:] < 0)[0]
            outro_start = (last_drop + neg[0]) if len(neg) else default_outro
        else:
            outro_start = default_outro

    # Build sections dict in seconds
    beat_len = 60.0 / float(bpm) if isinstance(bpm, (int,float)) and bpm else None
    to_time = (lambda b: round(b * beat_len, 3)) if beat_len else (lambda b: None)
    sections = {}
    if intro_end is not None:
        sections["intro"] = {"start_s": to_time(0), "end_s": to_time(intro_end), "beats": int(intro_end)}
    if drop_idx is not None:
        sections["drop_1"] = {"center_s": to_time(drop_idx), "window_beats": 8}
    if break_start is not None and break_end is not None:
        sections["break_1"] = {"start_s": to_time(break_start), "end_s": to_time(break_end)}
    if outro_start is not None:
        sections["outro"] = {"start_s": to_time(outro_start), "end_s": to_time(beats_n-1)}

    conf = 1.0
    if kd is None or not len(kd):
        conf *= 0.7
    if beats_n < 48:
        conf *= 0.8
    return sections, {"sections_confidence": round(conf, 3),
                      "intro_end_idx": intro_end,
                      "drop_idx": drop_idx,
                      "break_idx": (break_start, break_end),
                      "outro_idx": outro_start}

def _compute_section_features(data, envelope, frame_ms=50):
    # Beat grid (slot times in seconds)
    grid_times, slots_per_beat = _beat_grid_from_json(data)
    if not grid_times:
        return {}, {}
    total_slots = len(grid_times)
    if total_slots < 16:
        return {}, {}

    # Align energy to slots and aggregate per beat
    slot_energy = _align_envelope_to_slots(envelope, frame_ms, grid_times)
    beat_energy = _aggregate_per_beat(slot_energy)

    # Kick / hat tracks from Stage 5
    instr = data.get("instrument_tracks") or {}
    kick_slots = instr.get("kick")
    hat_slots = instr.get("hihat") or instr.get("hats")  # alias
    if isinstance(kick_slots, list) and len(kick_slots) >= total_slots:
        beat_kick = _aggregate_per_beat(kick_slots)
    else:
        beat_kick = None
    if isinstance(hat_slots, list) and len(hat_slots) >= total_slots:
        beat_hat = _aggregate_per_beat(hat_slots)
    else:
        beat_hat = None

    # Boundaries
    bpm = data.get("bpm")
    sections, info = _section_boundaries(data, beat_energy, beat_kick, frame_ms, bpm)
    if not sections:
        return {}, {}

    # drop_energy_crest around first drop (±4 beats)
    drop_idx = info.get("drop_idx")
    drop_energy_crest = None
    if isinstance(drop_idx, int):
        lo = max(0, drop_idx - 4)
        hi = min(len(beat_energy), drop_idx + 5)
        win = beat_energy[lo:hi]
        if len(win) >= 3:
            drop_energy_crest = float(np.max(win) / (np.mean(win) + 1e-9))

    # intro_hat_ratio over intro beats vs track
    intro_end = info.get("intro_end_idx")
    intro_hat_ratio = None
    if beat_hat is not None and isinstance(intro_end, int) and intro_end > 0:
        intro_mean = float(np.mean(beat_hat[:intro_end])) if intro_end <= len(beat_hat) else None
        track_mean = float(np.mean(beat_hat)) if len(beat_hat) else None
        if intro_mean is not None and track_mean and track_mean > 0:
            intro_hat_ratio = float(min(1.0, max(0.0, intro_mean / track_mean)))

    # break_syncopation: prefer label-aware proxy if labels available, else hat offbeat within break
    break_syncopation = None
    br = info.get("break_idx")
    if isinstance(br, tuple) and all(isinstance(x, int) or x is None for x in br) and br[0] is not None and br[1] is not None:
        b0, b1 = br
        # try rhythm_pattern.labels if present
        labels = _safe_get(data, ["rhythm_pattern", "labels"], None)
        if isinstance(labels, list) and len(labels) >= total_slots:
            # compute hat offbeat ratio on /16 inside break beats
            start_slot = b0 * 4
            end_slot = min(len(labels), b1 * 4 + 4)
            seg = labels[start_slot:end_slot]
            if seg:
                spb = 16  # /16 grid per bar → 4 slots per beat; here we only need beat quarters & offbeats
                # offbeat indices (2 mod 4 in slot units)
                offs = [i for i in range(len(seg)) if (i % 4) == 2]
                ons  = [i for i in range(len(seg)) if (i % 4) == 0]
                off_hits = sum(1 for i in offs if any("hat" in x for x in (seg[i] or [])))
                on_hits  = sum(1 for i in ons  if any("hat" in x for x in (seg[i] or [])))
                total = off_hits + on_hits
                break_syncopation = float(off_hits / total) if total > 0 else 0.0
        elif beat_hat is not None:
            # fallback: hat offbeat bias from confidences
            start_slot = b0 * 4
            end_slot = min(total_slots, b1 * 4 + 4)
            # offbeat = slot indices where (i % 4) == 2
            off_vals = [hat_slots[i] for i in range(start_slot, end_slot) if (i % 4) == 2]
            on_vals  = [hat_slots[i] for i in range(start_slot, end_slot) if (i % 4) == 0]
            s_off = float(np.sum(off_vals))
            s_on  = float(np.sum(on_vals))
            denom = s_off + s_on
            break_syncopation = float(s_off / denom) if denom > 0 else 0.0

    # Assemble section features
    section_feats = {
        "drop_energy_crest": float(drop_energy_crest) if drop_energy_crest is not None else None,
        "intro_hat_ratio": float(intro_hat_ratio) if intro_hat_ratio is not None else None,
        "break_syncopation": float(break_syncopation) if break_syncopation is not None else None,
        "sections_confidence": float(info.get("sections_confidence", 0.0))
    }

    return sections, section_feats


# --- Command Line Entry ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="Path to track folder")
    args = parser.parse_args()
    final_energy_analysis(args.folder)
