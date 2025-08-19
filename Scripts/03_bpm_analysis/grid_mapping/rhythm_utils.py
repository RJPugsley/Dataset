"""Rhythm utilities for DJmate

Design goals
------------
- Standardize *how* envelopes and onsets are computed (timebase, windows, filters),
  without forcing a giant onset list into JSON.
- Lazy, memoized computation so downstream stages can call the same API repeatedly
  with deterministic results.
- Lightweight dependencies: uses numpy/scipy; optionally librosa if present (for HPSS).

Typical usage
-------------
from rhythm_utils import RhythmFeatures, RhythmParams

rf = RhythmFeatures.from_files({
    "mix": "/path/to/mix.wav",
    "drums": "/path/to/drums.wav",
    "bass": "/path/to/bass.wav",
})

env = rf.envelope("drums")                 # 20 Hz RMS envelope (numpy array)
ons = rf.onsets("drums")                   # broadband onsets (seconds)
kick_ons = rf.kick_onsets("drums", bpm=130)  # kick-band onsets (seconds)

JSON contract (store once in Stage 1)
-------------------------------------
{
  "rhythm_timebase_hz": 20,
  "onset_params": rf.params.to_json_dict(),
  "envelope_stems": ["mix", "drums", "bass", "vocals"]
}
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple, Iterable, Any
import numpy as np
from functools import lru_cache

try:
    import soundfile as sf
except Exception:  # pragma: no cover
    sf = None

try:
    import librosa  # optional, used for HPSS if available
except Exception:  # pragma: no cover
    librosa = None

from scipy.signal import butter, sosfiltfilt


# -----------------------------
# Parameters (single source of truth)
# -----------------------------
@dataclass(frozen=True)
class RhythmParams:
    sr: int = 44100                # analysis sample rate
    window: int = 2048             # STFT/flux window size
    hop: int = 512                 # hop size
    env_hz: float = 20.0           # envelope timebase (Hz)
    method: str = "spectral_flux+hpss"  # descriptor for JSON
    # Peak picking defaults (loosely inspired by librosa defaults, tuned for EDM/UKG)
    pre_max: int = 10
    post_max: int = 10
    pre_avg: int = 50
    post_avg: int = 50
    delta: float = 0.07
    wait_sec: float = 0.03
    min_sep_sec: float = 0.04
    kick_band_hz: Tuple[float, float] = (35.0, 120.0)

    def to_json_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


# -----------------------------
# Utility DSP helpers
# -----------------------------

def _to_mono(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return x
    if x.shape[0] < x.shape[1]:  # (channels, samples)
        return np.mean(x, axis=0)
    return np.mean(x, axis=1)


def _resample_if_needed(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x
    # Use polyphase resampling via numpy (fallback) to avoid adding heavy deps.
    # We'll do simple linear interpolation which is fine for envelopes/onsets pre-STFT.
    import math
    ratio = sr_out / float(sr_in)
    n_out = int(math.ceil(len(x) * ratio))
    t_in = np.linspace(0, 1, num=len(x), endpoint=False)
    t_out = np.linspace(0, 1, num=n_out, endpoint=False)
    return np.interp(t_out, t_in, x).astype(x.dtype)


def _sos_bandpass(low_hz: float, high_hz: float, sr: int, order: int = 4):
    ny = 0.5 * sr
    low = max(1e-3, low_hz / ny)
    high = min(0.999, high_hz / ny)
    if not (0 < low < high < 1):
        raise ValueError("Invalid bandpass range for given sr")
    return butter(order, [low, high], btype="band", output="sos")


def _frame_signal(x: np.ndarray, frame: int, hop: int) -> np.ndarray:
    if len(x) < frame:
        pad = frame - len(x)
        x = np.pad(x, (0, pad))
    n = 1 + int(np.floor((len(x) - frame) / hop))
    idx = np.tile(np.arange(frame), (n, 1)) + np.tile(np.arange(n) * hop, (frame, 1)).T
    return x[idx]


def _rms_envelope(x: np.ndarray, sr: int, env_hz: float) -> np.ndarray:
    # Compute RMS in sliding window sized to achieve ~env_hz samples/sec
    hop = max(1, int(round(sr / env_hz)))
    win = hop * 2
    frames = _frame_signal(x, win, hop)
    rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)
    return rms.astype(np.float32)


def _spectral_flux(x: np.ndarray, sr: int, window: int, hop: int, hpss: bool = True,
                   band: Optional[Tuple[float, float]] = None) -> np.ndarray:
    y = x
    if band is not None:
        sos = _sos_bandpass(band[0], band[1], sr)
        y = sosfiltfilt(sos, y)
    if hpss and librosa is not None:
        try:
            y = librosa.effects.hpss(y)[1]  # percussive component
        except Exception:
            pass
    # STFT magnitude
    n = window
    hop = hop
    win = np.hanning(n).astype(np.float32)
    # Pad to center
    pad = n // 2
    y_pad = np.pad(y, (pad, pad))
    n_frames = 1 + (len(y_pad) - n) // hop
    flux = np.zeros(n_frames, dtype=np.float32)
    prev = None
    for i in range(n_frames):
        start = i * hop
        seg = y_pad[start:start + n]
        mag = np.abs(np.fft.rfft(seg * win))
        if prev is None:
            prev = mag
            continue
        # Rectified spectral flux
        diff = mag - prev
        diff[diff < 0] = 0
        flux[i] = np.sum(diff)
        prev = mag
    # Normalize
    if np.max(flux) > 0:
        flux /= np.max(flux)
    return flux


def _peak_pick(env: np.ndarray, sr_env: float, pre_max: int, post_max: int,
               pre_avg: int, post_avg: int, delta: float, wait_sec: float,
               min_sep_sec: float) -> np.ndarray:
    """Simple peak picker similar in spirit to librosa.util.peak_pick.
    Returns times (seconds) of detected peaks.
    """
    x = env
    n = len(x)
    peaks = []
    i = 0
    wait = int(round(wait_sec * sr_env))
    last_peak = -np.inf
    min_sep = int(round(min_sep_sec * sr_env))

    while i < n:
        # Local max window
        start_max = max(0, i - pre_max)
        end_max = min(n, i + post_max + 1)
        local_max = x[i] == np.max(x[start_max:end_max])

        # Local average window
        start_avg = max(0, i - pre_avg)
        end_avg = min(n, i + post_avg + 1)
        local_avg = np.mean(x[start_avg:end_avg])

        if local_max and (x[i] >= local_avg + delta):
            if i - last_peak >= min_sep:
                peaks.append(i)
                last_peak = i
                i += wait
                continue
        i += 1

    if not peaks:
        return np.array([], dtype=np.float32)
    times = np.array(peaks, dtype=np.float32) / float(sr_env)
    return times


# -----------------------------
# Main API
# -----------------------------
class RhythmFeatures:
    def __init__(self, stems: Dict[str, Tuple[np.ndarray, int]], params: Optional[RhythmParams] = None):
        self.params = params or RhythmParams()
        # Normalize stems to mono @ params.sr
        self.stems: Dict[str, np.ndarray] = {}
        for name, (audio, sr) in stems.items():
            y = _to_mono(audio.astype(np.float32))
            y = _resample_if_needed(y, sr, self.params.sr)
            # Normalize to -1..1
            if np.max(np.abs(y)) > 1e-9:
                y = y / np.max(np.abs(y))
            self.stems[name] = y
        # internal caches
        self._env_cache: Dict[Tuple[str, float, str], np.ndarray] = {}
        self._onsenv_cache: Dict[Tuple[str, str, Tuple[float, float] | None, bool], Tuple[np.ndarray, float]] = {}
        self._onsets_cache: Dict[Tuple[str, str, Tuple[float, float] | None], np.ndarray] = {}

    # ---------- Construction helpers ----------
    @classmethod
    def from_files(cls, stem_paths: Dict[str, str], params: Optional[RhythmParams] = None):
        if sf is None:
            raise RuntimeError("soundfile is required for from_files(). Install pysoundfile.")
        stems: Dict[str, Tuple[np.ndarray, int]] = {}
        for name, path in stem_paths.items():
            y, sr = sf.read(path, always_2d=False)
            # soundfile can return (n, channels); ensure 1D or 2D channel-first
            if y.ndim == 2 and y.shape[0] < y.shape[1]:
                y = y.T
            stems[name] = (y, sr)
        return cls(stems, params=params)

    # ---------- Public API ----------
    def envelope(self, stem: str, hz: Optional[float] = None, mode: str = "rms") -> np.ndarray:
        """Per-stem energy envelope at given rate (default params.env_hz).
        mode: "rms" (default) or "flux" (spectral flux downsampled to hz).
        Returns np.ndarray[float32] of length ~duration * hz.
        """
        hz = hz or self.params.env_hz
        key = (stem, float(hz), mode)
        if key in self._env_cache:
            return self._env_cache[key]
        y = self._get_stem(stem)
        if mode == "rms":
            env = _rms_envelope(y, self.params.sr, hz)
        elif mode == "flux":
            flux = _spectral_flux(y, self.params.sr, self.params.window, self.params.hop, hpss=False)
            # Resample flux to target hz
            sr_env = self.params.sr / self.params.hop
            env = _resample_if_needed(flux, int(round(sr_env)), int(round(hz)))
        else:
            raise ValueError("mode must be 'rms' or 'flux'")
        self._env_cache[key] = env
        return env

    def onset_envelope(self, stem: str, mode: str = "broadband",
                       band: Optional[Tuple[float, float]] = None,
                       hpss: bool = True) -> Tuple[np.ndarray, float]:
        """Compute an onset envelope for a stem.
        mode: "broadband" (default) or "kickband" (uses band if provided).
        Returns (env, sr_env) where sr_env = params.sr / params.hop.
        """
        key = (stem, mode, band, bool(hpss))
        if key in self._onsenv_cache:
            return self._onsenv_cache[key]
        y = self._get_stem(stem)
        use_band = band if (mode == "kickband" and band is not None) else None
        env = _spectral_flux(
            y, self.params.sr, self.params.window, self.params.hop,
            hpss=(hpss and (librosa is not None)), band=use_band
        )
        sr_env = self.params.sr / self.params.hop
        self._onsenv_cache[key] = (env, sr_env)
        return env, sr_env

    def onsets(self, stem: str = "drums", mode: str = "broadband", **peak_opts) -> np.ndarray:
        """Detect onsets (seconds) using spectral flux + peak picking.
        mode: "broadband" or "kickband" (requires band argument via peak_opts or params.kick_band_hz).
        peak_opts can override peak picking parameters.
        """
        band = peak_opts.pop("band", None)
        key = (stem, mode, band)
        if key in self._onsets_cache:
            return self._onsets_cache[key]
        env, sr_env = self.onset_envelope(
            stem, mode=mode,
            band=(band or (self.params.kick_band_hz if mode == "kickband" else None)),
            hpss=True,
        )
        p = self.params
        # Resolve peak picking parameters with overrides
        pre_max = int(peak_opts.get("pre_max", p.pre_max))
        post_max = int(peak_opts.get("post_max", p.post_max))
        pre_avg = int(peak_opts.get("pre_avg", p.pre_avg))
        post_avg = int(peak_opts.get("post_avg", p.post_avg))
        delta = float(peak_opts.get("delta", p.delta))
        wait_sec = float(peak_opts.get("wait_sec", p.wait_sec))
        min_sep_sec = float(peak_opts.get("min_sep_sec", p.min_sep_sec))

        times = _peak_pick(env, sr_env, pre_max, post_max, pre_avg, post_avg,
                           delta, wait_sec, min_sep_sec)
        self._onsets_cache[key] = times
        return times

    # in RhythmFeatures.kick_onsets(...)
    def kick_onsets(self, stem="drums", bpm=None, adaptive_band=False, **peak_opts):
        band = self.params.kick_band_hz
        if adaptive_band:
            # Estimate dominant kick band from first 15s
            y = self._get_stem(stem)
            sr = self.params.sr
            n = min(len(y), int(15*sr))
            seg = y[:n]
            # high-pass to remove rumble
            seg = sosfiltfilt(_sos_bandpass(25, 200, sr), seg)
            # Welch PSD
            from scipy.signal import welch
            f, p = welch(seg, fs=sr, nperseg=4096)
            mask = (f >= 30) & (f <= 180)
            f0 = f[mask][np.argmax(p[mask])]
            lo, hi = max(30.0, f0/np.sqrt(2)), min(180.0, f0*np.sqrt(2))
            band = (lo, hi)
        if bpm is not None and "min_sep_sec" not in peak_opts:
            slot = 60.0 / bpm / 4.0
            peak_opts["min_sep_sec"] = max(self.params.min_sep_sec, 0.5 * slot)
        return self.onsets(stem=stem, mode="kickband", band=band, **peak_opts)


    # ---------- Introspection ----------
    def json_contract(self) -> Dict[str, Any]:
        return {
            "rhythm_timebase_hz": self.params.env_hz,
            "onset_params": self.params.to_json_dict(),
            "envelope_stems": list(self.stems.keys()),
        }

    # ---------- Private ----------
    def _get_stem(self, stem: str) -> np.ndarray:
        if stem not in self.stems:
            raise KeyError(f"Stem '{stem}' not loaded. Available: {list(self.stems.keys())}")
        return self.stems[stem]


# -----------------------------
# Minimal self-test (optional)
# -----------------------------
if __name__ == "__main__":
    # This block is for quick smoke testing with small arrays; adjust paths if needed.
    sr = 44100
    t = np.linspace(0, 2.0, int(sr*2.0), endpoint=False)
    # Synthetic kick train at 2 Hz
    sig = np.sin(2*np.pi*50*t) * (np.sin(2*np.pi*2*t) > 0.99).astype(float)
    stems = {"drums": (sig.astype(np.float32), sr)}
    rf = RhythmFeatures(stems)
    env = rf.envelope("drums")
    ons = rf.onsets("drums")
    kons = rf.kick_onsets("drums", bpm=120)
    print("env_len=", len(env), "onsets=", ons[:5], "kick_onsets=", kons[:5])
