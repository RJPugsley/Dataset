import json
import numpy as np
from pathlib import Path

def load_onsets_from_folder(folder):
    json_path = next((f for f in Path(folder).glob("*.json") if not f.name.startswith("._")), None)
    if not json_path:
        print("❌ No JSON file found in folder.")
        return None, None, None

    with open(json_path) as f:
        data = json.load(f)

    onsets = data.get("onsets", [])
    bpm = data.get("rhythm_pattern", {}).get("bpm")
    return onsets, bpm, json_path.name

def estimate_bpm_autocorr(onsets):
    if len(onsets) < 2:
        return None

    iois = np.diff(onsets)
    bins = np.arange(0, 5.0, 0.01)  # up to 5 sec lag
    hist, _ = np.histogram(iois, bins=bins)

    autocorr = np.correlate(hist, hist, mode="full")
    autocorr = autocorr[len(autocorr)//2:]

    peak_lag = np.argmax(autocorr[10:]) + 10
    interval_sec = bins[peak_lag]
    bpm = 60.0 / interval_sec if interval_sec > 0 else None

    return round(bpm, 2)

def estimate_bpm_fft(onsets):
    if len(onsets) < 2:
        return None

    signal = np.zeros(int(onsets[-1] * 100))  # 10ms resolution
    for t in onsets:
        idx = int(t * 100)
        if idx < len(signal):
            signal[idx] = 1

    spectrum = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(len(signal), d=0.01)
    bpm_freqs = freqs * 60

    valid = (bpm_freqs > 50) & (bpm_freqs < 200)
    if not np.any(valid):
        return None

    peak_bpm = bpm_freqs[valid][np.argmax(spectrum[valid])]
    return round(peak_bpm, 2)

def main():
    print("🎚 DJmate BPM Estimator (Test Mode)\n")
    folder = input("📂 Enter path to track folder: ").strip()
    folder = Path(folder)

    if not folder.exists() or not folder.is_dir():
        print("❌ Invalid folder path.")
        return

    onsets, existing_bpm, name = load_onsets_from_folder(folder)
    if not onsets:
        print("⚠️ No onsets found in JSON.")
        return

    print(f"\n📄 File: {name}")
    print(f"🎼 Beatgrid BPM (existing): {existing_bpm}")

    bpm_ac = estimate_bpm_autocorr(onsets)
    bpm_fft = estimate_bpm_fft(onsets)

    print(f"\n🔁 BPM via Autocorrelation: {bpm_ac}")
    print(f"📊 BPM via FFT: {bpm_fft}")

if __name__ == "__main__":
    main()
