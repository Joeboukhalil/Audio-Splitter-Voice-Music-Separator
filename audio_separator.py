import os
import sys
import time
import queue
import numpy as np
import soundfile as sf
import sounddevice as sd
import librosa
import librosa.display
from scipy.signal import butter, sosfilt

# -------------------------
# Utility / I/O
# -------------------------
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def timestamp():
    return time.strftime("%Y%m%d-%H%M%S")

def save_audio(path, y, sr):
    # Normalize gently to avoid clipping
    if np.max(np.abs(y)) > 1e-6:
        y = y / max(1.001*np.max(np.abs(y)), 1.0)
    sf.write(path, y, sr)

def load_audio(path, target_sr=44100, mono=True):
    y, sr = librosa.load(path, sr=target_sr, mono=mono)
    return y, sr

# -------------------------
# Recording
# -------------------------
def record_audio(seconds=5, sr=44100, channels=1):
    print(f"Recording {seconds} s at {sr}Hz ...")
    y = sd.rec(int(seconds * sr), samplerate=sr, channels=channels, dtype='float32')
    sd.wait()
    y = np.squeeze(y)
    print("Done.")
    return y, sr

# -------------------------
# STFT helpers
# -------------------------
def stft(y, n_fft=2048, hop_length=512):
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window='hann', center=True)
    mag = np.abs(D)
    phase = np.angle(D)
    return D, mag, phase, n_fft, hop_length

def istft(mag, phase, hop_length=512):
    D = mag * np.exp(1j * phase)
    y = librosa.istft(D, hop_length=hop_length, window='hann', center=True)
    return y

def hz_to_bin(f_hz, n_fft, sr):
    return int(np.clip(np.round(f_hz * n_fft / sr), 0, n_fft//2))

# -------------------------
# Simple band split (bass/mid/treble)
# -------------------------
def split_bands(y, sr, bass_hz=150, treble_hz=4000, n_fft=2048, hop=512):
    D, mag, phase, n_fft, hop = stft(y, n_fft=n_fft, hop_length=hop)
    f_bass = hz_to_bin(bass_hz, n_fft, sr)
    f_treb = hz_to_bin(treble_hz, n_fft, sr)

    bass_mask = np.zeros_like(mag);    bass_mask[:f_bass, :] = 1.0
    mid_mask  = np.zeros_like(mag);    mid_mask[f_bass:f_treb, :] = 1.0
    treb_mask = np.zeros_like(mag);    treb_mask[f_treb:, :] = 1.0

    y_bass = istft(mag * bass_mask, phase, hop)
    y_mid  = istft(mag * mid_mask,  phase, hop)
    y_treb = istft(mag * treb_mask, phase, hop)
    return y_bass, y_mid, y_treb

# -------------------------
# HPSS (harmonic/percussive)
# -------------------------
def hpss(y, sr, n_fft=2048, hop=512):
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop)
    H, P = librosa.decompose.hpss(D, kernel_size=(31, 31))
    y_harm = librosa.istft(H, hop_length=hop)
    y_perc = librosa.istft(P, hop_length=hop)
    return y_harm, y_perc

# -------------------------
# Vocals-ish (non-AI):
# Harmonic + vocal band-pass (100–4000 Hz) mask
# -------------------------
def isolate_vocalsish(y, sr, n_fft=2048, hop=512):
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop)
    mag = np.abs(D); phase = np.angle(D)

    # HPSS masks in magnitude domain
    H_mag, P_mag = librosa.decompose.hpss(mag, kernel_size=(31, 31))

    # Frequency mask for typical vocal band
    f_low = hz_to_bin(100, n_fft, sr)
    f_high = hz_to_bin(4000, n_fft, sr)
    freq_mask = np.zeros_like(mag)
    freq_mask[f_low:f_high, :] = 1.0

    vocal_mag = H_mag * freq_mask
    vocal = istft(vocal_mag, phase, hop)

    # "Accompaniment-ish": percussive + harmonic outside vocal band
    accomp_mag = np.minimum(mag, (P_mag + (H_mag * (1.0 - freq_mask))))
    accomp = istft(accomp_mag, phase, hop)
    return vocal, accomp

# -------------------------
# Simple speech background reduction via spectral gating
# 1) Find quietest 10% frames to estimate noise floor
# 2) Gate below alpha * noise_profile
# -------------------------
def speech_focus(y, sr, n_fft=1024, hop=256, alpha=1.5):
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop)
    mag = np.abs(D); phase = np.angle(D)

    # Frame energy
    frame_energy = mag.mean(axis=0)
    k = max(1, int(0.1 * frame_energy.size))
    idx = np.argpartition(frame_energy, k)[:k]
    noise_profile = np.median(mag[:, idx], axis=1, keepdims=True)

    # Build mask
    thresh = alpha * noise_profile
    mask = (mag > thresh).astype(np.float32)

    # Slight smoothing across frequency
    mask = librosa.decompose.nn_filter(mask, aggregate=np.median, metric='cosine', width=5)

    enhanced_mag = mag * mask
    enhanced = istft(enhanced_mag, phase, hop)
    residual_mag = mag * (1.0 - mask)
    residual = istft(residual_mag, phase, hop)
    return enhanced, residual

# -------------------------
# Biquad helpers (optional extra bandpass for speech)
# -------------------------
def bandpass(y, sr, low_hz=100, high_hz=4000, order=4):
    sos = butter(order, [low_hz, high_hz], btype='bandpass', fs=sr, output='sos')
    return sosfilt(sos, y)

# -------------------------
# Processing pipeline
# -------------------------
def process_and_save(y, sr, base_name, outdir="outputs"):
    ensure_dir(outdir)
    stem_prefix = os.path.join(outdir, base_name + "_")

    # 1) HPSS
    y_harm, y_perc = hpss(y, sr)
    save_audio(stem_prefix + "harmonic.wav", y_harm, sr)
    save_audio(stem_prefix + "percussive.wav", y_perc, sr)

    # 2) Bands
    y_bass, y_mid, y_treb = split_bands(y, sr)
    save_audio(stem_prefix + "bass.wav", y_bass, sr)
    save_audio(stem_prefix + "mid.wav",  y_mid,  sr)
    save_audio(stem_prefix + "treble.wav", y_treb, sr)

    # 3) Vocals-ish vs accompaniment-ish
    y_voc, y_accomp = isolate_vocalsish(y, sr)
    save_audio(stem_prefix + "vocals_ish.wav", y_voc, sr)
    save_audio(stem_prefix + "accompaniment_ish.wav", y_accomp, sr)

    # 4) Speech focus (noise-reduced speech vs residual background)
    y_speech, y_back = speech_focus(y, sr)
    # Optional extra bandpass for speech clarity
    y_speech_bp = bandpass(y_speech, sr, 100, 4000)
    save_audio(stem_prefix + "speech_focus.wav", y_speech_bp, sr)
    save_audio(stem_prefix + "background_residual.wav", y_back, sr)

    print(f"\nSaved stems to: {os.path.abspath(outdir)}")
    print("  - harmonic.wav / percussive.wav")
    print("  - bass.wav / mid.wav / treble.wav")
    print("  - vocals_ish.wav / accompaniment_ish.wav")
    print("  - speech_focus.wav / background_residual.wav")

# -------------------------
# Simple CLI menu
# -------------------------
def main():
    print("\n=== Traditional DSP Audio Separator ===")
    print("1) Upload a file (enter path)")
    print("2) Record from microphone")
    print("3) Quit")
    choice = input("Select (1/2/3): ").strip()

    if choice == "1":
        path = input("Enter audio file path: ").strip().strip('"').strip("'")
        if not os.path.isfile(path):
            print("File not found.")
            return
        y, sr = load_audio(path)
        base = os.path.splitext(os.path.basename(path))[0] + "_" + timestamp()
        process_and_save(y, sr, base)

    elif choice == "2":
        try:
            secs = float(input("Duration seconds (e.g., 6): ").strip() or "6")
        except:
            secs = 6.0
        y, sr = record_audio(seconds=secs)
        base = "recording_" + timestamp()
        # Optionally save the raw recording too
        ensure_dir("outputs")
        save_audio(os.path.join("outputs", base + "_raw.wav"), y, sr)
        process_and_save(y, sr, base)

    else:
        print("Bye.")

if __name__ == "__main__":
    main()
