import os
import time
import numpy as np
import soundfile as sf
import sounddevice as sd
from scipy.signal import butter, sosfilt
import librosa

# -------------------------
# Utilities
# -------------------------
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def timestamp():
    return time.strftime("%Y%m%d-%H%M%S")

def save_audio(path, y, sr):
    if np.max(np.abs(y)) > 1e-6:
        y = y / max(1.001*np.max(np.abs(y)), 1.0)
    sf.write(path, y, sr)

# -------------------------
# Recording
# -------------------------
def record_audio(seconds=5, sr=44100, channels=1):
    print(f"🎤 Recording {seconds} seconds at {sr} Hz...")
    y = sd.rec(int(seconds * sr), samplerate=sr, channels=channels, dtype='float32')
    sd.wait()
    y = np.squeeze(y)
    print("✅ Recording complete.")
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

# -------------------------
# HPSS + Speech Focus for voice isolation
# -------------------------
def isolate_voice(y, sr):
    # HPSS
    D = librosa.stft(y)
    H, P = librosa.decompose.hpss(D, kernel_size=(31, 31))
    y_harm = librosa.istft(H)
    y_perc = librosa.istft(P)

    # Spectral gating to reduce background
    D = librosa.stft(y_harm)
    mag = np.abs(D)
    phase = np.angle(D)
    frame_energy = mag.mean(axis=0)
    k = max(1, int(0.1 * frame_energy.size))
    idx = np.argpartition(frame_energy, k)[:k]
    noise_profile = np.median(mag[:, idx], axis=1, keepdims=True)
    alpha = 1.5
    mask = (mag > alpha * noise_profile).astype(np.float32)

    # Slight smoothing
    mask = librosa.decompose.nn_filter(mask, aggregate=np.median, metric='cosine', width=5)

    enhanced_mag = mag * mask
    enhanced = istft(enhanced_mag, phase)
    
    # Optional extra bandpass 100-4000 Hz for speech clarity
    sos = butter(4, [100, 4000], btype='bandpass', fs=sr, output='sos')
    voice_only = sosfilt(sos, enhanced)
    return voice_only

# -------------------------
# Main
# -------------------------
def main():
    ensure_dir("outputs")
    print("\n=== Voice Isolation Recorder ===")
    try:
        secs = float(input("Enter recording duration in seconds (e.g., 6): ") or 6.0)
    except:
        secs = 6.0

    y, sr = record_audio(secs)
    voice_only = isolate_voice(y, sr)
    
    filename = os.path.join("outputs", "voice_only_" + timestamp() + ".wav")
    save_audio(filename, voice_only, sr)
    print(f"\n✅ Voice-only recording saved at: {filename}")

if __name__ == "__main__":
    main()
