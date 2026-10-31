import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, spectrogram
from pathlib import Path

def process_audio_filtering(input_file="hydrophone_recording.wav", output_folder="filtered_audio", low_cut=2000, high_cut=90000, filter_order=4):
    output_path = Path(output_folder)
    output_path.mkdir(exist_ok=True)

    # Load audio
    audio, fs = sf.read(input_file)
    print(f"Sampling Frequency: {fs} Hz")
    print(f"Audio Duration: {len(audio) / fs:.3f} seconds")

    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)

    audio = audio.astype(np.float64)

    if len(audio) == 0:
        raise ValueError("Audio file is empty.")

    if not np.all(np.isfinite(audio)):
        raise ValueError("Audio contains invalid values.")

    # Remove DC offset
    audio = audio - np.mean(audio)

    # Dynamically adjust cutoffs based on Nyquist frequency
    nyquist = fs / 2
    
    if high_cut >= nyquist:
        high_cut = max(low_cut + 100, nyquist - 100)
        print(f"Notice: HIGH_CUT capped at {high_cut} Hz (Nyquist limit: {nyquist} Hz)")

    if low_cut >= high_cut or low_cut <= 0:
        low_cut = max(50.0, high_cut / 4.0)
        print(f"Notice: LOW_CUT adjusted to {low_cut} Hz")

    # Design Butterworth band-pass filter
    sos = butter(filter_order, [low_cut, high_cut], btype="bandpass", fs=fs, output="sos")

    # Apply zero-phase filtering
    filtered_audio = sosfiltfilt(sos, audio)

    # Save filtered audio separately
    output_file = output_path / "filtered_recording.wav"
    sf.write(output_file, filtered_audio, fs, subtype="PCM_24")
    print("Filtered audio saved:", output_file)

    # Plot helpers
    time = np.arange(len(audio)) / fs

    def plot_waveform(signal, title, filename):
        plt.figure(figsize=(14, 5))
        plt.plot(time, signal, linewidth=0.5)
        plt.xlabel("Time (seconds)")
        plt.ylabel("Amplitude")
        plt.title(title)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_path / filename, dpi=300)
        plt.close()

    def plot_spectrogram(signal, title, filename):
        nperseg = min(1024, len(signal))
        noverlap = min(768, nperseg // 2)
        f, t, Sxx = spectrogram(signal, fs=fs, nperseg=nperseg, noverlap=noverlap)
        plt.figure(figsize=(14, 6))
        plt.pcolormesh(t, f / 1000, 10 * np.log10(Sxx + 1e-15), shading="auto")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Frequency (kHz)")
        plt.title(title)
        plt.colorbar(label="Power/Frequency (dB/Hz)")
        plt.ylim(0, fs / 2000)
        plt.tight_layout()
        plt.savefig(output_path / filename, dpi=300)
        plt.close()

    # Generate plots
    plot_waveform(audio, "Original Hydrophone Recording", "original_waveform.png")
    plot_waveform(filtered_audio, "Filtered Hydrophone Recording", "filtered_waveform.png")
    plot_spectrogram(audio, "Original Audio Spectrogram", "original_spectrogram.png")
    plot_spectrogram(filtered_audio, "Filtered Audio Spectrogram", "filtered_spectrogram.png")

    print("Audio filtering and visualization completed.")
    return output_file, fs

if __name__ == "__main__":
    process_audio_filtering()
