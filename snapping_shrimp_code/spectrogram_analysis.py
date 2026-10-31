import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from pathlib import Path

def plot_spectrogram_analysis(audio_file="hydrophone_recording.wav", output_folder="analysis_plots"):
    output_path = Path(output_folder)
    output_path.mkdir(exist_ok=True)

    audio, Fs = sf.read(audio_file)

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    audio = audio - np.mean(audio)

    window_length = 1024
    overlap = 768
    nfft = 2048

    f, t, Sxx = spectrogram(
        audio,
        fs=Fs,
        window=np.hanning(window_length),
        nperseg=window_length,
        noverlap=overlap,
        nfft=nfft,
        mode="psd"
    )

    plt.figure(figsize=(12, 6))
    plt.pcolormesh(t, f / 1000, 10 * np.log10(Sxx + 1e-15), shading="auto", cmap="viridis")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Frequency (kHz)")
    plt.title("Spectrogram of Hydrophone Recording")
    plt.colorbar(label="Power/Frequency (dB/Hz)")
    plt.ylim([0, Fs / 2000])
    plt.tight_layout()

    out_file = output_path / "spectrogram_analysis.png"
    plt.savefig(out_file, dpi=300)
    plt.close()

    print(f"Saved spectrogram analysis plot to {out_file}")

if __name__ == "__main__":
    plot_spectrogram_analysis()
