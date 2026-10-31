import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from pathlib import Path

def plot_fft_spectrum(audio_file="hydrophone_recording.wav", output_folder="analysis_plots"):
    output_path = Path(output_folder)
    output_path.mkdir(exist_ok=True)

    audio, Fs = sf.read(audio_file)

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    audio = audio - np.mean(audio)

    N = len(audio)
    window = np.hanning(N)
    windowed_audio = audio * window

    Y = np.fft.fft(windowed_audio)
    P2 = np.abs(Y) / np.sum(window)
    P1 = P2[:N // 2 + 1]

    if len(P1) > 2:
        P1[1:-1] = 2 * P1[1:-1]

    f = Fs * np.arange(N // 2 + 1) / N

    plt.figure(figsize=(10, 5))
    plt.plot(f, P1, linewidth=0.8)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.title("Frequency Spectrum of Hydrophone Recording")
    plt.grid(True)
    plt.xlim([0, Fs / 2])
    plt.tight_layout()

    out_file = output_path / "frequency_spectrum.png"
    plt.savefig(out_file, dpi=300)
    plt.close()

    print(f"Saved FFT frequency spectrum plot to {out_file}")

if __name__ == "__main__":
    plot_fft_spectrum()
