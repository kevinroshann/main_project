import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from pathlib import Path

def plot_normalized_amplitude(audio_file="hydrophone_recording.wav", output_folder="analysis_plots"):
    output_path = Path(output_folder)
    output_path.mkdir(exist_ok=True)

    audio, Fs = sf.read(audio_file)

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    audio = audio - np.mean(audio)

    N = len(audio)
    t = np.arange(N) / Fs

    max_amplitude = np.max(np.abs(audio))
    if max_amplitude == 0:
        raise ValueError("Audio contains only zero values.")

    audio_normalized = audio / max_amplitude

    plt.figure(figsize=(12, 5))
    plt.plot(t, audio_normalized, linewidth=0.8)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Normalized Amplitude")
    plt.title("Normalized Amplitude vs Time")
    plt.ylim([-1.05, 1.05])
    plt.grid(True)
    plt.tight_layout()

    out_file = output_path / "normalized_amplitude.png"
    plt.savefig(out_file, dpi=300)
    plt.close()

    print(f"Sampling Frequency: {Fs} Hz")
    print(f"Recording Duration: {N/Fs:.3f} seconds")
    print(f"Saved normalized amplitude plot to {out_file}")

if __name__ == "__main__":
    plot_normalized_amplitude()
