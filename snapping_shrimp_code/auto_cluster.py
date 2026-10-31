import os
import glob
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def extract_clip_features(wav_path):
    audio, fs = sf.read(wav_path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    peak_amp = np.max(np.abs(audio))
    rms_amp = np.sqrt(np.mean(audio ** 2))
    zcr = np.mean(np.abs(np.diff(np.sign(audio)))) / 2.0

    N = len(audio)
    fft_vals = np.abs(np.fft.rfft(audio * np.hanning(N)))
    freqs = np.fft.rfftfreq(N, 1 / fs)

    total_energy = np.sum(fft_vals) + 1e-12
    spectral_centroid = np.sum(freqs * fft_vals) / total_energy
    spectral_bandwidth = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * fft_vals) / total_energy)

    cum_energy = np.cumsum(fft_vals)
    rolloff_idx = np.searchsorted(cum_energy, 0.85 * total_energy)
    spectral_rolloff = freqs[min(rolloff_idx, len(freqs) - 1)]

    return {
        "filename": os.path.basename(wav_path),
        "peak_amplitude": peak_amp,
        "rms_amplitude": rms_amp,
        "zero_crossing_rate": zcr,
        "spectral_centroid_hz": spectral_centroid,
        "spectral_bandwidth_hz": spectral_bandwidth,
        "spectral_rolloff_hz": spectral_rolloff,
        "spectral_skewness": skew(fft_vals),
        "spectral_kurtosis": kurtosis(fft_vals),
    }

def run_unsupervised_clustering(snap_folder="snap_events", num_clusters=3, output_csv="clustered_events.csv"):
    wav_files = glob.glob(os.path.join(snap_folder, "*.wav"))
    if not wav_files:
        print(f"No WAV files found in '{snap_folder}'. Please check path.")
        return

    print(f"Extracting features from {len(wav_files)} candidate clips in '{snap_folder}'...")
    features_list = []
    for wav_file in wav_files:
        try:
            feat = extract_clip_features(wav_file)
            features_list.append(feat)
        except Exception as e:
            print(f"Error reading {wav_file}: {e}")

    df = pd.DataFrame(features_list)

    feature_cols = [
        "peak_amplitude", "rms_amplitude", "zero_crossing_rate",
        "spectral_centroid_hz", "spectral_bandwidth_hz",
        "spectral_rolloff_hz", "spectral_skewness", "spectral_kurtosis"
    ]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[feature_cols])

    print(f"Running K-Means Clustering with k={num_clusters}...")
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    df["cluster_id"] = kmeans.fit_predict(X_scaled)

    df.to_csv(output_csv, index=False)
    print(f"Clustering complete! Results saved to '{output_csv}'.")

    print("\n--- Cluster Summary ---")
    summary = df.groupby("cluster_id")[["spectral_centroid_hz", "peak_amplitude", "zero_crossing_rate"]].mean()
    summary["clip_count"] = df.groupby("cluster_id").size()
    print(summary)

if __name__ == "__main__":
    run_unsupervised_clustering()