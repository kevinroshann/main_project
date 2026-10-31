import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def generate_project_plots(csv_file="clustered_events.csv", min_snap_amp=0.15, min_snap_freq=2500):
    df = pd.read_csv(csv_file)
    if df.empty:
        print("Error: 'clustered_events.csv' is empty.")
        return

    # Dynamic filter for biological snaps based on signal thresholds
    df['is_true_snap'] = (df['peak_amplitude'] >= min_snap_amp) & (df['spectral_centroid_hz'] >= min_snap_freq)

    strong_snaps = df[df['is_true_snap']]
    noise_artifacts = df[~df['is_true_snap']]

    # Plot Cluster Feature Separation
    plt.figure(figsize=(10, 6))
    plt.scatter(
        noise_artifacts['spectral_centroid_hz'], 
        noise_artifacts['peak_amplitude'], 
        c='#1f77b4', 
        label=f'Ambient Noise / Low Energy ({len(noise_artifacts)})', 
        alpha=0.5, 
        edgecolors='none', 
        s=40
    )

    if not strong_snaps.empty:
        plt.scatter(
            strong_snaps['spectral_centroid_hz'], 
            strong_snaps['peak_amplitude'], 
            c='#d62728', 
            label=f'Verified Biological Snaps ({len(strong_snaps)})', 
            alpha=0.9, 
            edgecolors='k', 
            s=100
        )

    plt.axhline(y=min_snap_amp, color='r', linestyle='--', linewidth=1, label=f'Min Snap Amp Threshold ({min_snap_amp})')
    plt.xlabel('Spectral Centroid (Hz)')
    plt.ylabel('Peak Amplitude')
    plt.title('Biological Snap Verification vs Ambient Background Noise')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('cluster_feature_separation.png', dpi=300)
    plt.close()

    print("Saved updated 'cluster_feature_separation.png'")

    print("\n--- Verified Dataset Summary ---")
    print(f"Total Candidate Events Detected Across All Files: {len(df)}")
    print(f"Verified Biological Snaps (Amp >= {min_snap_amp}): {len(strong_snaps)}")
    print(f"Filtered Ambient Noise / Background Artifacts: {len(noise_artifacts)}")

    if "filename" in df.columns:
        print("\n--- Breakdown Per Audio File ---")
        df['file_stem'] = df['filename'].apply(lambda x: str(x).split('_snap_')[0] if '_snap_' in str(x) else 'unknown')
        file_summary = df.groupby('file_stem')['is_true_snap'].agg(
            total_candidates='count',
            verified_snaps='sum'
        ).reset_index()
        print(file_summary.to_string(index=False))

if __name__ == "__main__":
    generate_project_plots()