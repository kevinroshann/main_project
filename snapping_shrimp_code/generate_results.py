import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def generate_project_plots(csv_file="clustered_events.csv"):
    df = pd.read_csv(csv_file)

    # 1. Plot Cluster Feature Separation
    plt.figure(figsize=(10, 6))
    clusters = df['cluster_id'].unique()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    labels = ['Cluster 0: Background Noise', 'Cluster 1: Distant Snaps', 'Cluster 2: Strong Biological Snaps']

    for cluster_id, color, label in zip(sorted(clusters), colors, labels):
        subset = df[df['cluster_id'] == cluster_id]
        plt.scatter(
            subset['spectral_centroid_hz'], 
            subset['peak_amplitude'], 
            c=color, 
            label=label, 
            alpha=0.7, 
            edgecolors='k', 
            s=50 if cluster_id != 2 else 120
        )

    plt.xlabel('Spectral Centroid (Hz)')
    plt.ylabel('Peak Amplitude')
    plt.title('K-Means Clustering of Underwater Acoustic Events')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('cluster_feature_separation.png', dpi=300)
    plt.close()
    print("Saved 'cluster_feature_separation.png'")

    # 2. Extract verified snap counts (Clusters 1 & 2)
    verified_snaps = df[df['cluster_id'].isin([1, 2])]
    
    print("\n--- Final Summary ---")
    print(f"Total Candidate Events Detected: {len(df)}")
    print(f"Strong Verified Snaps (Cluster 2): {len(df[df['cluster_id'] == 2])}")
    print(f"Distant / Secondary Snaps (Cluster 1): {len(df[df['cluster_id'] == 1])}")
    print(f"Filtered Noise / Artifacts (Cluster 0): {len(df[df['cluster_id'] == 0])}")

if __name__ == "__main__":
    generate_project_plots()