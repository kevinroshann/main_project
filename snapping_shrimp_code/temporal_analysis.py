import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def analyze_temporal_patterns(csv_file="clustered_events.csv", output_plot="temporal_snap_analysis.png", min_snap_amp=0.15):
    df = pd.read_csv(csv_file)
    if df.empty:
        print("Error: CSV file is empty.")
        return

    # Check for start_time_s column
    if 'start_time_s' not in df.columns:
        print("\n[Temporal Analysis] 'start_time_s' column not found in CSV. Assigning sequential time index.")
        df['start_time_s'] = np.arange(len(df)) * 0.1

    # Filter for verified biological snaps
    snaps = df[df['peak_amplitude'] >= min_snap_amp].sort_values(by='start_time_s').reset_index(drop=True)

    if len(snaps) < 2:
        print(f"\n[Temporal Analysis] Only {len(snaps)} verified snap(s) found. Skipping Inter-Snap Interval (ISI) calculation.")
        return

    # Compute Inter-Snap Intervals (ISI) in milliseconds
    snaps['isi_ms'] = snaps['start_time_s'].diff() * 1000.0
    isi_clean = snaps['isi_ms'].dropna()

    # Generate Visualization Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Subplot 1: Snap Arrival Timeline
    ax1.eventplot(snaps['start_time_s'], lineoffsets=1, linelengths=0.5, colors='#d62728')
    ax1.set_xlabel('Recording Timestamp (seconds)')
    ax1.set_yticks([])
    ax1.set_title('Biological Snap Occurrence Timeline')
    ax1.grid(True, axis='x')

    # Subplot 2: Inter-Snap Interval (ISI) Distribution
    ax2.hist(isi_clean, bins=min(20, max(5, len(isi_clean))), color='#1f77b4', edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Inter-Snap Interval (ms)')
    ax2.set_ylabel('Frequency Count')
    ax2.set_title(f'Inter-Snap Interval (ISI) Distribution (Median ISI: {isi_clean.median():.2f} ms)')
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(output_plot, dpi=300)
    plt.close()

    print(f"\n--- Temporal & ISI Analysis Summary ---")
    print(f"Verified Snaps Analyzed: {len(snaps)}")
    print(f"Median Inter-Snap Interval (ISI): {isi_clean.median():.2f} ms")
    print(f"Minimum ISI: {isi_clean.min():.2f} ms | Maximum ISI: {isi_clean.max():.2f} ms")
    print(f"Saved temporal analysis plot to '{output_plot}'")

if __name__ == "__main__":
    analyze_temporal_patterns()