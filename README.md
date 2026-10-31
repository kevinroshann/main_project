**Stage 1 — Signal Preprocessing**
- Converts multi-channel audio to mono and removes DC offset.
- Applies a dynamic 4th-order zero-phase Butterworth band-pass filter (2000–3806 Hz for 7812 Hz recordings).

**Stage 2 — Transient Event Detection**
- Computes RMS energy using 0.5 ms windows.
- Applies adaptive thresholding: `Median + 6 × MAD`.
- Merges events within 2 ms and removes glitches shorter than 0.25 ms.
- Extracts padded WAV clips and exports `all_detected_events.csv`.

**Stage 3 — Feature Extraction**
Extracts Peak Amplitude, RMS, Zero-Crossing Rate, Spectral Centroid, Bandwidth, Roll-off, Skewness, and Kurtosis.

**Stage 4 — K-Means Classification**
- Standardizes features using `StandardScaler`.
- Performs K-Means clustering (`k=3`).
- Identifies three preliminary acoustic categories:
  - **Cluster 2:** High-amplitude, nearby biological snaps.
  - **Cluster 1:** Attenuated, distant snapping shrimp clicks.
  - **Cluster 0:** Background noise and electrical artifacts.
