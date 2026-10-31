import numpy as np
import pandas as pd
import soundfile as sf
from pathlib import Path

def detect_snapping_events(input_file="filtered_audio/filtered_recording.wav", output_folder="snap_events", frame_ms=0.5, hop_ms=0.1, threshold_mad=6.0, min_event_ms=0.25, merge_gap_ms=2.0, pre_event_ms=2.0, post_event_ms=5.0):
    output_path = Path(output_folder)
    output_path.mkdir(exist_ok=True)

    # Load filtered audio
    audio, fs = sf.read(input_file)
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)
    audio = audio.astype(float)

    # RMS envelope
    frame_size = max(1, int(frame_ms * fs / 1000))
    hop_size = max(1, int(hop_ms * fs / 1000))
    frame_starts = np.arange(0, max(1, len(audio) - frame_size + 1), hop_size)

    rms = np.array([
        np.sqrt(np.mean(audio[s:s + frame_size] ** 2))
        for s in frame_starts
    ])

    # Adaptive threshold (median + k * robust sigma)
    median_value = np.median(rms)
    mad = np.median(np.abs(rms - median_value))
    robust_sigma = 1.4826 * mad

    threshold = max(median_value + threshold_mad * robust_sigma, np.finfo(float).eps)
    active = rms > threshold

    # Find event regions
    edges = np.diff(np.r_[False, active, False].astype(int))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]

    events = []
    for start, end in zip(starts, ends):
        sample_start = frame_starts[start]
        sample_end = min(len(audio), frame_starts[end - 1] + frame_size)
        events.append([sample_start, sample_end])

    # Merge nearby events
    merge_gap = int(merge_gap_ms * fs / 1000)
    merged = []
    for start, end in events:
        if not merged:
            merged.append([start, end])
        elif start - merged[-1][1] <= merge_gap:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    # Extract events and calculate features
    minimum_samples = int(min_event_ms * fs / 1000)
    pre_samples = int(pre_event_ms * fs / 1000)
    post_samples = int(post_event_ms * fs / 1000)

    metadata = []
    for number, (start, end) in enumerate(merged, 1):
        if end - start < minimum_samples:
            continue

        clip_start = max(0, start - pre_samples)
        clip_end = min(len(audio), end + post_samples)
        event = audio[clip_start:clip_end]

        peak = np.max(np.abs(event))
        event_rms = np.sqrt(np.mean(event ** 2))

        spectrum = np.abs(np.fft.rfft(event * np.hanning(len(event))))
        frequencies = np.fft.rfftfreq(len(event), d=1/fs)
        peak_frequency = frequencies[np.argmax(spectrum)]

        filename = output_path / f"candidate_snap_{number:04d}.wav"
        sf.write(filename, event, fs)

        metadata.append({
            "event_number": number,
            "start_time_s": start / fs,
            "end_time_s": end / fs,
            "duration_ms": (end - start) / fs * 1000,
            "peak_amplitude": peak,
            "rms_amplitude": event_rms,
            "peak_frequency_hz": peak_frequency,
            "filename": str(filename),
        })

    # Save CSV
    df = pd.DataFrame(metadata)
    df.to_csv(output_path / "event_metadata.csv", index=False)
    print("Candidate events detected:", len(df))
    print("Metadata saved successfully.")

if __name__ == "__main__":
    detect_snapping_events()
