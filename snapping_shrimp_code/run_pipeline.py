import sys
from pathlib import Path
from audio_filtering import process_audio_filtering
from snapping_event_detection import detect_snapping_events
from normalized_amplitude import plot_normalized_amplitude
from fft_analysis import plot_fft_spectrum
from spectrogram_analysis import plot_spectrogram_analysis

def run_all(audio_file="hydrophone_recording.wav"):
    input_path = Path(audio_file)
    if not input_path.exists():
        print(f"Error: Input audio file '{audio_file}' not found.")
        sys.exit(1)

    print("=" * 60)
    print(f"STARTING ANALYSIS PIPELINE FOR: {audio_file}")
    print("=" * 60)

    # Step 1: Analysis on Original File
    print("\n[1/5] Generating Normalized Amplitude Plot...")
    plot_normalized_amplitude(audio_file=str(input_path), output_folder="analysis_plots")

    print("\n[2/5] Generating FFT Frequency Spectrum...")
    plot_fft_spectrum(audio_file=str(input_path), output_folder="analysis_plots")

    print("\n[3/5] Generating STFT Spectrogram...")
    plot_spectrogram_analysis(audio_file=str(input_path), output_folder="analysis_plots")

    # Step 2: Audio Filtering
    print("\n[4/5] Preprocessing and Band-Pass Filtering Audio...")
    filtered_wav, fs = process_audio_filtering(input_file=str(input_path), output_folder="filtered_audio")

    # Step 3: Snapping Event Detection
    print("\n[5/5] Running Adaptive RMS/MAD Snapping Event Detector...")
    detect_snapping_events(input_file=str(filtered_wav), output_folder="snap_events")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("Outputs generated:")
    print(" - Filtered Audio: filtered_audio/filtered_recording.wav")
    print(" - Analysis Plots: analysis_plots/, filtered_audio/")
    print(" - Extracted Snap Clips & CSV: snap_events/")
    print("=" * 60)

if __name__ == "__main__":
    target_file = sys.argv[1] if len(sys.argv) > 1 else "hydrophone_recording.wav"
    run_all(target_file)
