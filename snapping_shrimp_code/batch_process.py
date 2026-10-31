import os
import glob
import pandas as pd
from pathlib import Path
from audio_filtering import process_audio_filtering
from snapping_event_detection import detect_snapping_events

def run_batch_pipeline(input_dir="audio_files_folder", output_csv="all_detected_events.csv"):
    audio_files = glob.glob(os.path.join(input_dir, "*.wav"))
    print(f"Found {len(audio_files)} audio files to process in '{input_dir}'.")

    all_metadata = []

    for idx, audio_file in enumerate(audio_files, 1):
        file_stem = Path(audio_file).stem
        print(f"\n[{idx}/{len(audio_files)}] Processing: {Path(audio_file).name}")
        
        try:
            # Step 1: Filter audio
            filtered_wav, fs = process_audio_filtering(
                input_file=audio_file, 
                output_folder="filtered_audio"
            )
            
            # Step 2: Detect events with unique file prefix
            detect_snapping_events(
                input_file=str(filtered_wav), 
                output_folder="snap_events",
                file_prefix=file_stem
            )
            
            # Step 3: Accumulate metadata
            csv_path = Path(f"snap_events/{file_stem}_metadata.csv")
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                df["source_file"] = Path(audio_file).name
                all_metadata.append(df)

        except Exception as e:
            print(f"Error processing {audio_file}: {e}")

    # Combine into single master CSV
    if all_metadata:
        combined_df = pd.concat(all_metadata, ignore_index=True)
        combined_df.to_csv(output_csv, index=False)
        print(f"\nBatch processing complete! Saved {len(combined_df)} total candidate events across all files to '{output_csv}'.")

if __name__ == "__main__":
    import sys
    folder = sys.argv[1] if len(sys.argv) > 1 else "audio_files_folder"
    run_batch_pipeline(input_dir=folder)