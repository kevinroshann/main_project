import os
import sys
import shutil
import subprocess

def clean_previous_outputs():
    """Deletes old generated directories and output files to prevent data contamination."""
    print("\n[Step 1/5] Cleaning previous run artifacts...")
    folders_to_remove = ["snap_events", "filtered_audio", "analysis_plots"]
    files_to_remove = [
        "all_detected_events.csv", 
        "clustered_events.csv", 
        "cluster_feature_separation.png",
        "temporal_snap_analysis.png"
    ]

    for folder in folders_to_remove:
        if os.path.exists(folder):
            shutil.rmtree(folder)
            print(f" Removed folder: {folder}/")

    for file in files_to_remove:
        if os.path.exists(file):
            os.remove(file)
            print(f" Removed file: {file}")

def run_script(command):
    """Executes a terminal command and stops if an error occurs."""
    print(f"\nExecuting: {' '.join(command)}")
    result = subprocess.run(command)
    if result.returncode != 0:
        print(f"\nError: Command failed with return code {result.returncode}")
        sys.exit(result.returncode)

def main():
    audio_folder = sys.argv[1] if len(sys.argv) > 1 else "singleaudio/"

    print("=" * 60)
    print("      STARTING SNAPPING SHRIMP ANALYSIS PIPELINE (WITH TEMPORAL)  ")
    print("=" * 60)

    # 1. Clean workspace
    clean_previous_outputs()

    # 2. Run Batch Processing & Event Detection
    print("\n[Step 2/5] Running Batch Preprocessing & Event Detection...")
    run_script([sys.executable, "batch_process.py", audio_folder])

    # 3. Run Auto Feature Extraction & Clustering
    print("\n[Step 3/5] Running Feature Extraction & K-Means Clustering...")
    run_script([sys.executable, "auto_cluster.py"])

    # 4. Generate Physical Feature Separation Results
    print("\n[Step 4/5] Generating Physical Verification & Cluster Plots...")
    run_script([sys.executable, "generate_results.py"])

    # 5. Run Temporal & Inter-Snap Interval Analysis
    print("\n[Step 5/5] Running Temporal Activity & ISI Analysis...")
    run_script([sys.executable, "temporal_analysis.py"])

    print("\n" + "=" * 60)
    print("          PIPELINE COMPLETED SUCCESSFULLY!                  ")
    print("Outputs generated:")
    print(" - Candidate CSV & Clips: snap_events/ & all_detected_events.csv")
    print(" - Cluster Results: clustered_events.csv & cluster_feature_separation.png")
    print(" - Temporal & ISI Plot: temporal_snap_analysis.png")
    print("=" * 60)

if __name__ == "__main__":
    main()
