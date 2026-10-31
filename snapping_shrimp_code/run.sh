#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

AUDIO_FOLDER="singleaudio/"

rm -rf snap_events filtered_audio analysis_plots all_detected_events.csv clustered_events.csv cluster_feature_separation.png

python3 batch_process.py "$AUDIO_FOLDER"

python3 auto_cluster.py

python3 generate_results.py
