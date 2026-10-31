#!/bin/bash

OUTPUT="combined.wav"

# Generate sorted WAV file list
find "$PWD" -maxdepth 1 -type f -iname "*.wav" \
    ! -name "$OUTPUT" | sort > files.txt

# Create concat list
sed "s|.*|file '&'|" files.txt > concat.txt

# Merge WAV files without re-encoding
ffmpeg -f concat -safe 0 -i concat.txt -c:a copy "$OUTPUT"

# Cleanup
rm -f files.txt concat.txt

echo "Successfully created $OUTPUT"
