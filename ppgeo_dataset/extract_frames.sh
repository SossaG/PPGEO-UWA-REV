#!/bin/bash

# Exit on Ctrl+C
trap "echo 'Interrupted by user. Exiting...'; exit 1" INT

# Output directory for extracted frames
output_root="extracted_frames"
mkdir -p "$output_root"

# Folder containing downloaded videos
video_root="/media/sim/data/ivan/ppgeo_dataset/downloads"

index=0

# Loop through .mp4 and .webm videos by number prefix
video_list=($(find "$video_root" -maxdepth 1 -type f \( -iname "*.mp4" -o -iname "*.webm" \) | sort))


for video in "${video_list[@]}"; do
    [ -e "$video" ] || continue  # skip if no matches

    video_id="$(basename "${video%.*}")"
    output_dir="$output_root/dir-$index"

    if [ -d "$output_dir" ] && [ "$(ls -A "$output_dir")" ]; then
        echo "Skipping $video — already extracted."
        index=$((index + 1))
        continue
    fi

    echo "Extracting from $video → $output_dir"
    mkdir -p "$output_dir"
    rm -f "$output_dir"/*.jpg

    # Extract 1 frame per second starting from time 0
    ffmpeg -loglevel error -ss 0 -i "$video" -r 1 -qscale:v 2 "$output_dir/tmp_%d.jpg"

    # Rename to sequential filenames: 0.jpg, 1.jpg, ...
    n=0
    for f in $(ls "$output_dir"/tmp_*.jpg | sort -V); do
        mv "$f" "$output_dir/$n.jpg"
        n=$((n + 1))
    done

    echo "Finished dir-$index with $n frames."
    index=$((index + 1))
    #break #FOR NOW WHILE I TEST ONE SINGULAR VIDEO EXTRACTION
done

