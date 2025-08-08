#!/bin/bash

# Exit on Ctrl+C
trap "echo 'Interrupted by user. Exiting...'; exit 1" INT

output_root="extracted_frames"
mkdir -p "$output_root"

video_root="/media/sim/data/ivan/ppgeo_dataset/downloads"
video_list=($(find "$video_root" -maxdepth 1 -type f \( -iname "*.mp4" -o -iname "*.webm" \) | sort))

index=0

for video in "${video_list[@]}"; do
    [ -e "$video" ] || continue

    # Extract video ID (e.g., 00033) and compute dir index
    video_id="$(basename "${video%.*}")"

    # We only want to process video 33
    if [[ "$video_id" != "00033" ]]; then
        index=$((index + 1))
        continue
    fi

    output_dir="$output_root/dir-$index"

    if [ -d "$output_dir" ] && [ "$(ls -A "$output_dir")" ]; then
        echo "Skipping $video — already extracted."
        break
    fi

    echo "Extracting from $video → $output_dir"
    mkdir -p "$output_dir"
    rm -f "$output_dir"/*.jpg

    ffmpeg -loglevel error -ss 0 -i "$video" -r 1 -qscale:v 2 "$output_dir/tmp_%d.jpg"

    n=0
    for f in $(ls "$output_dir"/tmp_*.jpg | sort -V); do
        mv "$f" "$output_dir/$n.jpg"
        n=$((n + 1))
    done

    echo "Finished dir-$index with $n frames."
    break
done

