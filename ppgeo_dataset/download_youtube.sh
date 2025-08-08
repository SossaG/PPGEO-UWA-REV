#!/bin/bash

# === Configuration ===
URL_FILE="numbered_urls.txt"
BASE_DIR="/media/sim/data/ivan/ppgeo_dataset"
OUTPUT_DIR="${BASE_DIR}/downloads"
CACHE_DIR="${BASE_DIR}/yt_cache"
COOKIE_FILE="${BASE_DIR}/cookies.txt"  # optional — comment out below if not using

# === Setup ===
mkdir -p "$OUTPUT_DIR"
mkdir -p "$CACHE_DIR"

# === Download loop ===
while IFS=$'\t' read -r number url; do
    if [[ -n "$number" && -n "$url" ]]; then
        echo "Downloading $url as $number.mp4..."

        XDG_CACHE_HOME="$CACHE_DIR" yt-dlp \
            --cookies "$COOKIE_FILE" \
            -o "${OUTPUT_DIR}/${number}.%(ext)s" \
            "$url"
    fi
done < "$URL_FILE"

echo " All downloads attempted. Saved to: $OUTPUT_DIR"

