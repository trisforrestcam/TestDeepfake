#!/bin/bash

video="$1"
no_audio="$2"
output="${3:-output_with_audio.mp4}"

if [ -z "$video" ] || [ -z "$no_audio" ]; then
    echo "Usage: ./merge_audio.sh <video_co_am_thanh> <video_ko_am_thanh> [output]"
    echo ""
    echo "Example:"
    echo "  ./merge_audio.sh ComeMyWayVideo.mp4 output_deepfake.mp4"
    echo "  ./merge_audio.sh input.mp4 no_audio.mp4 final.mp4"
    exit 1
fi

ffmpeg -y \
    -i "$no_audio" \
    -i "$video" \
    -c:v copy \
    -c:a aac \
    -map 0:v:0 \
    -map 1:a:0? \
    -shortest \
    "$output"

echo "Done: $output"
