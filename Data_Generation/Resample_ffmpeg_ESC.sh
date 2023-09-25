#!/bin/bash

# set input and output directories
input_dir="mp3"
output_dir="mp3_resampled"

# loop through language folders

  # loop through speaker folders
  #for speaker in {0..19}; do
speaker_input_dir=$input_dir 
speaker_output_dir=$output_dir

    # create output directory if it doesn't exist
if [ ! -d "${speaker_output_dir}" ]; then
      mkdir -p "${speaker_output_dir}"
fi

    # loop through input files and resample with ffmpeg in parallel
find "${speaker_input_dir}" -name "*.mp3" | parallel -j+0 ffmpeg -i {} -ar 16000 "${speaker_output_dir}/{/.}.wav"

    # remove MP3 files from input directory if the corresponding output file exists
find "${speaker_input_dir}" -name "*.mp3" -type f | while read mp3file; do
    wavfile="${speaker_output_dir}/$(basename "${mp3file}" .mp3).wav"
    if [ -f "${wavfile}" ]; then
        rm "${mp3file}"
    fi
done
