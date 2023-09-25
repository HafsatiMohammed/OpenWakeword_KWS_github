#!/bin/bash

# set input and output directories
input_dir="Common_voice_Train"
output_dir="Common_voice_Train_resampled"

# loop through language folders
for lang in "en" "de" "fr" "es" "it" "pl" "pt" "ru" "nl" "zh-CN"; do
  lang_input_dir="${input_dir}/${lang}"
  lang_output_dir="${output_dir}/${lang}"
  
  # create output directory if it doesn't exist
  if [ ! -d "${lang_output_dir}" ]; then
    mkdir -p "${lang_output_dir}"
  fi
  
  # loop through speaker folders
  for speaker in {0..19}; do
    speaker_input_dir="${lang_input_dir}/${speaker}"
    speaker_output_dir="${lang_output_dir}/${speaker}"
    
    # create output directory if it doesn't exist
    if [ ! -d "${speaker_output_dir}" ]; then
      mkdir -p "${speaker_output_dir}"
    fi

    # loop through input files and resample with ffmpeg in parallel
    find "${speaker_input_dir}" -name "*.mp3" | parallel -j+0 ffmpeg -i {} -ar 16000 "${speaker_output_dir}/{/.}.wav"


    if [ -f "${speaker_output_dir}/{/.}.wav"]; then
        rm "${speaker_input_dir}" -name "*.mp3"
    fi


    
  done
done
