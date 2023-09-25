#!/bin/bash

for language in en de fr es it pl pt ru nl zh-CN ; do
    nohup python3.9 Download_Common_Voice.py -l $language > output_$language.log 2>&1 &
done

echo "Downloading Common Voice dataset in background. Check log files for progress."