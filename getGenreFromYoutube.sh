#!/bin/bash

if [ "$#" -ne 1 ]; then
	echo "pass youtube url as parameter"
else
	cd $(dirname "$0")
	youtube-dl $1 -x  --audio-format mp3 --restrict-filenames -o 'testfile.mp3' #download file
	printf "File downloaded \n"
	python3 ./extract_features.py testfile.mp3 #extract features
	printf "Features extracted \n"
	python3 ./parse_features.py testfile.mp3 #decode features
	printf "parsing features \n"
	# $1[0:-3]*.csv #remove csv files
	printf "Cleanup failed \n"
fi

