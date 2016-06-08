#!/bin/bash

#example url http://youtube.com/watch?v=PNjG22Gbo6U
if [ "$#" -ne 1 ]; then
	echo "pass youtube url as parameter"
else
	cd $(dirname "$0")
	youtube-dl --extract-audio --audio-format mp3 -o "testfile.%(ext)s" $1 #download file
	printf "\nFile downloaded \n"
	#python3 ./extract_features.py testfile.mp3 #extract features
	#delete file
	#rm ./testfile.mp3
	#printf "\nFeatures extracted \n"
	python3 ./querying_genre.py testfile.mp3 #decode features
	printf "\nparsed features \n"
	find *.csv -type f -delete -depth 1
	find *.mp3 -type f -delete -depth 1
	printf "Cleanup performed \n"
fi