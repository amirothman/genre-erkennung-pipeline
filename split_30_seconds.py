#!/usr/local/bin/python3
import os
import re
import subprocess
import os.path
import sys

def iterate_audio(path="."):
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            # print(name)
            if re.search(".mp3$",name):
                # print(name+"found")
                song_path = (os.path.join(root,name))
                yield song_path

def ffmpeg_process(filepath,cmd,delete_original=True):
    filepath = filepath.replace("./","")
    tupelPath = os.path.splitext(filepath)
    commandos = [cmd.format(tupelPath[0],tupelPath[1])]
    if delete_original:
        del_commando = "rm "+filepath
        commandos.append(del_commando)
    for cmd in commandos:
        print(cmd)
        subprocess.call(cmd.split())

def thirty_seconds(filepath):
    """
    split audio to 30 seconds each.

    ffmpeg -i in.mp3 -f segment -segment_time 30 -c copy out%03d.mp3"""
    split_commando = "ffmpeg -i {0}{1} -f segment -segment_time 30 -c copy {0}-%03d{1}"
    ffmpeg_process(filepath,split_commando)

def ten_seconds(filepath):
    """
    split audio to 30 seconds each.

    ffmpeg -i in.mp3 -f segment -segment_time 30 -c copy out%03d.mp3"""
    split_commando = "ffmpeg -i {0}{1} -f segment -segment_time 10 -c copy {0}-%03d{1}"
    ffmpeg_process(filepath,split_commando)

def to_mono(filepath):
    "to mono"
    "ffmpeg -i stereo.flac -ac 1 mono.flac"
    mono_commando = "ffmpeg -i {0}.{1} -ac 1 {0}-mono.{1}"
    ffmpeg_process(filepath,mono_commando)

#for every file in folder splits in thirty second parts
def batch_thirty_seconds(folder_path):
    print("batch thirty_seconds")
    for song_path in iterate_audio(folder_path):
        thirty_seconds(song_path)

#for every file in folder makes files mono
def batch_mono(folder_path):
    print("batch mono")
    for song_path in iterate_audio(folder_path):
        to_mono(song_path)

if __name__=="__main__":
    # ffmpeg_process("02-Dreaming.mp3")
    # file_format = "mp3"
    # folder_paths = ["dataset/test/rock","dataset/train/rock"]
    # print("proses")
    # for folder_path in folder_paths:
    #     batch_thirty_seconds(folder_path,file_format)
    #     batch_mono(folder_path,file_format)

    file_format = "mp3"
    if len(sys.argv) < 2:
        print("missing parameter for dataset path")
    else:
        batch_thirty_seconds(sys.argv[1])
