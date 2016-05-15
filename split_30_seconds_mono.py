import os
import re
import subprocess

def iterate_audio(format_ending="",path="."):
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            if re.search(".{0}".format(format_ending),name):
                song_path = (os.path.join(root,name))
                yield song_path

def ffmpeg_process(filepath,format_ending,cmd,delete_original=True):
    filepath = filepath.replace("./","")
    filepath = filepath.replace(".{0}".format(format_ending),"")
    commandos = [cmd.format(filepath,format_ending)]
    if delete_original:
        del_commando = "rm {0}".format("".join([filepath,".",format_ending]))
        commandos.append(del_commando)
    for cmd in commandos:
        print(cmd)
        subprocess.call(cmd.split())

def thirty_seconds(filepath,format_ending):
    """
    split audio to 30 seconds each.

    ffmpeg -i in.mp3 -f segment -segment_time 30 -c copy out%03d.mp3"""
    split_commando = "ffmpeg -i {0}.{1} -f segment -segment_time 30 -c copy {0}-%03d.{1}"
    ffmpeg_process(filepath,format_ending,split_commando)

def to_mono(filepath,format_ending):
    "to mono"
    "ffmpeg -i stereo.flac -ac 1 mono.flac"
    mono_commando = "ffmpeg -i {0}.{1} -ac 1 {0}-mono.{1}"
    ffmpeg_process(filepath,format_ending,mono_commando)

def batch_thirty_seconds(file_format,folder_path):
    print("batch thirty_seconds")
    for song_path in iterate_audio(file_format,folder_path):
        print(song_path)
        thirty_seconds(song_path,file_format)

def batch_mono(file_format,folder_path):
    print("batch mono")
    for song_path in iterate_audio(file_format,folder_path):
        print(song_path)
        to_mono(song_path,file_format)

if __name__=="__main__":
    # ffmpeg_process("02-Dreaming.mp3")
    file_format = "mp3"
    folder_path = "ngetest"
    print("proses")
    batch_thirty_seconds(file_format,folder_path)
    batch_mono(file_format,folder_path)
