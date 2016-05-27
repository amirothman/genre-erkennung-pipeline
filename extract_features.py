#!/usr/local/bin/python
import subprocess
from subprocess import Popen, PIPE
import sys
from split_30_seconds import iterate_audio

def extract_features(path="."):
    audio_features = [
                    #   "vamp:qm-vamp-plugins:qm-tempotracker:tempo",
                      "vamp:qm-vamp-plugins:qm-mfcc:coefficients",
                      "vamp:bbc-vamp-plugins:bbc-spectral-contrast:peaks",
                      ]

    for feature in audio_features:
        cmd = "sonic-annotator -d {0} {1} -r -w csv --csv-stdout".format(feature,path)
        p = Popen(cmd.split(), stdin=PIPE, stdout=PIPE, stderr=PIPE)
        output, err = p.communicate(b"input data that is passed to subprocess' stdin")
        print(output)

def extract_features_single(path="."):
    audio_features = [
                      "vamp:qm-vamp-plugins:qm-tempotracker:tempo",
                    #   "vamp:qm-vamp-plugins:qm-mfcc:coefficients",
                    #   "vamp:bbc-vamp-plugins:bbc-spectral-contrast:peaks",
                      ]

    for feature in audio_features:
        cmd = "sonic-annotator -d {0} {1} -w csv --csv-stdout".format(feature,path)
        subprocess.call(cmd.split())
        resultCSV = subprocess.communicate()[0]
        print(resultCSV)

if __name__=="__main__":
    # extract_features("dataset/gztan_split_10sec")
    #for audio_path in iterate_audio(format_ending="au",path="dataset/gztan_split_10sec"):
    #    extract_features_single(audio_path)
    # extract_features("dataset/train/hiphop")
    # extract_features("dataset/train/rock")
    # extract_features("dataset/train/pop")

    # extract_features("dataset/train")
    if len(sys.argv) < 2:
        print("missing parameter for dataset path")
    else:
        extract_features(sys.argv[1])
