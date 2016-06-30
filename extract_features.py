#!/usr/local/bin/python3
import subprocess
from subprocess import Popen, PIPE
import sys
from split_30_seconds import iterate_audio
import os

def extract_features(path=".",audio_features=["vamp:qm-vamp-plugins:qm-mfcc:coefficients"]):
    for feature in audio_features:
        cmd = "sonic-annotator -d {0} {1} -r -w csv".format(feature,path)
        subprocess.call(cmd.split())
        #p = Popen(cmd.split(), stdin=PIPE, stdout=PIPE, stderr=PIPE)
        #output, err = p.communicate()

def extract_features_single(path=".",audio_features=["vamp:qm-vamp-plugins:qm-mfcc:coefficients"]):
    for feature in audio_features:
        cmd = "sonic-annotator -d {0} {1} -w csv".format(feature,path)
        subprocess.call(cmd.split())

def extract_features_single_custom_transform(path=".",ttl="test.ttl"):
    cmd = "sonic-annotator -t {0} {1} -w csv".format(ttl,path)
    subprocess.call(cmd.split())

if __name__=="__main__":
    # extract_features("dataset/gztan_split_10sec")

    #    extract_features_single(audio_path)
    # extract_features("dataset/train/hiphop")
    # extract_features("dataset/train/rock")
    # extract_features("dataset/train/pop")

    # extract_features("dataset/train")
    _features = [
                        #   "vamp:qm-vamp-plugins:qm-tempotracker:tempo",
                          "vamp:qm-vamp-plugins:qm-mfcc:coefficients",
                          "vamp:bbc-vamp-plugins:bbc-spectral-contrast:peaks",
                          ]
    if len(sys.argv) < 2:
        print("missing parameter for dataset or file path")
    else:
        if os.path.isdir(sys.argv[1]):
            for _file in iterate_audio(path=sys.argv[1]):
                print("extracing" + _file)
                extract_features_single(_file,_features)
        else:
            extract_features_single(sys.argv[1],_features)

    #extract features for single file
