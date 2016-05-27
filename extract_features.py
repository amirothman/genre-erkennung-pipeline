import subprocess
from split_30_seconds_mono import iterate_audio

def extract_features(path="."):
    audio_features = [
                    #   "vamp:qm-vamp-plugins:qm-tempotracker:tempo",
                    #   "vamp:qm-vamp-plugins:qm-mfcc:coefficients",
                    #   "vamp:bbc-vamp-plugins:bbc-spectral-contrast:peaks",
                        "vamp:bbc-vamp-plugins:bbc-spectral-flux:spectral-flux"
                      ]

    for feature in audio_features:
        cmd = "sonic-annotator -d {0} {1} -r -w csv --csv-force".format(feature,path)
        subprocess.call(cmd.split())

def extract_features_single(path="."):
    audio_features = [
                      "vamp:qm-vamp-plugins:qm-tempotracker:tempo",
                    #   "vamp:qm-vamp-plugins:qm-mfcc:coefficients",
                    #   "vamp:bbc-vamp-plugins:bbc-spectral-contrast:peaks",
                      ]

    for feature in audio_features:
        cmd = "sonic-annotator -d {0} {1} -w csv --csv-force".format(feature,path)
        subprocess.call(cmd.split())

if __name__=="__main__":
    # extract_features("dataset/gztan_split_10sec")
    # for audio_path in iterate_audio(format_ending="au",path="dataset/gztan_split_10sec"):
    #     extract_features_single(audio_path)
    extract_features("dataset/gztan_split_10sec")
    # extract_features("dataset/train/rock")
    # extract_features("dataset/train/pop")
    # extract_features("dataset/train")
