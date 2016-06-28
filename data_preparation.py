from split_30_seconds import batch_thirty_seconds
from extract_features import extract_features_single
from parse_features import build_vectors
from split_30_seconds import iterate_audio

_path = "dataset/3_genres/"

# batch_thirty_seconds(_path)

_features = [
                       "vamp:qm-vamp-plugins:qm-adaptivespectrogram:output",
                      "vamp:qm-vamp-plugins:qm-mfcc:coefficients",
                      "vamp:bbc-vamp-plugins:bbc-spectral-contrast:peaks",
                      ]

for _file in iterate_audio(path=_path):
    print("extracting" + _file)
    extract_features_single(_file,_features)

build_vectors(folder_path=_path,
              keyword="adaptivespectrogram_output")
build_vectors(folder_path=_path,
              keyword="spectral-contrast_peaks")
build_vectors(folder_path=_path,
              keyword="mfcc_coefficients",
              lower_limit=1)
