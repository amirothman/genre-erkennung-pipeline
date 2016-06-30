from split_30_seconds import batch_thirty_seconds
from extract_features import extract_features_single,extract_features_single_custom_transform
from parse_features import build_vectors
from split_30_seconds import iterate_audio

_path = "dataset/3_genres_copy"
_data_label = "3_genres_no_lonely"
# batch_thirty_seconds(_path)

_features = [
                      "vamp:qm-vamp-plugins:qm-mfcc:coefficients",
                      "vamp:bbc-vamp-plugins:bbc-spectral-contrast:peaks",
                      "vamp:bbc-vamp-plugins:bbc-spectral-contrast:valleys"
                      ]

for _file in iterate_audio(path=_path):
    print("extracting" + _file)
    extract_features_single_custom_transform(_file,"test.ttl")

build_vectors(folder_path=_path,
              keyword="adaptivespectrogram_output",
              data_label=_data_label,
              lower_limit=1)
#
build_vectors(folder_path=_path,
              keyword="spectral-contrast_peaks",
              data_label=_data_label,
              lower_limit=1)
#
build_vectors(folder_path=_path,
              keyword="mfcc_coefficients",
              data_label=_data_label,
              lower_limit=1)
#
build_vectors(folder_path=_path,
              keyword="spectral-contrast_valleys",
              data_label=_data_label,
              lower_limit=1)
