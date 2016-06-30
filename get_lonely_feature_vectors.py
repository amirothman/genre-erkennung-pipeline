from split_30_seconds import iterate_audio
import os
import re

path_to_check = "dataset/3_genres_copy"

feature_csv_patterns = ["_vamp_bbc-vamp-plugins_bbc-spectral-contrast_peaks.csv",
                        "_vamp_bbc-vamp-plugins_bbc-spectral-contrast_valleys.csv",
                        "_vamp_qm-vamp-plugins_qm-adaptivespectrogram_output.csv",
                        "_vamp_qm-vamp-plugins_qm-mfcc_coefficients.csv"]

for _file in iterate_audio(path_to_check):
    # print("rm {0}".format(_file))
    _file_no_end_format = re.sub(r".mp3$","",_file)
    _csv_good = True

    for feature in feature_csv_patterns:
        _csv = "{0}{1}".format(_file_no_end_format,feature)
        if not os.path.isfile(_csv):
            _csv_good = False

    if not _csv_good:
        # print("# no good")
        # print(_file)
        for feature in feature_csv_patterns:
            _csv = "{0}{1}".format(_file_no_end_format,feature)
            if os.path.isfile(_csv):
                print("rm {0}".format(_csv))
