#!/bin/bash

while true
do
  # sh keras_gpu.sh mfcc_model.py
  # sh keras_gpu.sh spectral_contrast_model.py
  sh keras_gpu.sh spectral_contrast_valleys_model.py
  sh keras_gpu.sh spectral_contrast_peaks_model.py
done
