#!/bin/bash
# The dictionary of converted WAVEFORM of DIHARD (16K,16bit)
dihard_wav_dir=/disk1/leisun/data/LDC2018E31_First_DIHARD_Challenge_Development_Data/data/wav/
# Specify the output dictionary
output_dir=../output/
# For GPU with not sufficient memory, long audio sentences should be splitted into sub-audios in case of `out of GPU memory` when decoding LSTM model.
# --gpu_id : indicate which GPU to use
# --truncate_minutes: how many minutes per chunk
python main_denoising.py --wav_dir  $dihard_wav_dir  --output_dir $output_dir --gpu_id  0  --truncate_minutes 10