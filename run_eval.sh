#!/bin/bash
# The dictionary of converted WAVEFORM of DIHARD (16K,16bit)
dihard_wav_dir=/disk1/leisun/data/LDC2018E32_First_DIHARD_Challenge_Evaluation_Data_V1.1/LDC2018E32_First_DIHARD_Challenge_Evaluation_Data_V1.1/data/wav/
# Specify the output dictionary
output_dir=/disk1/leisun/data/LDC2018E32_First_DIHARD_Challenge_Evaluation_Data_V1.1/LDC2018E32_First_DIHARD_Challenge_Evaluation_Data_V1.1/data/wav_pn_enhanced/
# For GPU with not sufficient memory, long audio sentences should be splitted into sub-audios in case of `out of GPU memory` when decoding LSTM model.

#---- Parmeters:
##### --wav_dir: original wav dictionary. (16K,16bit)
##### --output_dir: specify the output dictionary
##### --use_gpu: whether using GPU, if False, it will choose CPU
##### --gpu_id : choose which GPU card to use
##### --truncate_minutes: how many minutes per chunk. (turn it down when meets GPU memory deficiency )
GPU_available=True
python main_denoising.py --wav_dir  $dihard_wav_dir  --output_dir $output_dir --use_gpu $GPU_available   --gpu_id 0  --truncate_minutes 5

## Get the vad information of all wave file from a specified dictionary
#---- Parmeters:
##### wav_dir : str, specify a dictionary
##### mode: int, the vad aggressiveness in webrtcvad
##### hoplength: int, step size in milli-second.
python main_get_vad.py --wav_dir $output_dir --mode 3  --hoplength 30
