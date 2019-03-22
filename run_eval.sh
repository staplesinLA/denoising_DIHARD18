#!/bin/bash
# This script demonstrates how to run speech enhancement and VAD. For full documentation,
# please consult the docstrings of ``main_denoising.py`` and ``main_get_vad.py``.


###################################
# Run speech enhancement
###################################
WAV_DIR=/data/wav/  # Directory of WAV files (16 kHz, 16 bit) to enhance.
SE_WAV_DIR=/data/wav_pn_enhanced  # Output directory for enhanced WAV.
USE_GPU=true  # Use GPU instead of CPU. To instead use CPU, set to 'false'.
GPU_DEVICE_ID=0  # Use GPU with device id 0. Irrelevant if using CPU.
TRUNCATE_MINUTES=10  # Duration in minutes of chunks for enhancement. If you experience
                     # OOM errors with your GPU, try reducing this.
python main_denoising.py \
       --verbose \
       --wav_dir $WAV_DIR --output_dir $SE_WAV_DIR \
       --use_gpu $USE_GPU --gpu_id $GPU_DEVICE_ID \
       --truncate_minutes $TRUNCATE_MINUTES || exit 1


###################################
# Perform VAD using enhanced audio
###################################
VAD_DIR=/data/vad  # Output directory for label files containing VAD output.
HOPLENGTH=30  # Duration in milliseconds of frames for VAD. Also controls step size.
MODE=3  # WebRTC aggressiveness. 0=least agressive and  3=most aggresive.
NJOBS=1  # Number of parallel processes to use.
python main_get_vad.py \
       --verbose \
       --wav_dir $SE_WAV_DIR --output_dir $VAD_DIR \
       --mode $MODE --hoplength $HOPLENGTH \
       --n_jobs $NJOBS || exit 1

exit 0
