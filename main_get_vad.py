# -*- coding:UTF-8 -*-
from warnings import warn
import numpy as np
import webrtcvad
from librosa.core import resample
from librosa.util import frame
from librosa import load
import utils
import argparse
import os
#main_vad(wav_dir = args.wav_dir, mode = args.mode, hop= args.hoplength)
def main_vad(wav_dir ,mode, hop):   
    if not os.path.exists(wav_dir):   
        raise RuntimeError("cannot locate the original dictionary !")

    wav_files = [os.path.join(wav_dir,line)  for line in os.listdir(wav_dir) ]
    
    for wav in wav_files:
        if wav.endswith('.wav'):
            data, fs = load(wav, sr=16000)
            vad_info = utils.vad(data, fs, fs_vad = 16000, hoplength = hop, vad_mode=mode)
            
            segments = utils.get_segments(vad_info,fs)          
            output_file = open(wav.replace('.wav','.sad' ),'w')
            for i in range(segments.shape[0]):
                start_time = segments[i][0]
                end_time =  segments[i][1]
                output_file.write( "%.3f  %.3f \n" %(start_time, end_time) )

parser = argparse.ArgumentParser(description='Getting VAD information based on webrtcvad module.')
parser.add_argument('--wav_dir',type=str, default=None)
parser.add_argument('--mode', type=int, default=3)
parser.add_argument('--hoplength',type=int,default=30)
args = parser.parse_args()

main_vad(wav_dir = args.wav_dir, mode = args.mode, hop= args.hoplength)
















