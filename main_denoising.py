# coding: utf-8

import numpy as np
import scipy.io.wavfile as wav_io
import scipy.io as sio
import math
import os
import utils
import pdb
import argparse
import sys

from decode_model import decode_model


HERE = os.path.abspath(os.path.dirname(__file__))

def main_denoising(wav_dir ,out_dir, gpu_id , truncate_minutes):

    if not os.path.exists(wav_dir):
        raise RuntimeError(
            "cannot locate the original dictionary: %s" % wav_dir)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # print "Since the some clips in DHAHRD are long, it's better to
    # split the long sentences to several sub-clips, in case of
    # causing GPU memory problem during LSTM inference.\n "

    # loading global MVN statistics
    glo_mean_var= sio.loadmat(os.path.join(HERE, 'model/global_mvn_stats.mat'))
    mean= glo_mean_var['global_mean']
    var = glo_mean_var['global_var']


    wav_files = [os.path.join(wav_dir,line)  for line in os.listdir(wav_dir) ]

    # feature_extraction
    for wav in wav_files:
        if wav.endswith('.wav'):
            rate, wav_data = wav_io.read(wav)
            sample_length= wav_data.size

            # apply peak-normalization first.
            #pdb.set_trace()
            wav_data = utils.peak_normalization(wav_data)

            chunk_length = truncate_minutes * rate * 60
            total_chunks = int( math.ceil ( float(sample_length) / float(chunk_length) ) )
            se_data_total= np.array([],dtype=np.int16)
            #pdb.set_trace()
            for  i  in range(1, total_chunks + 1):

                if (i==1 and total_chunks==1): # if it only contains 1 chunk
                    temp = wav_data
                elif (i==total_chunks):  # if it's the last chunk
                    temp = wav_data[(i-1)*chunk_length-1:]
                else:
                    if(i==1):
                        temp= wav_data[0:chunk_length-1]
                    else:
                        temp = wav_data[((i-1)*chunk_length-1):(i*chunk_length-1)]
                print("Current processing wav: %s, segment: %d/%d ."%(wav,i,total_chunks))

                if temp.shape[0] < 256: # if it's not enough for one half of frame
                    #pdb.set_trace()
                    data_se = temp # do not process
                    se_data_total = np.append(se_data_total, data_se.astype(np.int16))
                    continue

                # Process the audio in separate temporary files
                noisy_normed_lps='temp_normed.lps';
                enhanced_wav = 'temp_se.wav'

                # extract lps feature from waveform
                noisy_htkdata = utils.wav2logspec(temp,window=np.hamming(512)) ##!!!!!!!!!!!!

                # Do MVN before decoding
                normed_noisy = (noisy_htkdata-mean)/var
                utils.writeHtk(noisy_normed_lps, normed_noisy, sampPeriod=160000, parmKind=9)

                # make the decoding list in CNTK-determined format
                cntk_len = noisy_htkdata.shape[0]-1
                flist=open('./test_normed.scp','w');
                flist.write('test.normedlsp=temp_normed.lps[0,'+str(cntk_len)+"]\n")
                flist.close()

                # Start CNTK model-decoding
                decode_model(gpu_id)

                # Read decoded data
                SE_mat=sio.loadmat('enhanced_norm_fea_mat/test.normedlsp.mat')
                IRM= SE_mat['SE']
                # Directly mask the original feature
                masked_lps = noisy_htkdata + np.log(IRM)

                wave_recon = utils.logspec2wav(
                    masked_lps, temp, window=np.hamming(512), nperseg=512, noverlap=256)
                wav_io.write(enhanced_wav,16000,np.asarray(wave_recon))

                # # # Back to time domain
                rate, data_se = wav_io.read(enhanced_wav)
                se_data_total = np.append(se_data_total, data_se)

            output_wav = os.path.join( out_dir,  wav.split('/')[-1] )
            wav_io.write(output_wav,16000,np.asarray(se_data_total))
            print("Processing wav: %s, done ."%(wav))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Decoding parameters of speech denosing model')
    parser.add_argument('--wav_dir',type=str, default=None)
    parser.add_argument('--output_dir',type= str, default=None)
    parser.add_argument('--gpu_id', type=int, default = 0)
    parser.add_argument('--truncate_minutes', type=int, default = 10)
    args = parser.parse_args()

    main_denoising(wav_dir = args.wav_dir, out_dir = args.output_dir,
                   gpu_id = args.gpu_id, truncate_minutes= args.truncate_minutes)
