# -*- coding:UTF-8 -*-
import numpy as np
import scipy.io.wavfile as wav_io
import scipy.io as sio
import math
import os
import HTK
import pdb
import argparse
import sys

def main_denoising(wav_dir ,out_dir, gpu_id , truncate_minutes):
    
    if not os.path.exists(wav_dir):   
        raise RuntimeError("cannot locate the original dictionary !")
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # print "Since the some clips in DHAHRD are long, it's better to split the long sentences to several sub-clips, in case of causing GPU memory problem during LSTM inference.\n "

    # loading global MVN statistics
    glo_mean_var= sio.loadmat('./model/global_mvn_stats.mat')
    mean= glo_mean_var['global_mean']
    var = glo_mean_var['global_var']
    
    
    wav_files = [os.path.join(wav_dir,line)  for line in os.listdir(wav_dir) ]
    
    # feature_extraction
    for wav in wav_files:
        if wav.endswith('.wav'):
            
            rate, wav_data = wav_io.read(wav)
            sample_length= wav_data.size
            #minutes = float(sample_length) / float((60*rate)) # actual minutes of wav_data
            
            chunk_length = truncate_minutes * rate * 60
            if (sys.version_info.major==2):
                total_chunks =  int (math.ceil ( sample_length/ chunk_length)+1 )
            elif (sys.version_info.major==3):
                total_chunks =  int (math.ceil ( sample_length/ chunk_length) )

            se_data_total= np.array([],dtype=np.int16)
            
            for  i  in range(1, total_chunks + 1):
                if (i==1 and total_chunks==1): # if it only contains 1 chunk
                    temp = wav_data
                elif (i==total_chunks):  # if it's the last chunk
                    temp = wav_data[(i-1)*chunk_length-1-int(0.016*rate):]
                else:
                    if(i==1):
                        temp= wav_data[0:chunk_length-1]
                    else:
                        temp = wav_data[((i-1)*chunk_length-1-int(0.016*rate)):(i*chunk_length-1)]
                
                print("Current processing wav: %s, segment: %d/%d ."%(wav,i,total_chunks))
                
                # Process the audio in separate temporary files
                noisy_wav = 'temp.wav'
                noisy_raw = 'temp.raw'
                noisy_lps = 'temp.lps';
                noisy_normed_lps='temp_normed.lps';
                
                enhanced_lps = 'denoised.lps'
                enhanced_raw = 'temp_se.raw'
                enhanced_wav = 'temp_se.wav'
                    
                wav_io.write(noisy_wav,rate,np.asarray(temp)) #write temp.wav
                os.system('sox %s  %s' %(noisy_wav,noisy_raw )) # convert to temp.raw
                

                # Feature extraction : temp.lps
                os.system("./tools/Wav2LogSpec_be -F RAW -fs 16 %s %s" %(noisy_raw,noisy_lps) )
                nSamples, sampPeriod, sampSize, parmKind, data = HTK.readHtk(noisy_lps)
                noisy_htkdata= np.array(data).reshape(nSamples, int(sampSize / 4))
        
                # Do MVN for temporary feature before decoding
                normed_noisy = (noisy_htkdata-mean)/var
                HTK.writeHtk(noisy_normed_lps, normed_noisy, sampPeriod, parmKind)
                
                # make the decoding list in CNTK-determined format
                cntk_len = noisy_htkdata.shape[0]-1
                flist=open('./test_normed.scp','w');
                flist.write('test.normedlsp=temp_normed.lps[0,'+str(cntk_len)+"]\n")
                flist.close()
                
                ## Start decoding 
                os.system('python decode_model.py  %d ' %(gpu_id))                
                
                SE_mat=sio.loadmat('enhanced_norm_fea_mat/test.normedlsp.mat')
                IRM= SE_mat['SE']
                # Directly mask the original feature
                masked_lps = noisy_htkdata + np.log(IRM)
                
                htkdata  =   masked_lps
                HTK.writeHtk(enhanced_lps, htkdata, sampPeriod, parmKind)
                # # Back to time domain
                os.system('./tools/LogSpec2Wav_ReadBe %s %s %s info.txt %s -F RAW -fs 16' %(noisy_raw,noisy_raw,enhanced_lps,enhanced_raw))
                os.system('sox -t raw -e signed-integer -r 16000 -c 1 -b 16 %s  %s' %(enhanced_raw,enhanced_wav))
                rate, data_se = wav_io.read(enhanced_wav)
                se_data_total = np.append(se_data_total, data_se)
            
            output_wav = os.path.join( out_dir,  wav.split('/')[-1] )
            wav_io.write(output_wav,16000,np.asarray(se_data_total))


            
            
parser = argparse.ArgumentParser(description='Decoding parameters of speech denosing model')
parser.add_argument('--wav_dir',type=str, default=None)
parser.add_argument('--output_dir',type= str, default=None)
parser.add_argument('--gpu_id', type=int, default = 0)
parser.add_argument('--truncate_minutes', type=int, default = 10)
args = parser.parse_args()

main_denoising(wav_dir = args.wav_dir, out_dir = args.output_dir, gpu_id = args.gpu_id, truncate_minutes= args.truncate_minutes)
















