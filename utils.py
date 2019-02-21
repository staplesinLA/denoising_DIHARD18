import struct
import scipy.io.wavfile
import os
import numpy as np
from scipy import signal
import webrtcvad
from librosa.core import resample
from librosa.util import frame
from librosa import load
import pdb  


def stft(x, window, nperseg=512, noverlap=256):
    if len(window) != nperseg:
        raise ValueError('window length must equal nperseg')
    x = np.array(x)
    nadd = noverlap - (len(x) - nperseg) % noverlap
    x = np.concatenate((x, np.zeros(nadd)))
    step = nperseg - noverlap
    shape = x.shape[:-1] + ((x.shape[-1] - noverlap) // step, nperseg)
    strides = x.strides[:-1] + (step * x.strides[-1], x.strides[-1])
    x = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
    x = x * window
    result = np.fft.rfft(x, n=nperseg)
    return result


def istft(x, window, nperseg=512, noverlap=256):
    x = np.fft.irfft(x)
    y = np.zeros((len(x) - 1) * noverlap + nperseg)
    C1 = window[0:256]
    C2 = window[0:256] + window[256:512]
    C3 = window[256:512]
    y[0:noverlap] = x[0][0:noverlap] / C1   
    for i in range(1, len(x)):
        y[i * noverlap:(i + 1) * noverlap] = (x[i - 1][noverlap:nperseg] + x[i][0:noverlap]) / C2
    y[-noverlap:] = x[len(x) - 1][noverlap:] / C3
    # return y[0:len(wave)]
    return y


def wav2logspec(x, window, nperseg=512, noverlap=256):
    y = stft(x, window, nperseg=nperseg, noverlap=noverlap)
    return np.log(np.square(abs(y))+1e-8)


def logspec2wav(lps, wave, window, nperseg=512, noverlap=256):
    z = stft(wave, window)
    angle = z / (np.abs(z) + 1e-8)
    x = np.sqrt(np.exp(lps)) * angle
    x = np.fft.irfft(x)
    y = np.zeros((len(x)-1) * noverlap + nperseg)
    C1 = window[0:256]
    C2 = window[0:256] + window[256:512]
    C3 = window[256:512]
    y[0:noverlap] = x[0][0:noverlap] / C1  #
    for i in range(1, len(x)):
        y[i*noverlap:(i+1)*noverlap] = (x[i-1][noverlap:nperseg] + x[i][0:noverlap]) / C2
    y[-noverlap:] = x[len(x)-1][noverlap:] / C3
    # return y[0:len(wave)]
    return np.int16(y[0:len(wave)])

    
    
def peak_normalization(wave):
    norm = wave.astype(float)
    norm = norm / ( max (abs(norm))) * (np.exp2(15)-1)
    return norm.astype(int)

def readHtk(filename):
    """
    Reads the features in a HTK file, and returns them in a 2-D numpy array.
    """
    with open(filename, "rb") as f:
        # Read header
        nSamples, sampPeriod, sampSize, parmKind = struct.unpack(">iihh", f.read(12))
        # Read data
        data = struct.unpack(">%df" % (nSamples * sampSize / 4), f.read(nSamples * sampSize))
      #  return numpy.array(data).reshape(nSamples, int(sampSize / 4))
        return nSamples, sampPeriod, sampSize, parmKind, data

def writeHtk(filename, feature, sampPeriod, parmKind):
    """
    Writes the features in a 2-D numpy array into a HTK file.
    """
    with open(filename, "wb") as f:
        # Write header
        nSamples = feature.shape[0]
        sampSize = feature.shape[1] * 4
        f.write(struct.pack(">iihh", nSamples, sampPeriod, sampSize, parmKind))
        f.write(struct.pack(">%df" % (nSamples * sampSize / 4), *feature.ravel()))
       
        

def vad(data, fs, fs_vad=16000, hoplength=30, vad_mode=0):
    """ Voice activity detection.
    This was implementioned for easier use of py-webrtcvad.
    Thanks to: https://github.com/wiseman/py-webrtcvad.git
    Parameters
    ----------
    data : ndarray
        numpy array of mono (1 ch) speech data.
        1-d or 2-d, if 2-d, shape must be (1, time_length) or (time_length, 1).
        if data type is int, -32768 < data < 32767.
        if data type is float, -1 < data < 1.
    fs : int
        Sampling frequency of data.
    fs_vad : int, optional
        Sampling frequency for webrtcvad.
        fs_vad must be 8000, 16000, 32000 or 48000.
        Default is 16000.
    hoplength : int, optional
        Step size[milli second].
        hoplength must be 10, 20, or 30.
        Default is 0.1.
    vad_mode : int, optional
        set vad aggressiveness.
        As vad_mode increases, it becomes more aggressive.
        vad_mode must be 0, 1, 2 or 3.
        Default is 0.
    Returns
    -------
    vact : ndarray
        voice activity. time length of vact is same as input data.
        If 0, it is unvoiced, 1 is voiced.
    """

    # check argument
    if fs_vad not in [8000, 16000, 32000, 48000]:
        raise ValueError('fs_vad must be 8000, 16000, 32000 or 48000.')

    if hoplength not in [10, 20, 30]:
        raise ValueError('hoplength must be 10, 20, or 30.')

    if vad_mode not in [0, 1, 2, 3]:
        raise ValueError('vad_mode must be 0, 1, 2 or 3.')

    # check data
    if data.dtype.kind == 'i':
        if data.max() > 2**15 - 1 or data.min() < -2**15:
            raise ValueError(
                'when data type is int, data must be -32768 < data < 32767.')
        data = data.astype('f')

    elif data.dtype.kind == 'f':
        if np.abs(data).max() >= 1:
            data = data / np.abs(data).max() * 0.9
            warn('input data was rescaled.')
        data = (data * 2**15).astype('f')
    else:
        raise ValueError('data dtype must be int or float.')

    data = data.squeeze()
    if not data.ndim == 1:
        raise ValueError('data must be mono (1 ch).')

    # resampling
    if fs != fs_vad:
        resampled = resample(data, fs, fs_vad)
    else:
        resampled = data

    resampled = resampled.astype('int16')

    hop = fs_vad * hoplength // 1000
    framelen = resampled.size // hop + 1
    padlen = framelen * hop - resampled.size
    paded = np.lib.pad(resampled, (0, padlen), 'constant', constant_values=0)
    framed = frame(paded, frame_length=hop, hop_length=hop).T
    
    
    vad = webrtcvad.Vad()
    vad.set_mode(vad_mode)
    valist = [vad.is_speech(tmp.tobytes(), fs_vad) for tmp in framed]

    hop_origin = fs * hoplength // 1000
    va_framed = np.zeros([len(valist), hop_origin])
    va_framed[valist] = 1

    return va_framed.reshape(-1)[:data.size]
    
    
    
    

def get_segments(vad_info,fs):
    vad_index = np.where( vad_info==1.0) # find the speech index   
    vad_diff = np.diff( vad_index)
            
    vad_temp = np.zeros_like(vad_diff)
    vad_temp[ np.where(vad_diff==1) ]  = 1       
    vad_temp =  np.column_stack( (np.array([0]), vad_temp, np.array([0]) ))
    final_index = np.diff(vad_temp)
            
    starts = np.where( final_index == 1)
    ends = np.where( final_index == -1)
      
    sad_info = np.column_stack( (starts[1], ends[1]) ) 
    vad_index = vad_index[0]
    
    segments = np.zeros_like(sad_info,dtype=np.float)
    for i in range(sad_info.shape[0]):
        segments[i][0] = float( vad_index [ sad_info[i][0] ] ) / fs
        segments[i][1] =  float( vad_index[ sad_info[i][1] ]  +1 ) / fs
 
    return  segments  # present in seconds
















