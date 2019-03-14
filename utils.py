"""Various utility functions."""
from __future__ import print_function
from __future__ import unicode_literals
import struct
import sys

from librosa.core import resample
from librosa.util import frame
import numpy as np
import webrtcvad

EPS = 1e-8


def warn(msg):
    """Print warning message to STERR."""
    msg = 'WARN: %s' % msg
    print(msg, file=sys.stderr)


# TODO: See if there is any reason for not just calling librosa.core.stft.
# n_per_seg = nfft
# should default to window.size
def stft(x, window, n_per_seg=512, noverlap=256):
    """Return short-time Fourier transform (STFT) for signal.

    Parameters
    ----------
    x : ndarray, (n_samps,)
        Input signal.

    window : ndarray, (wl,)
        Array of weights to use when windowing the signal.

    n_per_seg : int, optional

    """
    if len(window) != n_per_seg:
        raise ValueError('window length must equal n_per_seg')
    x = np.array(x)
    nadd = noverlap - (len(x) - n_per_seg) % noverlap
    x = np.concatenate((x, np.zeros(nadd)))
    step = n_per_seg - noverlap
    shape = x.shape[:-1] + ((x.shape[-1] - noverlap) // step, n_per_seg)
    strides = x.strides[:-1] + (step * x.strides[-1], x.strides[-1])
    x = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
    x = x * window
    result = np.fft.rfft(x, n=n_per_seg)
    return result


def istft(x, window, n_per_seg=512, noverlap=256):
    """TODO"""
    x = np.fft.irfft(x)
    y = np.zeros((len(x) - 1) * noverlap + n_per_seg)
    C1 = window[0:256]
    C2 = window[0:256] + window[256:512]
    C3 = window[256:512]
    y[0:noverlap] = x[0][0:noverlap] / C1
    for i in range(1, len(x)):
        y[i * noverlap:(i + 1) * noverlap] = (x[i - 1][noverlap:n_per_seg] + x[i][0:noverlap]) / C2
    y[-noverlap:] = x[len(x) - 1][noverlap:] / C3
    # return y[0:len(wave)]
    return y


def wav2logspec(x, window, n_per_seg=512, noverlap=256):
    """TODO"""
    y = stft(x, window, n_per_seg=n_per_seg, noverlap=noverlap)
    return np.log(np.square(abs(y)) + EPS)


def logspec2wav(lps, wave, window, n_per_seg=512, noverlap=256):
    "Convert log-power spectrum back to time domain."""
    z = stft(wave, window)
    angle = z / (np.abs(z) + EPS) # Recover phase information
    x = np.sqrt(np.exp(lps)) * angle
    x = np.fft.irfft(x)
    y = np.zeros((len(x) - 1) * noverlap + n_per_seg)
    C1 = window[0:256]
    C2 = window[0:256] + window[256:512]
    C3 = window[256:512]
    y[0:noverlap] = x[0][0:noverlap] / C1
    for i in range(1, len(x)):
        y[i*noverlap:(i + 1)*noverlap] = (x[i-1][noverlap:n_per_seg] + x[i][0:noverlap]) / C2
    y[-noverlap:] = x[len(x)-1][noverlap:] / C3
    return np.int16(y[0:len(wave)])


MAX_PCM_VAL = 32767
def peak_normalization(wave):
    """Perform peak normalization."""
    norm = wave.astype(float)
    norm = norm / max(abs(norm)) * MAX_PCM_VAL
    return norm.astype(int)


def read_htk(filename):
    """Return features from HTK file a 2-D numpy array."""
    with open(filename, 'rb') as f:
        # Read header
        n_samples, samp_period, samp_size, parm_kind = struct.unpack(
            '>iihh', f.read(12))

        # Read data
        data = struct.unpack(
            '>%df' % (n_samples * samp_size / 4), f.read(n_samples * samp_size))

        return n_samples, samp_period, samp_size, parm_kind, data


def write_htk(filename, feature, samp_period, parm_kind):
    """Write array of frame-level features to HTK binary file."""
    with open(filename, 'wb') as f:
        # Write header
        n_samples = feature.shape[0]
        samp_size = feature.shape[1] * 4
        f.write(struct.pack('>iihh', n_samples, samp_period, samp_size, parm_kind))
        f.write(struct.pack('>%df' % (n_samples * samp_size / 4), *feature.ravel()))


VALID_VAD_SRS = {8000, 16000, 32000, 48000}
VALID_VAD_HOPLENGTHS = {10, 20, 30}
VALID_VAD_MODES = {0, 1, 2, 3}
def vad(data, fs, fs_vad=16000, hoplength=30, vad_mode=0):
    """ Voice activity detection.

    This was implementioned for easier use of py-webrtcvad.

    Thanks to: https://github.com/wiseman/py-webrtcvad.git

    Parameters
    ----------
    data : ndarray, (n_samps,)
        Input signal.

    fs : int
        Sample rate in Hz of ``data``.

    fs_vad : int, optional
        Sample rate in Hz for webrtcvad.
        (Default: 16000)

    hoplength : int, optional
        Step size in milliseconds.
        (Default: 30)

    vad_mode : int, optional
        Set vad aggressiveness. As ``vad_mode`` increases, it becomes more
        aggressive.
        (Default: 0)

    Returns
    -------
    vact : ndarray
        voice activity. time length of vact is same as input data. If 0, it is
        unvoiced, 1 is voiced.
    """
    # Check arguments.
    if fs_vad not in VALID_VAD_SRS:
        raise ValueError('fs_vad must be one of %s' % VALID_VAD_SRS)
    if hoplength not in VALID_VAD_HOPLENGTHS:
        raise ValueError('hoplength must be one of %s' % VALID_VAD_FRAME_LENGTHS)
    if vad_mode not in VALID_VAD_MODES:
        raise ValueError('vad_mode must be one of %s' % VALID_VAD_MODES)
    if data.dtype.kind == 'i':
        if data.max() > 2**15 - 1 or data.min() < -2**15:
            raise ValueError(
                'when data type is int, data must be in range [-32768, 32767].')
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

    # Resample.
    if fs != fs_vad:
        resampled = resample(data, fs, fs_vad)
    else:
        resampled = data
    resampled = resampled.astype('int16')
 
    # Perform VAD using resampled data.
    hop = fs_vad * hoplength // 1000   
    framelen = resampled.size // hop + 1
    padlen = framelen * hop - resampled.size
    padded = np.lib.pad(resampled, (0, padlen), 'constant', constant_values=0)
    framed = frame(padded, frame_length=hop, hop_length=hop).T
    vad = webrtcvad.Vad()
    vad.set_mode(vad_mode)
    valist = [vad.is_speech(tmp.tobytes(), fs_vad) for tmp in framed]
    hop_origin = fs * hoplength // 1000
    va_framed = np.zeros([len(valist), hop_origin])
    va_framed[valist] = 1

    return va_framed.reshape(-1)[:data.size]


def get_segments(vad_info, fs):
    """Convert array of VAD labels into segmentation."""
    vad_index = np.where(vad_info == 1.0) # Find the speech index.
    vad_diff = np.diff(vad_index)

    vad_temp = np.zeros_like(vad_diff)
    vad_temp[np.where(vad_diff == 1)] = 1
    vad_temp = np.column_stack((np.array([0]), vad_temp, np.array([0])))
    final_index = np.diff(vad_temp)

    starts = np.where(final_index == 1)
    ends = np.where(final_index == -1)

    sad_info = np.column_stack([starts[1], ends[1]])
    vad_index = vad_index[0]

    segments = np.zeros_like(sad_info, dtype=np.float)
    for i in range(sad_info.shape[0]):
        segments[i][0] = float(vad_index[sad_info[i][0]]) / fs
        segments[i][1] = float(vad_index[sad_info[i][1]] + 1) / fs

    return segments  # Present in seconds.
