"""Various utility functions."""
from __future__ import print_function
from __future__ import unicode_literals
import numbers
import os
import sndhdr
import struct
import sys

import librosa.core
import librosa.util
import numpy as np
import scipy.signal
import webrtcvad

EPS = 1e-8


def warn(msg):
    """Print warning message to STERR."""
    msg = 'WARN: %s' % msg
    print(msg, file=sys.stderr)


def error(msg):
    """Print warning message to STERR."""
    msg = 'ERROR: %s' % msg
    print(msg, file=sys.stderr)


# TODO: Find out why this duplicates functionality of librosa.core.stft.
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


# TODO: Find out why this duplicates functionality of librosa.core.istft.
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
def peak_normalization(x):
    """Perform peak normalization."""
    norm = x.astype(float)
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
VALID_VAD_FRAME_LENGTHS = {10, 20, 30}
VALID_VAD_MODES = {0, 1, 2, 3}
def vad(data, fs, fs_vad=16000, frame_length=30, vad_mode=0, med_filt_width=1):
    """Perform voice activity detection using WebRTC.

    VAD is performed by splitting the input into non-overlapping frames
    of size ``frame_length`` ms and then applying a classifier to each
    frame. The classifier is based on the VAD deveoped by Google for
    WebRTC as implemented in ``py-webrtcvad``.

    Parameters
    ----------
    data : ndarray, (n_samples,)
        Input signal.

    fs : int
        Sample rate in Hz of ``data``.

    fs_vad : int, optional
        Sample rate resampled to prior to performing VAD.
        (Default: 16000)

    frame_length : int, optional
        Frame length in milliseconds.
        (Default: 30)

    vad_mode : int, optional
        VAD aggressiveness. As ``vad_mode`` increases, it becomes more aggressive
        about filtering out nonspeech.
        (Default: 0)

    med_filt_width : int, optional
        Window size for median filter used to smooth frame level VAD labels. *MUST*
        be an odd number. Large values lead to more aggressive smoothing. When
        <=1, label smoothing is disabled.
        (Default: 1)

    Returns
    -------
    vact : ndarray, (n_samples,)
        ``vact[i]`` is 1 if voicing detected at sample ``i`` and 0 otherwise.

    References
    ----------
    - https://github.com/wiseman/py-webrtcvad.git
    - https://webrtc.org/
    """
    # Check arguments.
    if fs_vad not in VALID_VAD_SRS:
        raise ValueError('fs_vad must be one of %s' % VALID_VAD_SRS)
    if frame_length not in VALID_VAD_FRAME_LENGTHS:
        raise ValueError(
            'frame_length must be one of %s' % VALID_VAD_FRAME_LENGTHS)
    if vad_mode not in VALID_VAD_MODES:
        raise ValueError('vad_mode must be one of %s' % VALID_VAD_MODES)
    if data.dtype.kind == 'i':
        if data.max() > 2**15 - 1 or data.min() < -2**15:
            raise ValueError(
                'when data type is int, data must be in range [-32768, 32767].')
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
    if not isinstance(med_filt_width, numbers.Integral):
        raise TypeError('med_filt_width must be an odd integer')
    if med_filt_width % 2 == 0:
        raise ValueError('med_filt_width must be an odd integer')


    # Resample.
    if fs != fs_vad:
        data = data.astype('f', copy=False)
        resampled = librosa.core.resample(data, fs, fs_vad)
    else:
        resampled = data
    resampled = resampled.astype('int16')

    # Convert from milliseconds to samples.
    def ms_to_samples(t, sr):
        return t*sr // 1000
    frame_length_resamp = ms_to_samples(frame_length, fs_vad)
    frame_length = ms_to_samples(frame_length, fs)

    # Enframe downsampled signal.
    hop_length_resamp = frame_length_resamp
    n_frames = resampled.size // hop_length_resamp + 1
    n_pad = n_frames * hop_length_resamp - resampled.size
    padded = np.pad(resampled, (0, n_pad), 'constant', constant_values=0)
    framed = librosa.util.frame(
        padded, frame_length=frame_length_resamp, hop_length=hop_length_resamp)
    framed = framed.T # Convert to (n_frames, frame_length_resamp).

    # Classify frames as speech/nonspeech.
    vad = webrtcvad.Vad()
    vad.set_mode(vad_mode)
    valist = [vad.is_speech(frame.tobytes(), fs_vad) for frame in framed]

    # Smooth labels.
    if med_filt_width > 1:
        valist = scipy.signal.medfilt(valist, med_filt_width)
        valist = valist.astype(np.bool)

    # Convert to sample-level labels.
    va_framed = np.zeros((n_frames, frame_length), dtype='uint8')
    va_framed[valist] = 1
    vact = va_framed.reshape(-1)[:data.size]

    return vact


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


def write_segments(fn, segs, n_digits=3, label=''):
    """Write segmentation to file."""
    fmt_str = '%%.%df %%.%df %%s\n' % (n_digits, n_digits)
    with open(fn, 'wb') as f:
        for onset, offset in segs:
            line = fmt_str % (onset, offset, label)
            f.write(line.encode('utf-8'))


def listdir(dirpath, abspath=True, ext=None):
    """List contents of directory."""
    fns = os.listdir(dirpath)
    if ext is not None:
        fns = [fn for fn in fns if fn.endswith(ext)]
    if abspath:
        fns = [os.path.abspath(os.path.join(dirpath, fn))
               for fn in fns]
    fns = sorted(fns)
    return fns


def load_script_file(fn, ext=None):
    """Load HTK script file of paths."""
    with open(fn, 'rb') as f:
        paths = [line.decode('utf-8').strip() for line in f]
    paths = sorted(paths)
    if ext is not None:
        filt_paths = []
        for path in paths:
            if not path.endswith(ext):
                warn('Skipping file "%s" that does not match extension "%s"' %
                     (path, ext))
                continue
            filt_paths.append(path)
        paths = filt_paths
    return paths


def xor(x, y):
    """Return truth value of ``x`` XOR ``y``."""
    return bool(x) != bool(y)


def is_wav(fn):
    """Returns True if ``fn`` is a WAV file."""
    hinfo = sndhdr.what(fn)
    if hinfo is None:
        return False
    elif hinfo[0] != 'wav':
        return False
    return True


def get_sr(fn):
    """Return sample rate in Hz of WAV file."""
    if not is_wav(fn):
        raise ValueError('File "%s" is not a valid WAV file.' % fn)
    hinfo = sndhdr.what(fn)
    return hinfo[1]


def get_num_channels(fn):
    """Return number of channels present in  WAV file."""
    if not is_wav(fn):
        raise ValueError('File "%s" is not a valid WAV file.' % fn)
    hinfo = sndhdr.what(fn)
    return hinfo[2]


def get_bitdepth(fn):
    """Return bitdepth of WAV file."""
    if not is_wav(fn):
        raise ValueError('File "%s" is not a valid WAV file.' % fn)
    hinfo = sndhdr.what(fn)
    return hinfo[4]
