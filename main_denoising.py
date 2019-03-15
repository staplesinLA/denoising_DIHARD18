#!/usr/bin/env python
"""Perform denoising of audio stored in WAV files.

References
----------
Sun, Lei, et al. "Speaker diarization with enhancing speech for the First DIHARD
Challenge." Proceedings of INTERSPEECH 2019. 2793-2797.
"""
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import argparse
import glob
import math
from multiprocessing import Process
import os
import shutil
import tempfile

import numpy as np
import scipy.io.wavfile as wav_io
import scipy.io as sio

from decode_model import decode_model
import utils

HERE = os.path.abspath(os.path.dirname(__file__))
GLOBAL_MEAN_VAR_MATF = os.path.join(HERE, 'model', 'global_mvn_stats.mat')


SR = 16000 # Sample rate of files in Hz.
WL = 512 # Analysis window length in samples for feature extraction.
WL2 = WL // 2
NFREQS = 257 # Number of positive frequencies in FFT output.


def main_denoising(wav_dir, out_dir, use_gpu, gpu_id, truncate_minutes):
    """Perform speech enhancement for WAV files in ``wav_dir``.

    Parameters
    ----------
    wav_dir : str
        Path to directory of WAV files to enhance.

    out_dir : str
        Path to output directory for enhanced WAV files.

    use_gpu : bool, optional
        If True and GPU is available, perform all processing on GPU.
        (Default: True)

    gpu_id : int, optional
         Id of GPU on which to do computation.
         (Default: 0)

    truncate_minutes: float
        Maximimize size in minutes to process at a time. The enhancement will
        be done on chunks of audio no greather than ``truncate_minutes``
        minutes duration.
    """
    if not os.path.exists(wav_dir):
        raise RuntimeError(
            'Cannot locate the original dictionary: %s' % wav_dir)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Load global MVN statistics.
    glo_mean_var = sio.loadmat(GLOBAL_MEAN_VAR_MATF)
    mean = glo_mean_var['global_mean']
    var = glo_mean_var['global_var']

    # Perform speech enhancement.
    wav_files = sorted(glob.glob(os.path.join(wav_dir, '*.wav')))
    for wav in wav_files:
        # Read noisy audio WAV file.
        rate, wav_data = wav_io.read(wav)
        if rate != SR:
            print('ERROR: Sample rate of file "%s" is not %d Hz. Skipping.' %
                  (wav, SR))
            continue

        # Apply peak-normalization first.
        wav_data = utils.peak_normalization(wav_data)

        # Perform denoising in chunks of size chunk_length samples.
        chunk_length = int(truncate_minutes * rate * 60)
        total_chunks = int(
            math.ceil(wav_data.size / chunk_length))
        data_se = [] # Will hold enhanced audio data for each chunk.
        for i in range(1, total_chunks + 1):
            tmp_dir = tempfile.mkdtemp()
            try:
                # Get samples for this chunk.
                bi = (i-1)*chunk_length # Index of first sample of this chunk.
                ei = bi + chunk_length # Index of last sample of this chunk + 1.
                temp = wav_data[bi:ei]
                print('Processing file: %s, segment: %d/%d.' %
                      (wav, i, total_chunks))

                # Skip denoising if chunk is too short.
                if temp.shape[0] < WL2:
                    data_se.append(temp)
                    continue

                # Determine paths to the temporary files to be created.
                noisy_normed_lps_fn = os.path.join(
                    tmp_dir, 'noisy_normed_lps.htk')
                noisy_normed_lps_scp_fn = os.path.join(
                    tmp_dir, 'noisy_normed_lps.scp')
                irm_fn = os.path.join(
                    tmp_dir, 'irm.mat')
                enhanced_wav = os.path.join(tmp_dir, 'se.wav')

                # Extract LPS features from waveform.
                noisy_htkdata = utils.wav2logspec(temp, window=np.hamming(WL))

                # Do MVN before decoding.
                normed_noisy = (noisy_htkdata - mean) / var

                # Write features to HTK binary format making sure to also
                # create a script file.
                utils.write_htk(
                    noisy_normed_lps_fn, normed_noisy, samp_period=SR,
                    parm_kind=9)
                cntk_len = noisy_htkdata.shape[0] - 1
                with open(noisy_normed_lps_scp_fn, 'w') as f:
                    f.write('irm=%s[0,%d]\n' % (noisy_normed_lps_fn, cntk_len))

                # Apply CNTK model to determine ideal ratio mask (IRM), which will
                # be output to the temp directory as irm.mat. In order to avoid a
                # memory leak, must do this in a separate process which we then
                # kill.
                p = Process(
                    target=decode_model,
                    args=(noisy_normed_lps_scp_fn, tmp_dir, NFREQS, use_gpu,
                          gpu_id))
                p.start()
                p.join()

                # Read in IRM and directly mask the original LPS features.
                irm = sio.loadmat(irm_fn)['IRM']
                masked_lps = noisy_htkdata + np.log(irm)

                # Reconstruct audio.
                wave_recon = utils.logspec2wav(
                    masked_lps, temp, window=np.hamming(WL), n_per_seg=WL,
                    noverlap=WL2)
                wav_io.write(enhanced_wav, SR, np.asarray(wave_recon))

                # Back to time domain.
                rate, chunk_data_se = wav_io.read(enhanced_wav)
                data_se.append(chunk_data_se)
            finally:
                shutil.rmtree(tmp_dir)
        data_se = np.concatenate(data_se)
        bn = os.path.basename(wav)
        output_wav = os.path.join(out_dir, bn)
        wav_io.write(output_wav, SR, data_se)
        print('Finished processing file "%s".' % wav)


def main():
    """Main."""
    parser = argparse.ArgumentParser(
        description='Denoise WAV files.', add_help=True)
    parser.add_argument(
        '--wav_dir', nargs=None, type=str, metavar='STR',
        help='directory containing WAV files to denoise '
             '(default: %(default)s')
    parser.add_argument(
        '--output_dir', nargs=None, type=str, metavar='STR',
        help='output directory for denoised WAV files (default: %(default)s)')
    parser.add_argument(
        '--use_gpu', nargs=None, type=str, metavar='STR',
        choices=['true', 'false'], default='true',
        help='whether or not to use GPU (default: %(default)s)')
    parser.add_argument(
        '--gpu_id', nargs=None, type=int, metavar='INT', default=0,
        help='device id of GPU to use (default: %(default)s)')
    parser.add_argument(
        '--truncate_minutes', nargs=None, type=float,
        metavar='FLOAT', default=10,
        help='maximum chunk size in minutes (default: %(default)s)')
    args = parser.parse_args()
    use_gpu = args.use_gpu == 'true'
    main_denoising(
        args.wav_dir, args.output_dir, use_gpu, args.gpu_id,
        args.truncate_minutes)


if __name__ == '__main__':
    main()
