#!/usr/bin/env python
"""TODO

To change the aggressiveness mode, how aggressive WebRTC is about filtering out
non-speech, use the ``--mode`` parameter with an integer in the range [0, 3]:

    python main_get_vad.py --mode 0 --wav_dir some_dir

Higher values for ``--mode`` translate into more aggressive filtering.

"""
from __future__ import print_function
from __future__ import unicode_literals
import argparse
import os
import sys

from joblib import delayed, Parallel
import librosa

import utils
from utils import VALID_VAD_SRS, VALID_VAD_FRAME_LENGTHS, VALID_VAD_MODES


def perform_vad(wav_file, segs_file, **kwargs):
    """Perform VAD for WAV file.

    Parameters
    ----------
    wav_file : str
        Path to WAV file to perform VAD for.

    segs_file : str
        Path to output segments file.

    kwargs
        Keyword arguments to pass to ``utils.vad``.

    Returns
    -------
    e : Exception
        If an exception is raised during processing, it is returned. Otherwise,
        returns ``None``.
    """
    try:
        data, fs = librosa.load(wav_file, sr=16000)
        vad_info = utils.vad(data, fs, **kwargs)
        segments = utils.get_segments(vad_info, fs)
        utils.write_segments(segs_file, segments)
        return None
    except Exception as e:
        return e


def main():
    """Main."""
    # Parse command line arguments.
    parser = argparse.ArgumentParser(
        description='Perform VAD using webrtcvad.', add_help=True)
    parser.add_argument(
        '--wav_dir', nargs=None, type=str, metavar='STR',
        help='directory containing WAV files to perform VAD for '
             '(default: %(default)s)')
    parser.add_argument(
        '--output_dir', nargs=None, type=str, metavar='STR',
        help='output directory for denoised WAV files (default: %(default)s)')
    parser.add_argument(
        '-S', dest='scpf', nargs=None, type=str, metavar='STR',
        help='script file of paths to WAV files to denosie (default: %(default)s)')
    parser.add_argument(
        '--fs_vad', nargs=None, default=16000, type=int, metavar='INT',
        help='target sample rate in Hz for VAD (default: %(default)s)')
    parser.add_argument(
        '--hoplength', nargs=None, default=30, type=int, metavar='INT',
        help='duration between frames in ms (default: %(default)s)')
    parser.add_argument(
        '--mode', nargs=None, default=3, type=int, metavar='INT',
        help='WebRTC VAD aggressiveness (default: %(default)s)')
    parser.add_argument(
        '--n_jobs', nargs=None, default=1, type=int, metavar='INT',
        help='number of parallel jobs (default: %(default)s)')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    if not utils.xor(args.wav_dir, args.scpf):
        parser.error('Exactly one of --wav_dir and -S must be set.')
        sys.exit(1)
    if not (args.wav_dir or args.output_dir):
        parser.error(
            'At least one of --wav_dir or --output_dir must be set.')
        sys.exit(1)
    if args.fs_vad not in VALID_VAD_SRS:
        parser.error(
            '--fs_vad must be one of %s' % VALID_VAD_SRS)
        sys.exit(1)
    if args.hoplength not in VALID_VAD_FRAME_LENGTHS:
        parser.error(
            '--hop_length must be one of %s' % VALID_VAD_FRAME_LENGTHS)
        sys.exit(1)
    if args.mode not in VALID_VAD_MODES:
        parser.add_argument('--mode must be one of %s' % VALID_VAD_MODES)
        sys.exit(1)
    args.frame_length = args.hoplength # Retain hoplength argument for compatibility.

    # Determine files to perform VAD on.
    if args.scpf is not None:
        wav_files = utils.load_script_file(args.scpf, '.wav')
    else:
        wav_files = utils.listdir(args.wav_dir, ext='.wav')

    # Determine output directory for VAD.
    if args.output_dir is None and args.wav_dir is not None:
        utils.warn('Output directory not specified. Defaulting to "%s"' %
                   args.wav_dir)
        args.output_dir = args.wav_dir

    # Perform VAD.
    def kwargs_gen():
        for wav_file in wav_files:
            bn = os.path.basename(wav_file)
            segs_file = os.path.join(args.output_dir, bn.replace('.wav', '.sad'))
            yield dict(wav_file=wav_file, segs_file=segs_file, fs_vad=args.fs_vad,
                       frame_length=args.frame_length, vad_mode=args.mode)
    f = delayed(perform_vad)
    res = Parallel(n_jobs=args.n_jobs)(f(**kwargs) for kwargs in kwargs_gen())
    for e, wav_file in zip(res, wav_files):
        if e is None:
            continue
        utils.error('Problem encountered while processing file "%s". Skipping.' % wav_file)


if __name__ == '__main__':
    main()
