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

import librosa

import utils


def main():
    """Main."""
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
        '--mode', nargs=None, default=3, type=int, metavar='INT',
        help='WebRTC VAD aggressiveness (default: %(default)s)')
    parser.add_argument(
        '--hoplength', nargs=None, default=30, type=int, metavar='INT',
        help='duration between frames in ms (default: %(default)s)')
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

    args.frame_length = args.hoplength # Retain hoplength argument for compatibility.
    args.fs_vad = 16000

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
    for wav in wav_files:
        if wav.endswith('.wav'):
            data, fs = librosa.load(wav, sr=16000)
            vad_info = utils.vad(
                data, fs, args.fs_vad, args.frame_length, args.mode)
            segments = utils.get_segments(vad_info, fs)
            bn = os.path.basename(wav)
            segsf = os.path.join(args.output_dir, bn.replace('.wav', '.sad'))
            utils.write_segments(segsf, segments)


if __name__ == '__main__':
    main()
