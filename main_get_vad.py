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

from utils import get_segments, vad, write_segments



def main():
    """Main."""
    parser = argparse.ArgumentParser(
        description='Perform VAD using webrtcvad.', add_help=True)
    parser.add_argument(
        '--wav_dir', nargs=None, default=None, type=str, metavar='STR',
        help='directory containing WAV files to perform VAD for '
             '(default: %(default)s)')
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
    args.frame_length = args.hoplength # Retain hoplength argument for compatibility.
    args.fs_vad = 16000

    if not os.path.exists(args.wav_dir):
        raise RuntimeError('cannot locate the original dictionary !')

    wav_files = [os.path.join(args.wav_dir, line)  for line in os.listdir(args.wav_dir)]
    wav_files = sorted(wav_files)
    for wav in wav_files:
        if wav.endswith('.wav'):
            data, fs = librosa.load(wav, sr=16000)
            vad_info = vad(
                data, fs, args.fs_vad, args.frame_length, args.mode)
            segments = get_segments(vad_info, fs)
            segsf = wav.replace('.wav', '.sad')
            write_segments(segsf, segments)


if __name__ == '__main__':
    main()
