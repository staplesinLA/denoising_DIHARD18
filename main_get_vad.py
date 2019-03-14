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

from librosa import load

import utils


def main_vad(wav_dir, mode, hoplength):
    """TODO"""
    if not os.path.exists(wav_dir):
        raise RuntimeError('cannot locate the original dictionary !')

    wav_files = [os.path.join(wav_dir, line)  for line in os.listdir(wav_dir)]
    wav_files = sorted(wav_files)
    for wav in wav_files:
        if wav.endswith('.wav'):
            data, fs = load(wav, sr=16000)
            vad_info = utils.vad(
                data, fs, fs_vad=16000, hoplength=hoplength, vad_mode=mode)
            segments = utils.get_segments(vad_info, fs)
            with open(wav.replace('.wav', '.sad'), 'w') as f:
                for i in range(segments.shape[0]):
                    start_time = segments[i][0]
                    end_time = segments[i][1]
                    f.write('%.3f  %.3f \n' % (start_time, end_time))


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
    main_vad(wav_dir=args.wav_dir, mode=args.mode, hoplength=args.hoplength)


if __name__ == '__main__':
    main()
