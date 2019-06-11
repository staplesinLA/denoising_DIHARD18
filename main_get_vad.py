#!/usr/bin/env python
"""Perform voice activity detection (VAD) using WebRTC's implementation.

To perform VAD for all WAV files under the directory ``wav_dir/`` and write
the output to the directory ``vad_dir/`` as HTK label files:

    python main_get_vad.py --wav_dir wav_dir/ --output_dir vad_dir/

For each file with the ``.wav`` extension under ``wav_dir/``, there will now be
a corresponding label file with the extension ``.sad`` under ``vad_dir/``. Each
label file will contain one speech segment per line, each consisting of three
space-delimited fields:

- onset  --  the onset of the segment in seconds
- offset --  the offset of the segment in seconds
- label  --  the label for the segment; controlled by the ``--speech_label`` flag

If ``--output_dir`` is not specified, these files will be output to ``wav_dir/``.

Alternately, you may specify the files to process via a script file of paths to
WAV files with one path per line:

    /path/to/file1.wav
    /path/to/file2.wav
    /path/to/file3.wav
    ...

This functionality is enabled via the ``-S`` flag, as in the following:

   python main_get_vad.py -S some.scp --output_dir vad_dir/

which will perform VAD for those file listed in ``some.scp`` and output label files
to ``vad_dir. Note that if you use a script file, you *MUST* specify an output
directory.

WebRTC exposes several parameters for tuning it's output, which may be adjusted via
the following flags:

- ``--fs_vad``  --  controls the sample rate the audio is resampled to prior to
  performing VAD; possible values are 8 kHz, 16 kHz, 32 kHz, and 48 kHz
- ``--hoplength``  --  the duration in milliseconds of the frames for VAD; possible
  values are 10 ms, 20 ms, and 30 ms
- ``--mode``  --   the WebRTC aggressiveness mode, which controls how aggressive
  WebRTC is about filter out non-speech; 0 is least aggressive and 3 most aggressive

Optionally, label smoothing may be applied to the output of WebRTC to eliminate short,
irregular silences and speech segments. Label smoothing is done using a median filter
applied to the frame-level labeling produced by WebRTC and is controlled by the 
``--med_filt_width`` parameter.

When processing large batches of audio, it may be desireable to parallelize the
computation, which may be done by specifying the number of parallel processes to
employ via the ``--n_jobs`` flag:

   python main_get_vad.py --n_jobs 40 -S some.scp --output_dir vad_dir/

References
----------
- https://github.com/wiseman/py-webrtcvad.git
- https://webrtc.org/
"""
from __future__ import print_function
from __future__ import unicode_literals
import argparse
import numbers
import os
import sys
import traceback

from joblib import delayed, Parallel
import librosa

import utils
from utils import VALID_VAD_SRS, VALID_VAD_FRAME_LENGTHS, VALID_VAD_MODES


def perform_vad(wav_file, segs_file, speech_label, **kwargs):
    """Perform VAD for WAV file.

    If an exception is raised during processing, it returns the exception as well as
    the full traceback. Otherwise, returns ``None``.

    Parameters
    ----------
    wav_file : str
        Path to WAV file to perform VAD for.

    segs_file : str
        Path to output segments file.

    speech_label : str
        Label for speech segments.

    kwargs
        Keyword arguments to pass to ``utils.vad``.
    """
    try:
        data, fs = librosa.load(wav_file, sr=None)
        vad_info = utils.vad(data, fs, **kwargs)
        segments = utils.get_segments(vad_info, fs)
        utils.write_segments(segs_file, segments, label=speech_label)
        return None
    except Exception as e:
        tb = traceback.format_exc()
        return e, tb


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
        '-S', dest='scpf', nargs=None, type=str, metavar='STR',
        help='script file of paths to WAV files to perform VAD for (default: %(default)s)')
    parser.add_argument(
        '--output_dir', nargs=None, type=str, metavar='STR',
        help='output directory for label files (default: None)')
    parser.add_argument(
        '--output_ext', nargs=None, default='.sad', type=str, metavar='STR',
        help='extension for output label files (default: %(default)s)')
    parser.add_argument(
        '--speech_label', nargs=None, default='', type=str, metavar='STR',
        help='label for speech segments (default: %(default)s)')
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
        '--med_filt_width', nargs=None, default=1, type=int, metavar='INT',
        help='window size in frames for median smoothing of VAD output; '
             '<=1 disables (default: %(default)s')
    parser.add_argument(
        '--verbose', default=False, action='store_true',
        help='print full stacktrace for files with errors')
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
        parser.error('--mode must be one of %s' % VALID_VAD_MODES)
        sys.exit(1)
    if (not isinstance(args.med_filt_width, numbers.Integral) or
        args.med_filt_width % 2 == 0):
        parser.error('--med_filt_width must be an odd integer')
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
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    # Perform VAD.
    def kwargs_gen():
        for wav_file in wav_files:
            bn = os.path.basename(wav_file)
            segs_file = os.path.join(
                args.output_dir, bn.replace('.wav', args.output_ext))
            yield dict(
                wav_file=wav_file, segs_file=segs_file,
                speech_label=args.speech_label, fs_vad=args.fs_vad,
                frame_length=args.frame_length, vad_mode=args.mode,
                med_filt_width=args.med_filt_width)
    f = delayed(perform_vad)
    res = Parallel(n_jobs=args.n_jobs)(f(**kwargs) for kwargs in kwargs_gen())
    for res_, wav_file in zip(res, wav_files):
        if res_ is None:
            continue
        e, tb = res_
        msg = 'Problem encountered while processing file "%s". Skipping.' % wav_file
        if args.verbose:
            msg = '%s Full error output:\n%s' % (msg, tb)
        utils.error(msg)


if __name__ == '__main__':
    main()
