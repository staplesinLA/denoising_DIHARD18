"""Functions for deriving ideal ratio masks for purposes of speech denoising.

References
----------
Sun, Lei, et al. "Speaker diarization with enhancing speech for the First DIHARD
Challenge." Proceedings of INTERSPEECH 2019. 2793-2797.
"""
from __future__ import print_function
from __future__ import unicode_literals
import os
import re
import sys
import warnings

warnings.filterwarnings(
    'ignore', message=r'[\s\S]+Missing optional dependency')
warnings.filterwarnings(
    'ignore', message='Unsupported Linux distribution')
warnings.filterwarnings(
    'ignore', message='HTKDeserializer')
warnings.filterwarnings(
    'ignore', message='Insufficiently recent colorama version found')

from cntk.io import MinibatchSource, HTKFeatureDeserializer, StreamDef, StreamDefs
from cntk import load_model, combine
from cntk.device import try_set_default_device, gpu, cpu
import numpy as np
import scipy.io as sio
import wurlitzer


HERE = os.path.abspath(os.path.dirname(__file__))
MODELF = os.path.join(HERE, "model", "speech_enhancement.model")
PY2 = sys.version_info[0] == 2


def decode_model(features_file, irm_mat_dir, feature_dim, use_gpu=True,
                 gpu_id=0):
    """Applies model to LPS features to generate ideal ratio mask.

    Parameters
    ----------
    features_file : str
        Path to HTK script file for chunks of LPS features to be processed.

    irm_mat_dir : str
        Path to output directory for ``.mat`` files containing ideal ratio
        masks.

    feature_dim : int
        Feature dimensionality. Needed to parse HTK binary file containing
        features.

    use_gpu : bool, optional
        If True and GPU is available, perform all processing on GPU.
        (Default: True)

    gpu_id : int, optional
         Id of GPU on which to do computation.
         (Default: 0)
    """
    if not os.path.exists(irm_mat_dir):
        os.makedirs(irm_mat_dir)

    # Load model.
    with wurlitzer.pipes() as (stdout, stderr):
        try_set_default_device(gpu(gpu_id) if use_gpu else cpu())
        model_dnn = load_model(MODELF)

    # Compute ideal ratio masks for all chunks of LPS features specified in
    # the script file and save as .mat files in irm_mat_dir.
    with wurlitzer.pipes() as (stdout, stderr):
        test_reader = MinibatchSource(
            HTKFeatureDeserializer(StreamDefs(
                amazing_features=StreamDef(shape=feature_dim, context=(3, 3),
                                           scp=features_file))),
            randomize=False, frame_mode=False, trace_level=0)
    eval_input_map = {input: test_reader.streams.amazing_features}
    with open(features_file, 'r') as f:
        for line in f:
            # Parse line of script file to get id for chunk and location of
            # corresponding LPS features. Each line has the format:
            #
            #     {CHUNK_ID}={PATH_TO_HTK_BIN}[{START_FRAME_INDEX},{END_FRAME_INDEX}]
            line = line.strip()
            chunk_id, htk_bin_path, start_ind, end_ind = re.match(
                r'(\S+)=(\S+)\[(\d+),(\d+)\]$', line).groups()
            start_ind = int(start_ind)
            end_ind = int(end_ind)
            mb_size = end_ind - start_ind + 1

            # Determine IRM features for frames in chunk.
            noisy_fea = test_reader.next_minibatch(
                mb_size, input_map=eval_input_map)
            real_noisy_fea = noisy_fea[input].data
            node_name = b'irm' if PY2 else 'irm'
            node_in_graph = model_dnn.find_by_name(node_name)
            output_nodes = combine([node_in_graph.owner])
            with wurlitzer.pipes() as (stdout, stderr):
                irm = output_nodes.eval(real_noisy_fea)
            irm = np.concatenate((irm), axis=0)

            # Write .mat file.
            sio.savemat(
                os.path.join(irm_mat_dir, chunk_id + '.mat'), {'IRM' : irm})
