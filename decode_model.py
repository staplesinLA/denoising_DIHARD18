# -*- coding:UTF-8 -*-
import cntk as C
import numpy as np
from cntk.io import MinibatchSource, HTKFeatureDeserializer, StreamDef, StreamDefs
from cntk import load_model, combine
from cntk.device import gpu,try_set_default_device,cpu
from cntk.ops import as_composite
import scipy.io as sio
import os
import sys

GPU_id = int(sys.argv[1])
try_set_default_device(gpu(GPU_id))
model_dnn= load_model("./model/speech_enhancment.model")
features_file = "./test_normed.scp" 
feature_dim = 257
test_reader = MinibatchSource(HTKFeatureDeserializer(StreamDefs(
    amazing_features = StreamDef(shape=feature_dim,context=(3,3), scp=features_file))),randomize = False,frame_mode=False)
eval_input_map = {input :test_reader.streams.amazing_features}

f = open(features_file)
line = f.readline() 
while line:
	temp_input_path = line.split(']')[0]
	mb_size = temp_input_path.split(',')[-1]
	mb_size = int(mb_size) + 1
	noisy_fea=test_reader.next_minibatch(mb_size, input_map = eval_input_map)
	real_noisy_fea=noisy_fea[input].data

	node_in_graph = model_dnn.find_by_name('irm')
	output_nodes = combine([node_in_graph.owner])
	out_noisy_fea = output_nodes.eval(real_noisy_fea)
	# out_noisy_fea = as_composite(model_dnn.output1[0].owner).eval(real_noisy_fea)

	out_SE_noisy_fea = np.concatenate((out_noisy_fea),axis=0)

	out_file_path = line.split('=')[0]
	out_file_name = os.path.join('enhanced_norm_fea_mat',out_file_path)
	out_file_fullpath = os.path.split(out_file_name)[0]
	# print (out_file_fullpath)
	if not os.path.exists(out_file_fullpath):
		os.makedirs(out_file_fullpath)
	sio.savemat(out_file_name, {'SE': out_SE_noisy_fea})
	line = f.readline()

f.close() 