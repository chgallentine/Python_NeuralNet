# -*- coding: utf-8 -*-
# @Author: Charlie Gallentine
# @Date:   2019-12-07 11:46:22
# @Last Modified by:   Charlie Gallentine
# @Last Modified time: 2019-12-08 15:21:06

import NeuralNet as NN 
import numpy as np 
from matplotlib import pyplot as plt 


architecture = [
	{"num_nodes" : 2, "type" : "input", "activation" : None},
	{"num_nodes" : 2, "type" : "hidden", "activation" : "sigmoid"},
	{"num_nodes" : 1, "type" : "output", "activation" : "sigmoid"},
]

tset = [
	[[0.0,0.0]],
	[[0.0,1.0]],
	[[1.0,0.0]],
	[[1.0,1.0]],
]

tkey = [
	[[0.0]],
	[[1.0]],
	[[1.0]],
	[[0.0]],
]


network = NN.NeuralNet(architecture)

data = NN.Dataset("xor_data.txt", 2, 1.0)
network.train(data, 10000, 0.75, 3)

for entry in tset:
	network.set_input(np.array(entry))
	network.feed_foward()

	print(entry)
	print(network.arch[-1].data)

