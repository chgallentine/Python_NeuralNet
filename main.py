# -*- coding: utf-8 -*-
# @Author: Charlie Gallentine
# @Date:   2019-12-07 11:46:22
# @Last Modified by:   Charlie Gallentine
# @Last Modified time: 2019-12-08 15:06:21

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

def ntrain(network, data, epochs, learning_rate, batch_size):
	train_set = data.training
	i = 0
	while i < epochs:
		i += 1
		total_err = 0.0

		for j,val in enumerate(train_set):
			network.set_input(np.array(val[0]))
			total_err += 0.5 * ((network.feed_foward() - val[1]) ** 2)
			network.backpropagate(val[1])
			network.update_weight(learning_rate)

		if (i % 100 == 0):
			print(": ".join((str(i),str(total_err))))

		if (total_err < 0.00001):
			print("Done at: ")
			print(i)
			break

network = NN.NeuralNet(architecture)

data = NN.Dataset("xor_data.txt", 2, 1.0)
ntrain(network, data, 10000, 0.75, 4)

for entry in tset:
	network.set_input(np.array(entry))
	network.feed_foward()

	print(entry)
	print(network.arch[-1].data)



# i = 0
# while i < 500:
# 	i += 1
# 	total_err = 0.0

# 	for j,val in enumerate(tset):
# 		network.set_input(np.array(val))
# 		total_err += 0.5 * ((network.feed_foward() - tkey[j]) ** 2)
# 		network.backpropagate(tkey[j], 0.5)

# 	if (total_err < 0.00001):
# 		print("Done at: ")
# 		print(i)
# 		break



# for entry in tset:
# 	network.set_input(np.array(entry))
# 	network.feed_foward()

# 	print(entry)
# 	print(network.arch[-1].data)

