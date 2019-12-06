# -*- coding: utf-8 -*-
# @Author: Charlie Gallentine
# @datae:   2019-11-28 09:45:11
# @Last Modified by:   Charlie Gallentine
# @Last Modified time: 2019-12-06 12:07:59

import numpy as np 
from matplotlib import pyplot as plt 

architecture = [
	{"num_nodes" : 2, "type" : "input", "activation" : None},
	{"num_nodes" : 2, "type" : "hidden", "activation" : "sigmoid"},
	{"num_nodes" : 1, "type" : "output", "activation" : "sigmoid"},
]

class Layer:
	def __init__(self, rows, cols, type, activation):
		self.type = type

		if (type == "weight"):
			self.activation = None
			self.data = np.random.rand(rows,cols)
			self.d_data = np.zeros((rows,cols))
		elif (type == "input"):
			self.activation = None
			self.data = np.zeros((1,cols))
			self.d_data = None
		elif (type == "output"):
			self.activation = activation
			self.data = np.zeros((1,cols))
			self.d_data = None
		elif (type == "hidden"):
			self.activation = activation
			self.data = np.zeros((1,cols))
			self.d_data = None	
		else:
			self.activation = None
			self.data = None
			self.d_data = None
			self.type = None

		np.atleast_2d(self.data)
		np.atleast_2d(self.d_data)


class NerualNet:
	arch = []

	def __init__(self, architecture):
		for i,layer in enumerate(architecture):
			if (i < len(architecture) - 1):
				self.arch.append( Layer(1, layer["num_nodes"], layer["type"], layer["activation"]) )
				self.arch.append( Layer(layer["num_nodes"], architecture[i+1]["num_nodes"], "weight", None) )

		self.arch.append( Layer(1, architecture[-1]["num_nodes"], architecture[-1]["type"], architecture[-1]["activation"]) )

	#  Set an array as the input to the network
	def set_input(self, arr):
		self.arch[0].data = arr

	def activate(self, x, type):
		if (type == "sigmoid"):
			return 1.0 / (1.0 + np.exp(-x))
		elif (type == "ReLU"):
			return np.max(0,x)
		elif (type == "tanh"):
			return 2.0 * (2.0 * (1.0/(1.0 + np.exp(-2.0 * x)))) - 1.0
		else:
			return x #no-op

	def d_activate(self, x, type):
		if (type == "sigmoid"):
			return self.activate(x,"sigmoid") * (1.0 - self.activate(x,"sigmoid"))
		elif (type == "ReLU"):
			return 0 if x < 0 else 1
		elif (type == "tanh"):
			return 1.0 - (self.activate(x,"tanh") ** 2)
		else:
			return x #no-op

	# Go through neural network performing the forward pass
	def feed_foward(self):
		for i, layer in enumerate(self.arch):
			if (i < len(self.arch) - 1):
				self.activate(layer.data, layer.type)

				if ( layer.type != "weight" ):
					self.arch[i+2].data = np.copy(np.matmul(layer.data, self.arch[i+1].data))
		return self.arch[-1].data


	def backpropagate(self, expected, learning_rate):
		# Output minus expected
		dzlplus1_dal = np.subtract(self.arch[-1].data, expected)
		# Derivative of activation for layer L
		dal_dzl = self.d_activate(self.arch[-1].data, self.arch[-1].activation)
		# Get the activation for layer L-1
		dzl_dwl = np.copy(np.transpose(self.arch[-3].data))
		# Track the first two terms I.E. dE_dzL
		passdown_mat = np.multiply(dzlplus1_dal, dal_dzl)

		tmp = np.matmul(dzl_dwl, passdown_mat)

		self.arch[-2].d_data = np.copy(learning_rate * tmp)

		for i, layer in reversed(list(enumerate(self.arch))):
			if (layer.type != "weight" and layer.type != "output" and i > 0):
				dzlplus1_dal = np.copy(np.transpose(self.arch[i+1].data))
				dal_dzl = self.d_activate(layer.data, layer.activation)
				dzl_dwl = np.copy(np.transpose(self.arch[i-2].data))
				pd_x_dzlplus1_dal = np.copy(np.matmul(passdown_mat, dzlplus1_dal))
				pd_elmwise_dal_dzl = np.copy(np.multiply(pd_x_dzlplus1_dal, dal_dzl))
				passdown_mat = np.copy(pd_elmwise_dal_dzl)
				tmp = np.matmul(dzl_dwl, pd_elmwise_dal_dzl)
				self.arch[i-1].d_data = np.copy(tmp);

		for layer in self.arch:
			if (layer.type == "weight"):
				layer.data = layer.data - learning_rate * layer.d_data


NN = NerualNet(architecture)

NN.set_input(np.array([[1,0]]))

train_set = [
	[[0,0]],
	[[0,1]],
	[[1,0]],
	[[0,0]],
]

train_key = [
	[[0]],
	[[1]],
	[[1]],
	[[0]],
]

i = 0
while i < 5000000000:
	i += 1
	total_err = 0.0

	for j,val in enumerate(train_set):
		NN.set_input(np.array(val))
		total_err += 0.5 * ((NN.feed_foward() - train_key[j]) ** 2)
		print("ERROR")
		print(total_err)
		print("\n")
		NN.backpropagate(train_key[j], 0.5)

	if (total_err < 0.00001):
		break

print("Done at: ")
print(i)

for entry in train_set:
	NN.set_input(np.array(entry))
	NN.feed_foward()

	print(entry)
	print(NN.arch[-1].data)





















