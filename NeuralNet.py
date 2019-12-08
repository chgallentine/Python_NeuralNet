# -*- coding: utf-8 -*-
# @Author: Charlie Gallentine
# @datae:   2019-11-28 09:45:11
# @Last Modified by:   Charlie Gallentine
# @Last Modified time: 2019-12-08 15:05:15

import numpy as np 
import random

class Layer:
	def __init__(self, rows, cols, type, activation):
		self.type = type

		if (type == "weight"):
			self.activation = None
			self.data = np.random.rand(rows,cols)
			self.d_data = np.zeros((rows,cols))
			self.bias = np.random.rand(1,cols)
			self.d_bias = np.zeros((1,cols))
		elif (type == "input"):
			self.activation = None
			self.data = np.zeros((1,cols))
			self.d_data = None
			self.bias = np.ones((1,1))
			self.d_bias = None
		elif (type == "output"):
			self.activation = activation
			self.data = np.zeros((1,cols))
			self.d_data = None
			self.bias = None
			self.d_bias = None
		elif (type == "hidden"):
			self.activation = activation
			self.data = np.zeros((1,cols))
			self.d_data = None	
			self.bias = np.ones((1,1))
			self.d_bias = None
		else:
			self.activation = "other"
			self.data = None
			self.d_data = None
			self.type = None
			self.bias = None

		np.atleast_2d(self.data)
		np.atleast_2d(self.d_data)
		np.atleast_2d(self.bias)
		np.atleast_2d(self.d_bias)


class NeuralNet:
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
			return np.divide(np.exp(x), (1.0 + np.exp(x)))
		elif (type == "ReLU"):
			return np.max(0,x)
		elif (type == "tanh"):
			return 2.0 * (2.0 * (1.0/(1.0 + np.exp(-2.0 * x)))) - 1.0
		else:
			return x #no-op

	def d_activate(self, x, type):
		if (type == "sigmoid"):
			return np.multiply(x, 1.0 - x)
		elif (type == "ReLU"):
			return 0 if x < 0 else 1
		elif (type == "tanh"):
			return 1.0 - (x ** 2)
		else:
			return x #no-op

	def cost(self, expected):
		return np.sum(0.5 * ((self.arch[-1].data - expected) ** 2))

	def d_cost(self, expected):
		return np.subtract(self.arch[-1].data, expected)

	# Go through neural network performing the forward pass
	def feed_foward(self):
		for i, layer in enumerate(self.arch):
			if (i < len(self.arch) - 1):
				if ( layer.type != "weight" ):
					self.arch[i+2].data = np.copy(
						self.activate(
							np.matmul(
								self.activate(layer.data, layer.activation), 
								self.arch[i+1].data) + 
							np.matmul(
								layer.bias, 
								self.arch[i+1].bias), 
							self.arch[i+2].activation
						)
					)

		return self.arch[-1].data


	def backpropagate(self, expected):
		# Output minus expected
		dzlplus1_dal = np.subtract(self.arch[-1].data, expected)
		# Derivative of activation for layer L
		dal_dzl = self.d_activate(self.arch[-1].data, self.arch[-1].activation)
		# Get the activation for layer L-1
		dzl_dwl = np.copy(np.transpose(self.arch[-3].data))
		# Track the first two terms I.E. dE_dzL
		passdown_mat = np.multiply(dzlplus1_dal, dal_dzl)

		tmp = np.matmul(dzl_dwl, passdown_mat)
		self.arch[-2].d_data = self.arch[-2].d_data + tmp
		self.arch[-2].d_bias = self.arch[-2].d_bias + passdown_mat

		for i, layer in reversed(list(enumerate(self.arch))):
			if (layer.type != "weight" and layer.type != "output" and i > 0):
				dzlplus1_dal = np.copy(np.transpose(self.arch[i+1].data))
				dal_dzl = self.d_activate(layer.data, layer.activation)
				dzl_dwl = np.copy(np.transpose(self.arch[i-2].data))
				pd_x_dzlplus1_dal = np.copy(np.matmul(passdown_mat, dzlplus1_dal))
				pd_elmwise_dal_dzl = np.copy(np.multiply(pd_x_dzlplus1_dal, dal_dzl))
				passdown_mat = np.copy(pd_elmwise_dal_dzl)
				tmp = np.matmul(dzl_dwl, pd_elmwise_dal_dzl)

				self.arch[i-1].d_data = self.arch[i-1].d_data + tmp
				self.arch[i-1].d_bias = self.arch[i-1].d_bias + passdown_mat

	def update_weight(self, learning_rate):
		for layer in self.arch:
			if (layer.type == "weight"):
				layer.data = layer.data - learning_rate * layer.d_data
				layer.bias = layer.bias - learning_rate * layer.d_bias
				layer.d_data = np.zeros(layer.d_data.shape)
				layer.d_bias = np.zeros(layer.d_bias.shape)



class Dataset:
	training = []
	validation = []

	def __init__(self, fname, inlen, trainpercent):
		datafile = open(fname, "r")

		for dp in datafile:
			tmp = []
			tmp.append([list(np.fromstring(dp, dtype=float, sep=' ')[:inlen])])
			tmp.append([list(np.fromstring(dp, dtype=float, sep=' ')[inlen:])])

			if (random.uniform(0, 1) <= trainpercent):
				self.training.append(tmp)
			else:
				self.validation.append(tmp)

















