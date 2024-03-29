import sys
sys.path.append('mytorch')

from Conv2d import *
from activation import *
from batchnorm2d import *

import numpy as np
import os


class ConvBlock(object):
	def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
		#TODO	
		self.layers = []


	def forward(self, A):
		#TODO
		return NotImplemented

	def backward(self, grad): 
		#TODO
		return NotImplemented


class ResBlock(object):
	def __init__(self, in_channels, out_channels, filter_size, stride=3, padding=1):
		self.convolution_layers =  [ ] #TODO Initialize all layers in this list.				
		self.final_activation =	None				#TODO 

		if stride != 1 or in_channels != out_channels or filter_size!=1 or padding!=0:
			self.residual_connection = None 		#TODO
		else:
			self.residual_connection = None			#TODO 


	def forward(self, A):
		Z = A
		'''
		Implement the forward for convolution layer.

		'''
		#TODO 
			

		'''
		Add the residual connection to the output of the convolution layers

		'''
		#TODO 
		

		'''
		Pass the the sum of the residual layer and convolution layer to the final activation function
		'''
		#TODO 

		return NotImplemented
	

	def backward(self, grad):

		'''
		Implement the backward of the final activation
		'''
		#TODO 


		'''
		Implement the backward of residual layer to get "residual_grad"
		'''
		#TODO 


		'''
		Implement the backward of the convolution layer to get "convlayers_grad"
		'''
		#TODO 


		'''
		Add convlayers_grad and residual_grad to get the final gradient 
		'''
		#TODO 



		return NotImplementedError
