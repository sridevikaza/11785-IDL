# DO NOT import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

from flatten import *
from Conv1d import *
from linear import *
from activation import *
from loss import *
import numpy as np
import os
import sys
import pdb

sys.path.append('mytorch')


class CNN_SimpleScanningMLP():
    def __init__(self):
        # Your code goes here -->
        # self.conv1 = ???
        # self.conv2 = ???
        # self.conv3 = ???
        # ...
        # <---------------------   
        self.conv1 = Conv1d(in_channels=24, out_channels=8, kernel_size=8, stride=4)
        self.conv2 = Conv1d(in_channels=8, out_channels=16, kernel_size=1, stride=1)
        self.conv3 = Conv1d(in_channels=16, out_channels=4, kernel_size=1, stride=1)
        self.layers = [self.conv1, ReLU(), self.conv2, ReLU(), self.conv3, Flatten()]

    def init_weights(self, weights):
        # Load the weights for your CNN from the MLP Weights given
        # w1, w2, w3 contain the weights for the three layers of the MLP
        # Load them appropriately into the CNN
        w1, w2, w3 = weights
        w1 = np.transpose(w1).reshape((self.conv1.conv1d_stride1.out_channels, self.conv1.conv1d_stride1.kernel_size, self.conv1.conv1d_stride1.in_channels))
        w2 = np.transpose(w2).reshape((self.conv2.conv1d_stride1.out_channels, self.conv2.conv1d_stride1.kernel_size, self.conv2.conv1d_stride1.in_channels))
        w3 = np.transpose(w3).reshape((self.conv3.conv1d_stride1.out_channels, self.conv3.conv1d_stride1.kernel_size, self.conv3.conv1d_stride1.in_channels))
        self.conv1.conv1d_stride1.W = np.transpose(w1, (0,2,1))
        self.conv2.conv1d_stride1.W = np.transpose(w2, (0,2,1))
        self.conv3.conv1d_stride1.W = np.transpose(w3, (0,2,1))

    def forward(self, A):
        """
        Do not modify this method

        Argument:
            A (np.array): (batch size, in channel, in width)
        Return:
            Z (np.array): (batch size, out channel , out width)
        """

        Z = A
        for layer in self.layers:
            Z = layer.forward(Z)
        return Z

    def backward(self, dLdZ):
        """
        Do not modify this method

        Argument:
            dLdZ (np.array): (batch size, out channel, out width)
        Return:
            dLdA (np.array): (batch size, in channel, in width)
        """

        for layer in self.layers[::-1]:
            dLdA = layer.backward(dLdA)
        return dLdA


class CNN_DistributedScanningMLP():
    def __init__(self):
        # Your code goes here -->
        # self.conv1 = ???
        # self.conv2 = ???
        # self.conv3 = ???
        # ...
        # <---------------------
        self.conv1 = Conv1d(in_channels=24, out_channels=2, kernel_size=2, stride=2)
        self.conv2 = Conv1d(in_channels=2, out_channels=8, kernel_size=2, stride=2)
        self.conv3 = Conv1d(in_channels=8, out_channels=4, kernel_size=2, stride=1)
        self.layers = [self.conv1, ReLU(), self.conv2, ReLU(), self.conv3, Flatten()]

    def __call__(self, A):
        # Do not modify this method
        return self.forward(A)

    def init_weights(self, weights):
        # Load the weights for your CNN from the MLP Weights given
        # w1, w2, w3 contain the weights for the three layers of the MLP
        # Load them appropriately into the CNN

        w1, w2, w3 = weights
        # pdb.set_trace()
        w1 = np.reshape(w1[:48,:2].T, (2,2,24)) #2,2,24
        w2 = np.reshape(w2[:4,:8].T, (8,2,2))   #8,2,2
        w3 = np.reshape(w3[:16,:4].T, (4,2,8))  #4,2,8
        # pdb.set_trace()
        w1 = np.transpose(w1, (0,2,1))  #2,24,2
        w2 = np.transpose(w2, (0,2,1))  #8,2,2
        w3 = np.transpose(w3, (0,2,1))  #4,8,2
        self.conv1.conv1d_stride1.W = w1
        self.conv2.conv1d_stride1.W = w2
        self.conv3.conv1d_stride1.W = w3

    def forward(self, A):
        """
        Do not modify this method

        Argument:
            A (np.array): (batch size, in channel, in width)
        Return:
            Z (np.array): (batch size, out channel , out width)
        """

        Z = A
        for layer in self.layers:
            Z = layer.forward(Z)
        return Z

    def backward(self, dLdZ):
        """
        Do not modify this method

        Argument:
            dLdZ (np.array): (batch size, out channel, out width)
        Return:
            dLdA (np.array): (batch size, in channel, in width)
        """
        dLdA = dLdZ
        for layer in self.layers[::-1]:
            dLdA = layer.backward(dLdA)
        return dLdA
