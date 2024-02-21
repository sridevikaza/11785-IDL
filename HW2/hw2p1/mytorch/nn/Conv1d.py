# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
from resampling import *
import pdb


class Conv1d_stride1():
    def __init__(self, in_channels, out_channels, kernel_size,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(
                0, 1.0, (out_channels, in_channels, kernel_size))
        else:
            self.W = weight_init_fn(out_channels, in_channels, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

        self.input_size = None # added this

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        self.A = A
        batch_size, _, self.input_size = A.shape
        output_size = self.input_size - self.kernel_size + 1

        Z = np.zeros((batch_size, self.out_channels, output_size))

        for i in range(output_size):
            section = A[:,:,i:i+self.kernel_size]
            # pdb.set_trace()
            Z[:, :, i] = np.tensordot(section, self.W, axes=([1, 2], [1, 2]))
            Z[:, :, i] += self.b

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """

        # dLdW
        batch_size = dLdZ.shape[0]
        for i in range(self.kernel_size):
            section = self.A[:,:,i:i+dLdZ.shape[2]]
            result = np.tensordot(section, dLdZ, axes=([0,2], [0,2])).T
            self.dLdW[:,:,i] = result

        # dLdb
        self.dLdb = np.sum(dLdZ, axis=(0,2))

        # dLdA
        dLdA = np.zeros((batch_size, self.in_channels, self.input_size))
        dLdZ_pad = np.pad(dLdZ, pad_width=((0, 0), (0, 0), (self.kernel_size-1, self.kernel_size-1)), mode='constant', constant_values=(0, 0))
        W_flipped = np.flip(self.W, axis=2)

        for i in range(self.input_size):
            section = dLdZ_pad[:,:,i:i+self.kernel_size]
            result = np.tensordot(section, W_flipped, axes=([1,2], [0,2]))
            dLdA[:,:,i] = result

        return dLdA


class Conv1d():
    def __init__(self, in_channels, out_channels, kernel_size, stride,padding = 0,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names

        self.stride = stride
        self.pad = padding

        # Initialize Conv1d() and Downsample1d() isntance
        self.conv1d_stride1 = Conv1d_stride1(in_channels, out_channels, kernel_size, weight_init_fn, bias_init_fn)
        self.downsample1d = Downsample1d(stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """

        # Pad the input appropriately using np.pad() function
        A = np.pad(A, pad_width=((0, 0), (0, 0), (self.pad, self.pad)), mode='constant', constant_values=(0, 0))

        # Call Conv1d_stride
        Z = self.conv1d_stride1.forward(A)

        # downsample
        Z = self.downsample1d.forward(Z)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        # Call downsample1d backward
        dLdZ = self.downsample1d.backward(dLdZ)

        # Call Conv1d_stride1 backward
        dLdA = self.conv1d_stride1.backward(dLdZ)

        # Unpad the gradient
        dLdA = dLdA[:,:,self.pad:dLdA.shape[-1]-self.pad]

        return dLdA