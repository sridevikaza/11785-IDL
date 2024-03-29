import numpy as np
from resampling import *


class Conv2d_stride1():
    def __init__(self, in_channels, out_channels,
                 kernel_size, weight_init_fn=None, bias_init_fn=None):

        # Do not modify this method

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(
                0, 1.0, (out_channels, in_channels, kernel_size, kernel_size))
        else:
            self.W = weight_init_fn(
                out_channels,
                in_channels,
                kernel_size,
                kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

        self.input_height = None
        self.input_width = None

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
        """
        self.A = A
    
        batch_size, _, self.input_height, self.input_width = A.shape
        output_height = self.input_height - self.kernel_size + 1
        output_width = self.input_width - self.kernel_size + 1

        Z = np.zeros((batch_size, self.out_channels, output_height, output_width))

        for i in range(output_height):
            for j in range(output_width):
                section = A[:,:,i:i+self.kernel_size,j:j+self.kernel_size]
                Z[:, :, i, j] = np.tensordot(section, self.W, axes=([1,2,3], [1,2,3]))
                Z[:, :, i, j] += self.b

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """

        # dLdW
        batch_size = dLdZ.shape[0]
        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                section = self.A[:,:,i:i+dLdZ.shape[2],j:j+dLdZ.shape[3]]
                result = np.tensordot(section, dLdZ, axes=([0,2,3], [0,2,3])).T
                self.dLdW[:,:,i,j] = result

        # dLdb
        self.dLdb = np.sum(dLdZ, axis=(0,2,3))

        # dLdA
        dLdA = np.zeros((batch_size, self.in_channels, self.input_height, self.input_width))
        dLdZ_pad = np.pad(dLdZ, pad_width=((0, 0), (0, 0), (self.kernel_size-1, self.kernel_size-1), (self.kernel_size-1, self.kernel_size-1)), mode='constant', constant_values=(0, 0))
        W_flipped = np.flip(self.W, axis=(2,3))

        for i in range(self.input_height):
            for j in range(self.input_width):
                section = dLdZ_pad[:,:,i:i+self.kernel_size, j:j+self.kernel_size]
                result = np.tensordot(section, W_flipped, axes=([1,2,3], [0,2,3]))
                dLdA[:,:,i,j] = result

        return dLdA


class Conv2d():
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names
        self.stride = stride
        self.pad = padding

        # Initialize Conv2d() and Downsample2d() isntance
        self.conv2d_stride1 = Conv2d_stride1(in_channels, out_channels, kernel_size, weight_init_fn, bias_init_fn)
        self.downsample2d = Downsample2d(stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
        """

        # Pad the input appropriately using np.pad() function
        A = np.pad(A, pad_width=((0, 0), (0, 0), (self.pad, self.pad), (self.pad, self.pad)), mode='constant', constant_values=(0, 0))

        # Call Conv2d_stride1
        Z = self.conv2d_stride1.forward(A)

        # downsample
        Z = self.downsample2d.forward(Z)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """

        # Call downsample1d backward
        dLdZ = self.downsample2d.backward(dLdZ)

        # Call Conv2d_stride1 backward
        dLdA = self.conv2d_stride1.backward(dLdZ)

        # Unpad the gradient
        dLdA = dLdA[:,:,self.pad:-self.pad,self.pad:-self.pad]

        return dLdA
