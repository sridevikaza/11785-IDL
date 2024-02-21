import numpy as np
from resampling import *
# import pdb

class MaxPool2d_stride1():

    def __init__(self, kernel):
        self.kernel = kernel
        self.input_height = None
        self.input_width = None
        self.max_idx_h = None
        self.max_idx_w = None

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        self.A = A
        batch_size, in_channels, self.input_height, self.input_width = A.shape
        output_height = self.input_height - self.kernel + 1
        output_width = self.input_width - self.kernel + 1
        Z = np.zeros((batch_size, in_channels, output_height, output_width))
        self.max_idx_h = np.zeros((batch_size, in_channels, output_height, output_width), dtype=int)
        self.max_idx_w = np.zeros((batch_size, in_channels, output_height, output_width), dtype=int)

        for i in range(output_height):
            for j in range(output_width):

                # get max value
                section = A[:,:,i:i+self.kernel,j:j+self.kernel]
                Z[:, :, i, j] = np.max(section, axis=(2,3))

                # save indices of max value in A for backward pass
                flat_idx = np.argmax(section.reshape(batch_size, in_channels, -1), axis=2)
                multi_dim_idx = np.unravel_index(flat_idx, (self.kernel, self.kernel))
                # pdb.set_trace()
                self.max_idx_h[:,:,i,j] = multi_dim_idx[0] + i
                self.max_idx_w[:,:,i,j] = multi_dim_idx[1] + j

        return Z


    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        dLdA = np.zeros(self.A.shape)
        batch_size, in_channels, output_height, output_width = dLdZ.shape

        for i in range(batch_size):
            for j in range(in_channels):
                for x in range(output_height):
                    for y in range(output_width):
                        h_idx = self.max_idx_h[i,j,x,y]
                        w_idx = self.max_idx_w[i,j,x,y]
                        # pdb.set_trace()
                        dLdA[i,j,h_idx,w_idx] += dLdZ[i,j,x,y]

        return dLdA


class MeanPool2d_stride1():

    def __init__(self, kernel):
        self.kernel = kernel
        self.input_height = None
        self.input_width = None

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        self.A = A
        batch_size, in_channels, self.input_height, self.input_width = A.shape
        output_height = self.input_height - self.kernel + 1
        output_width = self.input_width - self.kernel + 1
        Z = np.zeros((batch_size, in_channels, output_height, output_width))

        for i in range(output_height):
            for j in range(output_width):
            
                section = A[:,:,i:i+self.kernel,j:j+self.kernel]
                Z[:, :, i, j] = np.mean(section, axis=(2,3))

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        batch_size, in_channels, output_width, output_height = dLdZ.shape
        dLdA = np.zeros((batch_size, in_channels, self.input_width, self.input_height))
        
        for i in range(batch_size):
            for j in range(in_channels):
                for x in range(output_height):
                    for y in range(output_width):
                        
                        grad = dLdZ[i, j, x, y] / (self.kernel**2)
                        dLdA[i, j, x:x+self.kernel, y:y+self.kernel] += grad

        return dLdA

class MaxPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        # Create an instance of MaxPool2d_stride1
        self.maxpool2d_stride1 = MaxPool2d_stride1(kernel)
        self.downsample2d = Downsample2d(stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        # Call maxpool2d_stride1 forward
        Z = self.maxpool2d_stride1.forward(A)

        # Call downsample2d forward
        Z = self.downsample2d.forward(Z)

        return Z
    

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        # Call downsample2d backward
        dLdZ = self.downsample2d.backward(dLdZ)

        # Call maxpool2d_stride1 backward
        dLdA = self.maxpool2d_stride1.backward(dLdZ)

        return dLdA


class MeanPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        # Create an instance of MaxPool2d_stride1
        self.meanpool2d_stride1 = MeanPool2d_stride1(kernel)
        self.downsample2d = Downsample2d(stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        # Call meanpool2d_stride1 forward
        Z = self.meanpool2d_stride1.forward(A)

        # Call downsample2d forward
        Z = self.downsample2d.forward(Z)

        return Z
    
    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        # Call downsample2d backward
        dLdZ = self.downsample2d.backward(dLdZ)

        # Call meanpool2d_stride1 backward
        dLdA = self.meanpool2d_stride1.backward(dLdZ)

        return dLdA
