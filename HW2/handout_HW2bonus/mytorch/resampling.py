import numpy as np


class Upsample1d():

    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor
        self.input_width = None

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """

        batch_size, in_channels, self.input_width = A.shape
        output_width = self.upsampling_factor * (self.input_width-1) + 1
        Z = np.zeros((batch_size, in_channels, output_width))
        Z[:,:,::self.upsampling_factor] = A

        return Z


    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """

        batch_size, in_channels, _ = dLdZ.shape
        dLdA = np.zeros((batch_size, in_channels, self.input_width))

        for i in range(self.input_width):
            dLdA[:,:,i] = dLdZ[:,:,i*self.upsampling_factor]

        return dLdA


class Downsample1d():

    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor
        self.input_width = None

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """
        self.input_width = A.shape[2]
        Z = A[:,:,::self.downsampling_factor]

        return Z


    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """

        batch_size, in_channels, output_width = dLdZ.shape
        dLdA = np.zeros((batch_size, in_channels, self.input_width))

        for i in range(output_width):
            dLdA[:,:,i*self.downsampling_factor] = dLdZ[:,:,i]

        return dLdA


class Upsample2d():

    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor
        self.input_height = None
        self.input_width = None

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_height, output_width)
        """
        batch_size, in_channels, self.input_height, self.input_width = A.shape
        output_height = (self.upsampling_factor-1) * (self.input_height-1) + self.input_height
        output_width = (self.upsampling_factor-1) * (self.input_width-1) + self.input_width
        Z = np.zeros((batch_size, in_channels, output_height, output_width))
        Z[:,:,::self.upsampling_factor,::self.upsampling_factor] = A

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """

        batch_size, in_channels = dLdZ.shape[0], dLdZ.shape[1]
        dLdA = np.zeros((batch_size, in_channels, self.input_height, self.input_width))

        for i in range(self.input_height):
            for j in range(self.input_width):
                dLdA[:,:,i,j] = dLdZ[:,:,i*self.upsampling_factor,j*self.upsampling_factor]

        return dLdA


class Downsample2d():

    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor
        self.input_height = None
        self.input_width = None

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_height, output_width)
        """

        self.input_height, self.input_width = A.shape[2], A.shape[3]
        Z = A[:,:,::self.downsampling_factor,::self.downsampling_factor]

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """

        batch_size, in_channels, output_height, output_width = dLdZ.shape
        dLdA = np.zeros((batch_size, in_channels, self.input_height, self.input_width))

        for i in range(output_height):
            for j in range(output_width):
                dLdA[:,:,i*self.downsampling_factor, j*self.downsampling_factor] = dLdZ[:,:,i,j]

        return dLdA
