import numpy as np
from scipy.special import erf

class Identity:

    def forward(self, Z):

        self.A = Z

        return self.A

    def backward(self, dLdA):

        dAdZ = np.ones(self.A.shape, dtype="f")
        dLdZ = dLdA * dAdZ

        return dLdZ


class Sigmoid:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on Sigmoid.
    """
    def forward(self, Z):

        self.A = 1 / (1+np.exp(-Z))

        return self.A

    def backward(self, dLdA):

        dAdZ = (self.A - self.A * self.A)
        dLdZ = dLdA * dAdZ

        return dLdZ


class Tanh:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on Tanh.
    """
    def forward(self, Z):

        # self.A = (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))

        self.A = np.tanh(Z)

        return self.A

    def backward(self, dLdA):

        dAdZ = 1 - self.A**2
        dLdZ = dLdA * dAdZ

        return dLdZ


class ReLU:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on ReLU.
    """
    def forward(self, Z):

        self.A = np.maximum(0, Z)

        return self.A

    def backward(self, dLdA):

        dAdZ = np.where(self.A > 0, 1, 0)
        dLdZ = dLdA * dAdZ

        return dLdZ

class GELU:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on GELU.
    """
    def forward(self, Z):
        
        self.Z = Z
        self.A = Z/2 * (1+erf(Z/np.sqrt(2)))

        return self.A

    def backward(self, dLdA):

        dAdZ = 1/2*(1+erf(self.Z/np.sqrt(2))) + self.Z/np.sqrt(2*np.pi) * np.exp(-self.Z**2/2)
        dLdZ = dLdA * dAdZ

        return dLdZ

class Softmax:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on Softmax.
    """

    def forward(self, Z):
        """
        Remember that Softmax does not act element-wise.
        It will use an entire row of Z to compute an output element.
        """
        # N,C = Z.shape
        # self.A = np.zeros((N,C))
        # for i in range(N):
        #     self.A[i,:] = np.exp(Z[i,:]) / np.sum(np.exp(Z[i,:])) 
        
        self.A = np.exp(Z) / np.expand_dims(np.sum(np.exp(Z),axis=1),axis=1) # TODO

        return self.A
    
    def backward(self, dLdA):

        # Calculate the batch size and number of features
        N,C = self.A.shape # TODO

        # Initialize the final output dLdZ with all zeros. Refer to the writeup and think about the shape.
        dLdZ = np.zeros((N,C)) # TODO

        # Fill dLdZ one data point (row) at a time
        for i in range(N):

            # Initialize the Jacobian with all zeros.
            J = np.zeros((C,C)) # TODO

            # Fill the Jacobian matrix according to the conditions described in the writeup
            for m in range(C):
                for n in range(C):
                    a_m = self.A[i,m]
                    a_n = self.A[i,n]
                    if m==n:
                        J[m,n] = a_m * (1-a_m)
                    else:
                        J[m,n] = -a_m * a_n # TODO

            # Calculate the derivative of the loss with respect to the i-th input
            dLdZ[i,:] = np.matmul(dLdA[i,:],J) # TODO


        return dLdZ