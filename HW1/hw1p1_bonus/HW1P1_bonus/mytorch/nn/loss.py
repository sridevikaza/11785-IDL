import numpy as np


class MSELoss:

    def forward(self, A, Y):
        """
        Calculate the Mean Squared error
        :param A: Output of the model of shape (N, C)
        :param Y: Ground-truth values of shape (N, C)
        :Return: MSE Loss(scalar)

        """

        self.A = A
        self.Y = Y
        self.N, self.C = A.shape  # TODO
        se = (A-Y) * (A-Y)  # TODO
        sse = np.transpose(np.ones((self.N,1))) @ se @ np.ones((self.C,1)) # TODO
        mse = sse / (self.N * self.C)  # TODO

        return mse

    def backward(self):

        dLdA = 2 * ((self.A-self.Y)/(self.N * self.C))

        return dLdA


class CrossEntropyLoss:

    def forward(self, A, Y):
        """
        Calculate the Cross Entropy Loss
        :param A: Output of the model of shape (N, C)
        :param Y: Ground-truth values of shape (N, C)
        :Return: CrossEntropyLoss(scalar)

        Refer the the writeup to determine the shapes of all the variables.
        Use dtype ='f' whenever initializing with np.zeros()
        """
        self.A = A
        self.Y = Y
        N, C = self.A.shape  # TODO

        Ones_C = np.ones((C,1))  # TODO
        Ones_N = np.ones((N,1))  # TODO

        self.softmax = np.exp(A) / np.expand_dims(np.sum(np.exp(A),axis=1),axis=1)
        crossentropy = (-self.Y * np.log(self.softmax)) @ Ones_C # TODO
        sum_crossentropy = np.transpose(Ones_N) @ crossentropy # TODO
        L = sum_crossentropy / N

        return L

    def backward(self):

        dLdA = (self.softmax - self.Y)/self.A.shape[0]  # TODO

        return dLdA
