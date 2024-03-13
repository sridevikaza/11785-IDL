import numpy as np
from nn.activation import *
import pdb


class RNNCell(object):
    """RNN Cell class."""

    def __init__(self, input_size, hidden_size):

        self.input_size = input_size
        self.hidden_size = hidden_size

        # Activation function for
        self.activation = Tanh()

        # hidden dimension and input dimension
        h = self.hidden_size
        d = self.input_size

        # Weights and biases
        self.W_ih = np.random.randn(h, d)
        self.W_hh = np.random.randn(h, h)
        self.b_ih = np.random.randn(h)
        self.b_hh = np.random.randn(h)

        # Gradients
        self.dW_ih = np.zeros((h, d))
        self.dW_hh = np.zeros((h, h))

        self.db_ih = np.zeros(h)
        self.db_hh = np.zeros(h)

        self.x = None

    def init_weights(self, W_ih, W_hh, b_ih, b_hh):
        self.W_ih = W_ih
        self.W_hh = W_hh
        self.b_ih = b_ih
        self.b_hh = b_hh

    def zero_grad(self):
        d = self.input_size
        h = self.hidden_size
        self.dW_ih = np.zeros((h, d))
        self.dW_hh = np.zeros((h, h))
        self.db_ih = np.zeros(h)
        self.db_hh = np.zeros(h)

    def __call__(self, x, h_prev_t):
        return self.forward(x, h_prev_t)

    def forward(self, x, h_prev_t):
        """
        RNN Cell forward (single time step).

        Input (see writeup for explanation)
        -----
        x: (batch_size, input_size)
            input at the current time step

        h_prev_t: (batch_size, hidden_size)
            hidden state at the previous time step and current layer

        Returns
        -------
        h_t: (batch_size, hidden_size)
            hidden state at the current time step and current layer
        """

        """
        ht = tanh(Wihxt + bih + Whhht1 + bhh) 
        """
        self.x = x
        z = (np.matmul(self.W_ih, np.transpose(x)) + np.expand_dims(self.b_ih, axis=1) + 
             np.matmul(self.W_hh, np.transpose(h_prev_t)) + np.expand_dims(self.b_hh, axis=1))
        h_t = self.activation.forward(np.transpose(z)) # H_out x N
        return h_t

    def backward(self, delta, h_t, h_prev_l, h_prev_t):
        """
        RNN Cell backward (single time step).

        Input (see writeup for explanation)
        -----
        delta: (batch_size, hidden_size)
                Gradient w.r.t the current hidden layer

        h_t: (batch_size, hidden_size)
            Hidden state of the current time step and the current layer

        h_prev_l: (batch_size, input_size)
                    Hidden state at the current time step and previous layer

        h_prev_t: (batch_size, hidden_size)
                    Hidden state at previous time step and current layer

        Returns
        -------
        dx: (batch_size, input_size)
            Derivative w.r.t.  the current time step and previous layer

        dh_prev_t: (batch_size, hidden_size)
            Derivative w.r.t.  the previous time step and current layer

        """
        batch_size = delta.shape[0]
        # 0) Done! Step backward through the tanh activation function.
        # Note, because of BPTT, we had to externally save the tanh state, and
        # have modified the tanh activation function to accept an optionally input.
        dz = self.activation.backward(h_prev_t, state=h_t) # TODO

        # 1) Compute the averaged gradients of the weights and biases
        pdb.set_trace()
        self.dW_ih += 1/batch_size * dz * np.matmul(np.transpose(delta), self.x)
        self.dW_hh += 1/batch_size * dz * np.matmul(np.transpose(delta), h_prev_l)
        self.db_ih += 1/batch_size * dz * delta
        self.db_hh += 1/batch_size * dz * delta

        # # 2) Compute dx, dh_prev_t
        dx        = dz * np.matmul(delta, self.W_ih)
        dh_prev_t = dz * np.matmul(delta, self.W_hh)

        # 3) Return dx, dh_prev_t
        return dx, dh_prev_t