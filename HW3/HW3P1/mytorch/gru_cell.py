import numpy as np
from nn.activation import *

class GRUCell(object):
    """GRU Cell class."""

    def __init__(self, input_size, hidden_size):
        self.d = input_size
        self.h = hidden_size
        h = self.h
        d = self.d
        self.x_t = 0

        self.Wrx = np.random.randn(h, d)
        self.Wzx = np.random.randn(h, d)
        self.Wnx = np.random.randn(h, d)

        self.Wrh = np.random.randn(h, h)
        self.Wzh = np.random.randn(h, h)
        self.Wnh = np.random.randn(h, h)

        self.brx = np.random.randn(h)
        self.bzx = np.random.randn(h)
        self.bnx = np.random.randn(h)

        self.brh = np.random.randn(h)
        self.bzh = np.random.randn(h)
        self.bnh = np.random.randn(h)

        self.dWrx = np.zeros((h, d))
        self.dWzx = np.zeros((h, d))
        self.dWnx = np.zeros((h, d))

        self.dWrh = np.zeros((h, h))
        self.dWzh = np.zeros((h, h))
        self.dWnh = np.zeros((h, h))

        self.dbrx = np.zeros((h))
        self.dbzx = np.zeros((h))
        self.dbnx = np.zeros((h))

        self.dbrh = np.zeros((h))
        self.dbzh = np.zeros((h))
        self.dbnh = np.zeros((h))

        self.r_act = Sigmoid()
        self.z_act = Sigmoid()
        self.h_act = Tanh()

        # Define other variables to store forward results for backward here

    def init_weights(self, Wrx, Wzx, Wnx, Wrh, Wzh, Wnh, brx, bzx, bnx, brh, bzh, bnh):
        self.Wrx = Wrx
        self.Wzx = Wzx
        self.Wnx = Wnx
        self.Wrh = Wrh
        self.Wzh = Wzh
        self.Wnh = Wnh
        self.brx = brx
        self.bzx = bzx
        self.bnx = bnx
        self.brh = brh
        self.bzh = bzh
        self.bnh = bnh

    def __call__(self, x, h_prev_t):
        return self.forward(x, h_prev_t)

    def forward(self, x, h_prev_t):
        """GRU cell forward.

        Input
        -----
        x: (input_dim)
            observation at current time-step.

        h_prev_t: (hidden_dim)
            hidden-state at previous time-step.

        Returns
        -------
        h_t: (hidden_dim)
            hidden state at current time-step.

        """
        self.x = x
        self.hidden = h_prev_t
        
        # Add your code here.
        # Define your variables based on the writeup using the corresponding
        # names below.
        self.r = self.r_act.forward(self.Wrx @ x + self.brx + self.Wrh @ h_prev_t + self.brh)
        self.z = self.z_act.forward(self.Wzx @ x + self.bzx + self.Wzh @ h_prev_t + self.bzh)
        self.n = self.h_act.forward(self.Wnx @ x + self.bnx + self.r * (self.Wnh @ h_prev_t + self.bnh))
        h_t = (1 - self.z) * self.n + self.z * h_prev_t
        
        assert self.x.shape == (self.d,)
        assert self.hidden.shape == (self.h,)

        assert self.r.shape == (self.h,)
        assert self.z.shape == (self.h,)
        assert self.n.shape == (self.h,)
        assert h_t.shape == (self.h,) # h_t is the final output of you GRU cell.

        self.h_t = h_t
        return h_t

    def backward(self, delta):
        """GRU cell backward.

        This must calculate the gradients wrt the parameters and return the
        derivative wrt the inputs, xt and ht, to the cell.

        Input
        -----
        delta: (hidden_dim)
                summation of derivative wrt loss from next layer at
                the same time-step and derivative wrt loss from same layer at
                next time-step.

        Returns
        -------
        dx: (input_dim)
            derivative of the loss wrt the input x.

        dh_prev_t: (hidden_dim)
            derivative of the loss wrt the input hidden h.

        """

        # SOME TIPS:
        # 1) Make sure the shapes of the calculated dWs and dbs match the initalized shapes of the respective Ws and bs
        # 2) When in doubt about shapes, please refer to the table in the writeup.
        # 3) Know that the autograder grades the gradients in a certain order, and the local autograder will tell you which gradient you are currently failing.
        dz = delta * (-self.n + self.hidden)
        dn = delta * (1 - self.z)

        dtanh = self.h_act.backward(dn, state=self.n)
        self.dWnx = np.expand_dims(dtanh, axis=1) @ np.expand_dims(self.x, axis=1).T # h, d
        self.dbnx = dtanh # h
        dr = dtanh * (self.Wnh @ self.hidden + self.bnh)
        self.dWnh = np.expand_dims(dtanh, axis=1) * np.expand_dims(self.r, axis=1) @ np.expand_dims(self.hidden, axis=1).T # h, h
        self.dbnh = dtanh * self.r # h

        dsigz = self.z_act.backward(dz)
        self.dWzx = np.expand_dims(dsigz, axis=1) @ np.expand_dims(self.x, axis=1).T # h, d
        self.dbzx = dsigz # h
        self.dWzh = np.expand_dims(dsigz, axis=1) * np.expand_dims(self.hidden, axis=1).T # h, h
        self.dbzh = dsigz # h

        dsigr = self.r_act.backward(dr)
        self.dWrx = np.expand_dims(dsigr, axis=1) @ np.expand_dims(self.x, axis=1).T # h, d
        self.dbrx = dsigr # h
        self.dWrh = np.expand_dims(dsigr, axis=1) * np.expand_dims(self.hidden, axis=1).T # h, h
        self.dbrh = dsigr # h

        dx = np.squeeze((np.expand_dims(dtanh, axis=1).T @ self.Wnx + np.expand_dims(dsigz, axis=1).T @ self.Wzx + np.expand_dims(dsigr, axis=1).T @ self.Wrx).T)
        dh_prev_t = delta * self.z + dtanh * self.r @ self.Wnh + dsigz @ self.Wzh + dsigr @ self.Wrh

        assert dx.shape == (self.d,)
        assert dh_prev_t.shape == (self.h,)

        return dx, dh_prev_t