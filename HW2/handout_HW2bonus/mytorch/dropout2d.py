# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
import pdb 
np.random.seed(11785)

class Dropout2d(object):
    def __init__(self, p=0.5):
        # Dropout probability
        self.p = p
        self.mask = None

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, x, eval=False):
        """
        Arguments:
          x (np.array): (batch_size, in_channel, input_width, input_height)
          eval (boolean): whether the model is in evaluation mode
        Return:
          np.array of same shape as input x
        """
        # 1) Get and apply a per-channel mask generated from np.random.binomial
        # 2) Scale your output accordingly
        # 3) During test time, you should not apply any mask or scaling.
       
        if eval:
          return x
        else:
          self.mask = np.random.binomial(1, 1-self.p, size=(x.shape[0], x.shape[1], 1, 1)) # in_channel (0, 1)
          return x * self.mask * 1/(1-self.p)

    def backward(self, delta):
        """
        Arguments:
          delta (np.array): (batch_size, in_channel, input_width, input_height)
        Return:
          np.array of same shape as input delta
        """
        # 1) This method is only called during training.
        # 2) You should scale the result by chain rule
        return delta * self.mask * 1/(1-self.p)