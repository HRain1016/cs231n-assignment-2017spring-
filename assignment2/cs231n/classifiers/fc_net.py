from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


def all_forward(x, w, b, gamma=None, beta=None, bn_param=None, 
           dp_param=None, relu = None, bn_ln=None, dropout=None):
    """
    Convenience layer that perorms an affine transform followed by a batchnorm/layernorm, 
    a ReLU, or a dropout. Each of these following layers can be used or not.

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer
    - gamma, beta: Scale and shift parameters for the batchnorm/layernorm
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: giving running mean of features
      - running_var: giving running variance of features
    - dp_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'.
      - seed: Seed for the random number generator, which is needed for gradient checking.
    - relu: Use or not use (None) ReLU layer
    - bn_ln: 'batchnorm' or 'layernorm' or not use (for default)
    - dropout: None is default meaning that not use dropout

    Returns a tuple of:
    - out: Output from the last layer
    - cache: Object to give to the backward pass
    """
    # affine
    fc_out, fc_cache = affine_forward(x, w, b)
    
    # batchnorm/layernorm
    if bn_ln == 'batchnorm':
        bnln_out, bnln_cache = batchnorm_forward(fc_out, gamma, beta, bn_param)
    elif bn_ln == 'layernorm':
        bnln_out, bnln_cache = layernorm_forward(fc_out, gamma, beta, bn_param)
    else:
        bnln_out, bnln_cache = fc_out, None
        
    # ReLU
    if relu:
        relu_out, relu_cache = relu_forward(bnln_out)
    else:
        relu_out, relu_cache = bnln_out, None
    
    # dropout
    if dropout:
        out, dp_cache = dropout_forward(relu_out, dp_param)        
    else:
        out, dp_cache = relu_out, None
        
    cache = (fc_cache, bnln_cache, relu_cache, dp_cache)
    return out, cache

def all_backward(dout, cache, bn_ln=None):
    """
    Backward pass for the affine-batchnorm/layernorm-relu-dropout convenience layer
    """
    fc_cache, bnln_cache, relu_cache, dp_cache = cache
    
    if dp_cache != None:
        ddrop = dropout_backward(dout, dp_cache)
    else: 
        ddrop = dout
    
    if relu_cache is not None:
        drelu = relu_backward(ddrop, relu_cache)  
    else:
        drelu = ddrop   
    
    if bn_ln == 'batchnorm':
        dbnln, dgamma, dbeta = batchnorm_backward_alt(drelu, bnln_cache)
    elif bn_ln == 'layernorm':
        dbnln, dgamma, dbeta = layernorm_backward(drelu, bnln_cache)
    else:
        dbnln = drelu
        
    dx, dw, db = affine_backward(dbnln, fc_cache)
    
    if bn_ln:
        return dx, dw, db, dgamma, dbeta
    else:
        return dx, dw, db

    
class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        self.params['W1'] = np.random.normal(loc=0.0, scale=weight_scale, size=(input_dim, hidden_dim))
        self.params['W2'] = np.random.normal(loc=0.0, scale=weight_scale, size=(hidden_dim, num_classes))
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['b2'] = np.zeros(num_classes)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        out1, cache1 = affine_relu_forward(X, self.params['W1'], self.params['b1'])
        scores, cache2 = affine_forward(out1, self.params['W2'], self.params['b2'])
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        loss, dscores = softmax_loss(scores, y)
        loss += 0.5 * self.reg * (np.sum(self.params['W1']**2) + np.sum(self.params['W2']**2))
        dout1, grads['W2'], grads['b2'] = affine_backward(dscores, cache2)
        _, grads['W1'], grads['b1'] = affine_relu_backward(dout1, cache1)
        grads['W1'] += self.reg * self.params['W1']
        grads['W2'] += self.reg * self.params['W2']
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch/layer normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=1, normalization=None, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}
        all_dims = hidden_dims.copy()
        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        all_dims.insert(0, input_dim)
        all_dims.append(num_classes)
        for i in range(len(all_dims)-1):
            self.params['W%d'%(i+1)] = np.random.normal(loc=0.0, scale=weight_scale, size=(all_dims[i], all_dims[i+1]))  
            self.params['b%d'%(i+1)] = np.zeros(all_dims[i+1])
            if self.normalization and i < (self.num_layers - 1):
                self.params['gamma%d'%(i+1)] = np.ones(all_dims[i+1])
                self.params['beta%d'%(i+1)] = np.zeros(all_dims[i+1])
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization=='batchnorm':
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]
        if self.normalization=='layernorm':
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.normalization=='batchnorm':
            for bn_param in self.bn_params:
                bn_param['mode'] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        outs = []
        caches = []
        for i in range(self.num_layers-1):
            if i == 0: inputs = X
            else: inputs = outs[i-1]
            if self.normalization:
                out, cache = all_forward(inputs, self.params['W%d'%(i+1)], self.params['b%d'%(i+1)],
                             self.params['gamma%d'%(i+1)], self.params['beta%d'%(i+1)],
                             self.bn_params[i], self.dropout_param, 
                             relu=True, bn_ln=self.normalization, 
                             dropout=self.use_dropout )
            else:
                out, cache = all_forward(inputs, self.params['W%d'%(i+1)], self.params['b%d'%(i+1)],
                                 relu=True, dp_param=self.dropout_param, dropout=self.use_dropout)
            outs.append(out)
            caches.append(cache)
        scores, cache = all_forward(outs[-1], self.params['W%d'%self.num_layers], self.params['b%d'%self.num_layers])
        caches.append(cache)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        loss, dscores = softmax_loss(scores, y)
        reg_loss = 0
        for i in range(self.num_layers):
            reg_loss += 0.5 * self.reg * np.sum(self.params['W%d'%(i+1)]**2)
        loss += reg_loss
        
        douts = []
        caches = list(reversed(caches))
        dout, grads['W%d'%self.num_layers], grads['b%d'%self.num_layers] = all_backward(dscores, caches[0])
        grads['W%d'%self.num_layers] += self.reg * self.params['W%d'%self.num_layers]
        douts.append(dout)
        for i in range(self.num_layers-1):
            if self.normalization:
                dout, grads['W%d'%(self.num_layers-1-i)], grads['b%d'%(self.num_layers-1-i)], grads['gamma%d'%(self.num_layers\
                                                                                                          -1-i)], grads['beta%d'%(self.num_layers-1-i)] = all_backward(douts[i], caches[i+1], self.normalization)
            else:
                dout, grads['W%d'%(self.num_layers-1-i)], grads['b%d'%(self.num_layers-1-i)] = all_backward(douts[i], caches[i+1])
            grads['W%d'%(self.num_layers-1-i)] += self.reg * self.params['W%d'%(self.num_layers-1-i)]
            douts.append(dout)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
