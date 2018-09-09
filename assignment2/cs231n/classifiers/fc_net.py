from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


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
        self.params['W1'] = weight_scale*np.random.randn(input_dim, hidden_dim)
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['W2'] = weight_scale*np.random.randn(hidden_dim, num_classes)
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
#         print(self.params)
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
#         print(self.params['W1'])
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        X_reshaped = np.reshape(X, (X.shape[0], -1))
#         print(W1.shape)
#         print(X_reshaped.shape)
        h = X_reshaped.dot(W1) + b1
        h_relu = np.copy(h)
        h_relu[h < 0] = 0
        scores = (h_relu.dot(W2) + b2)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0.0, {}
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
        #Loss
        num_train = X.shape[0]
        exp_scores = np.exp(scores)
        num = np.sum(exp_scores, axis=1)[:,np.newaxis]
        den = np.reciprocal(num)
        exp_scores = exp_scores * den
        dy = np.copy(exp_scores)
        for i in range(num_train):
            dy[i][y[i]] += -1
            loss += -np.log(exp_scores[i][y[i]])
        loss /= num_train
        loss += 0.5 * self.reg * (np.sum(W1 * W1) + np.sum(W2 * W2))
        
        #Grads
        dW_temp = h_relu.T.dot(dy)
        dW_temp = np.true_divide(dW_temp, num_train)
        dW_temp += self.reg * W2
        grads['W2'] = dW_temp
        
        #b2
        grads['b2'] = np.true_divide(np.sum(dy, axis=0), num_train)
        
        #Relu
        grad_h_relu = dy.dot(W2.T)
        grad_h = np.copy(grad_h_relu)
        grad_h[h < 0] = 0
        
        #W1
        dW_temp = X_reshaped.T.dot(grad_h)
        dW_temp = np.true_divide(dW_temp, num_train)
        dW_temp += self.reg * W1
        grads['W1'] = dW_temp
        
        #b1
        grads['b1'] = np.true_divide(np.sum(grad_h, axis=0), num_train) 
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
    
 
    
def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        # 
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        #######################################################################
        sample_mean = np.mean(x, axis=0)
        sample_std = np.std(x, axis=0)
        sample_var = sample_std ** 2
        x_mu = x - sample_mean
        x_mu_pow = x_mu ** 2
        mu = sample_mean
        var = sample_std ** 2
        sqrt_var = np.sqrt(var + eps)
        i_var = np.reciprocal(sqrt_var)
        x_hat = x_mu * i_var
        out = x_hat * gamma + beta

        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var
        cache_list = (x, mu, x_mu, x_mu_pow, var, sqrt_var, i_var, x_hat, gamma, beta)
        cache = np.empty(len(cache_list), dtype=object)
        cache[:] = cache_list
        #x_hat, x_mu, x, x_mu_pow,
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        x_new = x - running_mean
        x_new = x_new / np.sqrt(running_var + eps) 
        out = x_new * gamma + beta
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    x, mu, x_mu, x_mu_pow, var, sqrt_var, i_var, x_hat, gamma, beta = cache
    N, D = x.shape
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################
    #dbeta = np.mean(dout, axis=0)
    dbeta = np.sum(dout, axis=0) 
    
    #dgamma = np.mean(dout * x_hat, axis=0)
    dgamma = np.sum(dout * x_hat, axis=0)
    
    dx_hat = np.multiply(dout,gamma)
    
    #di_var = np.mean(dx_hat * x_mu, axis=0)
    di_var = np.sum(dx_hat * x_mu, axis=0)
    
    dx_mu_1 = np.multiply(dx_hat, i_var)
    
    dsqrt_var = -di_var * np.reciprocal(sqrt_var ** 2) 
    
    d_var = dsqrt_var * 0.5 * np.reciprocal(sqrt_var)
    
    dx_mu_pow = (1.0 / N) * np.ones(x.shape) * d_var
    
    dx_mu_2 = 2.0 * np.multiply(x_mu,dx_mu_pow)
    
    dx_1 = dx_mu_2 + dx_mu_1
    
    #dmu = -1.0 * np.mean(dx_mu_2 + dx_mu_1, axis=0)
    dmu = -1.0 * np.sum(dx_mu_2 + dx_mu_1, axis=0)
    
    dx_2 = (1.0/N) * np.ones(x.shape) * dmu
    
    dx = dx_1 + dx_2
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta



def layernorm_forward(x, gamma, beta, ln_param):
    """
    Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.
    
    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get('eps', 1e-5)
    ###########################################################################
    # TODO: Implement the training-time forward pass for layer norm.          #
    # Normalize the incoming data, and scale and  shift the normalized data   #
    #  using gamma and beta.                                                  #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of  batch normalization, and inserting a line or two of  #
    # well-placed code. In particular, can you think of any matrix            #
    # transformations you could perform, that would enable you to copy over   #
    # the batch norm code and leave it almost unchanged?                      #
    ###########################################################################
    x = x.T
    sample_mean = np.mean(x, axis=0)
    sample_std = np.std(x, axis=0)
    sample_var = sample_std ** 2
    x_mu = x - sample_mean
    x_mu_pow = x_mu ** 2
    mu = sample_mean
    var = sample_std ** 2
    sqrt_var = np.sqrt(var + eps)
    i_var = np.reciprocal(sqrt_var)
    x_hat = x_mu * i_var
    x_hat = x_hat.T
    out = x_hat * gamma + beta

    cache_list = (x, mu, x_mu, x_mu_pow, var, sqrt_var, i_var, x_hat, gamma, beta)
    cache = np.empty(len(cache_list), dtype=object)
    cache[:] = cache_list
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache

def layernorm_backward(dout, cache):
    """
    Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    x, mu, x_mu, x_mu_pow, var, sqrt_var, i_var, x_hat, gamma, beta = cache
    N, D = x.shape
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################
    #dbeta = np.mean(dout, axis=0)
    dbeta = np.sum(dout, axis=0) 
    
    #dgamma = np.mean(dout * x_hat, axis=0)
    dgamma = np.sum(dout * x_hat, axis=0)
    
    dx_hat = np.multiply(dout,gamma)
    dx_hat = dx_hat.T
    
    #di_var = np.mean(dx_hat * x_mu, axis=0)
    di_var = np.sum(dx_hat * x_mu, axis=0)
    
    dx_mu_1 = np.multiply(dx_hat, i_var)
    
    dsqrt_var = -di_var * np.reciprocal(sqrt_var ** 2) 
    
    d_var = dsqrt_var * 0.5 * np.reciprocal(sqrt_var)
    
    dx_mu_pow = (1.0 / N) * np.ones(x.shape) * d_var
    
    dx_mu_2 = 2.0 * np.multiply(x_mu,dx_mu_pow)
    
    dx_1 = dx_mu_2 + dx_mu_1
    
    #dmu = -1.0 * np.mean(dx_mu_2 + dx_mu_1, axis=0)
    dmu = -1.0 * np.sum(dx_mu_2 + dx_mu_1, axis=0)
    
    dx_2 = (1.0/N) * np.ones(x.shape) * dmu
    
    dx = dx_1 + dx_2
    dx = dx.T
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta

def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.

    NOTE: Please implement **inverted** dropout, not the vanilla version of dropout.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    NOTE 2: Keep in mind that p is the probability of **keep** a neuron
    output; this might be contrary to some sources, where it is referred to
    as the probability of dropping a neuron output.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        mask = (np.random.rand(*x.shape) < p) / p
        out  = x * mask
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        out = x
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        dropout_param, mask = cache
        dx = dout * mask
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == 'test':
        dx = dout
    return dx


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
                 weight_scale=1e-3, dtype=np.float32, seed=None):
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
        weight_name = 'W' + str(1)
        bias_name = 'b' + str(1)
        self.params[weight_name] = weight_scale*np.random.randn(input_dim, hidden_dims[0])
        self.params[bias_name] = np.zeros(hidden_dims[0])
        
        ## If BatchNorm Used
        if self.normalization == 'batchnorm':
            gamma_name = 'gamma' + str(1)
            beta_name = 'beta' + str(1)   
            self.params[gamma_name] = np.ones(hidden_dims[0])
            self.params[beta_name] = np.zeros(hidden_dims[0])
        
        if self.normalization == 'layernorm':
            gamma_name = 'gamma' + str(1)
            beta_name = 'beta' + str(1)   
            self.params[gamma_name] = np.ones(hidden_dims[0])
            self.params[beta_name] = np.zeros(hidden_dims[0])

        for k in range(1, self.num_layers - 1):
            weight_name = 'W' + str(k+1)
            bias_name = 'b' + str(k+1)
            self.params[weight_name] = weight_scale*np.random.randn(hidden_dims[k-1], hidden_dims[k])
            self.params[bias_name] = np.zeros(hidden_dims[k])
            
            ## If BatchNorm Used
            if self.normalization == 'batchnorm':
                gamma_name = 'gamma' + str(k+1)
                beta_name = 'beta' + str(k+1)
                self.params[gamma_name] = np.ones(hidden_dims[k])
                self.params[beta_name] = np.zeros(hidden_dims[k])

            if self.normalization == 'layernorm':
                gamma_name = 'gamma' + str(k+1)
                beta_name = 'beta' + str(k+1)
                self.params[gamma_name] = np.ones(hidden_dims[k])
                self.params[beta_name] = np.zeros(hidden_dims[k])

        
        weight_name = 'W' + str(self.num_layers)
        bias_name = 'b' + str(self.num_layers)
        self.params[weight_name] = weight_scale*np.random.randn(hidden_dims[self.num_layers - 2], num_classes)
        self.params[bias_name] = np.zeros(num_classes)        

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
        cache = {}
        input_temp = np.reshape(X, (X.shape[0], -1))
        W1 = self.params['W' + str(1)]
        b1 = self.params['b' + str(1)]
        h = input_temp.dot(W1) + b1
        ## If BatchNorm Used
        if self.normalization == 'batchnorm':
            gamma1 = self.params['gamma' + str(1)]
            beta1 = self.params['beta' + str(1)]
            h_b, cache_BN = batchnorm_forward(h, gamma1, beta1, self.bn_params[0])
            cache['BN_cache' + '1']= np.copy(cache_BN)
            
        elif self.normalization == 'layernorm':
            gamma1 = self.params['gamma' + str(1)]
            beta1 = self.params['beta' + str(1)]
            h_b, cache_BN = layernorm_forward(h, gamma1, beta1, self.bn_params[0])
            cache['BN_cache' + '1']= np.copy(cache_BN)
            
        else:
            h_b = h
            
        h_relu = np.copy(h_b)
        h_relu[h_b < 0] = 0
        cache['Input' + '1'] = np.copy(input_temp)
        cache['Hidden' + '1'] = np.copy(h_b)
        
        if self.use_dropout:
            h_out, cache_dp = dropout_forward(h_relu, self.dropout_param)
            cache['Dp_Cache'  + '1'] = np.copy(cache_dp)
        else:
            h_out = h_relu
        
        reg_loss = np.sum(W1 * W1)
        for k in range(1, self.num_layers - 1):
            W_temp = self.params['W' + str(k+1)]
            b_temp = self.params['b' + str(k+1)]
            reg_loss += np.sum(W_temp * W_temp)
            input_temp = np.copy(h_out)
            h = input_temp.dot(W_temp) + b_temp
            ## If BatchNorm Used
            if self.normalization == 'batchnorm':
                gamma_temp = self.params['gamma' + str(k+1)]
                beta_temp = self.params['beta' + str(k+1)]
                h_b, cache_BN = batchnorm_forward(h, gamma_temp, beta_temp, self.bn_params[k])
                cache['BN_cache' + str(k+1)]= np.copy(cache_BN)
            elif self.normalization == 'layernorm':
                gamma_temp = self.params['gamma' + str(k+1)]
                beta_temp = self.params['beta' + str(k+1)]
                h_b, cache_BN = layernorm_forward(h, gamma_temp, beta_temp, self.bn_params[k])
                cache['BN_cache' + str(k+1)]= np.copy(cache_BN)
                
            else:
                h_b = h
                
            h_relu = np.copy(h_b)
            h_relu[h_b < 0] = 0
            cache['Input' + str(k+1)] = np.copy(input_temp)
            cache['Hidden' + str(k+1)] = np.copy(h_b)
            if self.use_dropout:
                h_out, cache_dp = dropout_forward(h_relu, self.dropout_param)
                cache['Dp_Cache'  + str(k+1)] = np.copy(cache_dp)
            else:
                h_out = h_relu
            
            
        W_temp = self.params['W' + str(self.num_layers)]
        b_temp = self.params['b' + str(self.num_layers)]
        reg_loss += np.sum(W_temp * W_temp)
        input_temp = np.copy(h_out)
        cache['Input' + str(self.num_layers)] = np.copy(input_temp)
        scores = input_temp.dot(W_temp) + b_temp

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
        ## LOSS
        num_train = X.shape[0]
        exp_scores = np.exp(scores)
        num = np.sum(exp_scores, axis=1)[:,np.newaxis]
        den = np.reciprocal(num)
        if np.isnan(np.sum(den)):
            print(num)
            print(input_temp)
            print(exp_scores)
            raise ValueError('den has Nan')
        exp_scores = exp_scores * den
        dy = np.copy(exp_scores)
        for i in range(num_train):
            dy[i][y[i]] += -1
            loss += -np.log(exp_scores[i][y[i]])
        loss /= num_train
        loss += 0.5 * self.reg * reg_loss
        
        ##Grads Last Layers
        W_temp = self.params['W' + str(self.num_layers)]
        grads['W' + str(self.num_layers)] = np.copy(np.true_divide(cache['Input' + str(self.num_layers)].T.dot(dy), num_train) + self.reg * W_temp)
        grads['b' + str(self.num_layers)] = np.copy(np.true_divide(np.sum(dy, axis=0), num_train))
        dout = dy.dot(W_temp.T)
        
        #Hidden Layers
        for k in range(self.num_layers - 2, 0, -1):      
            if self.use_dropout:
                dx_d = dropout_backward(dout, cache['Dp_Cache'  + str(k+1)])
            else:
                dx_d = dout
                
            W_temp = self.params['W' + str(k+1)]
            b_temp = self.params['b' + str(k+1)]
            da = relu_backward(dx_d, cache['Hidden' + str(k+1)])
            
            ## If BatchNorm Used
            if self.normalization == 'batchnorm':
                dx_b, dgamma, dbeta = batchnorm_backward(da, cache['BN_cache' + str(k+1)])
                grads['gamma' + str(k+1)] = np.copy(np.true_divide(dgamma, num_train))
                grads['beta' + str(k+1)] = np.copy(np.true_divide(dbeta, num_train))
            elif self.normalization == 'layernorm':
                dx_b, dgamma, dbeta = layernorm_backward(da, cache['BN_cache' + str(k+1)])
                grads['gamma' + str(k+1)] = np.copy(np.true_divide(dgamma, num_train))
                grads['beta' + str(k+1)] = np.copy(np.true_divide(dbeta, num_train))
            else:
                dx_b = da

            
            dx, dw, db = affine_backward(dx_b, (cache['Input' + str(k+1)], W_temp, b_temp))
            grads['W' + str(k+1)] = np.copy(np.true_divide(dw, num_train) + self.reg * W_temp)
            grads['b' + str(k+1)] = np.copy(np.true_divide(db, num_train))
            dout = dx
            
        if self.use_dropout:
            dx_d = dropout_backward(dout, cache['Dp_Cache'  + str(1)])
        else:
            dx_d = dout
        W_temp = self.params['W' + str(1)]
        b_temp = self.params['b' + str(1)]
        da = relu_backward(dx_d, cache['Hidden' + str(1)])
        ## If BatchNorm Used
        if self.normalization == 'batchnorm':
            dx_b, dgamma, dbeta = batchnorm_backward(da, cache['BN_cache' + str(1)])
            grads['gamma' + str(1)] = np.copy(np.true_divide(dgamma, num_train))
            grads['beta' + str(1)] = np.copy(np.true_divide(dbeta, num_train))
        elif self.normalization == 'layernorm':
            dx_b, dgamma, dbeta = layernorm_backward(da, cache['BN_cache' + str(1)])
            grads['gamma' + str(1)] = np.copy(np.true_divide(dgamma, num_train))
            grads['beta' + str(1)] = np.copy(np.true_divide(dbeta, num_train))
            
        else :
            dx_b = da
        dx, dw, db = affine_backward(dx_b, (cache['Input' + str(1)], W_temp, b_temp))
        grads['W' + str(1)] = np.copy(np.true_divide(dw, num_train) + self.reg * W_temp)
        grads['b' + str(1)] = np.copy(np.true_divide(db, num_train))

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
