import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
    
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
    dy = np.zeros([1, W.shape[1]])
  # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        exp_scores = np.exp(scores)
        den = float(np.sum(exp_scores))
        exp_scores /= den
        dy = np.copy(exp_scores)
        dy[y[i]] += -1
        loss += -np.log(exp_scores[y[i]])
        dW += X[i][:,np.newaxis].dot(dy[:,np.newaxis].T)

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
    loss /= num_train
    dW = np.true_divide(dW, num_train)

  # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW += 2.0 * reg * W
  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


    return loss, dW

#   """
#   Softmax loss function, naive implementation (with loops)

#   Inputs have dimension D, there are C classes, and we operate on minibatches
#   of N examples.

#   Inputs:
#   - W: A numpy array of shape (D, C) containing weights.
#   - X: A numpy array of shape (N, D) containing a minibatch of data.
#   - y: A numpy array of shape (N,) containing training labels; y[i] = c means
#     that X[i] has label c, where 0 <= c < C.
#   - reg: (float) regularization strength

#   Returns a tuple of:
#   - loss as single float
#   - gradient with respect to weights W; an array of same shape as W
#   """

#   """
#   Structured SVM loss function, naive implementation (with loops).

#   Inputs have dimension D, there are C classes, and we operate on minibatches
#   of N examples.

#   Inputs:
#   - W: A numpy array of shape (D, C) containing weights.
#   - X: A numpy array of shape (N, D) containing a minibatch of data.
#   - y: A numpy array of shape (N,) containing training labels; y[i] = c means
#     that X[i] has label c, where 0 <= c < C.
#   - reg: (float) regularization strength

#   Returns a tuple of:
#   - loss as single float
#   - gradient with respect to weights W; an array of same shape as W
#   """
 
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################


def softmax_loss_vectorized(W, X, y, reg):

  # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
    num_classes = W.shape[1]
    num_train = X.shape[0]
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
    scores = X.dot(W)
    exp_scores = np.exp(scores)
    den = np.sum(exp_scores, axis=1)[:,np.newaxis]
    den = np.reciprocal(den)
    exp_scores = exp_scores * den
    dy = np.copy(exp_scores)
    for i in range(num_train):
        dy[i][y[i]] += -1
        loss += -np.log(exp_scores[i][y[i]])
    loss /= num_train
    loss += reg * np.sum(W * W)
    
    dW = X.T.dot(dy)
    dW = np.true_divide(dW, num_train)
    dW += 2.0 * reg * W    
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

    
    return loss, dW
#   """
#   Softmax loss function, vectorized version.

#   Inputs and outputs are the same as softmax_loss_naive.
#   """
