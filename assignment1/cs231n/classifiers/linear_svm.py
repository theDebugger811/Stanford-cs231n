import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
    dW = np.zeros(W.shape) # initialize the gradient as zero
    dy = np.zeros([1, W.shape[1]])
  # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        dy = np.zeros([1, W.shape[1]])
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1.0 # note delta = 1
            if margin > 0:
                dy[0][y[i]] += -1.0
                dy[0][j] = 1.0
                loss += margin
        dW += X[i][:,np.newaxis].dot(dy)

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
    

def svm_loss_vectorized(W, X, y, reg):
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero
    num_classes = W.shape[1]
    num_train = X.shape[0]
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
    scores = X.dot(W)
    class_scores = np.asarray([scores[i][y[i]] for i in range(num_train)])[:,np.newaxis]
    true_class_matrix = np.zeros(scores.shape)
    for i in range(num_train):
        true_class_matrix[i][y[i]] = 1
    class_scores = np.repeat(class_scores, num_classes, axis=1)
    loss_matrix = scores - class_scores + np.ones(scores.shape) - true_class_matrix
    loss_matrix[loss_matrix < 0] = 0
    loss = np.sum(loss_matrix)
    loss /= num_train
    loss += reg * np.sum(W * W)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
    #print(loss_matrix.shape, dW.shape)
    dy = np.zeros(loss_matrix.shape)
    dy[loss_matrix > 0] = 1.0
    error_per_eg = np.sum(dy, axis=1)
    for i in range(num_train):
        dy[i][y[i]] = -1.0 * error_per_eg[i]
    #print(dy[0,:])
    #print(loss_matrix)
    dW = X.T.dot(dy)
    dW = np.true_divide(dW, num_train)
    dW += 2.0 * reg * W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

    return loss, dW
#  """
#   Structured SVM loss function, vectorized implementation.

#   Inputs and outputs are the same as svm_loss_naive.
#   """