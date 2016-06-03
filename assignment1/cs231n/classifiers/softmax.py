import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W) # (3073,10)
  num_train = X.shape[0]
  num_classes = W.shape[1] # (3073,10)
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in xrange(num_train):
  	scores = X[i].dot(W) # (3073) * (3073,10) = (10,)
  	correct_class_score = scores[y[i]]
  	sum_score = np.sum(np.exp(scores))
  	loss += -np.log(np.exp(correct_class_score) / sum_score)
  	for j in xrange(num_classes):
  		p = np.exp(scores[j])/sum_score
  		dW[:, j] += (p-(j==y[i])) * X[i,:]
  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  dW /= num_train
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W) # 3073,10
  num_train = X.shape[0]
  num_classes = W.shape[1] # (3073,10)
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X.dot(W) # (500,10)
  correct_class_score = scores[np.arange(len(scores)), y] # (500,)
  sum_score = np.sum(np.exp(scores), axis=1) # (500,)
  loss_vec = -np.log(np.exp(correct_class_score) / sum_score)
  loss = np.sum(loss_vec) / num_train
  loss += 0.5 * reg * np.sum(W * W)
  
  p = np.exp(scores) / np.sum(np.exp(scores), axis=1)[:, None]
  ind = np.zeros(p.shape) # 500,10
  ind[np.arange(num_train), y] = 1
  dW = X.T.dot(p-ind)
  dW /= num_train
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

