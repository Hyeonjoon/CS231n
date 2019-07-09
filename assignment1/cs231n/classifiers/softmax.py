from builtins import range
import numpy as np
import math
from random import shuffle
from past.builtins import xrange

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
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    num_class = W.shape[1]

    for i in range(num_train):
        correct_class = y[i]

        # Create score arrays for Softmax loss function.
        # Biasing score arrays to make their max score 0,
        # which is helpful for the numerical stability.
        score = X[i].dot(W)
        biased_score = score - np.max(score)
        prob_score = np.exp(biased_score) / np.sum(np.exp(biased_score))
        loss -= math.log(prob_score[correct_class])
        for j in range(num_class):
            if (j != correct_class):
                dW[:, j] += X[i] * prob_score[j]
        dW[:, correct_class] -= X[i] * (1 - prob_score[correct_class])

    # Compute means.
    loss /= num_train
    dW /= num_train

    # Regularize.
    loss += reg * np.sum(W * W)
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]

    scores = X.dot(W)
    max_scores = np.max(scores, axis=1)[None].T
    biased_scores = scores - max_scores
    exp_scores = np.exp(biased_scores)
    prob_scores = exp_scores / np.sum(exp_scores, axis=1)[None].T

    loss -= np.sum(np.log(prob_scores[np.arange(num_train), y]))

    prob_scores[np.arange(num_train), y] -= 1       # x -> -(1 - x)
    dW += (X.T).dot(prob_scores)

    # Compute means.
    loss /= num_train
    dW /= num_train

    # Regularize.
    loss += reg * np.sum(W * W)
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
