from builtins import range
import numpy as np
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

    N, D = X.shape
    D, C = W.shape
    for i in range(N):
        f = W.T @ X[i]
        f -= np.max(f)
        f_y = f[y[i]]        
        alpha = 0.
        for j in range(C):
            alpha += np.exp(f[j])
        loss += -f_y + np.log(alpha)
        for j in range(C):
            dW[:,j] += np.exp(f[j]) / alpha * X[i]
        dW[:, y[i]] -= X[i]
    
    loss /= N
    loss += reg * np.sum(W * W)
    dW /= N
    dW += 2*W*reg
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
    N, D = X.shape
    D, C = W.shape
    f = X @ W
    f -= np.amax(f, axis=1, keepdims=True)
    alpha = np.sum(np.exp(f), axis=1, keepdims=True)
    loss += -np.sum(f[np.arange(N), y]) + np.sum(np.log(alpha))
    p = np.exp(f) / alpha
    p[np.arange(N), y] -= 1
    dW = X.T @ p

    loss /= N
    dW /= N
    loss += reg * np.sum(W * W)
    dW += 2*W*reg
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
