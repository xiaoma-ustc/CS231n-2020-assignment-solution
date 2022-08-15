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
    # regularization!                                         #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_train = X.shape[0]
    num_classes = W.shape[1]
    scores = X.dot(W)
    
    normalized_scores = (scores.T - np.max(scores, axis=1)).T # 防止指数爆炸溢出
    exp_normalized_scores = np.exp(normalized_scores)
    
    for i in range(num_train):
        temp = exp_normalized_scores[i][y[i]]
        sum_i = np.sum(exp_normalized_scores[i])
        loss -= np.log(temp*1.0 / sum_i)
    
    loss /= num_train
    loss += reg*np.sum(np.square(W))
    
    for i in range(num_train):
        sum_i = np.sum(exp_normalized_scores[i])
        for j in range(num_classes):
            dW[:, j] += (exp_normalized_scores[i][j]*1.0 / sum_i)*X[i]
            if j == y[i]:
                dW[:, j] -= X[i]
              
    dW /= num_train
    dW += 2*reg*W

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
    num_classes = W.shape[1]
    scores = X.dot(W)
    exp_normalized_scores = np.exp(scores.T - np.max(scores, axis=1)).T
    
    sum_i = np.sum(exp_normalized_scores, axis=1)
    p_i = exp_normalized_scores[range(num_train), y] / sum_i
    L_i = - np.log(p_i)
    loss = np.sum(L_i)
    
    loss /= num_train
    loss += reg*np.sum(W * W)
    acc_effect = (exp_normalized_scores.T / sum_i).T
    acc_effect[range(num_train), y] -= 1.0
    dW = X.T.dot(acc_effect)
    
    dW /= num_train
    dW += 2*reg*W


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
