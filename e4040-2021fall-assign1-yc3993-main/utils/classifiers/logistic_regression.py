import numpy as np
from random import shuffle

def sigmoid(x):
    """Sigmoid function implementation"""
    h = np.zeros_like(x)
        
    h=1/(1+np.exp(-x))
    return h 
    
    #############################################################################
    # TODO: Implement sigmoid function.                                         #         
    #############################################################################
    #############################################################################
    #                          START OF YOUR CODE                               #
    #############################################################################


    
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################



def logistic_regression_loss_naive(W, X, y, reg):
    """
      Logistic regression loss function, naive implementation (with loops)
      Use this linear classification method to find optimal decision boundary.

      Inputs have dimension D, there are C classes, and we operate on minibatches
      of N examples.

      Inputs:
      - W: a numpy array of shape (D, C) containing weights.
      - X: a numpy array of shape (N, D) containing a minibatch of data.
      - y: a numpy array of shape (N,) containing training labels; y[i] = c means
        that X[i] has label c, where c can be either 0 or 1.
      - reg: (float) regularization strength. For regularization, we use L2 norm.

      Returns a tuple of:
      - loss: (float) the mean value of loss functions over N examples in minibatch.
      - gradient: gradient wrt W, an array of same shape as W
    """
    # Set the loss to a random number
    loss = 0
    # Initialize the gradient to zero
    dW = np.zeros_like(W)
    num_train ,dimension = X.shape

    for i in range(num_train):
        xi = X[i,:]        
        f_x = 0 
        for j in range(xi.shape[0]):
            f_x += xi[j] * W[j]
        predict = sigmoid(f_x)      
        loss += -(y[i] * np.log(predict) + (1 - y[i]) * np.log(1 - predict))
        b=np.array([[i] for i in xi])
        dW += -(int(y[i])-int(predict)) * b 
    loss /= num_train
    loss += 0.5 * reg * np.sum(W * W) # add regularization

    dW =dW/num_train
    dW += reg * W # add regularization

    return loss, dW



def logistic_regression_loss_vectorized(W, X, y, reg):
    """
    Logistic regression loss function, vectorized version.
    Use this linear classification method to find optimal decision boundary.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Set the loss to a random number
    loss = 0
    # Initialize the gradient to zero
    dW = np.zeros_like(W)
    num_train ,dimension = X.shape
    f_x_mat = X.dot(W) 
    predict = 1.0 / (1.0 + np.exp(-f_x_mat)) # [N, 1]
    loss = -np.sum( np.log(predict)*y  +  np.log(1 - predict)*(1 - y) )
    loss = 1.0/num_train * loss + 0.5 * reg * np.sum(W * W)
    dW = (X.T).dot(predict - y) # [D, 1]
    dW = 1.0 / num_train * dW + reg * W


    

    return loss, dW
