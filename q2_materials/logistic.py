""" Methods for doing logistic regression."""

import numpy as np
from utils import sigmoid
import math

def logistic_predict(weights, data):
    """
    Compute the probabilities predicted by the logistic classifier.

    Note: N is the number of examples and 
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to the bias (intercepts).
        data:       N x M data matrix where each row corresponds 
                    to one data point.
    Outputs:
        y:          :N x 1 vector of probabilities. This is the output of the classifier.
    """
    N = len(data)
    M = len(data[0])
    insert_coln = np.ones((N, 1))
    data = np.append(data, insert_coln, axis = 1)
    z = np.dot(data, np.asarray(weights))
    y = 1/(1+np.exp((-1)*z))
    np.reshape(y, (-1, 1))
    return y

def evaluate(targets, y):
    """
    Compute evaluation metrics.
    Inputs:
        targets : N x 1 vector of targets.
        y       : N x 1 vector of probabilities.
    Outputs:
        ce           : (scalar) Cross entropy. CE(p, q) = E_p[-log q]. Here we want to compute CE(targets, y)
        frac_correct : (scalar) Fraction of inputs classified correctly.
    """
    ce = 0
    correct = 0
    for i in range(len(targets)):
        t = targets[i][0]
        y_ind = y[i][0]
        
        ce= ce-t*np.log(y_ind)-(1-t)*np.log(1-y_ind)
        if(y_ind > 0.5 and t==1) or (y_ind <= 0.5 and t==0):
            correct+=1
    frac_correct = correct/len(targets)
    return ce, frac_correct

def logistic(weights, data, targets, hyperparameters):
    """
    Calculate negative log likelihood and its derivatives with respect to weights.
    Also return the predictions.

    Note: N is the number of examples and 
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to bias (intercepts).
        data:       N x M data matrix where each row corresponds 
                    to one data point.
        targets:    N x 1 vector of targets class probabilities.
        hyperparameters: The hyperparameters dictionary.

    Outputs:
        f:       The sum of the loss over all data points. This is the objective that we want to minimize.
        df:      (M+1) x 1 vector of derivative of f w.r.t. weights.
        y:       N x 1 vector of probabilities.
    """

    N = len(data)
    M = len(data[0])
    y = logistic_predict(weights, data)
    insert_coln = np.ones((N, 1))
    data = np.append(data, insert_coln, axis = 1)
    df=np.zeros((M+1, 1))
    for i in range(N):
        df+=(y[i]-targets[i][0])*np.reshape(data[i], (-1, 1))
    f, frac_correct = evaluate(targets,y)
    return f, df, y


def logistic_pen(weights, data, targets, hyperparameters):
    """
    Calculate negative log likelihood and its derivatives with respect to weights.
    Also return the predictions.

    Note: N is the number of examples and 
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to bias (intercepts).
        data:       N x M data matrix where each row corresponds 
                    to one data point.
        targets:    N x 1 vector of targets class probabilities.
        hyperparameters: The hyperparameters dictionary.

    Outputs:
        f:             The sum of the loss over all data points. This is the objective that we want to minimize.
        df:            (M+1) x 1 vector of derivative of f w.r.t. weights.
    """

    f, df, y = logistic(weights, data, targets, hyperparameters)
    df = df + weights * hyperparameters["weight_regularization"]
    f = f+(hyperparameters["weight_regularization"]*np.linalg.norm(weights)**2)/2

    return f, df, y
