__author__ = 'yalguo'

import numpy as np
import math
import util
from util import load_data, feature_normalize, plot_data
from util import plot_decision_boundary, map_feature, load_mat
import scipy.optimize as op
import random
class logistic_regression:
    def cost_function(self, theta, X, y):
        #COSTFUNCTION Compute cost and gradient for logistic regression
        #   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
        #   parameter for logistic regression and the gradient of the cost
        #   w.r.t. to the parameters.

        # Initialize some useful values
        m = y.shape[0] # number of training examples

        # ====================== YOUR CODE HERE ======================
        # Instructions: Compute the cost of a particular choice of theta.
        #               You should set J to the cost.
        #               Compute the partial derivatives and set grad to the partial
        #               derivatives of the cost w.r.t. each parameter in theta
        #
        # Note: grad should have the same dimensions as theta
        #
        h = self.sigmoid(np.dot(X, theta))
        J1 = np.dot((-y), np.vectorize(math.log)(h))/m
        J2 = np.dot((-(1-y)), np.vectorize(math.log)(1-h))/m

        J = J1+J2

        return J

    def cost_function_reg(self, theta, X, y, lamda):
        #COSTFUNCTION Compute cost and gradient for logistic regression
        #   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
        #   parameter for logistic regression and the gradient of the cost
        #   w.r.t. to the parameters.

        # Initialize some useful values
        m = y.shape[0] # number of training examples

        # ====================== YOUR CODE HERE ======================
        # Instructions: Compute the cost of a particular choice of theta.
        #               You should set J to the cost.
        #               Compute the partial derivatives and set grad to the partial
        #               derivatives of the cost w.r.t. each parameter in theta
        #
        # Note: grad should have the same dimensions as theta
        #
        h = self.sigmoid(np.dot(X, theta))
        J1 = np.dot((-y), np.vectorize(math.log)(h))/m
        J2 = np.dot((-(1-y)), np.vectorize(math.log)(1-h))/m
        J3 = lamda*(np.dot(theta,theta))/(2*m)

        J = J1+J2+J3

        return J

    def gradient_function(self, theta, X, y):
        m = y.shape[0] # number of training examples
        h = self.sigmoid(np.dot(X, theta))
        grad = np.dot((h-y), X)/m

        return grad

    def gradient_function_reg(self, theta, X, y, lamda):
        m = y.shape[0] # number of training examples
        h = self.sigmoid(np.dot(X, theta))
        grad = np.dot((h-y), X)/m

        var = lamda*theta/m
        var[0] = 0
        grad = grad + var

        return grad

    def sigmoid(self, z):
        #SIGMOID Compute sigmoid functoon
        #   J = SIGMOID(z) computes the sigmoid of z.

        # You need to return the following variables correctly
        g = np.zeros(z.shape[0])

        # ====================== YOUR CODE HERE ======================
        # Instructions: Compute the sigmoid of each value of z (z can be a matrix,
        #               vector or scalar).


        g=(1+np.vectorize(math.exp)(-z))**-1
        return g

    def predict(self, theta, X):
        p = np.vectorize(round)(self.sigmoid(np.dot(theta, np.transpose(X))))
        return p

def test_lg():
    lg = logistic_regression()
    (X, y) = load_data(r'lg_data.txt')

    plot_data(X, y)

    # (X, mu, sigma) = feature_normalize(X)
    m = y.shape[0]
    # X = np.reshape(X, (m,1))
    X = np.concatenate((np.ones((m,1),np.float),X), axis=1)
    print X
    n = X.shape[1]
    theta = np.zeros(n)

    cost = lg.cost_function(theta, X, y)
    grad = lg.gradient_function(theta, X, y)
    print 'Cost at initial theta (zeros): %f\n' % cost
    print 'Gradient at initial theta (zeros):', grad

    Result = op.minimize(fun = lg.cost_function, x0 = theta, args = (X, y), method = 'TNC',
                                 jac = lg.gradient_function)
    optimal_theta = Result.x

    print optimal_theta

    plot_decision_boundary(optimal_theta, X, y)

    pred = lg.predict(optimal_theta, X)

    tup = (pred == y).nonzero()
    accuracy = float(tup[0].shape[0])/float(y.shape[0])
    print 'Accuracy:', accuracy

def test_lg_reg():
    lg = logistic_regression()
    (X, y) = load_data(r'lg_reg_data.txt')

    # plot_data(X, y)
    # (X, mu, sigma) = feature_normalize(X)
    X = map_feature(X[:,0], X[:,1])

    n = X.shape[1]
    theta = np.zeros(n)
    lamda =1

    cost = lg.cost_function_reg(theta, X, y, lamda)
    grad = lg.gradient_function_reg(theta, X, y,lamda)
    print 'Cost at initial theta (zeros): %f\n' % cost
    print 'Gradient at initial theta (zeros):', grad

    Result = op.minimize(fun = lg.cost_function_reg, x0 = theta, args = (X, y, lamda), method = 'BFGS')#,
                                 # jac = lg.gradient_function_reg)
    optimal_theta = Result.x

    print optimal_theta

    plot_decision_boundary(optimal_theta, X, y)

    pred = lg.predict(optimal_theta, X)

    accuracy = (np.vectorize(float)(pred==y)).mean()
    print 'Accuracy:', accuracy

def test_lg_one_vs_all():
    lg = logistic_regression()
    input_layer_size = 400  # 20x20 Input Images of Digits
    num_labels = 10         # 10 labels, from 1 to 10
                            # (note that we have mapped "0" to label 10)
    (X, y) = load_mat('lg_one_vs_all_data.mat', 'X', 'y')
    m = X.shape[0]

    ran_list = [i for i in range(m)]
    random.shuffle(ran_list)
    sel = X[ran_list[:100],:]
    util.display_data(sel)

    y = np.reshape(y,(m,))
    lamda = 0.1

    all_theta = one_vs_all(X, y, num_labels, lamda)

    print all_theta

    X = np.hstack((np.ones((m, 1)), X))
    pred = (lg.sigmoid(np.dot(all_theta, np.transpose(X))).argmax(0)+1)

    accuracy = (np.vectorize(float)(pred==y)).mean()

    print accuracy

def one_vs_all(X, y, num_labels, lamda):
    #ONEVSALL trains multiple logistic regression classifiers and returns all
    #the classifiers in a matrix all_theta, where the i-th row of all_theta
    #corresponds to the classifier for label i
    #   [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
    #   logisitc regression classifiers and returns each of these classifiers
    #   in a matrix all_theta, where the i-th row of all_theta corresponds
    #   to the classifier for label i

    # Some useful variables
    m = X.shape[0]
    n = X.shape[1]

    # You need to return the following variables correctly
    all_theta = np.zeros((num_labels, n + 1))
    J = np.zeros(num_labels)

    # Add ones to the X data matrix
    X = np.hstack((np.ones((m, 1)), X))
    # ====================== YOUR CODE HERE ======================
    # Instructions: You should complete the following code to train num_labels
    #               logistic regression classifiers with regularization
    #               parameter lambda.
    #
    # Hint: theta(:) will return a column vector.
    #
    # Hint: You can use y == c to obtain a vector of 1's and 0's that tell use
    #       whether the ground truth is true/false for this class.
    #
    # Note: For this assignment, we recommend using fmincg to optimize the cost
    #       function. It is okay to use a for-loop (for c = 1:num_labels) to
    #       loop over the different classes.
    #
    #       fmincg works similarly to fminunc, but is more efficient when we
    #       are dealing with large number of parameters.
    #
    # Example Code for fmincg:
    #
    #     % Set Initial theta
    #     initial_theta = zeros(n + 1, 1);
    #
    #     % Set options for fminunc
    #     options = optimset('GradObj', 'on', 'MaxIter', 50);
    #
    #     % Run fmincg to obtain the optimal theta
    #     % This function will return theta and the cost
    #     [theta] = ...
    #         fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), ...
    #                 initial_theta, options);
    #

    initial_theta = np.zeros(n+1)

    lg = logistic_regression()
    for i in range(1,num_labels+1):
        #  Run fminunc to obtain the optimal theta
        #  This function will return theta and the cost
        Result = op.minimize(fun = lg.cost_function_reg, x0 = initial_theta, args = (X, np.vectorize(int)(y==i), lamda),
                              method = 'TNC', jac = lg.gradient_function_reg)
        all_theta[i-1,:] = Result.x
        J[i-1] = Result.fun
        # Print theta to screen
        print 'Iteration:', i
        print 'Cost:', Result.fun
        # print ('Theta_%d:' % i), Result.x

    # =========================================================================
    return all_theta

if __name__=="__main__":
    # test_lg()
    # test_lg_reg()
    test_lg_one_vs_all()