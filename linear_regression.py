__author__ = 'yalguo'

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics, datasets, preprocessing
from util import load_data, feature_normalize

class linear_regression:
    def compute_cost(self, X, y, theta):
        #COMPUTECOST Compute cost for linear regression
        #   J = compute_cost(X, y, theta) computes the cost of using theta as the
        #   parameter for linear regression to fit the data points in X and y

        # Initialize some useful values
        m = y.shape[0] # number of training examples

        # You need to return the following variables correctly

        # ====================== YOUR CODE HERE ======================
        # Instructions: Compute the cost of a particular choice of theta
        #               You should set J to the cost.

        #h = y[i]-np.dot(theta,X[i])

        J = sum([(y[i]-np.dot(theta,X[i]))**2 for i in range(m)])/(2*m)

        return J

    def gradient_descnet(self, X, y, theta, alpha, num_iters):
        #GRADIENTDESCENT Performs gradient descent to learn theta
        #   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by
        #   taking num_iters gradient steps with learning rate alpha

        # Initialize some useful values
        m = y.shape[0] # number of training examples
        J_history = np.zeros((num_iters,), dtype=np.float);

        for iter in range(0,num_iters):

            # ====================== YOUR CODE HERE ======================
            # Instructions: Perform a single gradient step on the parameter vector
            #               theta.
            #
            # Hint: While debugging, it can be useful to print out the values
            #       of the cost function (computeCost) and gradient here.
            #
            derivative = sum([X[i]*(np.dot(theta, X[i])-y[i]) for i in range(m)])/m
            theta = theta - alpha*derivative
            # ============================================================

            # Save the cost J in every iteration
            J_history[iter] = self.compute_cost(X, y, theta)

        return (theta, J_history)


def test_lr_boston():
    data = datasets.load_boston()
    # X = data['data'][:,12]
    # y = data['target']
    X = preprocessing.scale(data['data'][:,12])
    y = preprocessing.scale(data['target'])

    m = y.shape[0]

    X = np.reshape(X, (m,1))
    X = np.concatenate((np.ones((m,1),np.float),X), axis=1)
    print X
    n = X.shape[1]
    theta = np.zeros(n)

    iteration = 1000
    alpha = 0.01

    train = X[:X.shape[0]/2]
    ytrain = y[:y.shape[0]/2]
    # test = X[X.shape[0]/2:]
    # ytest = y[y.shape[0]/2:]

    lr = linear_regression()
    first_j = lr.compute_cost(train, ytrain, theta)
    print first_j

    (theta, J) = lr.gradient_descnet(train, ytrain, theta, alpha, iteration)

    print theta, '\n'

    plt.plot(J, linewidth=2)
    plt.xlabel('Iteration, i', fontsize=16)
    plt.ylabel(r'J($\theta$)', fontsize=16)
    plt.show()

    #plot the error
    plt.plot(X[:,1], y, 'ro', label='Data')
    plt.plot(X[:,1], [np.dot(theta,i) for i in X], 'b-', linewidth=2, label='Model')
    plt.xlabel('lower status of the population', fontsize=16)
    plt.ylabel('home price', fontsize=14)
    plt.show()

def test_lr_data():
    lr = linear_regression()
    # (X, y) = load_data(r'lr_data.txt')
    (X, y) = load_data(r'lr_data.txt')

    (X, mu, sigma) = feature_normalize(X)
    m = y.shape[0]
    # X = np.reshape(X, (m,1))
    X = np.concatenate((np.ones((m,1),np.float),X), axis=1)
    print X
    n = X.shape[1]
    theta = np.zeros(n)

    iteration = 1500
    alpha = 0.01

    first_j = lr.compute_cost(X, y, theta)
    print first_j

    (theta, J) = lr.gradient_descnet(X, y, theta, alpha, iteration)

    print theta, '\n'

    plt.plot(J, linewidth=2)
    plt.xlabel('Iteration, i', fontsize=16)
    plt.ylabel(r'J($\theta$)', fontsize=16)
    plt.show()

    #plot the error
    plt.plot(X[:,1], y, 'ro', label='Data')
    plt.plot(X[:,1], [np.dot(theta,i) for i in X], 'b-', linewidth=2, label='Model')
    plt.xlabel('population', fontsize=16)
    plt.ylabel('profit', fontsize=14)
    plt.show()


if __name__=="__main__":
    test_lr_data()