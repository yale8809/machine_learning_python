__author__ = 'yalguo'

import numpy as np
from os.path import dirname
from os.path import join
import matplotlib.pyplot as plt
from scipy.io import loadmat
import math
import cv2

def load_data(file_name):
    module_path = dirname(__file__)

    data_file_name = join(module_path, 'data', file_name)
    with open(data_file_name) as f:
        lines = f.readlines()
        m = len(lines)
        n = len(lines[0].split(','))
        data = np.empty((m, n-1))
        target = np.empty((m,), dtype=np.int)
        for i, line in enumerate(lines):
            numbers = line.split(",")
            for j, number in enumerate(numbers):
                if j<n-1:
                    data[i][j] = float(number)
            target[i] = float(numbers[-1])

        return (data, target)

def load_mat(file_name, char1, char2):
    module_path = dirname(__file__)
    data_file_name = join(module_path, 'data', file_name)
    data = loadmat(data_file_name)

    return (data[char1], data[char2])


def feature_normalize(X):
    X_norm = X
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)

    for i in range(len(mu)):
        X_norm[:,i] -= mu[i]

    for i in range(len(sigma)):
        X_norm[:,i] /= mu[i]


    return X_norm, mu, sigma

def map_feature(X1, X2):
        # MAPFEATURE Feature mapping function to polynomial features
        #
        #   MAPFEATURE(X1, X2) maps the two input features
        #   to quadratic features used in the regularization exercise.
        #
        #   Returns a new feature array with more features, comprising of
        #   X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..
        #
        #   Inputs X1, X2 must be the same size
        #

        degree = 5
        out = np.ones(np.shape(X1))
        for i in range(1,degree+1):
            for j in range(i+1):
                out = np.vstack((out,(X1**(i-j))*(X2**j)))
        return np.transpose(out)


def plot_data(X, y, show = True):
    #PLOTDATA Plots the data points X and y into a new figure
    #   PLOTDATA(x,y) plots the data points with + for the positive examples
    #   and o for the negative examples. X is assumed to be a Mx2 matrix.

    # ====================== YOUR CODE HERE ======================
    # Instructions: Plot the positive and negative examples on a
    #               2D plot, using the option 'k+' for the positive
    #               examples and 'ko' for the negative examples.
    #

    # Find Indices of Positive and Negative Examples
    pos = (y==1).nonzero(); neg = (y == 0).nonzero()
    # Plot Examples
    X_pos = X.take(pos, axis=0)
    X_neg = X.take(neg, axis=0)
    plt.plot(X_pos[0,:,0], X_pos[0,:,1], 'b+', label='Data', markersize=10)
    plt.plot(X_neg[0,:,0], X_neg[0,:,1], 'ro', label='Data', linewidth=10)
    plt.xlabel('X1', fontsize=16)
    plt.ylabel('X2', fontsize=14)

    if show:
        plt.show()

def plot_decision_boundary(theta, X, y):
    plot_data(X[:,1:3], y, show = False)

    if X.shape[1] <= 3:
        # Only need 2 points to define a line, so choose two endpoints
        plot_x = np.asarray([min(X[:,1])-2,  max(X[:,1])+2])

        # Calculate the decision boundary line
        plot_y = (-1/theta[2])*(theta[1]*plot_x + theta[0])

        # Plot, and adjust axes for better viewing
        plt.plot(plot_x, plot_y)

        # Legend, specific for the exercise
        plt.legend(['Admitted', 'Not admitted', 'Decision Boundary'])
        plt.axis([30, 100, 30, 100])
        plt.show()

    else:
        # Here is the grid range
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)

        z = np.zeros((len(u), len(v)))
        # Evaluate z = theta*x over the grid
        for i in range(len(u)):
            for j in range(len(v)):
                z[i,j] = np.dot(map_feature(u[i], v[j]),theta)
        # Plot z = 0
        # Notice you need to specify the range [0, 0]
        plt.contour(u, v, z, [0, 0], linewidth = 2)
        plt.show()

def display_data(X):
    m = X.shape[0]
    n = X.shape[0]
    example_width = int(round(math.sqrt(n)))

    # Compute number of items to display
    display_rows = int(math.floor(math.sqrt(m)))
    display_cols = int(math.ceil(m / display_rows))

    # Between images padding
    pad = 1

    # Setup blank display
    display_array = - np.ones((pad + display_rows * (example_width + pad),
                           pad + display_cols * (example_width + pad)))

    # Copy each example into a patch on the display array
    curr_ex = 0
    for j in range(display_rows):
        for i in range(display_cols):
            if curr_ex >= m:
                break
            # Copy the patch

            # Get the max value of the patch
            max_val = max(abs(X[curr_ex, :]))
            row_start = pad + j*(example_width + pad)
            row_end = row_start+example_width

            col_start = pad + i*(example_width + pad)
            col_end = col_start+example_width
            display_array[row_start:row_end, col_start:col_end]= \
                            np.transpose(np.reshape(X[curr_ex, :], (example_width, example_width))) / max_val
            curr_ex = curr_ex + 1

        if curr_ex > m:
            break

    # cv2.imshow('image',display_array)
    # k = cv2.waitKey(0)

    plt.imshow(display_array, cmap = 'gray', interpolation = 'bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()
