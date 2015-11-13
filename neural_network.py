__author__ = 'yalguo'

import numpy as np
import math
import util
from util import load_data, feature_normalize, plot_data
from util import plot_decision_boundary, map_feature, load_mat
import scipy.optimize as op
import random
import matplotlib.pyplot as plt

class neutral_network:
    def nn_cost_function(self, nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lamda):
        #NNCOSTFUNCTION Implements the neural network cost function for a two layer
        #neural network which performs classification
        #   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
        #   X, y, lambda) computes the cost and gradient of the neural network. The
        #   parameters for the neural network are "unrolled" into the vector
        #   nn_params and need to be converted back into the weight matrices.
        #
        #   The returned parameter grad should be a "unrolled" vector of the
        #   partial derivatives of the neural network.
        #

        # Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
        # for our 2 layer neural network

        # Setup some useful variables
        theta1 = np.reshape(nn_params[:hidden_layer_size*(input_layer_size+1)],
                            (hidden_layer_size, (input_layer_size + 1)))

        theta2 = np.reshape(nn_params[hidden_layer_size*(input_layer_size+1):],
                            (num_labels, (hidden_layer_size + 1)))
        m = X.shape[0]

        # You need to return the following variables correctly
        J = 0

        # ====================== YOUR CODE HERE ======================
        # Instructions: You should complete the code by working through the
        #               following parts.
        #
        # Part 1: Feedforward the neural network and return the cost in the
        #         variable J. After implementing Part 1, you can verify that your
        #         cost function computation is correct by verifying the cost
        #         computed in ex4.m
        #
        # Part 2: Implement the backpropagation algorithm to compute the gradients
        #         Theta1_grad and Theta2_grad. You should return the partial derivatives of
        #         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
        #         Theta2_grad, respectively. After implementing Part 2, you can check
        #         that your implementation is correct by running checkNNGradients
        #
        #         Note: The vector y passed into the function is a vector of labels
        #               containing values from 1..K. You need to map this vector into a
        #               binary vector of 1's and 0's to be used with the neural network
        #               cost function.
        #
        #         Hint: We recommend implementing backpropagation using a for-loop
        #               over the training examples if you are implementing it for the
        #               first time.
        #
        # Part 3: Implement regularization with the cost function and gradients.
        #
        #         Hint: You can implement this around the code for
        #               backpropagation. That is, you can compute the gradients for
        #               the regularization separately and then add them to Theta1_grad
        #               and Theta2_grad from Part 2.
        #
        a1 = np.hstack((np.ones((m,1)), X))
        z2 = np.dot(a1, np.transpose(theta1))
        a2 = self.sigmoid(z2)
        ma2 = a2.shape[0]
        a2 = np.hstack((np.ones((ma2,1)), a2))
        z3 = np.dot(a2, np.transpose(theta2))
        h = a3 = self.sigmoid(z3)

        for i in range(num_labels):
            J += (np.dot(-(np.vectorize(int)(y==(i+1))),np.vectorize(math.log)(h[:,i]))+
                  np.dot(-(1-(np.vectorize(int)(y==(i+1)))), np.vectorize(math.log)(1-h[:,i])))/m

        temp_theta1 = theta1[:, 1:]
        temp_theta2 = theta2[:, 1:]
        J += lamda*(sum(sum(temp_theta1**2))+sum(sum(temp_theta2**2)))/(2*m)

        return J

    def gradient_function(self, nn_params, input_layer_size, hidden_layer_size,  num_labels, X, y, lamda):

        theta1 = np.reshape(nn_params[:hidden_layer_size*(input_layer_size+1)],
                            (hidden_layer_size, (input_layer_size + 1)))

        theta2 = np.reshape(nn_params[hidden_layer_size*(input_layer_size+1):],
                            (num_labels, (hidden_layer_size + 1)))
        m = X.shape[0]
        label_list = [i+1 for i in range(num_labels)]

        Theta1_grad = np.zeros(np.shape(theta1))
        Theta2_grad = np.zeros(np.shape(theta2))
        temp_theta1 = theta1[:, 1:]
        temp_theta2 = theta2[:, 1:]
        a1 = np.hstack((np.ones((m,1)), X))

        for t in range(m):
            a_1 =  a1[t,:]
            z_2 = np.dot(theta1, a_1)
            a_2 = self.sigmoid(z_2)
            a_2 = np.hstack((1,a_2))
            z_3 = np.dot(theta2, a_2)
            a_3 = self.sigmoid(z_3)

            delta_3 = (a_3-(np.vectorize(int)(y[t] == label_list)))
            delta_2 = np.dot(np.transpose(temp_theta2), delta_3)*self.sigmoid_gradient(z_2)
            #
            # delta_2 = np.reshape(delta_2,(np.size(delta_2),1))
            # a_1 = np.reshape(a_1,(np.size(a_1),1))
            # delta_3 = np.reshape(delta_3,(np.size(delta_3),1))
            # a_2 = np.reshape(a_2,(np.size(a_2),1))
            Theta1_grad += np.outer(delta_2, a_1)
            Theta2_grad += np.outer(delta_3, a_2)

        Theta1_grad = Theta1_grad/m
        Theta2_grad = Theta2_grad/m

        Theta1_grad[:,1:] += temp_theta1*lamda/m
        Theta2_grad[:,1:] += temp_theta2*lamda/m

        grad = np.hstack((np.reshape(Theta1_grad, (np.size(Theta1_grad),)), np.reshape(Theta2_grad, (np.size(Theta2_grad),))))
        # Unroll gradients
        return grad

    def compute_numerical_gradient(self, theta, input_layer_size, hidden_layer_size, num_labels, X, y, lamda):
        #COMPUTENUMERICALGRADIENT Computes the gradient using "finite differences"
        #and gives us a numerical estimate of the gradient.
        #   numgrad = COMPUTENUMERICALGRADIENT(J, theta) computes the numerical
        #   gradient of the function J around theta. Calling y = J(theta) should
        #   return the function value at theta.

        # Notes: The following code implements numerical gradient checking, and
        #        returns the numerical gradient.It sets numgrad(i) to (a numerical
        #        approximation of) the partial derivative of J with respect to the
        #        i-th input argument, evaluated at theta. (i.e., numgrad(i) should
        #        be the (approximately) the partial derivative of J with respect
        #        to theta(i).)
        #

        numgrad = np.zeros(np.shape(theta))
        perturb = np.zeros(np.shape(theta))
        e = 10**-4

        for i in range(theta.shape[0]):
            # Set perturbation vector
            perturb[i] = e
            loss1 = self.nn_cost_function(theta - perturb, input_layer_size, hidden_layer_size,num_labels, X, y, lamda)
            loss2 = self.nn_cost_function(theta + perturb, input_layer_size, hidden_layer_size,num_labels, X, y, lamda)
            # Compute Numerical Gradient
            numgrad[i] = (loss2 - loss1) / (2*e)
            perturb[i] = 0

        return numgrad

    def check_NN_gradients(self, lamda=0):
        #CHECKNNGRADIENTS Creates a small neural network to check the
        #backpropagation gradients
        #   CHECKNNGRADIENTS(lambda) Creates a small neural network to check the
        #   backpropagation gradients, it will output the analytical gradients
        #   produced by your backprop code and the numerical gradients (computed
        #   using computeNumericalGradient). These two gradient computations should
        #   result in very similar values.
        #
        input_layer_size = 3
        hidden_layer_size = 5
        num_labels = 3
        m = 5

        # We generate some 'random' test data
        Theta1 = self.debug_initialize_weights(hidden_layer_size, input_layer_size)
        Theta2 = self.debug_initialize_weights(num_labels, hidden_layer_size)
        # Reusing debugInitializeWeights to generate X
        X  = self.debug_initialize_weights(m, input_layer_size-1)
        y  = np.asarray([i%num_labels+1 for i in range(m)])

        nn_param = np.hstack((np.reshape(Theta1, (np.size(Theta1),)), np.reshape(Theta2, (np.size(Theta2),))))
        numgrad = self.compute_numerical_gradient(nn_param,input_layer_size, hidden_layer_size,num_labels,X,y,lamda)
        grad = self.gradient_function(nn_param, input_layer_size, hidden_layer_size, num_labels,X,y,lamda)
        # Visually examine the two gradient computations.  The two columns
        # you get should be very similar.
        print numgrad, '\n', grad
        print 'The above two columns you get should be very similar.', \
                 '(Left-Your Numerical Gradient, Right-Analytical Gradient)'

        # Evaluate the norm of the difference between two solutions.
        # If you have a correct implementation, and assuming you used EPSILON = 0.0001
        # in computeNumericalGradient.m, then diff below should be less than 1e-9
        diff = np.linalg.norm(numgrad-grad)/np.linalg.norm(numgrad+grad)

        print 'If your backpropagation implementation is correct, then\n',\
                 'the relative difference will be small (less than 1e-9).\n',\
                 'Relative Difference:', diff

    def predict_oneD(self, X, theta1, theta2):
        a1 = np.hstack((1, X))
        z2 = np.dot(a1, np.transpose(theta1))
        a2 = self.sigmoid(z2)
        a2 = np.hstack((1, a2))
        z3 = np.dot(a2, np.transpose(theta2))
        h = a3 = self.sigmoid(z3)
        return h.argmax(0) + 1

    def predict(self, theta1, theta2, X):
        m = X.shape[0]
        a1 = np.hstack((np.ones((m,1)), X))
        z2 = np.dot(a1, np.transpose(theta1))
        a2 = self.sigmoid(z2)
        ma2 = a2.shape[0]
        a2 = np.hstack((np.ones((ma2,1)), a2))
        z3 = np.dot(a2, np.transpose(theta2))
        h = a3 = self.sigmoid(z3)
        return h.argmax(1)+1

    def sigmoid(self, z):
        #SIGMOID Compute sigmoid functoon
        #   J = SIGMOID(z) computes the sigmoid of z.

        # You need to return the following variables correctly

        # ====================== YOUR CODE HERE ======================
        # Instructions: Compute the sigmoid of each value of z (z can be a matrix,
        #               vector or scalar).

        g=(1+np.vectorize(math.exp)(-z))**-1
        return g

    def sigmoid_gradient(self, z):
        #SIGMOIDGRADIENT returns the gradient of the sigmoid function
        #evaluated at z
        #   g = SIGMOIDGRADIENT(z) computes the gradient of the sigmoid function
        #   evaluated at z. This should work regardless if z is a matrix or a
        #   vector. In particular, if z is a vector or matrix, you should return
        #   the gradient for each element.

        # ====================== YOUR CODE HERE ======================
        # Instructions: Compute the gradient of the sigmoid function evaluated at
        #               each value of z (z can be a matrix, vector or scalar).

        g=self.sigmoid(z)*(1-self.sigmoid(z))

        return g

    def rand_initialize_weights(self, L_in, L_out):
        #RANDINITIALIZEWEIGHTS Randomly initialize the weights of a layer with L_in
        #incoming connections and L_out outgoing connections
        #   W = RANDINITIALIZEWEIGHTS(L_in, L_out) randomly initializes the weights
        #   of a layer with L_in incoming connections and L_out outgoing
        #   connections.
        #
        #   Note that W should be set to a matrix of size(L_out, 1 + L_in) as
        #   the column row of W handles the "bias" terms
        #

        # You need to return the following variables correctly

        # ====================== YOUR CODE HERE ======================
        # Instructions: Initialize W randomly so that we break the symmetry while
        #               training the neural network.
        #
        # Note: The first row of W corresponds to the parameters for the bias units
        #
        epsilon_init = 0.12
        W = (np.random.rand(L_out, 1+L_in)*2-1)*epsilon_init

        return W

    def debug_initialize_weights(self, fan_out, fan_in):
        #DEBUGINITIALIZEWEIGHTS Initialize the weights of a layer with fan_in
        #incoming connections and fan_out outgoing connections using a fixed
        #strategy, this will help you later in debugging
        #   W = DEBUGINITIALIZEWEIGHTS(fan_in, fan_out) initializes the weights
        #   of a layer with fan_in incoming connections and fan_out outgoing
        #   connections using a fix set of values
        #
        #   Note that W should be set to a matrix of size(1 + fan_in, fan_out) as
        #   the first row of W handles the "bias" terms
        #

        # Set W to zeros
        W = np.asarray([math.sin(i)/10 for i in range(1,fan_out*(1+fan_in)+1)])

        # Initialize W using "sin", this ensures that W is always of the same
        # values and will be useful for debugging
        W = np.reshape(W, (fan_out, 1+fan_in))

        return W


def display_and_predict(X, m, nn, theta1, theta2, y):
    pred = nn.predict(theta1, theta2, X)
    accuracy = (np.vectorize(float)(pred == y)).mean()
    print "Accuracy:", accuracy
    ran_list = [i for i in range(m)]
    random.shuffle(ran_list)
    for i in range(m):
        # Display
        # util.display_data(X[ran_list[i], :])
        Xi = X[ran_list[i], :]
        n = Xi.shape[0]
        width = int(round(math.sqrt(n)))

        image = -np.ones((width, width))
        max_val = max(abs(Xi))
        image = np.transpose(np.reshape(Xi, (width, width))) / max_val

        plt.imshow(image, cmap='gray', interpolation='bicubic')
        plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        plt.show()

        pred_one = nn.predict_oneD(theta1, theta2, Xi)
        print 'Neural Network Prediction:', pred_one % 10
        #
        # % Pause
        # fprintf('Program paused. Press enter to continue.\n');
        # pause;


def test_predict():
    nn = neutral_network()
    (X, y) = load_mat('lg_one_vs_all_data.mat', 'X', 'y')
    (theta1, theta2) = load_mat('nn_predict.mat', 'Theta1', 'Theta2')
    m = X.shape[0]
    y = np.reshape(y,(m,))
    display_and_predict(X, m, nn, theta1, theta2, y)

def test_nn():
    nn = neutral_network()
    # Setup the parameters you will use for this exercise
    input_layer_size  = 400  # 20x20 Input Images of Digits
    hidden_layer_size = 25   # 25 hidden units
    num_labels = 10          # 10 labels, from 1 to 10
                             # (note that we have mapped "0" to label 10)
    (X, y) = load_mat('lg_one_vs_all_data.mat', 'X', 'y')
    (theta1, theta2) = load_mat('nn_predict.mat', 'Theta1', 'Theta2')
    nn_param = np.hstack((np.reshape(theta1, (np.size(theta1),)), np.reshape(theta2, (np.size(theta2),))))
    m = X.shape[0]
    y = np.reshape(y,(m,))

    lamda = 0
    J = nn.nn_cost_function(nn_param, input_layer_size, hidden_layer_size, num_labels, X, y, lamda)

    print 'When lamda is 0, Cost at parameters (loaded from ex4weights):', J
    print '(this value should be about 0.287629)'

    lamda = 1
    J = nn.nn_cost_function(nn_param, input_layer_size, hidden_layer_size, num_labels, X, y, lamda)

    print 'When lamda is 1, Cost at parameters (loaded from ex4weights):', J
    print '(this value should be about 0.383770)'

    g = nn.sigmoid_gradient(np.asarray([1, -0.5, 0, 0.5, 1]))
    print 'Sigmoid gradient evaluated at [1 -0.5 0 0.5 1]:', g

    initial_theta1 = nn.rand_initialize_weights(input_layer_size, hidden_layer_size)
    initial_theta2 = nn.rand_initialize_weights(hidden_layer_size, num_labels)
    Initial_nn_param = np.hstack((np.reshape(initial_theta1, (np.size(initial_theta1),)), np.reshape(initial_theta2, (np.size(initial_theta2),))))

    nn.check_NN_gradients()

    lamda = 3
    nn.check_NN_gradients(lamda)

    debug_J = nn.nn_cost_function(nn_param, input_layer_size, hidden_layer_size, num_labels, X, y, lamda)
    print '\nCost at (fixed) debugging parameters (w/ lambda = 10): ', debug_J, \
         '\n(this value should be about 0.576051)\n\n'
    Result = op.minimize(fun = nn.nn_cost_function, x0 = Initial_nn_param, args = (input_layer_size, hidden_layer_size,
                               num_labels, X, y, lamda), method = 'TNC', jac = nn.gradient_function)
    optimal_theta = Result.x

    Theta1 = np.reshape(optimal_theta[:hidden_layer_size*(input_layer_size+1)],
                            (hidden_layer_size, (input_layer_size + 1)))

    Theta2 = np.reshape(optimal_theta[hidden_layer_size*(input_layer_size+1):],
                            (num_labels, (hidden_layer_size + 1)))


    display_and_predict(X, m, nn, Theta1, Theta2, y)

def test_lung_data():
    nn = neutral_network()
    (X, y) = load_data(r'nn_lung_data.txt')
    (X, mu, sigma) = feature_normalize(X)
    m = X.shape[0]
    # y = np.reshape(y,(m,))
    # Setup the parameters you will use for this exercise
    input_layer_size  = X.shape[1]
    hidden_layer_size = 10   # 25 hidden units
    num_labels = 4          # 10 labels, from 1 to 10
                             # (note that we have mapped "0" to label 10)
    lamda=0.01
    initial_theta1 = nn.rand_initialize_weights(input_layer_size, hidden_layer_size)
    initial_theta2 = nn.rand_initialize_weights(hidden_layer_size, num_labels)
    Initial_nn_param = np.hstack((np.reshape(initial_theta1, (np.size(initial_theta1),)), np.reshape(initial_theta2, (np.size(initial_theta2),))))
    Result = op.minimize(fun = nn.nn_cost_function, x0 = Initial_nn_param, args = (input_layer_size, hidden_layer_size,
                               num_labels, X, y, lamda), method = 'TNC', jac = nn.gradient_function)
    optimal_theta = Result.x

    theta1 = np.reshape(optimal_theta[:hidden_layer_size*(input_layer_size+1)],
                            (hidden_layer_size, (input_layer_size + 1)))

    theta2 = np.reshape(optimal_theta[hidden_layer_size*(input_layer_size+1):],
                            (num_labels, (hidden_layer_size + 1)))

    (X_test, y_test) = load_data(r'nn_lung_data_test.txt')

    (X_test, mu, sigma) = feature_normalize(X_test)
    pred = nn.predict(theta1, theta2, X_test)
    accuracy = (np.vectorize(float)(pred == y_test)).mean()
    print "Accuracy:", accuracy

if __name__=="__main__":
    # test_predict()
    # test_nn()
    test_lung_data()