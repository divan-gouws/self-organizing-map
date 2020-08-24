# 1. Initialize the weights of each node to small standardized random values (between 0 and 1)
#    - what does a node look like? there should be as many weights as input dimensions?
# 2. Choose a random vector from the training set and present it to the lattice
# 3. Examine every node's weight and determine which is most like the input vector. This is the BMU
# 4. Calculate the radius of the neighbourhood of the BMU
# 5. Weights of nodes within the radius are updated to make them more like the BMU. Closer nodes are adjusted more
# 6. Return to step 2 for N iterations

import math
import numpy as np
import random
random.seed(42)

class SOM:
    '''
    params
    =====
    num_rows = number of rows in map
    num_cols = number of columns in map
    map = a numpy array with shape and nesting defined as above
    input_dim = dimension of input vector
    sigma = initial radius
    learning_rate = proportionality coefficient for weight updates
    '''
    def __init__(self, num_rows, num_cols, input_dim, radius, learning_rate):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.input_dim = input_dim
        self.radius = radius
        self.learning_rate = learning_rate
        # create a map as a nested array : weights within nodes within rows. nodes are positions in a row
        self.map = np.random.rand(num_rows, num_cols, input_dim)

    def euclidean_distance(self, vector_1, vector_2):
        return np.linalg.norm(x - y)

    def calculate_decay(self, iteration, num_iterations):
        return np.exp(-iteration / num_iterations)

    def calculate_neighbour_penalty(self, distance, sigma):
        return np.exp(- distance ** 2 / (2 * (sigma ** 2)))

    def find_bmu(self, current_vector):
        # create a variable to keep track of the smallest distance, this is used to find the best matching unit (bmu)
        smallest_distance = None
        # iterate over all rows in the map
        for rownum, row in enumerate(self.map):
            # iterate over all nodes in the current row of the map
            for colnum, node in enumerate(row):
                # calculate the distance between the nodes' weights and the input vector
                distance = self.euclidean_distance(current_vector, node)
                # check if the node is the first one or if it's closer to the input vector than all prior nodes
                if (smallest_distance is None) or (distance < smallest_distance):
                    # update the shortest distance to the new shortest distance found
                    smallest_distance = distance
                    # set this node as the best matching unit (bmu)
                    bmu = node #this is not used further, only its position is used to place the radius
                    # track the bmu's position for use in radius calculations
                    bmu_position = np.array([rownum, colnum])
        return bmu_position

    def update_weights(self, current_vector, bmu_position):
        # iterate over all rows in the map 
        for rownum, row in enumerate(self.map):
            # iterate over all nodes in the current row of the map 
            for colnum, node in enumerate(row):
                # find the current position of the node
                current_position = np.array([rownum, colnum])
                # get distance from current node to bmu
                distance_to_bmu = euclidean_distance(current_position, bmu_position)
                # check whether current node is in the radius of the bmu
                if distance_to_bmu <= self.sigma:
                    # set the penalty coefficient proportional to how far the current (in radius) node is to the bmu
                    node_penalty = calculate_neighbour_penalty(distance_to_bmu, self.radius)
                    # update the node's weights
                    self.map[rownum][colnum] += node_penalty * self.learning_rate * (current_vector - node)

    def train(self, train_data, num_iter):
        # to choose a random vector from the training set, find a random number between 0 and the length of the training set
        random_range = len(train_data)
        # an iteration is an instance where a training vector gets presented to the network
        for iter in range(num_iter):
            # select a random vector from the training set
            current_vector = train_data[random.randrange(0, random_range)]
            # get position of bmu
            bmu_position = self.find_bmu(current_vector)
            # update sigma based on iteration - that is, the radius shrinks per iteration
            self.radius *= calculate_decay(iter, num_iter)
            # the learning rate also decays with each iteration
            self.learning_rate *= calculate_decay(iter, num_iter)
            # update weights in radius of bmu position using current training vector
            self.update_weights(current_vector, bmu_position)