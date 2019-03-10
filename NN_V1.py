import numpy as np
import matplotlib.pyplot as plt
import math

# Handwritten neural network with numpy, math and matplotlib libraries only
# Functionalities:         
#         weight initialization: "random", "he", "xavier" or "heuristic"
#         activation function: "sigmoid", "relu" or "leaky_relu"
#         gradient descent: "batch", "mini batch" or "stochastic"
#         optimization: "gradient descent", "momentum", "RMSProp" or "adam"
#         regulation: "L2", "dropout" or "batch normalization"
#         early stoping: stop training when cost is low


class NN:
        
    def __init__(self, L_dim, 
             initialization = "random",activation = "relu",
             learning_rate = 0.01, num_iterations = 3000, early_stop=True, cost_stop=0.005,
             batch=False,mini_batch=False, mini_batch_size=0, 
             optimizer="gd", beta = 0.9, beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8, 
             lambd=0, keep_prob = 1,
             batchnormalization=False,
             print_cost=False, print_cost_every_n_iterations=10):
        
        
        """
        Initialize a class for a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.

        Arguments:
        X -- input data (number of features, number of examples)
        Y -- label (1, number of examples)
        L_dim -- list contains each layer's neurals numbers
        initialization -- choices of initialization to use ("random", "he", "xavier" or "heuristic")
        activation -- activation function "sigmoid", "relu" or "leaky_relu"
        learning_rate -- learning rate of the gradient descent 
        num_iterations -- number of iterations of the optimization loop
        batch -- enable batch gradient descent (boolean)
        mini_batch -- enable mini batch gradient descent (boolean)
        mini_batch_size -- size of the mini-batches (int)
        optimizer -- choices of optimization ('gd', 'momentum', 'RMSProp' or 'adam')
        beta -- eomentum hyperparameter
        beta1 -- exponential decay hyperparameter for the past gradients estimates 
        beta2 -- exponential decay hyperparameter for the past squared gradients estimates 
        epsilon -- hyperparameter for preventing division by zero in Adam updates
        lambd -- L2 regulation (0 or other number)
        keep_prob - dropout regulation, probability of keeping a neuron
        batchnormalization -- enable batch normalization (boolean)
        print_cost -- if True, it prints the cost every 100 steps
        print_cost_every_n_iterations -- print cost every n iterations (int)
        """
        
        self.L_dim = L_dim
        self.initialization = initialization 
        self.activation = activation 
        self.learning_rate = learning_rate 
        self.num_iterations = num_iterations 
        self.early_stop = early_stop
        self.cost_stop = cost_stop
        self.batch = batch 
        self.mini_batch = mini_batch
        self.mini_batch_size = mini_batch_size
        self.optimizer = optimizer
        self.beta = beta
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.lambd = lambd
        self.keep_prob = keep_prob
        self.batchnormalization = batchnormalization
        self.print_cost = print_cost
        self.print_cost_every_n_iterations = print_cost_every_n_iterations
        return

    
    def fit(self, X_train, Y_train):

        """
        Implement a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID
            by updating parameters -- parameters learnt by the model

        Arguments:
        X_train -- input data (number of features, number of examples)
        y_train -- label (1, number of examples)
        """
        
        assert X_train.shape[1] == Y_train.shape[1], \
        'size of X_train and must be equal to that of Y_train'
    
        costs = []                         

        # Parameters initialization: self.parameters
        self.initialize_NN_parameters()

        # Initialize the optimizer for 'momentum', 'RMSProp' or 'adam': self.v, self.s
        self.initialize_optimizer()
        self.t = 0          # counter for Adam update


        if self.mini_batch == False and self.batch == False:
            # gradient descent
            print('Stochastic Gradient Descent...')
            for i in range(0, self.num_iterations):
                total_cost = 0
                for j in range(0, X_train.shape[1]):
                    # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
                    AL = self.NN_forward(X_train[:,j].reshape(-1, 1))

                    # Compute cost.
                    cost = self.compute_cost(AL, Y_train[:,j].reshape(-1, 1))

                    # Backward propagation.
                    self.NN_backward(Y_train[:,j].reshape(-1, 1),AL)

                    # Update parameters.
                    self.t+=1
                    self.gradient_descent()

                    total_cost += cost
                    costs.append(cost)

                # Print the cost every print_cost_every_n_iterations training example
                if self.print_cost and i % self.print_cost_every_n_iterations == 0:
                    print ("Total cost after iteration %i: %f" %(i, total_cost/(X_train.shape[1])))
                if self.early_stop == True and total_cost/(math.floor(X_train.shape[1]/(self.mini_batch_size)))<self.cost_stop:
                    break
                    
        elif self.batch == True:
            # batch
            print('Batch Gradient Descent...')
            for i in range(0, self.num_iterations):

                # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
                AL = self.NN_forward(X_train)

                # Compute cost.
                cost = self.compute_cost(AL, Y_train)

                # Backward propagation.
                self.NN_backward(Y_train, AL)

                # Update parameters.
                self.t+=1
                self.gradient_descent()

                # Print the cost every print_cost_every_n_iterations training example
                if self.print_cost and i % self.print_cost_every_n_iterations == 0:
                    print ("Cost after iteration %i: %f" %(i, cost))
                    costs.append(cost)
                if self.early_stop == True and total_cost/(math.floor(X_train.shape[1]/(self.mini_batch_size)))<self.cost_stop:
                    break
                    
        elif self.mini_batch == True:
            # mini batch
            print('Mini Batch Gradient Descent...')
            mini_batches = self.initialize_mini_batches(X_train, Y_train)
            for i in range(0, self.num_iterations):
                total_cost = 0
                for j in range(0, len(mini_batches)):
                    # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
                    AL = self.NN_forward(mini_batches[j][0])

                    # Compute cost.
                    cost = self.compute_cost(AL, mini_batches[j][1])

                    # Backward propagation.
                    self.NN_backward(mini_batches[j][1],AL)

                    # Update parameters.
                    self.t+=1
                    self.gradient_descent()

                    total_cost += cost
                    costs.append(cost)

                # Print the cost every print_cost_every_n_iterations training example
                if self.print_cost and i % self.print_cost_every_n_iterations == 0:
                    print ("Total cost after iteration %i: %f" %(i, total_cost/(math.floor(X_train.shape[1]/(self.mini_batch_size)))))
                if self.early_stop == True and total_cost/(math.floor(X_train.shape[1]/(self.mini_batch_size)))<self.cost_stop:
                    break
        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(self.learning_rate))
        plt.show()
    
        return self
    
    
    def initialize_NN_parameters(self):

        """
        Initialize parameters for a L-layer neural networ
        
        Needed Class Parameters:
        L_dim -- list contains each layer's neurals numbers
        initialization -- choices of initialization to use ("random", "he", "xavier" or "heuristic")

        Class Parameters Changed:
        parameters -- set contains parameters "W1", "b1", ..., "WL", "bL":
                            Wn -- weight matrix (L_dim[n], L_dim[n-1])
                            bn -- bias vector (L_dim[n], 1)
                            gamman -- batch normalization scale vector (L_dim[n], 1)
                            betan -- batch normalization shift vector (L_dim[n], 1)
        """

        # He initialization: https://arxiv.org/pdf/1502.01852.pdf
        self.parameters = {}
        L = len(self.L_dim) - 1 # integer representing the number of layers

        if self.initialization == "random":
            for l in range(1, L + 1):
                self.parameters['W' + str(l)] = np.random.randn(self.L_dim[l], self.L_dim[l-1]) *0.01
                self.parameters['b' + str(l)] = np.zeros((self.L_dim[l], 1))    
        elif self.initialization == "he":  # good for relu activation function
            for l in range(1, L + 1):
                self.parameters['W' + str(l)] = np.random.randn(self.L_dim[l], self.L_dim[l-1]) * np.sqrt(2/self.L_dim[l-1])
                self.parameters['b' + str(l)] = np.zeros((self.L_dim[l], 1))
        elif self.initialization == "xavier":  # good for tanh activation function
            for l in range(1, L + 1):
                self.parameters['W' + str(l)] = np.random.randn(self.L_dim[l], self.L_dim[l-1]) * np.sqrt(1/self.L_dim[l-1])
                self.parameters['b' + str(l)] = np.zeros((self.L_dim[l], 1))            
        elif self.initialization == "heuristic": 
            for l in range(1, L + 1):
                self.parameters['W' + str(l)] = np.random.randn(self.L_dim[l], self.L_dim[l-1]) * np.sqrt(1/(self.L_dim[l-1]+self.L_dim[l]))
                self.parameters['b' + str(l)] = np.zeros((self.L_dim[l], 1))       

        if self.batchnormalization == True:
            for l in range(1, L + 1):
                self.parameters['gamma'+str(l)] = np.random.randn(self.L_dim[l], 1)
                self.parameters['beta' + str(l)] = np.zeros((self.L_dim[l], 1))  

        return     

    
    def initialize_optimizer(self) :

        """
        Initialize parameters for optimizer
        
        Needed Class Parameters:
        L_dim -- list contains each layer's neurals numbers
        parameters -- set contains parameters "W1", "b1", ..., "WL", "bL":
                            Wn -- weight matrix (L_dim[n], L_dim[n-1])
                            bn -- bias vector (L_dim[n], 1)
                            gamman -- batch normalization scale vector (L_dim[n], 1)
                            betan -- batch normalization shift vector (L_dim[n], 1)

        Class Parameters Changed:
        v -- set contains the exponentially weighted average of the gradient.
                        v["dW" + str(l)]
                        v["db" + str(l)] 
                        v["dgamma" + str(l)]
                        v["dbeta" + str(l)] 
        s -- set contains the exponentially weighted average of the squared gradient.
                        s["dW" + str(l)]
                        s["db" + str(l)]
                        v["dgamma" + str(l)]
                        v["dbeta" + str(l)] 
        """

        L = len(self.L_dim) - 1 # integer representing the number of layers
        self.v = {}
        self.s = {}

        for l in range(L):
            self.v["dW" + str(l+1)] = np.zeros((self.parameters["W" + str(l+1)].shape[0],self.parameters["W" + str(l+1)].shape[1]))
            self.v["db" + str(l+1)] = np.zeros((self.parameters["b" + str(l+1)].shape[0],self.parameters["b" + str(l+1)].shape[1]))
            self.s["dW" + str(l+1)] = np.zeros((self.parameters["W" + str(l+1)].shape[0],self.parameters["W" + str(l+1)].shape[1]))
            self.s["db" + str(l+1)] = np.zeros((self.parameters["b" + str(l+1)].shape[0],self.parameters["b" + str(l+1)].shape[1]))
        
        if self.batchnormalization == True:
            for l in range(L):
                self.v["dgamma" + str(l+1)] = np.zeros((self.parameters["gamma" + str(l+1)].shape[0],self.parameters["gamma" + str(l+1)].shape[1]))
                self.v["dbeta" + str(l+1)] = np.zeros((self.parameters["beta" + str(l+1)].shape[0],self.parameters["beta" + str(l+1)].shape[1]))
                self.s["dgamma" + str(l+1)] = np.zeros((self.parameters["gamma" + str(l+1)].shape[0],self.parameters["gamma" + str(l+1)].shape[1]))
                self.s["dbeta" + str(l+1)] = np.zeros((self.parameters["beta" + str(l+1)].shape[0],self.parameters["beta" + str(l+1)].shape[1]))
        return
    

    def initialize_mini_batches(self,X, Y):

        """    
        Initialize datasets for mini batches
        
        Arguments:
        X -- input data (number of features, number of examples)
        Y -- label (1, number of examples)
        
        Needed Class Parameters:
        mini_batch_size -- size of the mini-batches (int)

        Returns:
        mini_batches -- list of set for mini batch gradient descent (mini_batch_X, mini_batch_Y)
        """
        np.random.seed(0) 
        m = X.shape[1]                  # number of training examples
        mini_batches = []

        # randomize indexes
        permutation = list(np.random.permutation(m))
        shuffled_X = X[:, permutation]
        shuffled_Y = Y[:, permutation]

        num_mini_batches = math.floor(m/(self.mini_batch_size))   # number of mini batches

        for i in range(num_mini_batches):
            mini_batches.append((shuffled_X[:, i*self.mini_batch_size : (i+1)*self.mini_batch_size],\
                                 shuffled_Y[:, i*self.mini_batch_size : (i+1)*self.mini_batch_size]))

        if m % self.mini_batch_size != 0:
            mini_batches.append((shuffled_X[:, num_mini_batches *self.mini_batch_size : ],\
                                shuffled_Y[:, num_mini_batches *self.mini_batch_size : ]))

        return mini_batches


    def forward(self, A_prev, D_prev, l):

        """
        Forword propogation for one layer
        
        Arguments:
        A_prev -- activation from the previous layer (size of previous layer, number of examples)
        D_prev -- dropout matrix D from the previous layer (size of previous layer, number of examples)
        l -- layer number
        
        Needed Class Parameters:
        parameters -- set contains parameters "W1", "b1", ..., "WL", "bL":
                        Wn -- weight matrix (L_dim[n], L_dim[n-1])
                        bn -- bias vector (L_dim[n], 1)
                        gamman -- batch normalization scale vector (L_dim[n], 1)
                        betan -- batch normalization shift vector (L_dim[n], 1)
        activation -- activation function "sigmoid" or "relu"
        keep_prob - dropout regulation, probability of keeping a neuron
        batchnormalization -- enable batch normalization (boolean)

        Returns:
        A --output of the activation function (size of current layer, number of examples)
        cache -- set contains  (A_prev, W, b, Z, D_prev), for backward propagation,
                               or (A_prev, (Z, D_prev, gamma, sigma_squared, Z_norm, eplison)) for batch normalization
        """

        W = self.parameters['W'+str(l+1)]
        b = self.parameters['b'+str(l+1)]
        Z = np.dot(W,A_prev)+b
        cache = (A_prev, W, b)

        if self.batchnormalization == True:
            gamma = self.parameters['gamma'+str(l+1)]
            beta = self.parameters['beta'+str(l+1)]
            mu = np.average(Z, axis=1).reshape(Z.shape[0],-1)
            sigma_squared = np.average((Z-mu)**2, axis=1).reshape(Z.shape[0],-1)
            eplison = 1e-8 
            Z_norm = (Z - mu)/np.sqrt(sigma_squared + eplison)
            Z = gamma*Z_norm + beta

        assert (Z.shape == (W.shape[0], A_prev.shape[1]))

        if self.activation == "sigmoid" or l+1 == len(self.L_dim) - 1  :
            A = 1/(1+np.exp(-Z))

        elif self.activation == "relu":
            A = np.maximum(0,Z)
            
        elif self.activation == "leaky_relu":
            A = np.maximum(0.01*Z,Z)

        
        if l+1 < len(self.L_dim) - 1  :
            D = np.random.rand(A.shape[0], A.shape[1])      # initialize dropout matrix D
            D = D < self.keep_prob                          # convert entries of D to 0 or 1 (using keep_prob as the threshold)
            A = A * D                                       # dropout neurals
            A = A / self.keep_prob                           # scale the value of neurons back to the non-shutoff version
        else:
            D = np.random.rand(A.shape[0], A.shape[1])      
            D = D < 1                       

#         print('A:',A,'Z:',Z)
        if self.batchnormalization == True:
            cache = (cache, (Z, D_prev, gamma, sigma_squared, Z_norm, eplison))  
        else:
            cache = (cache, (Z,D_prev))  

        assert (A.shape == Z.shape )

        return A, D, cache
    

    def NN_forward(self, X):

        """
        Forword propogation for all layers
        
        Arguments:
        X -- input data, np array (number of features, number of examples)
        
        Needed Class Parameters:
        parameters -- set contains parameters "W1", "b1", ..., "WL", "bL":
                        Wn -- weight matrix (L_dim[n], L_dim[n-1])
                        bn -- bias vector (L_dim[n], 1)
                        gamman -- batch normalization scale vector (L_dim[n], 1)
                        betan -- batch normalization shift vector (L_dim[n], 1)
        activation -- activation function "sigmoid" or "relu"
        keep_prob - dropout regulation, probability of keeping a neuron
        batchnormalization -- enable batch normalization (boolean)

        Returns:
        AL -- output of the last layer (1, number of examples)
        
        Class Parameters Changed:
        caches -- list of caches (A_prev, W, b, Z, D) (from 0 to L-1)
        """

        L = len(self.L_dim) - 1 # integer representing the number of layers
        
        self.caches = []
        A = X
        D_prev = np.ones_like(A)
        for l in range(L-1):
            A, D_prev, cache = self.forward(A, D_prev, l)
            self.caches.append(cache)

        # Output layer
        AL, D_prev, cache = self.forward(A, D_prev, L-1) #last layer, no dropout
        self.caches.append(cache)

        assert (AL.shape[1] == X.shape[1] )

        return AL


    def compute_cost(self, AL, Y):

        """
        Compute log cross-entropy cost
        
        Arguments:
        AL -- output of the last layer (1, number of examples)
        Y -- labels (1, number of examples)
        
        Needed Class Parameters:
        parameters -- parameters learnt by the model
        lambd -- L2 regulation (0 or other number)

        Returns:
        cost -- log cross-entropy cost
        """

        Y = Y.reshape(AL.shape) #Y should be the same shape as AL
        m = Y.shape[1]

        cost = -1/m*np.sum(Y*np.log(AL) + (1-Y)*np.log(1-AL), axis=1)
        assert(cost.shape[0] == Y.shape[0])

        if self.lambd != 0:
            L = len(self.L_dim) - 1 # integer representing the number of layers
            for l in range(L):
                cost = cost + 1/m * self.lambd/2 * np.sum(np.square(self.parameters['W'+str(l+1)])) 
#         print(cost[0])

        return np.sum(cost)

    
    def backward(self, dA, cache, l):

        """
        Back propogation for one layer
        
        Arguments:
        dA -- post-activation gradient
        cache -- set contains  (A_prev, W, b, Z, D), for backward propagation
        l -- layer number
        
        Needed Class Parameters:
        activation -- activation function "sigmoid" or "relu"
        lambd -- L2 regulation (0 or other number)
        keep_prob - dropout regulation, probability of keeping a neuron
        batchnormalization -- enable batch normalization (boolean)


        Returns:
        dA_prev -- Gradient of the current activation (of the previous layer l-1), same shape as A_prev
        dW -- Gradient of W, same shape as W
        db -- Gradient of b, same shape as b
        dgamma -- Gradient of gamma, same shape as gamma
        dbeta -- Gradient of beta, same shape as beta     
        """

        if self.batchnormalization == True:
            cache, (Z, D, gamma, sigma_squared, Z_norm, eplison) = cache
        else:
            cache, (Z,D) = cache
            

        if self.activation == "sigmoid" or l+1 == len(self.L_dim) - 1:
            s = 1/(1+np.exp(-Z))
            dZ = dA * s * (1-s)
        elif self.activation == "relu":
            dZ = np.array(dA, copy=True) 
            dZ[Z <= 0] = 0
        elif self.activation == "leaky_relu":
            dZ = np.array(dA, copy=True) 
            dZ[Z <= 0] = 0.01        
        
        A_prev, W, b=cache
        m = A_prev.shape[1]
        
        if self.batchnormalization == True:
            
            dZ_norm = dZ * gamma
            
            dbeta = np.sum(dZ, axis=1).reshape(-1,1)
            dgamma = np.sum(Z_norm*dZ, axis=1).reshape(-1,1)
            dZ = np.divide( (m*dZ_norm-np.sum(dZ_norm, axis=1).reshape(-1,1) - Z_norm*np.sum(dZ_norm*Z_norm, axis=1).reshape(-1,1)), m*np.sqrt(sigma_squared + eplison))
           
        dW = 1/m * np.dot(dZ,A_prev.T) + (self.lambd * W) / m
        db = 1/m * np.reshape(np.sum(dZ, axis=1), (-1, 1))
        dA_prev = np.dot(W.T, dZ)
        
        if l+1 < len(self.L_dim) - 1  :
            dA_prev = dA_prev * D               # dropdout
            dA_prev = dA_prev / self.keep_prob  # scale the value of neurons back to the non-shutoff version

        assert (dA_prev.shape == A_prev.shape )
        assert (dW.shape == W.shape )
        assert (db.shape == b.shape )
        
        if self.batchnormalization == True:
            assert (dgamma.shape == gamma.shape )
            assert (dbeta.shape == beta.shape )
            return dA_prev, dW, db, dgamma, dbeta
        
        return dA_prev, dW, db


    def NN_backward(self, Y, AL):

        """
        Back propogation for all layers
        
        Arguments:
        Y -- labels (1, number of examples)
        AL -- output of the last layer (1, number of examples)
        
        Needed Class Parameters:
        caches -- list of caches (A_prev, W, b, Z, D) (from 0 to L-1)
        activation -- activation function "sigmoid" or "relu"
        lambd -- L2 regulation (0 or other number)
        keep_prob - dropout regulation, probability of keeping a neuron
        batchnormalization -- enable batch normalization (boolean)

        Class Parameters Changed:
        grads -- set with the gradients
                 grads["dA" + str(l)] = ... 
                 grads["dW" + str(l)] = ...
                 grads["db" + str(l)] = ... 
                 grads["dgamma" + str(l)] = ...
                 grads["dbeta" + str(l)] = ... 
        """

        self.grads = {}

        L = len(self.L_dim) - 1 # integer representing the number of layers
        m = AL.shape[1]
        Y = Y.reshape(AL.shape) #Y should be the same shape as AL
        dAL =  -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

        if self.batchnormalization == True:
            # Lth layer:
            self.grads["dA" + str(L-1)], self.grads["dW" + str(L)], self.grads["db" + str(L)], self.grads["dgamma" + str(L)], self.grads["dbeta" + str(L)] = self.backward(dAL, self.caches[L-1], L-1)

            # from l=L-2 to l=0
            for l in reversed(range(L-1)):
                self.grads["dA" + str(l)], self.grads["dW" + str(l+1)], self.grads["db" + str(l+1)], self.grads["dgamma" + str(l+1)], self.grads["dbeta" + str(l+1)] = self.backward(self.grads["dA" + str(l+1)], self.caches[l], l)
        else:
            # Lth layer:
            self.grads["dA" + str(L-1)], self.grads["dW" + str(L)], self.grads["db" + str(L)] = self.backward(dAL, self.caches[L-1], L-1)

            # from l=L-2 to l=0
            for l in reversed(range(L-1)):
                self.grads["dA" + str(l)], self.grads["dW" + str(l+1)], self.grads["db" + str(l+1)] \
                    = self.backward(self.grads["dA" + str(l+1)], self.caches[l], l)

        return 
    
    

    def gradient_descent(self):

        """
        Gradient descent
        
        Needed Class Parameters:
        parameters -- set contains parameters "W1", "b1", ..., "WL", "bL":
                        Wn -- weight matrix (L_dim[n], L_dim[n-1])
                        bn -- bias vector (L_dim[n], 1)
                        gamman -- batch normalization scale vector (L_dim[n], 1)
                        betan -- batch normalization shift vector (L_dim[n], 1)   
        grads -- set with the gradients
                 grads["dA" + str(l)] = ... 
                 grads["dW" + str(l)] = ...
                 grads["db" + str(l)] = ... 
                 grads["dgamma" + str(l)] = ...
                 grads["dbeta" + str(l)] = ... 
        learning_rate -- learning rate of the gradient descent
        optimizer -- choices of optimization ('gd', 'momentum', 'RMSProp' or 'adam')
        beta -- momentum hyperparameter
        beta1 -- exponential decay hyperparameter for the past gradients estimates 
        beta2 -- exponential decay hyperparameter for the past squared gradients estimates 
        epsilon -- hyperparameter for preventing division by zero in Adam updates
        v -- set contains the exponentially weighted average of the gradient.
                        v["dW" + str(l)]
                        v["db" + str(l)] 
                        v["dgamma" + str(l)]
                        v["dbeta" + str(l)] 
        s -- set contains the exponentially weighted average of the squared gradient.
                        s["dW" + str(l)]
                        s["db" + str(l)]
                        v["dgamma" + str(l)]
                        v["dbeta" + str(l)] 
        t -- counter for adam update
        batchnormalization -- enable batch normalization (boolean)

        Class Parameters Changed:
        parameters -- set contains parameters "W1", "b1", ..., "WL", "bL":
                        Wn -- weight matrix (L_dim[n], L_dim[n-1])
                        bn -- bias vector (L_dim[n], 1)
                        gamman -- batch normalization scale vector (L_dim[n], 1)
                        betan -- batch normalization shift vector (L_dim[n], 1)   
        v -- set contains the exponentially weighted average of the gradient.
                        v["dW" + str(l)]
                        v["db" + str(l)] 
                        v["dgamma" + str(l)]
                        v["dbeta" + str(l)] 
        s -- set contains the exponentially weighted average of the squared gradient.
                        s["dW" + str(l)]
                        s["db" + str(l)]
                        v["dgamma" + str(l)]
                        v["dbeta" + str(l)] 
        """

        L = len(self.L_dim) - 1 # integer representing the number of layers

        v_corrected = {}                     
        s_corrected = {}           

        if self.optimizer == "gd":
            for l in range(L):
                self.parameters["W" + str(l+1)] = self.parameters["W" + str(l+1)] - self.learning_rate*self.grads["dW" + str(l + 1)]
                self.parameters["b" + str(l+1)] = self.parameters["b" + str(l+1)] - self.learning_rate*self.grads["db" + str(l + 1)] 

        elif self.optimizer == "momentum":
            for l in range(L):
                self.v["dW" + str(l+1)] = beta * self.v["dW" + str(l+1)] + (1-self.beta) * self.grads["dW" + str(l+1)]
                self.v["db" + str(l+1)] = beta * self.v["db" + str(l+1)] + (1-self.beta) * self.grads["db" + str(l+1)]

                self.parameters["W" + str(l+1)] -= self.learning_rate * self.v["dW" + str(l+1)]
                self.parameters["b" + str(l+1)] -= self.learning_rate * self.v["db" + str(l+1)]   

        elif self.optimizer == "RMSProp ":
            for l in range(L):
                self.s["dW" + str(l+1)] = self.beta2*self.s["dW" + str(l+1)]+(1-self.beta2)*np.power(self.grads['dW'+str(l+1)],2)
                self.s["db" + str(l+1)] = self.beta2*self.s["db" + str(l+1)]+(1-self.beta2)*np.power(self.grads['db'+str(l+1)],2)

                self.parameters["W" + str(l+1)] = self.parameters["W" + str(l+1)]-\
                    self.learning_rate*np.divide(self.grads["dW" + str(l+1)],np.sqrt(self.s["dW" + str(l+1)])+self.epsilon)
                self.parameters["b" + str(l+1)] = self.parameters["b" + str(l+1)]-\
                    self.learning_rate*np.divide(self.grads["db" + str(l+1)],np.sqrt(self.s["db" + str(l+1)])+self.epsilon)

        elif self.optimizer == "adam":
            for l in range(L):
                self.v["dW" + str(l+1)] = self.beta1*self.v["dW" + str(l+1)]+(1-self.beta1)*self.grads['dW'+str(l+1)]
                self.v["db" + str(l+1)] = self.beta1*self.v["db" + str(l+1)]+(1-self.beta1)*self.grads['db'+str(l+1)]

                v_corrected["dW" + str(l+1)] = self.v["dW" + str(l+1)]/(1-pow(self.beta1,self.t))
                v_corrected["db" + str(l+1)] = self.v["db" + str(l+1)]/(1-pow(self.beta1,self.t))

                self.s["dW" + str(l+1)] = self.beta2*self.s["dW" + str(l+1)]+(1-self.beta2)*np.power(self.grads['dW'+str(l+1)],2)
                self.s["db" + str(l+1)] = self.beta2*self.s["db" + str(l+1)]+(1-self.beta2)*np.power(self.grads['db'+str(l+1)],2)

                s_corrected["dW" + str(l+1)] = self.s["dW" + str(l+1)]/(1-pow(self.beta2,self.t))
                s_corrected["db" + str(l+1)] = self.s["db" + str(l+1)]/(1-pow(self.beta2,self.t))

                self.parameters["W" + str(l+1)] = self.parameters["W" + str(l+1)]-\
                    self.learning_rate*np.divide(v_corrected["dW" + str(l+1)],np.sqrt(s_corrected["dW" + str(l+1)])+self.epsilon)
                self.parameters["b" + str(l+1)] = self.parameters["b" + str(l+1)]-\
                    self.learning_rate*np.divide(v_corrected["db" + str(l+1)],np.sqrt(s_corrected["db" + str(l+1)])+self.epsilon)

        if self.batchnormalization == True:
            if self.optimizer == "gd":
                for l in range(L):
                    self.parameters["gamma" + str(l+1)] = self.parameters["gamma" + str(l+1)] - self.learning_rate*self.grads["dgamma" + str(l + 1)]
                    self.parameters["beta" + str(l+1)] = self.parameters["beta" + str(l+1)] - self.learning_rate*self.grads["dbeta" + str(l + 1)] 

            elif self.optimizer == "momentum":
                for l in range(L):
                    self.v["dgamma" + str(l+1)] = beta * self.v["dgamma" + str(l+1)] + (1-self.beta) * self.grads["dgamma" + str(l+1)]
                    self.v["dbeta" + str(l+1)] = beta * self.v["dbeta" + str(l+1)] + (1-self.beta) * self.grads["dbeta" + str(l+1)]

                    self.parameters["gamma" + str(l+1)] -= self.learning_rate * self.v["dgamma" + str(l+1)]
                    self.parameters["beta" + str(l+1)] -= self.learning_rate * self.v["dbeta" + str(l+1)]   

            elif self.optimizer == "RMSProp ":
                for l in range(L):
                    self.s["dgamma" + str(l+1)] = self.beta2*self.s["dgamma" + str(l+1)]+(1-self.beta2)*np.power(self.grads['dgamma'+str(l+1)],2)
                    self.s["dbeta" + str(l+1)] = self.beta2*self.s["dbeta" + str(l+1)]+(1-self.beta2)*np.power(self.grads['dbeta'+str(l+1)],2)

                    self.parameters["gamma" + str(l+1)] = self.parameters["gamma" + str(l+1)]-\
                        self.learning_rate*np.divide(self.grads["dgamma" + str(l+1)],np.sqrt(self.s["dgamma" + str(l+1)])+self.epsilon)
                    self.parameters["beta" + str(l+1)] = self.parameters["beta" + str(l+1)]-\
                        self.learning_rate*np.divide(self.grads["dbeta" + str(l+1)],np.sqrt(self.s["dbeta" + str(l+1)])+self.epsilon)

            elif self.optimizer == "adam":
                for l in range(L):
                    self.v["dgamma" + str(l+1)] = self.beta1*self.v["dgamma" + str(l+1)]+(1-self.beta1)*self.grads['dgamma'+str(l+1)]
                    self.v["dbeta" + str(l+1)] = self.beta1*self.v["dbeta" + str(l+1)]+(1-self.beta1)*self.grads['dbeta'+str(l+1)]

                    v_corrected["dgamma" + str(l+1)] = self.v["dgamma" + str(l+1)]/(1-pow(self.beta1,self.t))
                    v_corrected["dbeta" + str(l+1)] = self.v["dbeta" + str(l+1)]/(1-pow(self.beta1,self.t))

                    self.s["dgamma" + str(l+1)] = self.beta2*self.s["dgamma" + str(l+1)]+(1-self.beta2)*np.power(self.grads['dgamma'+str(l+1)],2)
                    self.s["dbeta" + str(l+1)] = self.beta2*self.s["dbeta" + str(l+1)]+(1-self.beta2)*np.power(self.grads['dbeta'+str(l+1)],2)

                    s_corrected["dgamma" + str(l+1)] = self.s["dgamma" + str(l+1)]/(1-pow(self.beta2,self.t))
                    s_corrected["dbeta" + str(l+1)] = self.s["dbeta" + str(l+1)]/(1-pow(self.beta2,self.t))

                    self.parameters["gamma" + str(l+1)] = self.parameters["gamma" + str(l+1)]-\
                        self.learning_rate*np.divide(v_corrected["dgamma" + str(l+1)],np.sqrt(s_corrected["dgamma" + str(l+1)])+self.epsilon)
                    self.parameters["beta" + str(l+1)] = self.parameters["beta" + str(l+1)]-\
                        self.learning_rate*np.divide(v_corrected["dbeta" + str(l+1)],np.sqrt(s_corrected["dbeta" + str(l+1)])+self.epsilon)                
                
        return
    
    
    def predict(self,X, y ):

        """    
        Make prediction with trained neural network
        
        Arguments:
        X -- input data (number of features, number of examples)
        Y -- labels (number of classes, number of examples)
        
        Needed Class Parameters:
        parameters -- parameters learnt of the model

        Returns:
        p -- predictions for the X
        """

        m = X.shape[1]
        n = len(self.parameters) // 2 # number of layers in the neural network
        p = np.zeros(m)
            
        # Forward propagation
        temp = self.keep_prob
        self.keep_prob = 1
        probas = self.NN_forward(X)  # no dropout for prediction
        self.keep_prob = temp
        
        # convert probas to 0/1 predictions
        for i in range(0, probas.shape[1]):
            p[i] = np.argmax(probas[:,i])
        #print results
        #print ("predictions: " + str(p))
        #print ("true labels: " + str(y))
        print("Accuracy: "  + str(np.sum((p == Y)/m)))

        return p
    
    