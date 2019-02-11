from utils import *
import numpy as np
from pyflann import *
from scipy.spatial.distance import cdist
import time

class SSNN:
    
    def __init__(self, n_nodes, fovea, method='auto'):
        
        self.n_nodes = n_nodes
        self.fovea = fovea
        self.weights = SSNN._init_weights(n_nodes)
        self.method = method
        self.use_gpu = False
        self.flann = FLANN()
                      
    
    def fit(self, num_iters=20000, initial_alpha=0.1, final_alpha=0.0005, verbose=True):
        
        learning_rate = SSNN._alpha_schedule(initial_alpha, final_alpha, num_iters, num_iters//4)

        start = time.time()

        get_neighbours = self.select_method(self.method)
        
        for i in range(num_iters):
            
            alpha = learning_rate[i]
            
            I = np.copy(self.weights)
                     
            I = cart2pol(I)
   
            # Dilation
            d = np.exp((2 * np.random.uniform() -1) * np.log(8))
            I[:,1] *= d
            
            I = pol2cart(I)
            
            # Translation Wondering if different translation for each point makes any difference
            delta_theta = 2 * np.random.uniform() * np.pi
            delta_rho = np.random.uniform() * self.fovea
            
            # Changes in the x and y direction
            I[:,0] += np.cos(delta_theta) * delta_rho
            I[:,1] += np.sin(delta_theta) * delta_rho
            
            # Random rotation
            I = cart2pol(I)
            I[:,0] += 2 * np.random.uniform() * np.pi
            
            # Remove the input vectors which are outside the bounds of the retina
            cull = np.where(I[:,1] <= 1)[0]
            I = pol2cart(I)
            I = I[cull]
            
            # Gets the nearest update vector for each weight
            index = get_neighbours(I, self.weights)
            
            # Update the weights 
            self.weights[index] -= (self.weights[index] - I) * alpha
 
            # Display progress
            if (verbose):
                sys.stdout.write('\r'+str(i + 1) +"/" + str(num_iters))
        
        # Constrain to unit circle
        normalize(self.weights)

        if(verbose):
            print("\nFinished.")
            print("\nTime taken: " + str(time.time()-start))

    def generative_fit(self, steps, num_iters=20000, initial_alpha=0.1, final_alpha=0.0005, verbose=True):
        
        start = time.time()

        self.n_nodes = self.n_nodes // (4 ** steps)
        self.weights = SSNN._init_weights(self.n_nodes)

        SSNN.fit(self, num_iters, initial_alpha, final_alpha, verbose)
        
        g_iters = 1250
        g_alpha = ((g_iters/num_iters) * initial_alpha) + final_alpha

        for i in range(steps):
            print("\nAnnealing new points...\n")
            self.weights = point_gen(self.weights, 1, 'sierpinski')
            self.weights = randomize(self.weights)
            self.n_nodes = self.weights.shape[0]

            if(i == steps-1):
                g_iters *= 2
                g_alpha = ((g_iters/num_iters) * initial_alpha) + final_alpha

            SSNN.fit(self, g_iters, g_alpha, final_alpha, verbose)
        
        if(verbose):
            print("\nFinal node count: " + str(self.weights.shape[0]))
            print("\nTotal time taken: " + str(time.time()-start))
        
        return 

    def display_tessellation(self, figsize=(15,15), s=1):
        display_tessellation(self.weights, figsize, s)
        return
    
    def display_stats(self, figsize=(15,8), k=7):
        display_stats(self.weights, figsize, k)
        return

    def select_method(self, method):

        if (method == 'default'):
            print("Using Scipy.")
            return self._bf_neighbours

        elif(method == 'flann'):
            print("Using FLANN.")
            return self._flann_neighbours

        elif(method == 'auto'):
            if(self.n_nodes <= 256):
                print("Using Scipy.")
                return self._bf_neighbours

            elif(self.n_nodes <= 20000 and self.use_gpu):
                print("Using GPU accelerated Pytorch.")
                return self._torch_neighbours

            else:
                print("Using FLANN.")
                return self._flann_neighbours

    def _bf_neighbours(self, X, Y): 
        dists = cdist(X, Y)
        indeces = np.argmin(dists, axis=1)
        return indeces

    def _flann_neighbours(self, X, Y):
        indeces, dists = self.flann.nn(Y, X, 1, algorithm="kdtree", branching=16, iterations=5, checks=16) 
        return indeces

    @staticmethod
    def _init_weights(n_nodes):
        r = np.random.uniform(1, 0, n_nodes)
        th = 2 * np.pi * np.random.uniform(1, 0, n_nodes)
        return pol2cart(np.array([th, r]).T)


    @staticmethod
    def _alpha_schedule(initial_alpha, final_alpha, num_iters, split):
        static = split
        decay = num_iters - static
        static_lr = np.linspace(initial_alpha, initial_alpha, static)
        decay_lr = np.linspace(initial_alpha, final_alpha, decay)
        return np.concatenate((static_lr, decay_lr))




