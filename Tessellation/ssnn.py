from utils import *
import numpy as np
from pyflann import *
from scipy.spatial.distance import cdist
import time

class SSNN:
    
    def __init__(self, n_nodes, fovea, method='auto'):
        
        self.n_nodes = n_nodes
        self.fovea = fovea
        
        r = np.random.uniform(1, 0, self.n_nodes)
        th = 2 * np.pi * np.random.uniform(1, 0, self.n_nodes)
        
        self.weights = pol2cart(np.array([th, r]).T)
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
            
            index = get_neighbours(I, self.weights)
            
            # Update the weights 
            self.weights[index] -= (self.weights[index] - I) * alpha
 
            # Display progress
            if (verbose):
                sys.stdout.write('\r'+str(i + 1) +"/" + str(num_iters))
        
        normalize(self.weights)

        if(verbose):
            print("\nFinished.")
            print("\nTime taken: " + str(time.time()-start))

    def generative_fit(self, steps, num_iters=20000, initial_alpha=0.1, final_alpha=0.0005, verbose=True):
        
        start = time.time()

        SSNN.fit(self, num_iters, initial_alpha, final_alpha, verbose)
        
        for i in range(steps):
            print("\nAnnealing new points...\n")
            self.weights = point_gen(self.weights, 1, 'sierpinski')
            self.weights = randomize(self.weights)
            self.n_nodes = self.weights.shape[0]
            SSNN.fit(self, 5000, 0.033, 0.0005, verbose)
        
        if(verbose):
            print("\nFinal node count: " + str(self.weights.shape[0]))
            print("\nTotal time taken: " + str(time.time()-start))
        
        return 

    def show_tessellation(self, figsize=(15,15), s=1, overlay=False):
        plt.figure(figsize=figsize)
        plt.scatter(self.weights[:,0], self.weights[:,1],s=s)
        if(overlay):
            plt.scatter(self.original_weights[:,0], self.original_weights[:,1], s=s+5, c='red')
        plt.show()
        return
    
    def display_stats(self, figsize=(15,8), k=7):
        display_stats(self.weights, figsize, k)
        return

    def select_method(self, method):

        if (method == 'default'):
            return self._bf_neighbours

        elif(method == 'flann'):
            return self._flann_neighbours

        elif(method == 'auto'):
            if(self.n_nodes <= 256):
                return self._bf_neighbours

            elif(self.n_nodes <= 20000 and self.use_gpu):
                return self._torch_neighbours

            else:
                return self._flann_neighbours

    def _bf_neighbours(self, X, Y): 
        dists = cdist(X, Y)
        indeces = np.argmin(dists, axis=1)
        return indeces

    def _flann_neighbours(self, X, Y):
        indeces, dists = self.flann.nn(Y, X, 1, algorithm="kdtree", branching=16, iterations=5, checks=16) 
        return indeces

    @staticmethod
    def _alpha_schedule(initial_alpha, final_alpha, num_iters, split):
        static = split
        decay = num_iters - static
        static_lr = np.linspace(initial_alpha, initial_alpha, static)
        decay_lr = np.linspace(initial_alpha, final_alpha, decay)
        return np.concatenate((static_lr, decay_lr))




