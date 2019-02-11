import numpy as np
from utils import *

def fibonacci_sunflower(n_nodes):
	""" Generates points using the golden ratio

		Parameters
		----------
		n_nodes: number of points to be generated

		Return: numpy array of points

	"""


	g_ratio = (np.sqrt(5) + 1) / 2

	nodes = np.arange(1,n_nodes+1)

	rho = np.sqrt(nodes-0.5)/np.sqrt(n_nodes)

	theta = np.pi * 2 * g_ratio * nodes

	x = rho * np.cos(theta)
	y = rho * np.sin(theta)

	return np.array([x,y]).T

def fibonacci_retina(n_nodes, fovea, fovea_density):

	""" Generates points using the fibonacci sunflower
		and dilates them with the dilate function found in utils.
		See README for more description of this dilate function.

		Parameters
		----------
		n_nodes: number of nodes in tessellation
		fovea: size of foveal region in tessellation; 0 < fovea <= 1
		fovea_density: scaling factor to affect the ratio of nodes
		in and outside the fovea.

		Return: numpy array of points

	"""
	retina = fibonacci_sunflower(n_nodes)
	return dilate(retina, fovea, fovea_density)
	




