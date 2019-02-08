import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

def cart2pol(coords):

    x = coords[:,0]
    y = coords[:,1]
    rho = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)

    return np.stack([theta, rho],1)

def pol2cart(coords):

    theta = coords[:,0]
    rho = coords[:,1]
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)

    return np.stack([x, y], 1)

def display_tessellation(points, figsize=(15,15), s=1, overlay=True):
    plt.figure(figsize=figsize)
    plt.scatter(points[:,0], points[:,1],s=s)
    plt.show()
    return

def display_stats(points, figsize=(15,8), k=7):
    
    nbs = KDTree(points)

    distance, index = nbs.query(points, k=k)

    eccentricity = np.linalg.norm(points, axis=1)
    distance = 1/np.mean(distance[:,1:],axis=1)

    distance = distance[np.argsort(eccentricity)]
    eccentricity = np.sort(eccentricity)

    plt.figure(figsize=figsize)
    plt.plot(eccentricity, distance)

    plt.xlabel("Eccentricity")
    plt.ylabel("1/mean distance to neighbours")
    plt.show()

    return

def normalize(points):
    points = cart2pol(points)
    points[:,1] /= np.max(points[:,1])
    points = pol2cart(points)

    return points


def randomize(points):

    nbs = KDTree(points)

    distance, index = nbs.query(points, k=5)

    eccentricity = np.linalg.norm(points, axis=1)
    distance = np.mean(distance[:,1:],axis=1)
    
    delta_rho = distance * np.sqrt(np.random.uniform())
    delta_theta = 2 * np.random.uniform(size = distance.shape[0]) * np.pi
    
    delta_x = delta_rho * np.cos(delta_theta)
    delta_y = delta_rho * np.sin(delta_theta)
    
    points[:,0] += (delta_x * 0.25)
    points[:,1] += (delta_y * 0.25)

    return normalize(points)

def point_gen(points, num_iters=1, mode='sierpinski'):
    points = points

    for i in range(num_iters):
        triangulation = Delaunay(points)
        d_tri = points[triangulation.simplices]

        if(mode == 'barycentre'):
            new_points = np.sum(d_tri,axis=1)/3

        elif(mode == 'sierpinski'):

            a = (d_tri[:,0] + d_tri[:,1])/2
            b = (d_tri[:,1] + d_tri[:,2])/2
            c = (d_tri[:,2] + d_tri[:,0])/2

            new_points = np.concatenate((a,b,c),axis=0)
            new_points = np.unique(new_points, axis=0)

        else:
            print("Unknown mode; Choose from 'barycentre' or 'sierpinski'.")
            return None

        points = np.concatenate((points, new_points),axis=0)

    return normalize(points)

