import numpy as np
from scipy.spatial.distance import pdist, squareform, euclidean
from skgstat.models import spherical, gaussian
from scipy.linalg import solve

class SpatialKriging():
    """ A simple spatial kriging class
    """
    def __init__(self, data):
        self.data = data
        self.dist_matrix = self.get_distance_matrix()                               # gets distance matrix only once at initialization to save computing
        self.variance_dist_matrix = self.get_semivariances(self.dist_matrix)        # gets variance distance matrix only once at initialization to save computing
        self.extend_variance_matrix()


    def get_distance_matrix(self):
        """ Calculates distance matrix among known points
        Return is a 2D square numpy array
        """
        # distance_matrix = pdist([s0] + list(zip(self.data['X'], self.data['Y'])))
        distance_matrix = pdist(list(zip(self.data['X'], self.data['Y'])))
        return squareform(distance_matrix)


    def get_distance_vector(self, s0):
        """ Calculates distance vector to the unknown point
        Return is a 1D numpy array
        """
        n = self.data.shape[0]
        distance_vector = [euclidean(s0, [xi, yi]) for xi, yi in zip(self.data['X'], self.data['Y'])]
        return np.array(distance_vector)

    
    def get_semivariances(self, dist_matrix):
        """ Gets variances
        """
        n = dist_matrix.shape[0]
        # range= 7. sill = 2. nugget = 0.
        variances = spherical(dist_matrix.flatten(), 7.0, 2.0, 0.0)
        if dist_matrix.size > len(dist_matrix):   # 2D numpy array
            return variances.reshape(n, n)
        else:
            return variances


    def extend_variance_matrix(self):
        """ Adds unitiy row and column to the simivariance matrix"""
        n = self.dist_matrix.shape[0]
        unity_row_vector = np.ones((1,n))
        unity_column_vector = np.concatenate((unity_row_vector.transpose(), np.array([[0.0]])))
        self.variance_dist_matrix = np.vstack((self.variance_dist_matrix, unity_row_vector))
        self.variance_dist_matrix = np.hstack((self.variance_dist_matrix, unity_column_vector))

    
    def estimate_with_ordinary_kriging(self, p0):
        """ Estimates with ordinary krigging for point p0"""
        #dist_matrix = self.get_distance_matrix()
        dist_vector = self.get_distance_vector(p0)
        #variance_dist_matrix = self.get_semivariances(dist_matrix)
        variance_dist_vector = self.get_semivariances(dist_vector)
        # extend variance_dist_vector with '1.0'
        variance_dist_vector = np.concatenate((variance_dist_vector, np.array([1.0])))

        # solve linear system
        weights = solve(self.variance_dist_matrix, variance_dist_vector)

        # estimate
        z_est = self.data.iloc[:,2].dot(weights[:-1])
        
        return z_est