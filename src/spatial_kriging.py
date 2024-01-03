import numpy as np
from scipy.spatial.distance import pdist, squareform, euclidean
import skgstat as skg
from skgstat.models import spherical, exponential, gaussian, matern
from scipy.linalg import solve
import copy

class SpatialKriging():
    """ A simple spatial kriging class
    """
    def __init__(self, data, range=7.0, sill=2.0, nugget=0.0, variogram_model=spherical):
        self.data = data
        self.range = range
        self.sill = sill
        self.nugget = nugget
        self.variogram_model = variogram_model
        self.dist_matrix = self.get_distance_matrix()                               # gets distance matrix only once at initialization to save computing
        self.variogram_matrix = self.get_semivariances(self.dist_matrix)        # gets variance distance matrix only once at initialization to save computing
        self.extend_variance_matrix()                                               # evaluate on initialization ...


    def build_experimental_variogram(self):
        """ Builds experimental variogram
        """
        coordinates = self.data[['X', 'Y']]
        values = self.data['Z']
        V = skg.Variogram(coordinates=coordinates, values=values, n_lags=6)

        return V
    

    def set_variogram_model_parameters(self, range, sill, nugget, model='spherical'):
        """ Sets variogram model parameters
        """
        self.range = range
        self.sill = sill
        self.nugget = nugget
        if model == 'spherical':
            self.variogram_model = spherical
        elif model == 'exponential':
            self.variogram_model = exponential
        elif model == 'gaussian':
            self.variogram_model = gaussian
        elif model == 'matern':
            self.variogram_model = matern

        # recalculate variogram matrix using the updated variogram model and parameters
        self.variogram_matrix = self.get_semivariances(self.dist_matrix)        # gets variance distance matrix only once at initialization to save computing
        self.extend_variance_matrix()                                               # evaluate on initialization ...



    def get_distance_matrix(self):
        """ Calculates distance matrix among known points
        Return is a 2D square numpy array
        """
        # distance_matrix = pdist([s0] + list(zip(self.data['X'], self.data['Y'])))
        #distance_matrix = pdist(list(zip(self.data['X'], self.data['Y'])))
        distance_matrix = pdist(list(zip(self.data['X'], self.data['Y'])), metric='euclidean')
        return squareform(distance_matrix)


    def get_distance_vector(self, s0):
        """ Calculates distance vector to the unknown point
        Return is a 1D numpy array
        """
        n = self.data.shape[0]
        distance_vector = [euclidean(s0, [xi, yi]) for xi, yi in zip(self.data['X'], self.data['Y'])]
        return np.array(distance_vector)

    
    def get_semivariances(self, dist_matrix):
        """ Gets variances (Variogram model) of values given in distance matrix

        Args:
            - distance matrix
        Returns:
            - Variogram model (Semi-variance matrix)
        """
        n = dist_matrix.shape[0]
        #variances = spherical(dist_matrix.flatten(), self.range, self.sill, self.nugget)
        variances = self.variogram_model(dist_matrix.flatten(), self.range, self.sill, self.nugget)
        if dist_matrix.size > len(dist_matrix):   # 2D numpy array
            return variances.reshape(n, n)
        else:
            return variances


    def extend_variance_matrix(self):
        """ Adds unitiy row and column to the simivariance matrix"""
        n = self.dist_matrix.shape[0]
        unity_row_vector = np.ones((1,n))
        unity_column_vector = np.concatenate((unity_row_vector.transpose(), np.array([[0.0]])))
        self.variogram_matrix = np.vstack((self.variogram_matrix, unity_row_vector))
        self.variogram_matrix = np.hstack((self.variogram_matrix, unity_column_vector))

    def get_covariance_matrix_and_vector(self, p0):
        """ Gets the covariance matrix and vector"""
        dist_vector = self.get_distance_vector(p0)
        variogram_vector = self.get_semivariances(dist_vector)
        self.covariance_matrix = copy.deepcopy(self.variogram_matrix)
        self.covariance_matrix[:-1,:-1] = self.nugget + self.sill - self.covariance_matrix[:-1,:-1]
        self.covariance_vector = self.nugget + self.sill - variogram_vector
        # extend variogram_vector with '1.0'
        self.covariance_vector = np.concatenate((self.covariance_vector, np.array([1.0])))
    
    def estimate_with_ordinary_kriging(self, p0):
        """ Estimates with ordinary krigging for point p0"""
        #dist_matrix = self.get_distance_matrix()
        dist_vector = self.get_distance_vector(p0)
        #variance_dist_matrix = self.get_semivariances(dist_matrix)
        variogram_vector = self.get_semivariances(dist_vector)

        # solve linear system
        #weights = solve(self.variogram_matrix, variogram_vector)
        covariance_matrix = copy.deepcopy(self.variogram_matrix)
        covariance_matrix[:-1,:-1] = self.nugget + self.sill - covariance_matrix[:-1,:-1]
        covariance_vector = self.nugget + self.sill - variogram_vector
        # extend variogram_vector with '1.0'
        covariance_vector = np.concatenate((covariance_vector, np.array([1.0])))

        weights = solve(covariance_matrix, covariance_vector)
        # estimate mean and variance
        z_est = self.data['Z'].dot(weights[:-1])                                                # mean
        var_est = self.nugget + self.sill - covariance_vector[:-1].dot(weights[:-1])            # variance

        #weights = solve(covariance_matrix[:-1,:-1], covariance_vector[:-1])
        ## estimate mean and variance
        #z_est = self.data['Z'].dot(weights)           # mean
        #var_est = self.nugget + self.sill - covariance_vector[:-1].dot(weights)          # variance
        
        return z_est, var_est, weights, variogram_vector