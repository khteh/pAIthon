import numpy, tensorflow as tf, scipy
from numpy.random import Generator, PCG64DXSM
rng = Generator(PCG64DXSM())

# https://jonathan-hui.medium.com/gan-how-to-measure-gan-performance-64b988c47732
# https://towardsdatascience.com/demystified-wasserstein-gan-with-gradient-penalty-ba5e9b905ead/#:~:text=The%20idea%20of%20Gradient%20Penalty,unit%20norm%20(Statement%201).&text=In%20Eq.,sum%20is%20the%20gradient%20penalty.
# https://ankittaxak5713.medium.com/wasserstein-gans-wgan-3b8031aebf53
# https://medium.com/@krushnakr9/gans-wasserstein-gan-with-gradient-penalty-wgan-gp-b8da816cb2d2
def frechet_distance(mu_x, mu_y, sigma_x, sigma_y):
    '''
    Function for returning the Fréchet distance between multivariate Gaussians,
    parameterized by their means and covariance matrices.
    Parameters:
        mu_x: the mean of the first Gaussian, (n_features)
        mu_y: the mean of the second Gaussian, (n_features) 
        sigma_x: the covariance matrix of the first Gaussian, (n_features, n_features)
        sigma_y: the covariance matrix of the second Gaussian, (n_features, n_features)
    '''
    print(f"mu_x: {mu_x.shape}, mu_y: {mu_y.shape}, sigma_x: {sigma_x.shape}, sigma_y: {sigma_y.shape}")
    return numpy.linalg.vector_norm(mu_x - mu_y)**2 + numpy.trace(sigma_x + sigma_y - 2 * scipy.linalg.sqrtm(sigma_x @ sigma_y))

if __name__ == "__main__":
    mean1 = numpy.array([0, 2]) # Center the mean at the origin
    covariance1 = numpy.array( # This matrix shows independence - there are only non-zero values on the diagonal
        [[1, 0],
        [0, 1]]
    )
    mean2 = numpy.array([0, 0]) # Center the mean at the origin
    covariance2 = numpy.array( # This matrix shows dependence 
        [[2, -1],
        [-1, 2]]
    )
    distance = frechet_distance(mean1, mean1, covariance1, covariance1)
    print(f"xx frechet_distance: {distance}")
    assert (distance == 0)

    distance = frechet_distance(mean1, mean2, covariance1, covariance2)
    print(f"xy frechet_distance: {distance}")
    assert numpy.isclose(distance, 8 - 2 * numpy.sqrt(3.))
    print("\n\033[92mAll test passed!")