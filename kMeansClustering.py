import numpy
import matplotlib.pyplot as plt
from numpy.random import Generator, PCG64DXSM
from utils.kMeans import *
rng = Generator(PCG64DXSM())

class kMeansClustering():
    _X = None
    _K: int = None
    _max_iters: int = None
    _original_img = None
    _initial_centroids = None

    def __init__(self, initial_centroids, K:int, max_iters: int):
        self._initial_centroids = initial_centroids
        self._K = K
        self._max_iters = max_iters

    def load_numpy(self, path: str):
        self._X = numpy.load(path)

    def _find_closest_centroids(self, centroids):
        """
        Computes the centroid memberships for every example
        
        Args:
            X (ndarray): (m, n) Input values      
            centroids (ndarray): (K, n) centroids
        
        Returns:
            idx (array_like): (m,) closest centroids
            output a one-dimensional array idx (which has the same number of elements as X) that holds the index of the closest centroid (a value in  {0,...,K-1}, where K is total number of centroids) to every training samples
        https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html
        """
        print(f"\n=== {self._find_closest_centroids.__name__} ===")
        # Set K
        K = centroids.shape[0]
        idx = numpy.zeros(self._X.shape[0], dtype=int)
        #print(f"K: {K}, idx: {idx}")
        #print(f"X: {X[0]}")
        for i in range(len(idx)):
            distance = numpy.inf
            for k in range(K):
                norm_ij = numpy.linalg.norm(self._X[i] - centroids[k])
                if norm_ij < distance:
                    distance = norm_ij
                    idx[i] = k
        #print(f"idx: {idx}")
        return idx

    def _compute_centroids(self, idx, K):
        """
        Returns the new centroids by computing the means of the 
        data points assigned to each centroid.
        
        Args:
            X (ndarray):   (m, n) Data points
            idx (ndarray): (m,) Array containing index of closest centroid for each 
                        example in X. Concretely, idx[i] contains the index of 
                        the centroid closest to example i
            K (int):       number of centroids
        
        Returns:
            centroids (ndarray): (K, n) New centroids computed
        """
        print(f"\n=== {self._compute_centroids.__name__} ===")
        # Useful variables
        m, n = self._X.shape
        
        # You need to return the following variables correctly
        centroids = np.zeros((K, n))
        # sum(x[]) / len(x[])
        for k in range(K):
            sum = 0
            count = 0
            for i in range(len(idx)):
                if idx[i] == k:
                    sum += self._X[i]
                    count += 1
            centroids[k] = sum / count
        return centroids

    def run_kMeans(self, plot_progress=False):
        """
        Runs the K-Means algorithm on data matrix X, where each row of X is a single example
        """   
        print(f"\n=== {self.run_kMeans.__name__} ===")
        # Initialize values
        m, n = self._X.shape
        #K = self._initial_centroids.shape[0]
        centroids = self._initial_centroids
        previous_centroids = centroids    
        idx = np.zeros(m)
        if plot_progress:
            plt.figure(figsize=(8, 6))

        # Run K-Means
        for i in range(self._max_iters):
            
            #Output progress
            print("K-Means iteration %d/%d" % (i, self._max_iters-1))
            
            # For each example in X, assign it to the closest centroid
            idx = self._find_closest_centroids(centroids)
            
            # Optionally plot progress
            if plot_progress:
                plot_progress_kMeans(self._X, centroids, previous_centroids, idx, i)
                previous_centroids = centroids
                
            # Given the memberships, compute new centroids
            centroids = self._compute_centroids(idx, self._K)
        if plot_progress:
            #print(f"plt.show() from run_kMeans()")
            plt.show()
        return centroids, idx

    def kMeans_init_centroids(self):
        """
        This function initializes K centroids that are to be 
        used in K-Means on the dataset X
        
        Args:
            X (ndarray): Data points 
            K (int):     number of centroids/clusters
        
        Returns:
            centroids (ndarray): Initialized centroids
        """
        print(f"\n=== {self.kMeans_init_centroids.__name__} ===")
        # Randomly reorder the indices of examples
        randidx = np.random.permutation(self._X.shape[0])
        
        # Take the first K examples as centroids
        self._initial_centroids = self._X[randidx[:self._K]]

    def kMeansImageCompression(self, path: str):
        print(f"\n=== {self.kMeansImageCompression.__name__} ===")
        # Load an image of a bird
        # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imread.html
        self._original_img = plt.imread(path)
        # Visualizing the image
        #plt.imshow(self._original_img)
        # Shape of original_img is: (128, 128, 3) (M, N, 4) (height, width, RGB+Apha)
        print("Shape of original_img is:", self._original_img.shape)

        # Divide by 255 so that all values are in the range 0 - 1 (not needed for PNG files)
        # original_img = original_img / 255

        # Reshape the image into an m x 3 matrix where m = number of pixels
        # (in this case m = 128 x 128 = 16384)
        # Each row will contain the Red, Green and Blue pixel values
        # This gives us our dataset matrix X_img that we will use K-Means on.
        self._X = np.reshape(self._original_img, (self._original_img.shape[0] * self._original_img.shape[1], 3))

        # k-Means on image pixels
        # Run your K-Means algorithm on this data
        # You should try different values of K and max_iters here
        #K = 16
        #max_iters = 10

        # Using the function you have implemented above. 
        self.kMeans_init_centroids()
        print(f"initial_centroids shape: {self._initial_centroids.shape}") # (K, 3)
        # Run K-Means - this can take a couple of minutes depending on K and max_iters
        centroids, idx = self.run_kMeans()
        print(f"Shape of idx: {idx.shape}") # image.shape[0] * image.shape[1]
        print(f"Shape of centroids: {centroids.shape}") # (K, 3)
        print("Closest centroid for the first five elements:", idx[:5])
        # Plot the colors of the image and mark the centroids
        plot_kMeans_RGB(self._X, centroids)
        # Visualize the 16 colors selected
        show_centroid_colors(centroids)

        # Compress the image
        """
        After finding the top K=16 colors to represent the image, you can now assign each pixel position to its closest centroid using the _find_closest_centroids function.

        This allows you to represent the original image using the centroid assignments of each pixel.
        Notice that you have significantly reduced the number of bits that are required to describe the image.
        The original image required 24 bits (i.e. 8 bits x 3 channels in RGB encoding) for each one of the  128x128
        pixel locations, resulting in total size of  128x128x24=393,216 bits.

        The new representation requires some overhead storage in form of a dictionary of 16 colors, each of which require 24 bits, but the image itself then only requires 4 bits (2^4=16) per pixel location.
        The final number of bits used is therefore  16x24+128x128x4=65,920 bits, which corresponds to compressing the original image by about a factor of 6.    
        """
        # Find the closest centroid of each pixel
        idx = self._find_closest_centroids(centroids)

        # Replace each pixel with the color of the closest centroid
        # https://numpy.org/doc/stable/user/basics.indexing.html#advanced-indexing
        # It's using idx to index (select) from the first dimension (rows) of centroids. The ,: part makes it clear that it will return all values along the second dimension.
        X_recovered = centroids[idx, :] 

        # Reshape image into proper dimensions
        X_recovered = np.reshape(X_recovered, self._original_img.shape)

        #Finally, you can view the effects of the compression by reconstructing the image based only on the centroid assignments.
        #Specifically, you replaced each pixel with the value of the centroid assigned to it.
        #Figure 3 shows a sample reconstruction. Even though the resulting image retains most of the characteristics of the original, you will also see some compression artifacts because of the fewer colors used.    
        # Display original image
        print(f"Showing original and compressed images...")
        fig, ax = plt.subplots(1,2, figsize=(16,16)) # figsize = (width, height)
        plt.axis('off')

        ax[0].imshow(self._original_img)
        ax[0].set_title('Original')
        ax[0].set_axis_off()

        # Display compressed image
        ax[1].imshow(X_recovered)
        ax[1].set_title('Compressed with %d colours'%K)
        ax[1].set_axis_off()
        plt.show()

if __name__ == "__main__":
    # Set initial centroids
    initial_centroids = numpy.array([[3,3],[6,2],[8,5]])
    # Number of iterations
    max_iters = 10
    kmeans = kMeansClustering(initial_centroids, initial_centroids.shape[0], max_iters)
    kmeans.load_numpy("data/ex7_X.npy")
    # Run K-Means
    centroids, idx = kmeans.run_kMeans(plot_progress=True)

    # Random initialization of centroids
    # Run this cell repeatedly to see different outcomes.
    # Set number of centroids and max number of iterations
    K = 3
    max_iters = 10
    kmeans = kMeansClustering(initial_centroids, K, max_iters)
    kmeans.load_numpy("data/ex7_X.npy")
    # Set initial centroids by picking random examples from the dataset
    initial_centroids = kmeans.kMeans_init_centroids()
    # Run K-Means
    centroids, idx = kmeans.run_kMeans(plot_progress=True)
    print(f"Shape of idx: {idx.shape}")

    # Image compression
    K = 16
    max_iters = 10
    kmeans = kMeansClustering([], K, max_iters)
    kmeans.kMeansImageCompression("images/bird_small.png")
