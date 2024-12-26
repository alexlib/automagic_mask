import numpy as np
from scipy.stats import chi2
from scipy.ndimage import median_filter
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from skimage.morphology import binary_closing, remove_small_holes

from scipy.ndimage import convolve

def majority_filter(mask, iterations=5):
    # Define a 3x3 kernel for the majority operation
    kernel = np.ones((3, 3))
    
    for _ in range(iterations):
        # Convolve the mask with the kernel
        convolved = convolve(mask.astype(int), kernel, mode='constant', cval=0)
        # Apply the majority rule
        mask = convolved >= 5
    
    return mask

# Apply the majority filter Nit times
# mask = majority_filter(mask, Nit)


def mask_detect(imdb, debug=False):
    CleanMask = True  # Morphological cleaning of the mask
    block_size = 100  # To evaluate statistics in blocks that fit into memory
    Nrange = imdb.shape[2]
    sample = imdb[:, :, 0]

    # Evaluate skewness and kurtosis in blocks
    skew = np.zeros_like(sample, dtype=float)
    kurt = np.zeros_like(sample, dtype=float)
    Nblock = int(np.ceil(sample.shape[1] / block_size))
    for i in range(Nblock):
        if debug:
            print(f'Block {i + 1} of {Nblock}')
        j1 = i * block_size
        j2 = min((i + 1) * block_size, sample.shape[1])
        block = imdb[:, j1:j2, :].astype(float)

        # Evaluate kurtosis and skewness
        kurt[:, j1:j2], skew[:, j1:j2] = my_stat(block, Nrange)

    # Pixels with constant intensity present a NaN kurtosis. Those pixels
    # should be surely masked out, set the kurtosis to 3 and skewness to 0
    kurt[np.isnan(kurt)] = 3
    skew[np.isnan(skew)] = 0

    # Evaluate the JB statistic
    jbtest = Nrange / 6 * (skew ** 2 + (kurt - 3) ** 2 / 4)
    # Evaluate the p-value
    p = 1 - chi2.cdf(jbtest, 2)

    if debug:
        plt.figure()
        plt.imshow(jbtest)
        plt.title('JB statistic')
        plt.figure()
        plt.imshow(np.log(p))
        plt.title('Logarithm of probability')
        plt.show()

    # Median filter the probability
    p_flt = median_filter(p, size=5, mode='symmetric')

    # Automatic clustering of bg and flow using kmeans
    bf = np.log(p_flt.flatten())
    mi = np.min(bf[np.isfinite(bf)])
    ma = np.max(bf[np.isfinite(bf)])
    bf = (bf - mi) / (ma - mi)
    kmeans = KMeans(n_clusters=2).fit(bf.reshape(-1, 1))
    id = kmeans.labels_
    cen = kmeans.cluster_centers_

    # Identify kmeans output using the probability
    idobj = np.argmax(cen[:, 0])
    mask = np.zeros_like(id)
    mask[id == idobj] = 1
    mask = mask.reshape(p.shape)

    if debug:
        plt.figure()
        plt.imshow(mask)
        plt.title('Image of the mask')
        plt.show()

    # Morphological operations
    if CleanMask:
        mask = binary_closing(mask)
        mask = remove_small_holes(mask)
        Nit = 5
        # Clean mask from spurious pixels
        for _ in range(Nit):
            mask = majority_filter(mask)

    return mask, p

def my_stat(block, Nrange):
    # Placeholder for statistical calculations
    kurt = np.mean(block, axis=2)  # Example logic
    skew = np.std(block, axis=2)  # Example logic
    return kurt, skew