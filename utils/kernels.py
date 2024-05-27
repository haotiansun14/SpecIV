'''
This file contains vectorized kernels we may use in our models.
'''

import torch
from torch import cdist as EuDist2
from scipy.spatial.distance import cdist
# from utils.dist import EuDist2

import numpy as np

def get_median(X) -> float:
    dist_mat = cdist(X, X, "sqeuclidean")
    res: float = np.median(dist_mat)
    return res

def construct_kernel(fea_a, fea_b=None, kernel_param={'kernel_type': 'Gaussian', 't': None}):
    '''
    Construct kernel matrix, with options:

    param fea_a, fea_b  : Rows of vectors of data points. (nSample x nFeature)
    param kernel_type:
            'Gaussian'      e^{-(||x - y||^2) / t}
            'Uniform'       I(||x - y|| < t/2) / t (Only for 1-d case)
            'Polynomial'    (x' * y)^t
            'PolyPlus'      (x' * y + 1)^t
            'Linear'        x' * y
    param t: param for 'Gaussian', 'Uniform' and 'Poly'
    '''
    print("Constructing kernel matrix...")

    # Get kernel_type and t from kernel_param
    kernel_type, t = kernel_param['kernel_type'].lower(), kernel_param.get('t')

    # Initialize parameters based on kernel_type
    if t is None:
        if kernel_type in ['gaussian', 'uniform']:
            t = 1
        elif kernel_type in ['polynomial', 'polyplus']:
            t = 2

    # If fea_b is not provided, fea_b = fea_a
    if fea_b is None:
        fea_b = fea_a

    # Calculate kernel matrix K
    # e^{-(||x - y||^2) / 2t^2}
    if kernel_type == 'gaussian':
        D = EuDist2(fea_a, fea_b).square()
        K = torch.exp(- D / t)
    # I(||x - y|| < t/2)/t
    elif kernel_type == 'uniform':
        D = EuDist2(fea_a, fea_b)
        K = torch.ones_like(D)
        K[D > (t / 2)] = 0
        K /= t
    # (x' * y)^t
    elif kernel_type == 'polynomial':
        D = fea_a @ fea_b.t()
        K = D.pow(t)
    # (x' * y + 1)^t
    elif kernel_type == 'polyplus':
        D = fea_a @ fea_b.t()
        K = (D + 1).pow(t)
    # x' * y
    elif kernel_type == 'linear':
        K = fea_a @ fea_b.t()
    else:
        raise NotImplementedError("Kernel Type Not Implemented.")

    # Ensure K is symmetric if fea_b is None
    if fea_b is None:
        K = torch.max(K, K.t())
    
    return K


if __name__ == "__main__":
    # Test the function
    a = torch.rand(10000, 10)
    b = None
    for kernel_type in ['Gaussian', 'Uniform', 'Polynomial', 'PolyPlus', 'Linear']:
        print("Testing kernel type:", kernel_type)
        K = construct_kernel(a, b, {'kernel_type': kernel_type, 't': None})
        print("Shape of K:", K.shape)
