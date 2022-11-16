import numpy as np

def find_row_in_matrix(A, x):
    return np.where((np.abs(A - x)<1.e-4).all(axis=1))[0]

def add_row_to_matrix(A, x):
    if isinstance(x, list):
        x = np.array(x)
    assert x.ndim == A.ndim or x.ndim == A.ndim-1
    assert A.ndim == 2
    if x.ndim == 1:
        x = x.reshape((1, -1))
    return np.concatenate((A, x), axis=0)

def confidence_interval(v, alpha=0.95):
    if isinstance(v,int) or v.ndim==0:
        mean=v
        err=0
    elif v.ndim==1:
        mean = np.mean(v)
        err = alpha*np.std(v)/np.sqrt(v.shape[0])
    else:
        mean = np.mean(v,axis=1)
        err = alpha*np.std(v,axis=1)/np.sqrt(v.shape[1])
    return mean, err