import numpy as np
import numpy.random as rand
import scipy.stats as stats
import numpy.random as npr
from numpy.linalg import inv, cholesky
from scipy.stats import chi2


multivariate_normal = np.random.multivariate_normal
def gaussian(mu, sig, size):
    return multivariate_normal(mu, sig, size)

def invwishartrand_prec(nu,phi):
    return inv(wishartrand(nu,phi))

def invwishartrand(nu, phi):
    return inv(wishartrand(nu, inv(phi)))

def wishartrand(nu, phi):
    dim = phi.shape[0]
    chol = cholesky(phi)
    #nu = nu+dim - 1
    #nu = nu + 1 - np.arange(1,dim+1)
    foo = np.zeros((dim,dim))

    for i in range(dim):
        for j in range(i+1):
            if i == j:
                foo[i,j] = np.sqrt(chi2.rvs(nu-(i+1)+1))
            else:
                foo[i,j]  = npr.normal(0,1)
    return np.dot(chol, np.dot(foo, np.dot(foo.T, chol.T)))
