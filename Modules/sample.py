import numpy as np
import numpy.random as rand
import scipy.stats as stats
import numpy.random as npr
from numpy.linalg import inv, cholesky
from scipy.stats import chi2

from copy import deepcopy

multivariate_normal = np.random.multivariate_normal
def gaussian(mu, sig, size=1):
    mu.resize(mu.size)
    if size == 1:
        return multivariate_normal(mu, sig)
    else:
        return multivariate_normal(mu, sig, size)

def invwishartrand_prec(nu,phi):
    return inv(wishart(nu,phi))

def invwishartrand(nu, phi):
    return inv(wishart(nu, inv(phi)))

def wishart(nu, phi):
    from scipy.stats import wishart
    return wishart.rvs(nu, phi)

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

def dirichlet(alpha, size):
    if size == 1:
        return np.random.dirichlet(alpha) 
    return np.random.dirichlet(alpha, size) 

def multinomial(n, pvals, size):
    return np.random.multinomial(n, pvals, size)

def normal_wishart(eps, W, rho, beta):
    S = wishart(beta, np.linalg.inv(beta*W))
    mu = gaussian(eps, np.linalg.inv(rho*S))

    return mu, S

def gamma(a, b, size=1):
    shape = 1.0*a/2
    scale = 1.0*2*b/a
    if size == 1:
        return np.random.gamma(shape, scale)
    else:
        return np.random.gamma(shape, scale, size)

def dicrete(probabilities, size):
    return np.random.choice(range(0,len(probabilities)), size, p=probabilities)











