from scipy import special
import numpy as np
#functions 
log = np.log
gammafn = special.gamma
trans = np.transpose
dot = np.dot
det = np.linalg.det
exp = np.exp 


def inv(X):
    return np.linalg.inv(X)

#X = num_observations * wordvec_dim
def update_parameters(X, sense):
    global S, eps, rho, W,beta, mu ,mean, tmp, scale_matrix, cov, X_sum
    n = len(X) #num observations
    X_sum = np.matrix(np.sum(X, axis = 0))

    eps = np.matrix(sense.word.eps)

    if eps.shape != X_sum.shape:
        X_sum.resize(eps.shape)
    rho = sense.word.rho
    W = sense.word.W
    beta = sense.word.beta
    S = sense.S

    #Update mu
    cov = inv((rho + n)*S)

    eps.resize(eps.size, 1)
    X_sum.resize(eps.size, 1)
    mean = np.dot(cov, np.dot(S, rho* eps + X_sum))
    sense.mu = gaussian(mean, cov)

    #update precision S
    mu = sense.mu
    tmp =  np.dot(np.transpose(X-mu), X-mu)
    scale_matrix = beta*W + tmp
    sense.S = wishart(n + beta, scale_matrix)
    
    S = sense.S

    eps_prior_cov = sense.word.eps_cov
    eps_prior_mean = np.matrix(sense.word.eps_mean)
    eps_prior_mean.resize(sense.dim)
    # eps = np.matrix(eps)
    eps.resize(sense.dim)

    global eps_mean, eps_cov
    eps_mean = inv(eps_prior_cov) + rho*S
    eps_mean = inv(eps_mean)

    print(S.shape, mu.shape)
    tmp = np.dot(inv(eps_prior_cov), eps_prior_mean)
    mu = np.matrix(mu)
    mu.resize(sense.dim, 1)
    tmp.resize(sense.dim, 1)
    tmp += np.dot(S, rho*mu)
    tmp = np.matrix(tmp)
    tmp.resize(sense.dim, 1)

    eps_mean = np.dot(eps_mean, tmp)

    eps_cov = inv(inv(eps_prior_cov) + rho*S)
    sense.word.eps = gaussian(eps_mean, eps_cov)
# def update_hyperparams(sense):


def update_indicators(model, word, x):
    global W_star, eps_star
    D = word.dim * 1.0
    rho = word.rho*1.0
    eps = word.eps
    beta = word.beta*1.0
    W = word.W
    pi = np.pi  
    alpha = word.alpha*1.0
    n = word.num_instances*1.0

    sense_prob = np.zeros(len(word.senses) + 1)

    #First data point for the word
    if n == 0:  
        word.senses[0].update_data_vars(x)
        word.num_instances += 1
        word.senses[0].num_instances += 1
        return 0

    #active classes + 1 inactive class
    for j in range(0, len(word.senses)+1):
        if j == len(word.senses):
            word.senses.append(Sense(word, model))
            s = word.senses[-1]
            print(s.num_instances)
            nj = alpha
        else:
            nj = word.senses[j].num_instances*1.0

        W_star = word.senses[j].W_star
        eps_star = word.senses[j].eps_star
        print(eps_star)
        det_W_star = np.linalg.det(W_star)

        #Probability of x_i = j given c(-i), rho, beta, eps, beta, W
        log_prob = D/2*np.log((rho+nj) / (rho+nj+1))
        log_prob += D/2*np.log(pi)
        log_prob += log( gammafn((beta+nj+1)/2) )
        log_prob -= log( gammafn( (beta+nj+1-D)/2 ) )
        log_prob += (beta+nj)/2*log(det_W_star)

        eps_star.resize(1, eps_star.size)
        x.resize(1, x.size)
        tmp = W_star + (rho+nj)/(rho+nj+1) * \
                dot( trans(x-eps_star), x -eps_star  )
        tmp = det(tmp)

        print("_____________________-")
        print(tmp)
        log_prob -= (beta+nj+1)/2*log(tmp)
        print("_____________________-")

        prob = exp(log_prob)
        prob *= (n - nj)/(n+alpha)

        sense_prob[j] = prob 

    ci = np.argmax(sense_prob)
    word.senses[ci].update_data_vars(x)
    if ci != len(word.senses) - 1:
        del word.senses[len(word.senses) - 1] 

#x = 1*dim
from model import *
from train import *
def xload(flag, c = None, word="state"):
    global s, t, C, cword, wword, m  
    if flag:
        m = pickle.load(open("./model.pkl", "rb"))
        c = Contexts(m)
    t = c.get_new_contexts(word)
    C = c
    cword = word
    wword = m.Words[cword]

    s = m.Words[word].senses[0]
    update_parameters(t, s)
    update_indicators(m, wword, t)
