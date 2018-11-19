from scipy import special
import numpy as np
from scipy.stats import multivariate_normal as multivariate_normal_fn
from ARS import *

#functions 
log = np.log
gammafn = special.gamma
digammafn = special.digamma
trans = np.transpose
dot = np.dot
det = np.linalg.det
exp = np.exp 
normPDF = multivariate_normal_fn.pdf

num_auxilary_classes = 10

def inv(X):
    return np.linalg.inv(X)

#X = num_observations * wordvec_dim
def update_parameters(X, sense):
    global S, eps, rho, W,beta, mu ,mean, tmp, scale_matrix, cov, X_sum
    n = len(X) #num observations
    X_sum = np.matrix(np.sum(X, axis = 0))

    eps = np.matrix(sense.word.eps)

    if eps.shape != X_sum.shape:
        X_sum = np.resize(X_sum, eps.shape)
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
    D = sense.dim

    eps_prior_cov = sense.word.eps_cov
    eps_prior_mean = np.matrix(sense.word.eps_mean)
    eps_prior_mean.resize(sense.dim, 1)
    # eps = np.matrix(eps)
    eps.resize(sense.dim)

    global eps_mean, eps_cov
    eps_mean = inv(eps_prior_cov) + rho*S
    eps_mean = inv(eps_mean)

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

    scale_matrix_W = beta*S + D*inv(sense.word.eps_cov)
    scale_matrix_W = inv(scale_matrix_W)

    sense.word.W = wishart(beta + D, scale_matrix_W )

    #REMOVE THIS PLEASE 
    sense.S = inv(np.eye(300)*0.001)

def ARS_fn(alpha, n, k):
    val =  (k-1.5)*log(alpha)
    val -= 0.5/alpha
    val += log(gammafn(alpha))
    val -= log(gammafn(n+alpha))
    return val

def ARS_fn_der(alpha, n, k):
    val =  (k-1.5)/alpha
    val += 0.5/np.power(alpha,2)
    val += digammafn(alpha)
    val -= digammafn(n+alpha)
    return val
    
def update_alpha(word):
    word.ARS = ARS(ARS_fn, ARS_fn_der, lb=0, xi = [1e-5, 1, 6], use_lower=True,\
                   n=word.num_instances,
                   k=len(word.senses))
    word.alpha = word.ARS.draw(1)[0]

def calculate_sense_prob(model, word, x):
    global S
    D = word.dim * 1.0
    rho = word.rho*1.0
    eps = word.eps
    beta = word.beta*1.0
    W = word.W
    pi = np.pi  
    alpha = word.alpha*1.0
    n = word.num_instances*1.0
 
    c = 1e-3
    invc = 1./c
    sense_prob = np.zeros(len(word.senses))

    for j in range(0, len(word.senses)):
        s = word.senses[j]
        mu = s.mu
        # S = s.S
        #Change THIS
        S = inv(word.eps_cov)

        print(mu[0])
        mu.resize(mu.size, 1)
        x.resize(x.size, 1)

        sense_prob[j] = log(1.0*s.num_instances / (n - 1 + alpha))
        # print(c*S)
        tmp = -D/2*log(2*pi) + 0.5*log(det(c*S)) + D/2*log(invc)
        tmp -= 0.5*np.dot( np.dot(trans(x-mu), S), x-mu )
        sense_prob[j] += tmp

        # print(sense_prob[j], log(det(c*S)) + D*D/2*log(invc))
    print(sense_prob)
    return sense_prob

def update_indicators(model, word):
    global W_star, eps_star
    global num_auxilary_classes

    # D = word.dim * 1.0
    # rho = word.rho*1.0
    # eps = word.eps
    # beta = word.beta*1.0
    # W = word.W
    # pi = np.pi  
    alpha = word.alpha*1.0
    # n = word.num_instances*1.0

    # sense_prob = np.zeros(len(word.senses) + num_auxilary_classes)
    # sense_prob = log(sense_prob)

    for j in range(0, num_auxilary_classes):
        mu, S = word.sample_params()
        #Remove this
        # mu = gaussian(word.eps_mean, eps_cov)
        # S = 1000*np.eye(300)
        sense = Sense(word, model)
        sense.mu = mu
        sense.S = S
        # print(S)
        sense.num_instances = alpha/num_auxilary_classes
        wword.senses.append(sense)

    # sense_prob = calculate_sense_prob(model, word, x)

    # max_prob = np.max(sense_prob)
    # j_max  = np.argmax(sense_prob)
    
    #Recalculate all indicators
    c = Contexts(model, model.dataDir + "_words/")
    x = c.get_all_contexts(word.word)
    n = len(x)
    k = len(word.senses)

    nk = np.zeros(k)
    ind_list = np.zeros(n)
    for i in range(0, n):
        sense_prob = calculate_sense_prob(model, word, x[i])
        argmax = np.argmax(sense_prob)
        ind_list[i] = argmax
        nk[argmax] += 1
    
    print(nk)
    print(ind_list)
    print("_________________")
    nk = list(nk)
    i = 0
    while i < len(word.senses):
        if nk[i] == 0:
            del word.senses[i]
            del nk[i]
            continue
        word.senses[i].num_instances = nk[i]
        i += 1

    word.num_instances = sum(nk)
    #First data point for the word
    # if n == 0:  
        # word.senses[0].update_data_vars(x)
        # return 0

    
    ##########active classes + 1 inactive class
    # for j in range(0, len(word.senses)+1):
        # if j == len(word.senses):
            # word.senses.append(Sense(word, model))
            # s = word.senses[-1]
            # nj = alpha
        # else:
            # nj = word.senses[j].num_instances*1.0

        # W_star = word.senses[j].W_star
        # eps_star = word.senses[j].eps_star
        # det_W_star = np.linalg.det(W_star)

        ####Probability of x_i = j given c(-i), rho, beta, eps, beta, W
        # log_prob = D/2*np.log((rho+nj) / (rho+nj+1))
        # log_prob += D/2*np.log(pi)

        # log_prob += log( gammafn((beta+nj+1)/2) )
        # log_prob -= log( gammafn( (beta+nj+1-D)/2 ) )
        # log_prob += (beta+nj)/2*log(det_W_star)

        # eps_star.resize(1, eps_star.size)
        # x.resize(1, x.size)
        # tmp = W_star + (rho+nj)/(rho+nj+1) * \
                # dot( trans(x-eps_star), x -eps_star  )
        # tmp = det(tmp)

        # log_prob -= (beta+nj+1)/2*log(tmp)

        # prob = exp(log_prob)


        #CHECK THIS - may be wrong  
        # prob *= (nj)/(n+alpha)

        # sense_prob[j] = prob 

    # ci = np.argmax(sense_prob)
    # word.senses[ci].update_data_vars(x)
    # if ci != len(word.senses) - 1:
        # del word.senses[len(word.senses) - 1] 

######x = 1*dim
from model import *
from train import *
def xload(flag, c = None, word="state"):
    global s, t, C, cword, wword, m  
    if flag:
        m = pickle.load(open("./model.pkl", "rb"))
        c = Contexts(m, m.dataDir)
    t = c.get_new_contexts(word)
    C = c
    cword = word
    wword = m.Words[cword]

    s = m.Words[word].senses[0]
    update_parameters(t, s)
    update_indicators(m, wword, t)
