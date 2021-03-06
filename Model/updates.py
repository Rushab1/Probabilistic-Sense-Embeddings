from scipy import special
import numpy as np
from scipy.stats import multivariate_normal as multivariate_normal_fn
from ARS import *
from train import *

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
sense_per = 0.1
nu = 0


def inv(X):
    return np.linalg.inv(X)

#X = num_observations * wordvec_dim
def update_parameters(X, sense):
    X = deepcopy(X)
    # print("__________________________")
    # get_neighbours_sense(10, sense.word.model, sense)
    global S, eps, rho, W,beta, mu ,mean, tmp, scale_matrix, cov, X_sum
    
    X.resize(1,sense.dim)
    n = len(X) #num observations
    X_sum = np.matrix(np.sum(X, axis = 0))
    assert X_sum.size == sense.dim

    eps = np.matrix(sense.word.eps)

    if eps.shape != X_sum.shape:
        X_sum = np.resize(X_sum, eps.shape)

    rho = sense.word.rho
    W = sense.word.W
    beta = sense.word.beta
    S = sense.S

    if n == X.size:
        print("Input should be of shape num_obs X dim")
        return
    #Update mu
    cov = inv((rho + n)*S)
    eps.resize(eps.size, 1)
    X_sum.resize(eps.size, 1)
    mean = dot(cov, dot(S, rho* eps + X_sum))
    
    mean.resize(sense.mu.shape)
    # print(sense.mu.shape)
    # print(mean.shape)
    # print("________________________")
    try:
        tmp = (1-nu)*gaussian(mean, cov)

        sense.mu.resize(sense.dim, 1)
        tmp.resize(sense.dim, 1)
        # print(sense.mu.shape, tmp.shape, "===")
        # sense.mu = deepcopy(tmp)
        sense.mu = nu*sense.mu + tmp
    except:
        print("except snese.mu")

    # get_neighbours_sense(10, sense.word.model, sense)

    #update precision S
    # print(sense.mu.shape)
    mu = sense.mu
    X.resize(1, mu.size)
    mu.resize(1, mu.size)
    tmp =  dot(trans(X-mu), X-mu)
    scale_matrix = beta*W + tmp
    try:
        # print(sense.S.shape)
        tmp = (1-nu)*inv(wishart(n + beta, scale_matrix))
        sense.S = inv(nu*inv(sense.S) + tmp)
        # sense.S = deepcopy(tmp)
        # print(sense.S.shape)
    except Exception as e:
        print(e)
        print("except sense.S")
        
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
 
    try:
        sense.word.eps = nu*sense.word.eps + (1-nu)*gaussian(eps_mean, eps_cov)
    except:
        print("Except sense.word.eps")

    try:
        update_rho(sense)
    except Exception as e:
        print(e)
        print("RHO ERROR")

    scale_matrix_W = beta*S + D*inv(sense.word.eps_cov)
    scale_matrix_W = inv(scale_matrix_W)
    


    try:
        sense.word.W = inv(nu*inv(sense.word.W) + (1-nu)*inv(wishart(beta + D, scale_matrix_W )))
    except:
        print("except W")

    #REMOVE THIS PLEASE 
    # sense.S = inv(np.eye(sense.dim)*0.001)

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
    
def update_rho(sense):
    mu = sense.mu
    S = sense.S
    eps = sense.word.eps
    D = int(sense.dim)

    mu_shape = mu.shape 
    eps_shape = eps.shape

    mu.resize(D,1)
    eps.resize(D,1)

    tmp = dot(transpose(mu-eps), S)
    tmp = dot(tmp, mu-eps)

    shape = 0.5*D+0.25
    #second 0.5 = alpha/(2beta) = 0.5 since alpha =beta = 0.5
    scale = 0.5*tmp + 0.5

    sense.word.rho = float(nu*sense.word.rho + (1-nu)*gamma(shape, scale, 1))
    # print(sense.word.word, sense.word.rho)
    mu.resize(mu_shape)
    eps.resize(eps_shape)

def update_alpha(word):
    word.ARS = ARS(ARS_fn, ARS_fn_der, lb=0, xi = [1e-1, 1, 2], use_lower=True,\
                   n=word.num_instances,
                   k=len(word.senses))
    try:
        word.alpha = nu*word.alpha + (1-nu)*word.ARS.draw(1)[0]
    except:
        print("execept alpha")

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
        S = s.S
        #Change THIS
        # S = inv(word.eps_cov)

        mu.resize(mu.size, 1)
        x.resize(x.size, 1)

        sense_prob[j] = log(1.0*s.num_instances / (n - 1 + alpha))
        tmp = -D/2*log(2*pi) + 0.5*log(det(c*S)) + D/2*log(invc)
        tmp -= 0.5*np.dot( np.dot(trans(x-mu), S), x-mu )
        sense_prob[j] += tmp

        # print(sense_prob[j], log(det(c*S)) + D*D/2*log(invc))
    # print(sense_prob)
    return sense_prob

def update_indicators(model, word):
    global num_auxilary_classes
    alpha = word.alpha*1.0

    for j in range(0, num_auxilary_classes):
        # mu, S = word.sample_params()
        sense = Sense(word, model)
        # sense.mu = mu
        # sense.S = S
        sense.num_instances = alpha/num_auxilary_classes
        word.senses.append(sense)

    #Recalculate all indicators
    try:
        from train import Contexts
        c = Contexts(model, model.dataDir + "_words/")
    except Exception as e:
        print("1----------------")
        print(e)
        print("2----------------")
        sys.exit(0)
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
    
    #delete senses with no data i.e. num_instances = 0
    n = int(np.sum(nk))
    nk = list(nk)
    i = 0
    while i < len(word.senses):
        if nk[i] < sense_per*n:
            del word.senses[i]
            del nk[i]
            continue
        word.senses[i].num_instances = nk[i]
        i += 1

    k = len(word.senses)
    nk = np.zeros(k)
    ind_list = np.zeros(n)
    for i in range(0, n):
        sense_prob = calculate_sense_prob(model, word, x[i])
        argmax = np.argmax(sense_prob)
        ind_list[i] = argmax
        nk[argmax] += 1
    
    word.num_instances = sum(nk)
    return nk

######x = 1*dim
from model import *
from train import *
def xload(flag, c = None, word="state"):
    global s, t, C, cword, wword, m  
    if flag:
        m = pickle.load(open("./model.pkl", "rb"))
        c = Contexts(m, m.dataDir+"_words/")
    t = c.get_new_contexts(word)
    C = c
    cword = word
    wword = m.Words[cword]

    s = m.Words[word].senses[0]
    update_parameters(t, s)
    update_indicators(m, wword)
