def update_indicators(word, x):
    D = word.wordvec_dim * 1.0
    rho = word.rho*1.0
    eps = word.eps
    beta = word.beta*1.0
    W = word.W
    pi = np.pi  
    alpha = word.alpha*1.0
    n = word.num_instances*1.0

    #functions 
    log = np.log
    gamma = scipy.special.gamma
    trans = np.transpose
    dot = np.dot
    det = np.linalg.det
    exp = np.exp 

    sense_prob = np.zeros(len(word.senses) + 1)

    #First data point for the word
    if n == 0:  
        
        return 0

    #Last class in sense list is the inactive class
    for j in range(0, len(word.senses)):
        nj = word.senses[j].num_instances*1.0
        W_star = word.senses[j].W_star
        eps_star = word.senses[j].eps_star
        det_W_star = np.linalg.det(W_star)

        #Probability of x_i = j given c(-i), rho, beta, eps, beta, W
        log_prob = D/2*np.log((rho+nj) / (rho+nj+1))
        log_prob += D/2*np.log(pi)
        log_prob += log( gamma((beta+nj+1)/2) )
        log_prob -= log( gamma( (beta+nj+1-D)/2 ) )
        log_prob += (beta+nj)/2*log(det_W_star)

        tmp = W_star + (rho+nj)/(rho+nj+1) * \
                dot( trans(x-eps_star), x -eps_star  )
        tmp = det(tmp)

        log_prob -= (beta+nj+1)/2*log(tmp)

        prob = exp(log_prob)
        prob *= (n - nj)/(n+alpha)

        sense_prob[i] = prob 

