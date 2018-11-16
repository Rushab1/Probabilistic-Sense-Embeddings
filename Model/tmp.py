
def calculate_computation_vars(sense ):
        beta = sense.word.beta
        W = sense.word.W
        eps = sense.word.eps
        rho = sense.word.rho
        #variables to ease computation
        sense.eps_star = rho*eps + sense.X_sum
        sense.eps_star /= rho + sense.num_instances

        eps.resize(1, eps.size)
        print(eps.shape)
        sense.W_star = beta*W 
        global tmp
        tmp = rho*np.dot(np.transpose(eps), eps)
        print(tmp.shape)
        sense.W_star += tmp
        sense.W_star += sense.XXT
        sense.W_star += (rho + sense.num_instances) * \
                       np.dot(np.transpose(sense.eps_star), \
                       sense.eps_star)
        sense.eps_star = 0

