import sys
sys.path.append("../")

from numpy import *
from Modules.sample import *
from gensim.models.keyedvectors import KeyedVectors
from collections import Counter
import nltk
from nltk.corpus import stopwords
from scipy import special
from ARS import *
from train import *

import gensim
import os
import pickle
import argparse

#functions 
log = np.log
gammafn = special.gamma
digammafn = special.digamma
trans = np.transpose
dot = np.dot
det = np.linalg.det
exp = np.exp 

remove_stopwords = True
min_count = 200

def sprint(x):
    if type(x) != str:
        x = str(x)
    sys.stdout.write(x)

def sflush():
    sys.stdout.flush()

def spf(x):
    sprint(x)
    sflush()

class Model:
    def __init__(self, 
            dataDir, 
            extract_google_vec, 
            google_vec_type = "google", #'google' OR 'glove'
            google_vec_dim = 300 #50, 100, 200, 300 for glove, 300 for google
            ):
        global remove_stopwords
        self.remove_stopwords = remove_stopwords
        self.stopwords = stopwords.words("english")

        self.dataDir = dataDir
        self.text = open(dataDir).read()
        self.hyperparams = Hyperparameters()
        self.hyperparams.wordvec_dim = google_vec_dim
        spf("Building Vocab ...")
        self.build_vocab()
        spf("DONE: Reduced Vocab Size = " + str(len(self.reduced_vocab)) + "/" + str(len(self.vocab)) + "\n")

        if google_vec_type == "google":
            google_vec_file = "../google/GoogleNews-vectors-negative300.bin"
        elif google_vec_type == "glove":
            google_vec_file = "../google/glove.6B." + str(google_vec_dim) + "d.txt"

        self.extract_global_word_vectors(extract_google_vec, google_vec_file, google_vec_type)
        spf("Building Word structure ...")
        self.build_word_sense_hierarchy()
        spf("DONE")


    def build_vocab(self):
        self.vocab = self.text.split()
            
        self.vocab_counter = Counter(self.vocab)
        self.reduced_vocab = [w for w in self.vocab_counter \
            if self.vocab_counter[w]>self.hyperparams.min_count]

        if self.remove_stopwords:
            self.reduced_vocab = [w for w in self.reduced_vocab if w not in self.stopwords]

        self.vocab = list(set(self.vocab))

    def extract_global_word_vectors(self, extract_google_vec = False, \
            google_vec_file = \
            "../google/GoogleNews-vectors-negative300.bin",
            google_vec_type = "google"):

        dataDir = self.dataDir
        if extract_google_vec==False and \
                os.path.exists(dataDir+"_dir_" + google_vec_type + "_vec.pkl"):
                    
            f = open(dataDir + "_dir_"+ google_vec_type + "_vec.pkl", "rb")
            self.google_vec = pickle.load(f)

        else:
            sprint(google_vec_type + "vectors for " + dataDir + " not found")
            sprint("...Extracting word vectors\n")

            if google_vec_type == "google":
                full_google_vec = KeyedVectors.load_word2vec_format(\
                    google_vec_file, binary=True)

            elif google_vec_type == "glove":
                from gensim.test.utils import datapath, get_tmpfile
                from gensim.scripts.glove2word2vec import glove2word2vec
                glove_file = datapath(os.path.abspath(google_vec_file))
                tmp_file = tmp_file = get_tmpfile("test_word2vec.txt")
                glove2word2vec(glove_file, tmp_file)
                full_google_vec = KeyedVectors.load_word2vec_format(tmp_file)

            self.google_vec = {}
            words_not_present = []

            for word in self.vocab:
                if word not in full_google_vec.vocab:
                    words_not_present.append(word)
                else:
                    self.google_vec[word] = full_google_vec[word]
            sprint("Words missing : " + str(len(words_not_present)))
            open(dataDir + "_words_not_present", "w").write(\
                    "\n".join(words_not_present))
            f = open(dataDir + "_dir_" + google_vec_type +"_vec.pkl", "wb")
            sprint("...Saving word vectors")
            pickle.dump(self.google_vec, f)
            sprint('...DONE\n')
            sflush()
        
        #average google_vec
        dim = self.hyperparams.wordvec_dim
        self.average_google_vec = np.zeros(dim)
        for word in self.google_vec:
            self.average_google_vec += self.google_vec[word]
        self.average_google_vec /= len(self.google_vec)

        #Covariance matrix of google_vec
        tmp = []
        for word in self.google_vec:
            vec = self.google_vec[word]
            if type(vec) != str:
                tmp.append(vec)
        tmp = np.matrix(tmp)
        # self.cov_google_vec = np.dot(np.transpose(tmp), tmp) * 1e-7

    def build_word_sense_hierarchy(self):
        self.Words = {}
        cnt = 1
        not_found = []
        print()

        for word in self.reduced_vocab:
            if os.path.exists(self.dataDir + "_words/" + word):
                spf(str(cnt) + "\r" )
                cnt += 1
                self.Words[word] = Word(word, self)
            else:
                not_found.append(word)
            
        i = 0
        while i < len(self.reduced_vocab):
            if self.reduced_vocab[i][0] in not_found:
                del self.reduced_vocab[i]
            else:
                i+=1


class Hyperparameters:
    def __init__(self):
        global min_count
        self.wordvec_dim = 300
        self.min_count = min_count #Vocab min_count
       
class Sense:
    def __init__(self, word, model):
        self.word = word
        self.word_str = word.word
        self.dim = model.hyperparams.wordvec_dim
        hyperparams = model.hyperparams
        self.mu, self.S = self.word.sample_params()
        self.mu = np.matrix(self.mu)

        #data = num_data_points X word_vec_dim
        #X_sum = sum(data, axis = 0)
        #XXT = transpose(data) X data
        #num instances = data points for this sense
        # self.X_sum = np.zeros(self.dim)
        # self.XXT = np.zeros([self.dim, self.dim])
        self.num_instances = 0

        # self.calculate_computation_vars()


    def update_data_vars(self, x):
        if len(x.shape) == 1:
            x.resize(1, x.shape[0])
        self.X_sum = self.X_sum*self.num_instances + \
                     np.sum(x, axis=0)
        self.X_sum /= (self.num_instances + len(x))
        self.XXT += dot(transpose(x), x)
        self.num_instances += len(x)
        self.word.num_instances += len(x)
        self.calculate_computation_vars()

    def calculate_computation_vars(self ):
        beta = self.word.beta
        W = self.word.W
        eps = self.word.eps
        rho = self.word.rho
        #variables to ease computation
        self.eps_star = rho*eps + self.X_sum
        self.eps_star /= rho + self.num_instances

        eps.resize(1, eps.size)
        self.W_star = beta*W +rho*np.dot(np.transpose(eps), eps)
        self.W_star += self.XXT
        self.W_star += (rho + self.num_instances) * \
                       np.dot(np.transpose(self.eps_star), \
                       self.eps_star)
        # self.eps_star = 0

class Word:
    def __init__(self, word, model):
        self.word = word
        self.model = model
        hyperparams = model.hyperparams
        self.alpha = gamma(1.0, 1.0)
        dim = hyperparams.wordvec_dim
        self.dim = dim
        self.K = 1
        self.pi = [1]
        self.c = 0 
        self.num_instances = 0

        self.set_param_priors()

        # try:
            # self.eps_mean = model.google_vec[self.word]
            # self.eps_cov = model.cov_google_vec
        # except:
            # self.eps_mean = model.average_google_vec
            # self.eps_cov = model.cov_google_vec
        
        # print("-----------------")
        self.eps = gaussian(self.eps_mean, self.eps_cov)
        # print("-----------------")
        self.eps = np.matrix(self.eps)
        # sig = model.cov_google_vec
        sig = self.eps_cov
        try:
            self.W = wishart(dim, 1.0/dim * sig )
        except:
            print("Error")
            self.W = wishart(dim, 1.0/dim * np.eye(self.dim)*0.001)

        self.rho = gamma(1.0/2, 1.0/2, 1)

        tmp = gamma(1.0, 1.0/dim)
        self.beta = 1.0/tmp + dim - 1
        self.senses = [Sense(self, model)]

    def set_param_priors(self):
        global x
        c = Contexts(self.model, self.model.dataDir + "_words/")
        contexts = c.get_all_contexts(self.word)
        self.eps_mean = np.mean(contexts, axis=0)

        x = contexts - self.eps_mean
        n = len(contexts)
        self.eps_cov = dot(trans(x), x) / n

        #Remove This
        # self.eps_cov = np.eye(self.dim)*0.01

    def sample_alpha(self):
        return self.ARS.draw()


    def new_sense(self, sense):
        #self.pi = UPDATE 
        self.K += 1
        self.senses.apend(sense)

    def get_global_vector(self, model):
        try:
            global_vector = model.google_vec[self.word]
        except:
            global_vector = "<UNK>"
        return global_vector

    def sample_params(self):
        try: 
            mu, S = normal_wishart( \
                self.eps,
                self.W, 
                self.rho, 
                self.beta)
        except:
            print("WISHART Error")
            S = np.eye(300)*0.001
            mu = gaussian(self.eps, inv(self.rho*S))

        return mu, S

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("-datadir", type=str, default="../data/small_text8")
    args.add_argument("-extract_google_vec", type=int, default=0)
    args.add_argument("-google_vec_type", type=str, default="google")
    args.add_argument("-google_vec_dim", type=int, default=300)
    opts = args.parse_args()

    m = Model(opts.datadir, opts.extract_google_vec, opts.google_vec_type, opts.google_vec_dim)
    pickle.dump(m, open("model.pkl", "wb"))













