import sys
sys.path.append("../")
from numpy import *
from Modules.sample import *
from gensim.models.keyedvectors import KeyedVectors
from collections import Counter
import gensim
import os
import pickle

global hyperparams 

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
    def __init__(self, dataDir):
        self.dataDir = dataDir
        self.text = open(dataDir).read()
        self.hyperparams = Hyperparameters()
        spf("Building Vocab ...")
        self.build_vocab()
        spf("DONE\n")
        self.extract_global_word_vectors()
        spf("Building Word structure ...")
        self.build_word_sense_hierarchy()
        spf("DONE")


    def build_vocab(self):
        self.vocab = self.text.split()

        self.vocab_counter = Counter(self.vocab)
        self.reduced_vocab = [w for w in self.vocab_counter \
            if self.vocab_counter[w]>self.hyperparams.min_count]

        self.vocab = list(set(self.vocab))

    def extract_global_word_vectors(self, \
            google_vec_file = \
            "../google/GoogleNews-vectors-negative300.bin"):
        dataDir = self.dataDir

        if os.path.exists(dataDir + "_dir_google_vec.pkl"):
            f = open(dataDir + "_dir_google_vec.pkl", "rb")
            self.google_vec = pickle.load(f)
        else:
            sprint("Google vectors for " + dataDir + " not found")
            sprint("...Extracting word vectors\n")
            full_google_vec = KeyedVectors.load_word2vec_format(\
                    google_vec_file, binary=True)
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
            f = open(dataDir + "_dir_google_vec.pkl", "wb")
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
        self.cov_google_vec = np.dot(np.transpose(tmp), tmp)


    def build_word_sense_hierarchy(self):
        self.Words = {}
        cnt = 1
        for word in self.reduced_vocab:
            spf(str(cnt) )
            cnt += 1
            self.Words[word] = Word(word, self)

class Hyperparameters:
    def __init__(self):
        self.wordvec_dim = 300
        self.min_count = 2 #Vocab min_count
       
class Sense:
    def __init__(self, Word, model):
        self.Word = Word
        self.word = Word.word
        self.dim = model.hyperparams.wordvec_dim
        hyperparams = model.hyperparams

        try:
            self.eps = model.google_vec[self.word]
        except:
            self.eps = model.average_google_vec

        sig = model.cov_google_vec
        self.W = wishart(self.dim, 1.0/self.dim * sig )
        self.rho = gamma(1.0/2, 1.0/2, 1)

        tmp = gamma(1.0, 1.0/self.dim)
        self.beta = 1.0/tmp + self.dim - 1
        self.mu, self.S = normal_wishart(
                self.eps, self.W, self.rho, self.beta)

class Word:
    def __init__(self, word, model):
        self.word = word
        hyperparams = model.hyperparams
        self.alpha = gamma(1.0, 1.0)
        dim = hyperparams.wordvec_dim
        self.pi = dirichlet(np.ones(dim) * self.alpha, 1)
        self.K = 1
        self.senses = [Sense(self, model)]

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


class Context:
    def get_context_vector(self, sent, model):
        if type(sent) == str:
            self.words = sent.split()

        dim = model.hyperparams.wordvec_dim
        context = np.zeros(dim)
        cnt = 0
        for word in sent:
            try:
                context += google_vec[word]
                cnt += 1
            except:
                continue
        context /= cnt




if __name__ == "__main__":
    m = Model("../data/small_text8")













