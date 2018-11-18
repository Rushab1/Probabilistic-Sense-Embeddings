#Parallelized training
import numpy as np

params = {"batch_size": 10,
          "num_jobs": 10}

class Contexts:
    def __init__(self, model, dataDir = "../data/small_text8_words/"):
        global params
        self.batch_size = params["batch_size"]
        self.num_jobs = params["num_jobs"]

        self.model = model
        self.vocab = model.reduced_vocab
        self.dataDir = dataDir
        self.dim = model.hyperparams.wordvec_dim
        
        self.wordfile_line = {}
        for word in self.vocab:
            self.wordfile_line[word] = 0

    def get_new_contexts(self, word):
        start = self.wordfile_line[word]
        f = open(self.dataDir + word).read().strip().split("\n")
        context_array = np.zeros([self.batch_size, self.dim])
        
        cnt = 0
        self.f = f
        i = start
        
        while 1:
            context_array[cnt] = self.get_context_vector(f[i].split())
            cnt += 1
            i += 1
            if i == len(f):
                i = 0
            if cnt == self.batch_size:
                break


        self.wordfile_line[word] = i
        return context_array

    def get_all_contexts(self, word):
        f = open(self.dataDir + word).read().strip().split("\n")
        context_array = np.zeros([len(f), self.dim])

        f = list(set(f))

        for i in range(0, len(f)):
            context_array[i] = self.get_context_vector(f[i].split())
        return context_array

    def get_context_vector(self, sent):
        if type(sent) == str:
            sent = sent.split()

        context = np.zeros(self.dim)
        cnt = 0
        model = self.model

        for word in sent:
            try:
                context += self.model.google_vec[word]
                cnt += 1
            except Exception as e:
                # print("______________________")
                # print(e)
                continue
        context /= cnt
        return context






