#Parallelized training
import numpy as np
norm = np.linalg.norm

params = {"batch_size": 10,
          "num_jobs": 10}

def get_neighbours_word(n, model, word):
    w = np.array(model.Words[word].eps)
    list1 = []
    list2 = []
    for j in model.Words:
        i = model.Words[j].word
        x = np.array(model.Words[j].eps)
        list1.append(norm(w-x))
        list2.append(i)
    list1, list2 = zip(*sorted(zip(list1, list2)))
    print(list2[:n])
    print(list1[:n])

def get_neighbours_sense(n, model, sense):
    w = sense.mu
    w.resize(1, w.size)
    list1 = []
    list2 = []
    for j in model.Words:
        i = model.Words[j].word
        x = model.Words[j].eps_mean
        x.resize(w.shape)
        assert(w.shape == x.shape)
        list1.append(norm(w-x))
        list2.append(i)
    list1, list2 = zip(*sorted(zip(list1, list2)))
    print(list2[:n])
    print(list1[:n])
    



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
            print(f[i])
            tmp = self.get_context_vector(f[i].split())
            i += 1
            if i == len(f):
                i = 0

            if tmp == "UNDEFINED":
                continue
            context_array[cnt] = tmp
            cnt += 1

            if cnt == self.batch_size:
                break


        self.wordfile_line[word] = i
        return context_array

    def get_all_contexts(self, word):
        f = open(self.dataDir + word).read().strip().split("\n")
        context_array = np.zeros([len(f), self.dim])

        f = list(set(f))

        for i in range(0, len(f)):
            tmp = self.get_context_vector(f[i].split())
            if tmp != "UNDEFINED":
                context_array[i] = tmp
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
        if cnt == 0:
            print(sent)
            return "UNDEFINED"
        else:
            context /= cnt
        return context






