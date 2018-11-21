#Parallelized training
import multiprocessing as mp
import numpy as np
import time

from updates import *
norm = np.linalg.norm

params = {"batch_size": 20,
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
    print(", ".join(list2[:n]))
    # print(list1[:n])

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
    print(", ".join(list2[:n]))
    # print(list1[:n])
    



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
        
        #randomize the contexts
        perm = np.random.permutation(len(f))
        while 1:
            # tmp = self.get_context_vector(f[i].split())
            tmp = self.get_context_vector(f[perm[i]].split())
            i += 1
            if i == len(f):
                i = 0

            if type(tmp) == str:
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
            print("UNDEFINED")
            return "UNDEFINED"
        else:
            context /= cnt
        return context


def Updater(JobQueue, word_str, model, contexts, num_batches = 20):
    try:
        word = model.Words[word_str]

        for i in range(0, num_batches):
            c = contexts.get_new_contexts(word_str)
            
            nk = update_indicators(model, word)
            ind  =[]
            for j in range(0, len(c)):
                sp = calculate_sense_prob(model, word, c[j])
                ind.append(argmax(sp))


            for j in range(0, len(c)):
                s = update_parameters(c[j], word.senses[ind[j]])  


            try:
                update_alpha(word)
            except:
                print("ALPHA ERROR")

            print("______________________")
            print(word_str)
            for j in range(0, len(word.senses)):
                get_neighbours_sense(10, model, word.senses[j])
    except Exception as e:
        print("Exception in Updater")
        print(e)
    print("Done")
    return word


def Writer(JobQueue, checkpoint_dir, model):
    while 1:
        try:
            time.sleep(200)

            f = open(checkpoint_dir + "last_checkpoint.txt", "r")
            chkpt_num = int(f.read())
            f.close()
            
            chkpt_num += 1
            chkpt_file = open(checkpoint_dir + \
                    "checkpoint_"+str(chkpt_num)+".pkl", "wb")
            pickle.dump(model,chkpt_file)

            f = open(checkpoint_dir+"last_checkpoint.txt", "w")
            f.write(str(chkpt_num))
            f.close()
        except Exception as e:
            print(e)

def main(opts):
    if not os.path.exists(opts.checkpoint_dir):
        os.mkdir(opts.checkpoint_dir)
        os.system("echo 0 >> " + opts.checkpoint_dir + "last_checkpoint.txt")
    if opts.restart:
        os.system("rm -rf "+opts.checkpoint_dir + "*")
        os.system("echo 0 >> " + opts.checkpoint_dir + "last_checkpoint.txt")
        f = open(opts.checkpoint_dir+"last_checkpoint.txt", "w")
        f.write('0')
        f.close()
    else:
        f = open(opts.checkpoint_dir+"last_checkpoint.txt")
        last_checkpoint = f.read()
        f.close()
        if last_checkpoint != 0:
            opts.modelfile = opts.checkpoint_dir+"checkpoint_" + \
                    str(last_checkpoint) + ".pkl"

    model = pickle.load(open(opts.modelfile, "rb"))
    contexts = Contexts(model)
    try:
        manager = mp.Manager()
        JobQueue = manager.Queue()
        pool = mp.Pool(mp.cpu_count() + 2)

        jobs = []
        Words = model.Words.keys()

        model_writer = pool.apply_async(Writer, \
                (JobQueue, opts.checkpoint_dir, model))
        
        for i in range(0, len(Words)):
            try:
                job = pool.apply_async(Updater, (JobQueue, Words[i], \
                model, contexts, opts.num_batches))
                pass
            except Exception as e:
                print(e)
            jobs.append(job)

        pool.close()
        pool.join()
    except Exception as e:
        print(e)
