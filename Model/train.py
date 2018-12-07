#Parallelized training
import multiprocessing as mp
import numpy as np
import time
from nltk.corpus import stopwords

from updates import *
norm = np.linalg.norm

params = {"batch_size": 20,
          "num_jobs": 22 }
remove_stopwords = True
stopwords_eng = stopwords.words("english")

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

def get_neighbours_sense(n, model, sense, test_set = "vocab"):
    w = sense.mu
    w.resize(1, w.size)
    list1 = []
    list2 = []
    for j in model.vocab:
        try:
            x = model.google_vec[j]
        except Exception as e:
            continue
        x.resize(w.shape)
        assert(w.shape == x.shape)
        list1.append(norm(w-x))
        list2.append(j)
    list1, list2 = zip(*sorted(zip(list1, list2)))
    print(", ".join(list2[:n]))
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
        
        #randomize the contexts
        perm = np.random.permutation(len(f))
        while 1:
            print(f[perm[i]])
            # tmp = self.get_context_vector(f[i].split())
            tmp = self.get_context_vector(f[perm[i]].split(), word)
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
            tmp = self.get_context_vector(f[i].split(), word )
            if tmp != "UNDEFINED":
                context_array[i] = tmp
        return context_array

    def get_context_vector(self, sent, context_word = None):
        if type(sent) == str:
            sent = sent.split()

        context = np.zeros(self.dim)
        model = self.model
        norm_factor = 0

        for word in sent:
            # print(word, model.google_vec[word])
            try:
                if remove_stopwords and word in stopwords_eng:
                    continue 

                self.model.google_vec[word].resize(context.shape)
                if context_word != None:
                    try:
                        self.model.google_vec[word].resize(self.dim)
                        self.model.google_vec[context_word].resize(self.dim)

                        # prob = dot(self.model.google_vec[word],  \
                            # self.model.google_vec[context_word])

                        # prob /= norm(self.model.google_vec[word])
                        # prob /= norm(self.model.google_vec[context_word])
                        prob = norm(self.model.google_vec[word] - self.model.google_vec[context_word])

                        # if prob == np.inf:
                            # prob = 10

                        if prob < 4:
                            prob = 1.05
                        elif prob >=4 and prob <= 5:
                            prob = 1 - (prob - 4)/4 +0.05
                        elif prob >=5 and prob <=7:
                            prob = 0.75*(1 - (prob - 5)/2) + 0.05
                        elif prob >= 7:
                            prob = 0.05



                        #NEELAKANTAN ET AL 
                        # prob = 1.0 / (1 + exp(-prob))

                        print(round(prob, 2), word, context_word)
                    except Exception as e:
                        print(e, "------->>>")
                else:
                    prob = 1

                prob = 1
                norm_factor += prob
                try:
                    context.resize(self.model.google_vec[word].shape)
                except:
                    print("oooooooooooooooooo")
                context += prob * self.model.google_vec[word]
            except Exception as e:
                # print("______________________")
                if len(str(e).split()) > 3:
                    print(e, "==>")
                continue
        if norm_factor == 0:
            print(sent)
            print("UNDEFINED")
            return "UNDEFINED"
        else:
            context /= norm_factor
        return context

cnt_tmp = 0
def Updater(JobQueue, word_str, model, contexts, num_batches = 20):
    word = model.Words[word_str]
    for i in range(0, num_batches):
        try:
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


            global cnt_tmp
            if cnt_tmp%1 == 0:
                print("______________________")
                print(word_str)
                for j in range(0, len(word.senses)):
                    get_neighbours_sense(10, model, word.senses[j])
                cnt_tmp = 0
                print(cnt_tmp, i)
            cnt_tmp += 1
        except Exception as e:
            print("Exception in Updater")
            print(e)
    pickle.dump(model, open("../modelfiles/chk_word.pkl"))
    print("Saved model")
    print("Done")
    JobQueue.put(word_str)
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
            print("\n\n\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
            print(JobQueue.qsize())
            print(len(model.Words))
            print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n\n\n")

            if JobQueue.qsize() == len(model.Words):
                break

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

    print("Loading Model")
    model = pickle.load(open(opts.modelfile, "rb"))
    print("Done")
    contexts = Contexts(model, model.dataDir + "_words/")
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
