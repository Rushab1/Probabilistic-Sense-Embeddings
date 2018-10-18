from collections import Counter
from nltk.tokenize import word_tokenize
import argparse 
import sys
import os
import re
import multiprocessing as mp
import time


def wordContextMaker(arg, JobQueue, opts, start, end):
    start_time = time.clock()
    global data_words, wc
    context_memory_threshold = 500 #write after 10000 word-context pairs parsed

    #Make word to create dataset for parallel processing
    word_contexts = {}
    numContexts = 0

    #create training contexts for every word
    for i in range(start, end):
        sys.stdout.write("\r" + str(i) )
        sys.stdout.flush()

        #ignore words below threshold number of observations 
        if wc[data_words[i]] <= opts.min_count:
            continue

        #create context
        left_context = " ".join(data_words[max(0, i-5):i])
        right_context = data_words[i+1: min(i+6,len(data_words) )]
        right_context = " ".join(right_context)
        context = left_context + " " + right_context
        
        try:
            word_contexts[data_words[i]].append(context)
        except:
            word_contexts[data_words[i]] = [context]
        numContexts += 1

        #Write when total number of contexts crosses threshold
        if numContexts >= context_memory_threshold:
            JobQueue.put(word_contexts)
            numContexts = 0
            wordContexts = {}

    JobQueue.put(word_contexts)

    done = time.clock() - start_time
    res = 'Process' + str(arg), str(100), done
    JobQueue.put(res)
    return res

def contextWriter(JobQueue, totalJobs):
    '''listens for messages on the q, writes to file. '''

    print("IN WRITER")
    storageDir = re.sub("\.*", "", opts.data_file) 
    storageDir += "_words/"

    if not os.path.exists(storageDir):
        os.mkdir(storageDir)
    else:
        os.system("rm -rf " + storageDir + "/*")

    completedJobCount = 0

    while 1:
        wordContexts = JobQueue.get()
        print("IN WRIter2 ")
        print(JobQueue.qsize())

        if type(wordContexts) == tuple:
            completedJobCount += 1
            if completedJobCount == totalJobs:
                break
            continue

        if wordContexts == 'kill':
            break
        for word in wordContexts:
            try:
                contextList = wordContexts[word]
                contextStr = "\n".join(contextList)
            except Exception as e:
                print(e)
                print("EXCEPTION HERE")
                print(wordContexts)
                sys.exit(0)

            #append context to word file
            wordFile = storageDir + word

            with  open(wordFile, "a") as f:
                try:
                    f.write("%s\n"%contextStr)
                except Exception as e:
                    print(e)
                f.close()

def main(opts):
    #must use Manager queue here, or will not work
    manager = mp.Manager()
    JobQueue = manager.Queue()    
    pool = mp.Pool(mp.cpu_count() + 2)

    #put listener to work first
    watcher = pool.apply_async(contextWriter, (JobQueue,opts.num_jobs))

    #fire off workers
    jobs = []
    for i in range(0, opts.num_jobs):
        l = len(data_words)
        start = int(1.0 * i * l / opts.num_jobs )
        end = int(1.0 * (i+1) * l / opts.num_jobs)
        job = pool.apply_async(wordContextMaker, (i, JobQueue, opts, start, end))
        jobs.append(job)

    #now we are done, kill the listener
    pool.close()
    pool.join()

#Test unit:
def TestWriter(opts):
    print("TESTING if EVERYTHinG WAS DONE CORERCTLY")
    f = open(opts.data_file).read().split()
    numContexts_expected = len(f)

    numContexts_found = 0

    for filename in os.listdir( opts.data_file + "_words/"):
        filename = opts.data_file + "_words/" + filename
        f = open(filename).read().strip()
        f = f.split("\n")
        numLines = len(f)

        numContexts_found += numLines

    print(numContexts_expected)
    print(numContexts_found)

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("-data_file", type = str, default = "data/text8")
    args.add_argument("-min_count", type = int, default = 0)
    args.add_argument("-num_jobs", type = int, default = 5)
    opts =args.parse_args()

    global data_text, data_words, wc
    data_text = open(opts.data_file).read()
    data_words = word_tokenize(data_text) 
    #word count
    wc = Counter(data_words)

    print("STATS: \n #Words = " + str(len(data_words)) + \
            "#Vocab = " + str(len(wc)))
    main(opts)
    TestWriter(opts)
