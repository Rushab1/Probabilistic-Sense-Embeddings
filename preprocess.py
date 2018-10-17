from collections import Counter
from nltk.tokenize import word_tokenize
import argparse 
import sys
import os
import re
import multiprocessing as mp


def make_word_files(opts):
    global data_words, data_text
    data_text = open(opts.data_file).read()
    data_words = word_tokenize(data_text) 
    #word count
    wc = Counter(data_words)
    
    #Make word to create dataset for parallel processing
    # word_contexts = {}
    words_dir = re.sub("\.*", "", opts.data_file) 
    words_dir += "_words/"
    if not os.path.exists(words_dir):
        os.mkdir(words_dir)
    else:
        os.system("rm -rf " +  words_dir + "/*")

    #create training contexts for every word
    for i in range(0, len(data_words)):
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
        
        # try:
            # word_contexts[data_words[i]].append(context)
        # except:
            # word_contexts[data_words[i]] = [context]

        #append context to word file
        word_file = words_dir + data_words[i]
        open(word_file, "a").write(context +"\n")

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("-data_file", type = str, default = "data/text8")
    args.add_argument("-min_count", type = int, default = 0)
    opts =args.parse_args()
    make_word_files(opts)
