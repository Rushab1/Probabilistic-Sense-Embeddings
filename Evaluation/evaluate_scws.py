import argparse
import sys
sys.path.append("../Model/")
from train import *
from updates import *
import pickle
import re
import scipy


def evaluate_scws(model):
     global f, g, ratings, tmp, context1, context2, avgRating
     global word1, word2
     g = open("../data/evaluation_data/SCWS/ratings.txt").read().strip().split("\n")

     human_sim = []
     model_sim = []

     for i in range(0, len(g)):
         f = g[i].split(" ")
         w1 = f[0]
         w2 = f[3]
         del f[0], f[0], f[0], f[0], f[0]
         ratings = f[-10:]
         f = " ".join(f)
         
         f = f.split("<b>")
         tmp1 = f[1].split("</b>")

         tmp2 = f[2].split("</b>")

         word1 = tmp1[0].strip().lower()
         context1= f[0] + "___" + tmp1[1]

         word2 = tmp2[0].strip().lower()
         context2 = f[1] + "___" + tmp2[1]
        
         ratings = tmp2[1].split(" ")
         tmp = ratings[0].split(" ")
         avgRating = float(ratings[-11])
         ratings = ratings[-10:]

         for i in range(0, len(ratings)):
             ratings[i] = float(ratings[i])
         
         context1 = re.sub("[!|,|.|;|\"|(|)|\[|\]]", "", context1)
         context2 = re.sub("[!|,|.|;|\"|(|)|\[|\]]", "", context2)

         context_list = [context1, context2]
         for i in range(0, len(context_list)):
             tmp = context_list[i].split("___")
             l = tmp[0]
             l = l.split()
             if len(l) < 5:
                 l = " ".join(l)
             else:
                 l = " ".join(l[-5:])

             r = tmp[1]
             r = r.split()
             if len(r) < 5:
                 r = " ".join(r)
             else:
                 r = " ".join(r[0:5])
                 
             context_list[i] = l + " " + r
         context1 = context_list[0]
         context2 = context_list[1]

         ####################################
         #Calculate Similarity
         c = Contexts(model)
         c1_vec = c.get_context_vector(context1)
         c2_vec = c.get_context_vector(context2)

         try:
             sp = calculate_sense_prob(model, model.Words[word1], c1_vec)
             mu1 = model.Words[word1].senses[argmax(sp)].mu

             sp = calculate_sense_prob(model, model.Words[word2], c2_vec)
             mu2 = model.Words[word2].senses[argmax(sp)].mu
         except:
             continue

         print(word1, word2, len(model.Words[word2].senses))
         human_sim.append(norm(avgRating))
         model_sim.append(norm(mu1-mu2))
         
     corr = scipy.stats.spearmanr(human_sim, model_sim)
     print(corr)
     
     return human_sim, model_sim, corr

if __name__ == "__main__":
     args = argparse.ArgumentParser()
     args.add_argument("-model", type=str, default="latest")
     opts = args.parse_args()

     print("Loading Model")
     if opts.model == "latest":
         f = open("../modelfiles/last_checkpoint.txt").read()
         latest_chkpt = f
         opts.model = "../modelfiles/checkpoint_"+latest_chkpt+".pkl"
     model = pickle.load(open(opts.model, "rb"))

     print("Done\nEvaluating")
     a = evaluate_scws(model)
     
