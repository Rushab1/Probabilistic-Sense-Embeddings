from train import *
m = pickle.load(open("model.pkl", "rb"))

wword = m.Words["brazil"]
s = wword.senses[0]

c = Contexts(m, "../data/text8_words/")
t = c.get_new_contexts("brazil")


