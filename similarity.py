from pyemd import emd
from time import time
from nltk.corpus import stopwords
from nltk import download
import os
import gensim.models.keyedvectors as word2vec

start_nb=time()

sa = 'hello man, what are you doing'
sb = 'hi there fellow, how are you'

#flag = True
#while flag:
#    for i in range(0,len(ulist)):
#        for j in range(0,len(ulist)):
#            dist(ulist[i],ulist[j])
#            if dist < 0.8:
#                lista.append(ulist[j])
#                ulist = ulist.pop(j)
#            if len(ulist=<1):
#                flag=False
                            
sa = sa.lower().split()
sb = sb.lower().split()

if not os.path.exists('GoogleNews-vectors-negative300.bin'):
    raise ValueError("SKIP: You need to download the google news model")
model = word2vec.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
dist = model.wmdistance(sa,sb)

print(dist)
