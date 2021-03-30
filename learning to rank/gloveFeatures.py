gloveFile = "C:/Users/Rozemarijn Veelers/Documents\Master/Information Retrieval/Core IR/information_retrieval/data/vectors.txt"

import numpy as np
import scipy.spatial.distance

def cosine_distance_wordembedding_method(s1, s2):
    model = []
    query = 'query'
    passage = 'passage'
    vector_1 = np.mean(model[word] for word in query)
    vector_2 = np.mean(model[word] for word in passage)
    cosine = scipy.spatial.distance.cosine(vector_1, vector_2)

