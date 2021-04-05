from whoosh.qparser import QueryParser

gloveFile = "C:/Users/Rozemarijn Veelers/Documents\Master/Information Retrieval/Core IR/information_retrieval/data/vectors.txt"

import numpy as np
import scipy.spatial.distance

def create_feature_file():
    schemaQ = Schema(title=TEXT(stored=True), content=TEXT(stored=True))
    ixQ = open_dir('../index/qvectorindex', schema=schemaQ)
    schemaP = Schema(title=TEXT(stored=True), content=TEXT(stored=True))
    ixP = open_dir('../index/pvectorindex', schema=schemaP)

    i = 0
    feature_file = open("output/glove_feature.txt", "a", newline='')
    for line in list(open("../data/2019qrels-pass.txt", encoding='utf8')):
        queryID = line.split(' ')[0]
        passageID = line.split(' ')[2]
        relevance = int(line.split(' ')[3])

        qp = QueryParser('title', schema=ixQ.schema)
        q = qp.parse(queryID)

        pp = QueryParser('title', schema=ixP.schema)
        p = qp.parse(passageID)

        with ixQ.searcher() as s:
            results = s.search(q)
            qvector = list(results[0].get('content'))

        with ixP.searcher() as s:
            results = s.search(p)
            pvector = list(results[0].get('content'))

        euclidian = np.linalg.norm(qvector - pvector)
        cosine = scipy.spatial.distance.cosine(qvector, pvector)

        if i % 10 == 0:
            print(i)
            print(cosine)
            print(euclidian)

        i = i+1

        feature_file.write('qid:' + str(queryID) + ' pid:' + str(passageID) + ' 1:' + str(cosine) + ' 2:' + str(euclidian) + '\n')
    feature_file.close()

