from whoosh.qparser import QueryParser
from whoosh.index import create_in, open_dir
from whoosh.fields import *
import numpy as np
import scipy.spatial.distance

def create_features(indexQ, indexP, outputfile, readfile):
    schemaQ = Schema(title=TEXT(stored=True), content=TEXT(stored=True))
    ixQ = open_dir('../index/' + indexQ, schema=schemaQ)
    schemaP = Schema(title=TEXT(stored=True), content=TEXT(stored=True))
    ixP = open_dir('../index/' + indexP, schema=schemaP)

    i = 0
    feature_file = open("../featuresets/" + outputfile, "a", newline='')
    for line in list(open("../data/" + readfile, encoding='utf8')):
        queryID = line.split(' ')[0]
        passageID = line.split(' ')[2]
        relevance = int(line.split(' ')[3])

        qp = QueryParser('title', schema=ixQ.schema)
        q = qp.parse(queryID)

        pp = QueryParser('title', schema=ixP.schema)
        p = qp.parse(passageID)

        with ixQ.searcher() as s:
            results = s.search(q)
            qvector = np.array(list(map(float, results[0].get('content').strip('[').strip(']').split(', '))))

        with ixP.searcher() as s:
            results = s.search(p)
            pvector = np.array(list(map(float, results[0].get('content').strip('[').strip(']').split(', '))))

        cosine = scipy.spatial.distance.cosine(qvector, pvector)
        euclidian = np.linalg.norm(qvector - pvector)

        if i % 10 == 0:
            print(i)
            print(cosine)
            print(euclidian)

        i = i+1

        feature_file.write('qid:' + str(queryID) + ' pid:' + str(passageID) + ' 1:' + str(cosine) + ' 2:' + str(euclidian) + '\n')
    feature_file.close()

if __name__ == '__main__':
    indexQ = "fastqvector-index"
    indexP = "fastpvector-index"
    outputfile = "fast_features.txt"
    readfile = "2019qrels-pass.txt"
    create_features(indexQ, indexP, outputfile, readfile)
