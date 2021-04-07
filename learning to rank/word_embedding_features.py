from whoosh.qparser import QueryParser
from whoosh.index import create_in, open_dir
from whoosh.fields import *
import numpy as np
import scipy.spatial.distance

def create_features(indexQ, indexP, outputfile, readfile):

    # Make dictionary of passage vectors with passageID as key
    passages = {}
    with open("../data/" + str(indexP), 'r', encoding='utf8') as infile:
        total_data_per_line = infile.read().split("\n")

        for line in total_data_per_line:
            if len(line) < 2:
                continue
            passages[line.split('\t')[0]] = line.split('\t')[2]

    queries = {}
    with open("../data/" + str(indexQ), 'r', encoding='utf8') as infile:
        total_data_per_line = infile.read().split("\n")

        for line in total_data_per_line:
            if len(line) < 2:
                continue
            queries[line.split('\t')[0]] = line.split('\t')[2]

    i = 0
    feature_file = open("../featuresets/" + outputfile, 'a', newline='')

    for line in list(open("../data/" + readfile, encoding='utf8')):
        queryID = line.split('\t')[0]
        passageID = str(line.split('\t')[2])

        if queries.get(queryID) == None:
            print("Has no queryID: " + str(queryID))
            continue
        if passages.get(passageID) == None:
            print("Has no passageID: " + str(passageID))
            continue

        qvector = np.array(list(map(float, queries.get(queryID).strip('[').strip(']').split(', '))))
        pvector = np.array(list(map(float, passages.get(passageID).strip('[').strip(']').split(', '))))

        cosine = scipy.spatial.distance.cosine(qvector, pvector)
        euclidian = np.linalg.norm(qvector - pvector)

        if i % 10 == 0:
            print(i)

        i = i+1
        feature_file.write('qid:' + str(queryID) + ' pid:' + str(passageID) + ' 1:' + str(cosine) + ' 2:' + str(euclidian) + '\n')
    feature_file.close()


if __name__ == '__main__':
    indexQ = "word_embeddings/fasttext_scripts/7500queries_fasttext_embedding.txt"
    indexP = "word_embeddings/passages_fasttext_embedding.txt"
    outputfile = "training/fast_features_training.txt"
    readfile = "extended.qrels.dev.small.tsv"
    create_features(indexQ, indexP, outputfile, readfile)




