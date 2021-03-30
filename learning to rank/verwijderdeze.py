from whoosh.index import create_in, open_dir
from whoosh.fields import *
from whoosh.qparser import QueryParser
from rank_bm25 import BM25Okapi



def formatGlove():
    schemaQ = Schema(title=TEXT(stored=True), content=TEXT(stored=True))
    ixQ = open_dir('../index/indexdir', schema=schemaQ)

    i = 0
    trainingFile = open("output/qrels-passcollection.txt", "a", newline='', encoding='utf8')
    for line in list(open("../data/2019qrels-pass.txt", encoding='utf8')):
        print(line)
        # i = i + 1
        queryID = line.split(' ')[0]
        passageID = int(line.split(' ')[2])
        print(passageID)

        with open("../data/collection.tsv", encoding='utf8') as collection2:
            for passage2 in collection2:
                checkID = int(passage2.split('\t')[0])
                if checkID == passageID:
                    # collected.insert(0, passage2.split('\t')[1])
                    print(passage2.split('\t')[0])
                    print(passage2.split('\t')[1])
                    trainingFile.write(str(passage2.split('\t')[0]) + '\t' + passage2.split('\t')[1])
                    trainingFile.close()

if __name__ == '__main__':
    formatGlove()