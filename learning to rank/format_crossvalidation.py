from whoosh.index import create_in,open_dir
from whoosh.fields import *
from whoosh.qparser import QueryParser
from rank_bm25 import BM25Okapi

if __name__ == '__main__':
    schemaQ = Schema(title=TEXT(stored=True), content=TEXT(stored=True))
    ixQ = open_dir('../index/testindexdir', schema=schemaQ)

    i = 0
    for line in list(open("../data/2019qrels-pass.txt", encoding='utf8')):
        print(i)
        i = i+1
        queryID = line.split(' ')[0]
        passageID = int(line.split(' ')[2])
        relevance = int(line.split(' ')[3])
        qp = QueryParser('title', schema=ixQ.schema)
        q = qp.parse(queryID)

        with ixQ.searcher() as s:
            results = s.search(q)
            query = results[0].get('content')

            collected = []
            collectedID = []
            passageLength = []
            queryLength = len(query.split(" "))

        with open("../data/collection.tsv", encoding='utf8') as collection2:
            counter = 0
            for passage2 in collection2:
                checkID = int(passage2.split('\t')[0])
                if checkID == passageID:
                    collected.insert(0, passage2.split('\t')[1])
                    collectedID.insert(0, checkID)
                    passageLength.insert(0, len(passage2.split('\t')[1].split(" ")))
                    counter = counter + 1
                if checkID > passageID:
                    collected.append(passage2.split('\t')[1])
                    collectedID.append(passage2.split('\t')[0])
                    passageLength.append(len(passage2.split('\t')[1].split(" ")))
                    counter = counter + 1
                if counter > 1000:
                    break
        tokenized_corpus = [doc.split(" ") for doc in collected]
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = query.split(" ")
        score = bm25.get_scores(tokenized_query)
        trainingFile = open("output/crosstrain.txt", "a", newline='')
        trainingFile.write(str(relevance) + ' qid:' + str(queryID) + ' 1:' + str(score[0]) + ' 2:' + str(passageLength[0]) + ' 3:' + str(queryLength) + ' #' + str(passageID) + '\n')
        trainingFile.close()
