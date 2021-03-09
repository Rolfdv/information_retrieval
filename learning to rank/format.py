from whoosh.index import create_in, open_dir
from whoosh.fields import *
from whoosh.qparser import QueryParser
from rank_bm25 import BM25Okapi

def formatTraining():
    schemaQ = Schema(title=TEXT(stored=True), content=TEXT(stored=True))
    ixQ = open_dir('../index/dataindexdir', schema=schemaQ)

    i = 0
    for line in list(open("../data/qrels.train.tsv", encoding='utf8')):
        print(i)
        i = i+1
        queryID = line.split('\t')[0]
        passageID = int(line.split('\t')[2])
        qp = QueryParser('title', schema=ixQ.schema)
        q = qp.parse(queryID)

        with ixQ.searcher() as s:
            results = s.search(q)
            query = results[0].get('content')

            collected = []
            collectedID = []
            passageLength = []
            queryLength = len(query.split(" "))

        with open('../data/collection.tsv', encoding='utf8') as collection2:
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
                if counter > 10000:
                    break

        tokenized_corpus = [doc.split(" ") for doc in collected]
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = query.split(" ")
        score = bm25.get_scores(tokenized_query)
        trainingFile = open("output/train.txt", "a", newline='')
        for num, trainingitem in enumerate(collectedID):
            if num == 0:
                trainingFile.write('1 qid:' + str(queryID) + ' 1:' + str(score[num]) + ' 2:' + str(passageLength[num]) + ' 3:' + str(queryLength) + ' #' + str(trainingitem) + '\n')
            else:
                trainingFile.write('0 qid:' + str(queryID) + ' 1:' + str(score[num]) + ' 2:' + str(passageLength[num]) + ' 3:' + str(queryLength) + ' #' + str(trainingitem) + '\n')
            if num == 99:
                break
        trainingFile.close()
        if i == 1000:
            break

def formatValidate():
    schemaQ = Schema(title=TEXT(stored=True), content=TEXT(stored=True))
    ixQ = open_dir('../index/dataindexdir', schema=schemaQ)

    i = 0
    for line in reversed(list(open("../data/qrels.train.tsv", encoding='utf8'))):
        print(i)
        i = i+1
        queryID = line.split('\t')[0]
        passageID = int(line.split('\t')[2])
        qp = QueryParser('title', schema=ixQ.schema)
        q = qp.parse(queryID)

        with ixQ.searcher() as s:
            results = s.search(q)
            query = results[0].get('content')
            collected = []
            collectedID = []
            passageLength = []
            queryLength = len(query.split(" "))

            with open('../data/collection.tsv', encoding='utf8') as collection2:
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
                        collectedID.append(checkID)
                        passageLength.append(len(passage2.split('\t')[1].split(" ")))
                        counter = counter + 1
                    if counter > 10:
                        break
                tokenized_corpus = [doc.split(" ") for doc in collected]
                bm25 = BM25Okapi(tokenized_corpus)
                tokenized_query = query.split(" ")
                score = bm25.get_scores(tokenized_query)
                print(collectedID)
                trainingFile = open("output/validate.txt", "a", newline='')
                for num, trainingitem in enumerate(collectedID):
                    if num == 0:
                        trainingFile.write('1 qid:' + str(queryID) + ' 1:' + str(score[num]) + ' 2:' + str(passageLength[num]) + ' 3:' + str(queryLength) + ' #' + str(trainingitem) + '\n')
                    else:
                        trainingFile.write('0 qid:' + str(queryID) + ' 1:' + str(score[num]) + ' 2:' + str(passageLength[num]) + ' 3:' + str(queryLength) + ' #' + str(trainingitem) + '\n')
                    if num == 100:
                        break
                trainingFile.close()
        if i == 100:
            break

def formatTesting():
    schemaQ = Schema(title=TEXT(stored=True), content=TEXT(stored=True))
    ixQ = open_dir('../index/dataindexdir', schema=schemaQ)

    i = 0
    j = 0
    for line in reversed(list(open("../data/qrels.train.tsv", encoding='utf8'))):
        print(i)
        j = j+1
        if j < 100:
            continue
        i = i + 1
        queryID = line.split('\t')[0]
        passageID = int(line.split('\t')[2])
        qp = QueryParser('title', schema=ixQ.schema)
        q = qp.parse(queryID)

        with ixQ.searcher() as s:
            results = s.search(q)
            query = results[0].get('content')
            collected = []
            collectedID = []
            passageLength = []
            queryLength = len(query.split(" "))

            with open('../data/collection.tsv', encoding='utf8') as collection2:
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
                        collectedID.append(checkID)
                        passageLength.append(len(passage2.split('\t')[1].split(" ")))
                        counter = counter + 1
                    if counter > 10:
                        break
                tokenized_corpus = [doc.split(" ") for doc in collected]
                bm25 = BM25Okapi(tokenized_corpus)
                tokenized_query = query.split(" ")
                score = bm25.get_scores(tokenized_query)
                print(collectedID)
                trainingFile = open("output/validate.txt", "a", newline='')
                for num, trainingitem in enumerate(collectedID):
                    if num == 0:
                        trainingFile.write('1 qid:' + str(queryID) + ' 1:' + str(score[num]) + ' 2:' + str(passageLength[num]) + ' 3:' + str(queryLength) + ' #' + str(trainingitem) + '\n')
                    else:
                        trainingFile.write('0 qid:' + str(queryID) + ' 1:' + str(score[num]) + ' 2:' + str(passageLength[num]) + ' 3:' + str(queryLength) + ' #' + str(trainingitem) + '\n')
                    if num == 100:
                        break
                trainingFile.close()
        if i == 100:
            break

if __name__ == '__main__':
    # formatTraining()
    formatValidate()
    # formatTesting()