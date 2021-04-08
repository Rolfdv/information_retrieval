from whoosh.index import create_in, open_dir
from whoosh.fields import *
from whoosh.qparser import QueryParser
from rank_bm25 import BM25Okapi
from random import randint


def formatTraining():
    # Open index for queries and passages
    schemaQ = Schema(title=TEXT(stored=True), content=TEXT(stored=True))
    ixQ = open_dir('../index/queries.dev.small-index', schema=schemaQ)

    schemaP = Schema(title=TEXT(stored=True), content=TEXT)
    ixP = open_dir('../index/extended.passages.dev.small-index', schema=schemaP)

    # Create corpus for calculating bm25
    passages = []
    for line in list(open("../collectionandqueries/extended.passages.dev.small.tsv", encoding='utf8')):
        passages.append(str(line.split('\t')[1]).strip('\n'))
    print('Passages are loaded in memory!')

    # Keep track of track of current index with i
    i = -1
    score = []

    # Open file to write baseline features to
    trainingFile = open("output/baseline_feature.txt", 'a')

    # Iterate over all items in extended.qrels.dev.small
    for line in list(open("../data/extended.qrels.dev.small.tsv", encoding='utf8')):
        i = i + 1
        queryID = str(line.split('\t')[0])
        passageID = str(line.split('\t')[2])
        relevancy = str(line.split('\t')[3])

        # Search query text on queryID
        qp = QueryParser('title', schema=ixQ.schema)
        q = qp.parse(queryID)
        with ixQ.searcher() as s:
            results = s.search(q)
            query = results[0].get('content')

        # Search passage text on passageID
        pp = QueryParser('title', schema=ixP.schema)
        p = pp.parse(passageID)
        with ixP.searcher() as s:
            results = s.search(p)
            passage = str(results[0].get('content'))

        # Store information on query and passage length
        queryLength = len(query.split(" "))
        passageLength = len(passage.split(" "))
        indexPassage = passages.index(passage)

        # Only calculate the bm25 for all passages every 10 times, since the query changes after 10 trainingitems
        if i%10 == 0:
            tokenized_corpus = [doc.split(" ") for doc in passages]
            bm25 = BM25Okapi(tokenized_corpus)
            tokenized_query = query.split(" ")
            score = bm25.get_scores(tokenized_query)

        # Add training item to the file.
        trainingitem = str(str(relevancy) +  ' qid:' + str(queryID) + ' 1:' + str(score[indexPassage]) + ' 2:' + str(passageLength) + ' 3:' + str(queryLength) + ' #' + str(passageID) + '\n')
        trainingFile.write(str(trainingitem))

    trainingFile.close()


if __name__ == '__main__':
    formatTraining()





    # # Generate 9 random passageIDs
    #
    # while len(passageIDs) < 10:
    #     randomID = randint(0, 8841823)
    #     if not randomID == passageID:
    #         passageIDs.append(randomID)
    #
    # for num, trainingitem in enumerate(passageIDs):
    #     if num == 0:
    #         trainingFile.write(str(queryID) + '\t0\t' + str(passageIDs[num]) + '\t1' + '\n')
    #     else:
    #         trainingFile.write(str(queryID) + '\t0\t' + str(passageIDs[num]) + '\t0' + '\n')