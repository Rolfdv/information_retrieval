from whoosh.index import create_in, open_dir
from whoosh.fields import *
from whoosh.qparser import QueryParser, OrGroup
from whoosh.scoring import BM25F
from whoosh import scoring, qparser
import time

""" Adds edit distance and prefix length to seperate words in query

:query: str or unicode
:suffix: amount of letters (starting at the end) of a word that can be edited
:editDistance: maximum edit distance
:returns: formatted query
:rtype: str
"""
def fuzzy(query, suffix, editDistance):
    check = query.split(" ")
    for word in check:
        if len(word) > 4:
            word = word.replace('\n', '')
            query = query.replace(word, word + '~' + str(editDistance) + '/' + str(len(word) - suffix))
    return query

""" Probabilistic model based on BM25F that ranks passages from collection.tsv on relevance to query. FuzzyTermPlugin
    is added to the probabilistic model
"""
if __name__ == '__main__':
    # Add fuzzy term to query
    for line in list(open("../data/43queries.txt", encoding='utf8')):
        query_text = line
        query_text = fuzzy(query_text, 0, 0)
        timestart = time.time()

        # Open existing index 'dataindexdir'
        schema = Schema(title=TEXT(stored=True), content=TEXT(stored=True))
        ix = open_dir('../index/dataindexdir', schema=schema)

        # Search collection.tsv to rank passages from collection on BM25F score
        with ix.searcher(weighting=scoring.BM25F()) as searcher:
            qp = qparser.QueryParser("content", ix.schema, group=OrGroup)
            qp.add_plugin(qparser.FuzzyTermPlugin())
            query = qp.parse(query_text)
            results = searcher.search(query)

            indices = [r["title"] for r in results]
            entries = {}
            with open('../data/collection.tsv', encoding='utf8') as infile:
                for line in infile:
                    if line:
                        index, text = line.split("\t")
                        entries[index] = text

            # Print result with scores
            for hit in results:
                print('{:.2f}'.format(hit.score), entries[hit["title"]], end='')

        # Print runtime
        print("time: " + str(time.time() - timestart))

