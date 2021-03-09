from whoosh.index import create_in, open_dir
from whoosh.fields import *
from whoosh.qparser import QueryParser
from whoosh.scoring import BM25F
from whoosh import scoring
import time

""" Probabilistic model based on BM25F that ranks passages from collection.tsv on relevance to query.
"""
if __name__ == '__main__':
    timestart = time.time()

    # Open existing index 'dataindexdir'
    schema = Schema(title=TEXT(stored=True), content=TEXT(stored=True))
    ix = open_dir('../index/dataindexdir', schema=schema)

    for query_text in ["viking", "knee", "chicken", "potato", "food"]:

        # Search collection.tsv to rank passages from collection on BM25F score
        with ix.searcher(weighting=scoring.BM25F()) as searcher:
            qp = QueryParser("content", ix.schema)
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
    print(time.time() - timestart)


