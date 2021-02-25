from whoosh.index import create_in, open_dir
from whoosh.fields import *
from whoosh.qparser import QueryParser
from whoosh.scoring import BM25F
from whoosh import scoring
import time


if __name__ == '__main__':
    query_text = "viking"

    timestart = time.time()
    schema = Schema(title=TEXT(stored=True), content=TEXT(stored=True))
    ix = open_dir('indexdir', schema=schema)

    with ix.searcher(weighting=scoring.BM25F()) as searcher:
        qp = QueryParser("content", ix.schema)
        query = qp.parse(query_text)
        results = searcher.search(query)

        indices = [r["title"] for r in results]
        entries = {}
        with open("./data/collection.tsv") as infile:
            for line in infile:
                if line:
                    index, text = line.split("\t")
                    entries[index] = text

        if results.has_matched_terms():
            print(results.matched_terms())

        for hit in results:
            print('{:.2f}'.format(hit.score), entries[hit["title"]], end='')


