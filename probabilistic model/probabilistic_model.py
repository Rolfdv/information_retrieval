from whoosh.index import open_dir
from whoosh.fields import *
from whoosh.qparser import QueryParser
from whoosh import scoring, qparser
from whoosh.query import Phrase
import string
from tqdm import tqdm

# Load in 100 popular boy and girl names
with open('../data/popular_names.txt') as names_file:
    popular_names = names_file.readlines()
    popular_names = [name.replace("\n", "").lower() for name in popular_names]

# Only use the 43 queries for which the TREC relevance rating is known
with open('../data/2019qrels-pass.txt') as qrels_file:
    qids = set()
    for line in qrels_file:
        line = line.replace("\n", "")
        if line:
            qid, _, _, _ = line.split(" ")
            qids.add(qid)


if __name__ == '__main__':
    # Load in the index
    schema = Schema(title=TEXT(stored=True), content=TEXT(stored=True))
    ix = open_dir('../dataindexdir', schema=schema)

    with open("improve_OR_results_msmarco-test2019-queries.tsv", "w+") as results_file:
        with ix.searcher(weighting=scoring.BM25F()) as searcher:
            qp = QueryParser("content", ix.schema, group=qparser.OrGroup)

            with open(f'../data/msmarco-test2019-queries.tsv', 'r', encoding='utf8') as qrels_file:
                # for line in tqdm(qrels_file):
                for line in tqdm(qrels_file):
                    if line:
                        line = line.replace("\n", "")
                        qid, query_text = line.split("\t")

                        # Only use the 43 queries for which the TREC relevance rating is known
                        if qid not in qids:
                            continue

                        # Remove punctuation from query
                        exclude = set(string.punctuation)
                        query_text = ''.join(char for char in query_text if char not in exclude)

                        # Check for popular first name in query
                        query_split = query_text.split(" ")
                        phrases = []
                        for i, word in enumerate(query_split):
                            # If last word is reached,
                            # no need to check if it's a first name
                            # because there is no potential last name that follows it
                            if i == len(query_split) - 1:
                                break
                            # If word is a popular name
                            # remove it and next word from query text
                            # and add as phrase later
                            if word in popular_names:
                                next_word = query_split[i + 1]
                                query_text = query_text.replace(word, "")
                                query_text = query_text.replace(next_word, "")
                                phrases.append(Phrase("content", [word, next_word]))

                        query = qp.parse(query_text)

                        # Add phrases (names) to the parsed query
                        for phrase in phrases:
                            query = query | phrase

                        # Search the index for documents that match with the query
                        # Returned 10 documents are sorted on BM25F score descendingly
                        results = searcher.search(query, limit=10)
                        indices = [(r["title"], r.score) for r in results]

                        for i, (title, score) in enumerate(indices):
                            indices[i] = (int(title), float('{:.2f}'.format(score)))

                        # Store so we can calculate the NDCG@10 of these 10 documents later
                        results_file.write(f"{qid}\t{indices}\n")




