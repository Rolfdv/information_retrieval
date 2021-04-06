from whoosh.index import create_in, open_dir
from whoosh.fields import *
import os

if __name__ == '__main__':
    fileName = "training-queries-index"
    newFile = "../index/" + fileName
    openFile = "../collectionandqueries/queries.dev.small.tsv"

    schema = Schema(title=TEXT(stored=True), content=TEXT(stored=True))
    ix = create_in(newFile, schema)
    writer = ix.writer()

    with open(openFile, 'r', encoding='utf8') as infile:
        total_data_per_line = infile.read().split("\n")

    for line in total_data_per_line:
        if line:
            index, text = line.split("\t")
            writer.add_document(title=str(index), content=text)
    writer.commit()