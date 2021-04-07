from whoosh.index import create_in, open_dir
from whoosh.fields import *
import os

if __name__ == '__main__':
    fileName = "extended.passages.dev.small-index"
    newFile = "../index/" + fileName
    openFile = "../data/word_embeddings/glove_scripts/7500queries_glove_embedding.txt"
    #
    schema = Schema(title=TEXT(stored=True), content=TEXT(stored=True))
    ix = create_in(newFile, schema)
    writer = ix.writer()
    with open(openFile, 'r', encoding='utf8') as infile:
        total_data_per_line = infile.read().split("\n")

    for line in total_data_per_line:
        if line:
            index, text, vector = line.split("\t")
            writer.add_document(title=str(index), content=vector)
    writer.commit()

