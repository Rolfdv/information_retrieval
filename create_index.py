from whoosh.index import create_in
from whoosh.fields import *

if __name__ == '__main__':
    schema = Schema(title=TEXT(stored=True), content=TEXT)

    ix = create_in("dataindexdir", schema)
    writer = ix.writer()

    with open('./data/collection.tsv', 'r') as infile:
        total_data_per_line = infile.read().split("\n")

        for line in total_data_per_line:
            if line:
                index, text = line.split("\t")
                writer.add_document(title=str(index), content=text)
        writer.commit()