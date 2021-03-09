from whoosh.index import create_in, open_dir
from whoosh.fields import *
import os

""" Create index for 200 test queries
"""

# Create schema for 200 test queries
schema = Schema(title=TEXT(stored=True), content=TEXT(stored=True))
ix = create_in('../index/testindexdir', schema)
writer = ix.writer()

with open('../data/msmarco-test2019-queries.tsv', 'r', encoding='utf8') as infile:
    total_data_per_line = infile.read().split("\n")

for line in total_data_per_line:
    if line:
        index, text = line.split("\t")
        writer.add_document(title=str(index), content=text)
writer.commit()