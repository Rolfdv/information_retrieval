# Core Information Retrieval Project

This repository contains the code used for the Core IR assignment of team 4.

## TREC 2019 Deep Learning Datasets
The datasets of TREC 2019 Deep Learning is necessary for training the learning to rank model, indexing, and testing. Therefor the following data files have to be added to a directory 'data' in the root folder:
* collections              
* queries.train            
* msmarco-test2019-queries 
* 2019qrels-pass          
* qrels.train              

## Indexes
To improve runtime on searching the dataset **collections** and the dataset **msmarco-test2019-queries** the scripts _information_retrieval/learning to rank/index_queries.py_ and  _create_index_ will create these indexes. They have to be run prior to running other scripts.
