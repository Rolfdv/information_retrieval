
# Useful website for running: https://sourceforge.net/p/lemur/wiki/RankLib%20How%20to%20use/
example_query = "java -jar bin/RankLib-2.15.jar -train bin/MQ2008/Fold1/train.txt -test bin/MQ2008/Fold1/test.txt -validate bin/MQ2008/Fold1/vali.txt -ranker 6 -metric2t NDCG@10 -metric2T ERR@10 -save mymodel.txt".split(" ")

