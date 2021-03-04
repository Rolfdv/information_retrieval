
# Please ensure bin/RankLib-2.15.jar is present.
# Useful website for running: https://sourceforge.net/p/lemur/wiki/RankLib%20How%20to%20use/
train_query = "java -jar bin/RankLib-2.15.jar -train own/iter3/train.txt -test own/iter3/test.txt -validate own/iter3/validate.txt -ranker 1 -metric2t MAP@10 -save map10baseline3.txt -epoch 200"
individual_query_scores = "java -jar bin/RankLib-2.15.jar -load map10baseline3.txt -test own/iter3/test.txt -metric2T MAP@10 -idv output/map10iter3_rn100.txt"
