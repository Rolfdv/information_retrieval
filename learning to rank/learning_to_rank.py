# Please ensure bin/RankLib-2.15.jar is present.
# Useful website for running: https://sourceforge.net/p/lemur/wiki/RankLib%20How%20to%20use/

# Running this script prints commands you can run from the base of this project using the RankLib-2.15.jar

node = 20
layer = 2

for i in range(0, 5):
    # baseline_query = "java -jar bin/RankLib-2.15.jar -train own/iter4/fold{0}/train.txt -test own/iter4/fold{0}/test.txt -validate own/iter4/fold{0}/vali.txt -ranker 1 -metric2t NDCG@10 -save map10baselinefold{0}.txt -epoch 100 -layer 1 -node 10".format(i)
    query = "java -jar bin/RankLib-2.15.jar -train own/iter4/fold{0}/train.txt -test own/iter4/fold{0}/test.txt -validate own/iter4/fold{0}/vali.txt -ranker 1 -metric2t NDCG@10 -save ndcg10model{1}-{2}fold{0}.txt -epoch 100 -layer {2} -node {1}".format(i, node, layer)
    print(query)

print('------------------------')

for i in range(0, 5):
    # baseline_score_query = "java -jar bin/RankLib-2.15.jar -load ndcg10model10-10fold{0}.txt -test own/iter4/fold{0}/test.txt -metric2T NDCG@10 -idv output/map10baselinefold{0}.txt".format(i)
    score_query = "java -jar bin/RankLib-2.15.jar -load ndcg10model{1}-{2}fold{0}.txt -test own/iter4/fold{0}/test.txt -metric2T NDCG@10 -idv output/ndcg10model{1}-{2}fold{0}.txt".format(i, node, layer)
    print(score_query)
