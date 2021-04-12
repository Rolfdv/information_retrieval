
# Please ensure bin/RankLib-2.15.jar is present.
# Useful website for running: https://sourceforge.net/p/lemur/wiki/RankLib%20How%20to%20use/
base_train_query = "java -jar bin/RankLib-2.15.jar -train learning_to_rank/data/training.txt -test learning_to_rank/data/testing.txt -ranker 1 -metric2t NDCG@10 -feature learning_to_rank/feature_spec_models/model_01.txt -save learning_to_rank/models/model_01.txt -epoch 100"
individual_query_scores = "java -jar bin/RankLib-2.15.jar -load twenty_epoch_models/{0}_{1}_{2} -test learning_to_rank/data/testing_plus_validation.txt -metric2T NDCG@10"


def scoring_commands():
    for i in range(1, 13):
        if i < 10:
            model_name = 'model_0{}'.format(i)
        else:
            model_name = 'model_{}'.format(i)
        layer = 2
        nodes = 30
        folder_name = 'threehundred_epoch_models'
        base_score_query = "java -jar bin/RankLib-2.15.jar -load learning_to_rank/{3}/{0}_{1}_{2}.txt -test learning_to_rank/data/testing_plus_validation.txt -metric2T NDCG@10".format(model_name, layer, nodes, folder_name)
        print(base_score_query)


def training_commands():
    for i in range(1, 20):
        if i < 10:
            model_name = 'model_0{}'.format(i)
        else:
            model_name = 'model_{}'.format(i)
        layer = 2
        nodes = 30
        folder_name = 'threehundred_epoch_models'
        base_train_query = "java -jar bin/RankLib-2.15.jar -train learning_to_rank/data/training.txt -test learning_to_rank/data/testing_plus_validation.txt -validate learning_to_rank/data/validation.txt -ranker 1 -metric2t NDCG@10 -feature learning_to_rank/feature_spec_models/{0} -save learning_to_rank/{3}/{0}_{1}_{2}.txt -epoch 300 -layer {1} -node {2}".format(model_name, layer, nodes, folder_name)
        print(base_train_query)


if __name__ == '__main__':
    scoring_commands()
    training_commands()
