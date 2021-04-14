import random
import numpy as np

# This method takes a file and creates a k-fold cross validation.
if __name__ == '__main__':
    random.seed(20)
    number_of_folds = 5
    file_base = 'own/iter4'

    # Open the file
    file_source = file_base + '/crosstrain.txt'
    file = open(file_source, 'r')
    lines = file.readlines()
    query_ids = []
    # This for loop defines takes all query IDs from the file
    for line in lines:
        row = line.split(' ')
        relevance = int(row[0])
        query_id = int(row[1][4:])  # Get rid of 'qid:'. Structure = qid:<query_id>
        features = [row[2], row[3], row[4]]
        comment = row[5]
        if query_id not in query_ids:
            query_ids.append(query_id)
    file.close()

    # Shuffle the queries
    random.shuffle(query_ids)
    folds = np.array(query_ids)
    # Split into the number_of_folds
    folds = np.array_split(folds, number_of_folds)

    for i in range(number_of_folds):
        saved_query_id = 0
        # i is going to be our test set.
        # i+1 is going to be our validation set.
        # The other three are our training set.
        test_fold = folds[i]
        vali_index = i + 1
        if vali_index > number_of_folds - 1:
            vali_index = 0
        vali_fold = folds[vali_index]
        save_folder = file_base + '/fold{}/'.format(i)

        test_file = save_folder + 'test.txt'
        vali_file = save_folder + 'vali.txt'
        train_file = save_folder + 'train.txt'

        for line in lines:
            row = line.split(' ')
            query_id = int(row[1][4:])  # Get rid of 'qid:'. Structure = qid:<query_id>
            if query_id in test_fold:
                with open(test_file, 'a+') as file_writer:
                    file_writer.write(line)
                    file_writer.close()
            elif query_id in vali_fold:
                with open(vali_file, 'a+') as file_writer:
                    file_writer.write(line)
                    file_writer.close()
            else:
                with open(train_file, 'a+') as file_writer:
                    file_writer.write(line)
                    file_writer.close()
