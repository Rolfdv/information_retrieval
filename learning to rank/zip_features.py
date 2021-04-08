

if __name__ == '__main__':
    # Read feature files
    bert_feature_index_start = open('output/test_set_bert_features.txt', 'r')
    main_features = open('output/features.txt', 'r')
    word_feature_index_start = open('output/word_distance_features.txt', 'r')

    # Create ultimate feature file
    feature_file = open("output/ultimate_features.txt", "a", newline='')

    while True:
        # Read individual lines
        bert_line = bert_feature_index_start.readline().split(' ')
        main_line = main_features.readline().split(' ')
        word_line = word_feature_index_start.readline().split(' ')

        if not bert_line:
            break  # If we reach the end of the file, break.

        # Check the query IDs
        bert_query_id = bert_line[0][4:]
        main_query_id = main_line[1][4:]
        word_query_id = word_line[0][4:]
        assert bert_query_id == main_query_id and main_query_id == word_query_id

        # Check the passage IDs
        bert_passage_id = bert_line[1][4:]
        main_passage_id = main_line[5][1:]
        word_passage_id = word_line[1][4:]
        assert bert_passage_id == main_passage_id and main_passage_id == word_passage_id

        bert_feature_index_start = 2
        res_feature_index_start = 4
        bert_feature_line = ''
        for i in range(8):
            bert_start = bert_feature_index_start + i
            res_pos = res_feature_index_start + i

            feature = bert_line[bert_start].split(':')[1]
            bert_feature_line = bert_feature_line + ' {}:{}'.format(res_pos, feature)

        main_feature_line = main_line[0] + ' ' + main_line[1] + ' ' + main_line[2] + ' ' + main_line[3] + ' ' + main_line[4]

        word_feature_index_start = 2
        res_feature_index_start = 12
        word_feature_line = ''
        for i in range(2):
            word_start = word_feature_index_start + i
            res_pos = res_feature_index_start + i

            feature = word_line[word_start].split(':')[1]
            word_feature_line = word_feature_line + ' {}:{}'.format(res_pos, feature)

        res_line = main_feature_line + bert_feature_line + word_feature_line + ' #' + main_passage_id
        print(res_line)
        feature_file.write(res_line)
    feature_file.close()
