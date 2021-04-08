

if __name__ == '__main__':
    # Read feature files
    baseline_glove_fasttext_features_training_by_qid = {}
    with open('../featuresets/training/baseline_glove_fasttext_features_training.txt') as baseline_glove_fasttext_features_training:
        for line in baseline_glove_fasttext_features_training:
            line = line.replace("\n", "")
            if line:
                relevant, qid, param1, param2, param3, param4, param5, param6, param7, pid = line.split(" ")
                _, qid_val = qid.split(":")
                if qid_val not in baseline_glove_fasttext_features_training_by_qid.keys():
                    baseline_glove_fasttext_features_training_by_qid[qid_val] = []
                baseline_glove_fasttext_features_training_by_qid[qid_val].append(line)

    bert_features_by_qid = {}
    with open('../featuresets/training/bert_features_training.txt') as bert_features_file:
        for line in bert_features_file:
            line = line.replace("\n", "")
            if line:
                qid, pid, param1, param2, param3, param4, param5, param6, param7, param8, param9, param10, param11, param12, param13, param14 = line.split(" ")
                _, qid_val = qid.split(":")
                if qid_val not in bert_features_by_qid.keys():
                    bert_features_by_qid[qid_val] = []
                bert_features_by_qid[qid_val].append(line)


    with open("../featuresets/training/mega_combined_feature_file.txt", "w+") as res_file:
        for key in baseline_glove_fasttext_features_training_by_qid.keys():
            if key in bert_features_by_qid.keys():
                baseline_lines = baseline_glove_fasttext_features_training_by_qid[key]
                bert_lines = bert_features_by_qid[key]

                baseline_lines_sorted = sorted(baseline_lines, key=lambda x: int(x.split(" ")[-1].replace("#", "")))
                bert_lines_sorted = sorted(bert_lines, key=lambda x: int(x.split(" ")[1].replace("pid:", "")))
                assert len(baseline_lines_sorted) == len(bert_lines_sorted)
                zipped = zip(baseline_lines_sorted, bert_lines_sorted)
                for combined_line in zipped:
                    baseline_line = combined_line[0]
                    bert_line = combined_line[1]
                    baseline_pid = baseline_line.split(" ")[-1].replace("#", "")
                    bert_pid = bert_line.split(" ")[1].replace("pid:", "")
                    assert baseline_pid == bert_pid

                    new_line_split = baseline_line.split(" ")[:-1]

                    for i, val in enumerate(bert_line.split(" ")[2:]):
                        new_line_split.append(val.replace(f"{i + 1}:", f"{i + 8}:"))

                    new_line_split.append(f"#{baseline_pid}")
                    res_file.write(f"{' '.join(new_line_split)}\n")
            else:
                print(f"Missing key:{key} in fasttext features or glove features")


