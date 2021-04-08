

if __name__ == '__main__':
    # Read feature files
    baseline_features_by_qid = {}
    with open('../featuresets/training/baseline_features_training.txt') as baseline_features_file:
        for line in baseline_features_file:
            line = line.replace("\n", "")
            if line:
                relevant, qid, param1, param2, param3, pid = line.split(" ")
                _, qid_val = qid.split(":")
                if qid_val not in baseline_features_by_qid.keys():
                    baseline_features_by_qid[qid_val] = []
                baseline_features_by_qid[qid_val].append(line)

    fasttext_features_by_qid = {}
    with open('../featuresets/training/fast_features_training.txt') as fasttext_features_file:
        for line in fasttext_features_file:
            line = line.replace("\n", "")
            if line:
                qid, pid, param1, param2 = line.split(" ")
                _, qid_val = qid.split(":")
                if qid_val not in fasttext_features_by_qid.keys():
                    fasttext_features_by_qid[qid_val] = []
                fasttext_features_by_qid[qid_val].append(line)

    glove_features_by_qid = {}
    with open('../featuresets/training/glove_features_training.txt') as glove_features_file:
        for line in glove_features_file:
            line = line.replace("\n", "")
            if line:
                qid, pid, param1, param2 = line.split(" ")
                _, qid_val = qid.split(":")
                if qid_val not in glove_features_by_qid.keys():
                    glove_features_by_qid[qid_val] = []
                glove_features_by_qid[qid_val].append(line)

    with open("../featuresets/training/combined_feature_file.txt", "w+") as res_file:
        for key in baseline_features_by_qid.keys():
            if key in fasttext_features_by_qid.keys() and key in glove_features_by_qid:
                baseline_lines = baseline_features_by_qid[key]
                glove_lines = glove_features_by_qid[key]
                fasttext_lines = fasttext_features_by_qid[key]

                baseline_lines_sorted = sorted(baseline_lines, key=lambda x: int(x.split(" ")[5].replace("#", "")))
                glove_lines_sorted = sorted(glove_lines, key=lambda x: int(x.split(" ")[1].replace("pid:", "")))
                fasttext_lines_sorted = sorted(fasttext_lines, key=lambda x: int(x.split(" ")[1].replace("pid:", "")))
                assert len(baseline_lines_sorted) == len(fasttext_lines_sorted)
                assert len(baseline_lines_sorted) == len(glove_lines_sorted)
                zipped = zip(baseline_lines_sorted, glove_lines_sorted, fasttext_lines_sorted)
                for combined_line in zipped:
                    baseline_line = combined_line[0]
                    glove_line = combined_line[1]
                    fasttext_line = combined_line[2]
                    baseline_pid = baseline_line.split(" ")[-1].replace("#", "")
                    glove_pid = glove_line.split(" ")[1].replace("pid:", "")
                    fasttext_pid = fasttext_line.split(" ")[1].replace("pid:", "")
                    assert baseline_pid == glove_pid
                    assert baseline_pid == fasttext_pid

                    new_line_split = baseline_line.split(" ")[:-1]

                    glove_cosine_param = glove_line.split(" ")[2].replace("1:", "4:")
                    glove_euclidian_param = glove_line.split(" ")[3].replace("2:", "5:")
                    new_line_split.append(glove_cosine_param)
                    new_line_split.append(glove_euclidian_param)

                    fasttext_cosine_param = glove_line.split(" ")[2].replace("1:", "6:")
                    fasttext_euclidian_param = glove_line.split(" ")[3].replace("2:", "7:")
                    new_line_split.append(fasttext_cosine_param)
                    new_line_split.append(fasttext_euclidian_param)

                    new_line_split.append(f"#{baseline_pid}")
                    res_file.write(f"{' '.join(new_line_split)}\n")
            else:
                print(f"Missing key:{key} in fasttext features or glove features")


