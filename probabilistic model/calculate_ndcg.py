import ast
import math

# Create for each query, for each of its relevancy ratings, the list of passages with that relevancy rating
per_query_per_relevancy_rating_list_of_passages = {}

with open("../data/2019qrels-pass.txt") as inf:
    for line in inf:
        line = line.replace("\n", "")
        if line:
            qid, _, pid, rating = line.split(" ")

            # Get dict of relevancy ratings to lists of passages for this qid
            relevancy_ratings_lists_of_passages = per_query_per_relevancy_rating_list_of_passages.get(qid, {})

            # Get the list of passages for this qid's relevancy rating
            list_of_passages_for_this_relevancy_rating = relevancy_ratings_lists_of_passages.get(rating, [])

            # Add pid to the list
            list_of_passages_for_this_relevancy_rating.append(pid)

            # Store list back in dict
            relevancy_ratings_lists_of_passages[rating] = list_of_passages_for_this_relevancy_rating

            # Store dict back in dict
            per_query_per_relevancy_rating_list_of_passages[qid] = relevancy_ratings_lists_of_passages

log_base = 2
ndcg_at_number = 10

with open("../data/results_msmarco-test2019-queries.tsv") as results_file:
    for line in results_file:
        line = line.replace("\n", "")
        if line:
            qid, list_of_passages_and_score = line.split("\t")

            if qid in per_query_per_relevancy_rating_list_of_passages.keys():
                print(qid)

                ideal_vector = []
                keys = ['3', '2', '1', '0']
                for key in keys:
                    if key in per_query_per_relevancy_rating_list_of_passages[qid].keys():
                        for _ in range(len(per_query_per_relevancy_rating_list_of_passages[qid][key])):
                            ideal_vector.append(float(key))
                            if len(ideal_vector) >= ndcg_at_number:
                                break
                    if len(ideal_vector) >= ndcg_at_number:
                        break
                while len(ideal_vector) < ndcg_at_number:
                    ideal_vector.append(float(0))

                ideal_cumulative_vector = []
                for i, value in enumerate(ideal_vector):
                    if i == 0:
                        ideal_cumulative_vector.append(value)
                    else:
                        value += ideal_cumulative_vector[i-1]
                        ideal_cumulative_vector.append(value)

                ideal_discounted_cumulative_vector = []
                for i, value in enumerate(ideal_cumulative_vector):
                    if i+1 >= log_base:
                        value = ideal_discounted_cumulative_vector[i-1] + (ideal_vector[i] / math.log(i+1, log_base))
                    else:
                        value = ideal_cumulative_vector[i]
                    ideal_discounted_cumulative_vector.append(value)

                gain_vector = []

                relevancy_ratings_lists_of_passages = per_query_per_relevancy_rating_list_of_passages[qid]

                list_of_passages_and_score = ast.literal_eval(list_of_passages_and_score)
                # For each passage that was found, add it's relevancy rating to actual_vector
                for i, (pid, _) in enumerate(list_of_passages_and_score):
                    pid = str(pid)

                    for relevancy_rating in ['3', '2', '1', '0']:
                        if relevancy_rating not in relevancy_ratings_lists_of_passages.keys():
                            continue
                        list_to_check = relevancy_ratings_lists_of_passages[relevancy_rating]
                        if (pid in list_to_check) or relevancy_rating == '0':
                            gain_vector.append(int(relevancy_rating))
                            break

                    if len(gain_vector) >= ndcg_at_number:
                        break

                cumulative_gain_vector = []
                for i, value in enumerate(gain_vector):
                    if i == 0:
                        cumulative_gain_vector.append(value)
                    else:
                        value = cumulative_gain_vector[i-1] + gain_vector[i]
                        cumulative_gain_vector.append(value)

                discounted_cumulative_gain_vector = []
                for i, value in enumerate(cumulative_gain_vector):
                    if i+1 >= log_base:
                        value = discounted_cumulative_gain_vector[i-1] + (gain_vector[i] / math.log(i+1, log_base))
                    else:
                        value = cumulative_gain_vector[i]
                    discounted_cumulative_gain_vector.append(value)
                print(discounted_cumulative_gain_vector)
                print(ideal_discounted_cumulative_vector)

                ndcg_vector = []
                for i in range(len(discounted_cumulative_gain_vector)):
                    ndcg_vector.append(discounted_cumulative_gain_vector[i] / ideal_discounted_cumulative_vector[i])

                with open("improve_OR_ndcg_results.txt", "a+") as outfile:
                    try:
                        outfile.write(f"{qid}\t{ndcg_vector[-1]}\t{ndcg_vector}\n")
                    except IndexError:
                        outfile.write(f"{qid}\t0.0\t{ndcg_vector}\n")



