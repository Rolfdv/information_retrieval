from torch.nn.functional import softmax
from transformers import BertForNextSentencePrediction, BertTokenizer, BertForSequenceClassification
from whoosh.index import create_in,open_dir
from whoosh.fields import *
from whoosh.qparser import QueryParser
import time
import torch
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# Taken from: https://stackoverflow.com/questions/55111360/using-bert-for-next-sentence-prediction
def bert_next_sentence(prompt, next_sentence):
    model = BertForNextSentencePrediction.from_pretrained('bert-base-cased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    # encode the two sequences. Particularly, make clear that they must be
    # encoded as "one" input to the model by using 'seq_B' as the 'text_pair'
    start = time.perf_counter()
    encoded = tokenizer.encode_plus(prompt, text_pair=next_sentence, return_tensors='pt')

    # a model's output is a tuple, we only need the output tensor containing
    # the relationships which is the first item in the tuple
    seq_relationship_logits = model(**encoded)[0]

    # we still need softmax to convert the logits into probabilities
    # index 0: sequence B is a continuation of sequence A
    # index 1: sequence B is a random sequence
    probs = softmax(seq_relationship_logits, dim=1)
    end = time.perf_counter()
    time_taken = end - start

    print('Prompt: ' + prompt)
    print('Next: ' + next_sentence)
    print('Chance continuation: ' + str(probs.data[0][0]))
    print('Chance random: ' + str(probs.data[0][1]))
    print('Time: ' + str(time_taken))
    return float(probs.data[0][0]), float(probs.data[0][1])


def bert_sentiment(query, passage):
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    model = BertForSequenceClassification.from_pretrained("bert-base-cased")

    start = time.perf_counter()

    encoded = tokenizer.encode_plus(query, text_pair=passage, return_tensors='pt')
    labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
    outputs = model(**encoded, labels=labels)
    probs = softmax(outputs.logits, dim=1)

    end = time.perf_counter()
    time_taken = end - start

    print('Prompt: ' + query)
    print('Next: ' + passage)
    print('Chance positive: ' + str(probs.data[0][0]))
    print('Chance negative: ' + str(probs.data[0][1]))
    print('Time: ' + str(time_taken))
    return float(probs.data[0][0]), float(probs.data[0][1])


def create_feature_file():
    next_sentence_model = BertForNextSentencePrediction.from_pretrained('bert-base-cased')
    sentiment_model = BertForSequenceClassification.from_pretrained("bert-base-cased")
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    schemaQ = Schema(title=TEXT(stored=True), content=TEXT(stored=True))
    ixQ = open_dir('../index/training-queries-index', schema=schemaQ)
    schemaP = Schema(title=TEXT(stored=True), content=TEXT(stored=True))
    ixP = open_dir('../index/training-passages-index', schema=schemaP)

    i = 0
    feature_file = open("output/training_set_bert_features.txt", "a", newline='')
    for line in list(open("../collectionandqueries/extended.qrels.dev.small.tsv", encoding='utf8')):
        queryID = line.split('\t')[0]
        passageID = line.split('\t')[2]
        relevance = int(line.split('\t')[3])

        qp = QueryParser('title', schema=ixQ.schema)
        q = qp.parse(queryID)

        pp = QueryParser('title', schema=ixP.schema)
        p = qp.parse(passageID)

        with ixQ.searcher() as s:
            results = s.search(q)
            query = results[0].get('content')

        with ixP.searcher() as s:
            results = s.search(p)
            passage = results[0].get('content')

        # NEXT SENTENCE PREDICTION
        # Encode prompt (query) and next_sentence (passage)
        encoded = tokenizer.encode_plus(query, text_pair=passage, return_tensors='pt')
        # We only need output tensor with relationships
        seq_relationship_logits = next_sentence_model(**encoded)[0]
        # We need Softmax to get probabilities. [P(continue), P(random)]
        next_sentence_probs = softmax(seq_relationship_logits, dim=1)

        prob_continue = round(float(next_sentence_probs.data[0][0]), 4)
        prob_random = round(float(next_sentence_probs.data[0][1]), 4)

        # SENTIMENT ANALYSIS
        # Query
        query_inputs = tokenizer(query, return_tensors="pt")
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = sentiment_model(**query_inputs, labels=labels)
        sentiment_probs = softmax(outputs.logits, dim=1)
        query_prob_positive = round(float(sentiment_probs.data[0][0]), 4)
        query_prob_negative = round(float(sentiment_probs.data[0][1]), 4)
        # Passage
        sentence_inputs = tokenizer(passage, return_tensors="pt")
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = sentiment_model(**sentence_inputs, labels=labels)
        sentiment_probs = softmax(outputs.logits, dim=1)
        passage_prob_positive = round(float(sentiment_probs.data[0][0]), 4)
        passage_prob_negative = round(float(sentiment_probs.data[0][1]), 4)
        # Query + Passage combo
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = sentiment_model(**encoded, labels=labels)
        sentiment_probs = softmax(outputs.logits, dim=1)
        qp_prob_positive = round(float(sentiment_probs.data[0][0]), 4)
        qp_prob_negative = round(float(sentiment_probs.data[0][1]), 4)

        if i % 1000 == 0:
            print(i)
            print('Prompt: ' + query)
            print('Next: ' + passage)
            print('Chance continuation: ' + str(prob_continue))
            print('Chance random: ' + str(prob_random))
            print('Chance query positive: ' + str(query_prob_positive))
            print('Chance query negative: ' + str(query_prob_negative))
            print('Chance passage positive: ' + str(passage_prob_positive))
            print('Chance passage negative: ' + str(passage_prob_negative))
            print('Chance qp positive: ' + str(qp_prob_positive))
            print('Chance qp negative: ' + str(qp_prob_negative))
            print('Relevancy rating: ' + str(relevance))
        i = i + 1

        next_line = """qid:{} pid:{} 1:{} 2:{} 3:{} 4:{} 5:{} 6:{} 7:{} 8:{}\n"""\
                    .format(str(queryID), str(passageID), str(prob_continue), str(prob_random),
                            str(query_prob_positive), str(query_prob_negative),
                            str(passage_prob_positive), str(passage_prob_negative),
                            str(qp_prob_positive), str(qp_prob_negative))
        feature_file.write(next_line)
    feature_file.close()


if __name__ == '__main__':
    # passage = "The presence of communication amid scientific minds was equally important to the success of the Manhattan Project as scientific intellect was. The only cloud hanging over the impressive achievement of the atomic researchers and engineers is what their success truly meant; hundreds of thousands of innocent lives obliterated."
    # query = "why was the Manhatten project succesful"
    # bert_sentiment(query, passage)
    #
    # passage = "The presence of communication amid scientific minds was equally important to the success of the Manhattan Project as scientific intellect was."
    # query = "why was the Manhatten project succesful"
    # bert_sentiment(query, passage)
    #
    # passage = "I bought a gallon of milk."
    # query = "This afternoon I went to the store."
    # bert_sentiment(query, passage)
    #
    # query = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
    # passage = "The sky is blue due to the shorter wavelength of blue light."
    # bert_sentiment(query, passage)
    #
    # query = 'I like cookies !'
    # passage = 'Do you like them ?'
    # bert_sentiment(query, passage)
    #
    # query = 'I hate cookies'
    # passage = 'Do you like them ?'
    # bert_sentiment(query, passage)

    create_feature_file()
