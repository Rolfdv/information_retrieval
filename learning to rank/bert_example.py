from torch.nn.functional import softmax
from transformers import BertForNextSentencePrediction, BertTokenizer, BertForSequenceClassification
from whoosh.index import create_in,open_dir
from whoosh.fields import *
from whoosh.qparser import QueryParser
import time
import torch
from transformers import pipeline
from transformers import AlbertTokenizer, AlbertForSequenceClassification
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification


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
    print('---------------------------------------------------')


def bert_sentiment(query, passage):
    distilbert_sentiment_model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
    distilbert_sentiment_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')

    albert_sentiment_model = AlbertForSequenceClassification.from_pretrained('textattack/albert-base-v2-CoLA')
    albert_sentiment_tokenizer = AlbertTokenizer.from_pretrained('textattack/albert-base-v2-CoLA')

    start = time.perf_counter()
    combination = query + ' ' + passage

    # SENTIMENT ANALYSIS - DistilBERT
    # Query
    query_inputs = distilbert_sentiment_tokenizer(query, return_tensors="pt")
    labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
    outputs = distilbert_sentiment_model(**query_inputs, labels=labels)
    sentiment_probs = softmax(outputs.logits, dim=1)
    dist_query_prob_positive = round(float(sentiment_probs.data[0][1]), 4)
    dist_query_prob_negative = round(float(sentiment_probs.data[0][0]), 4)
    # Passage
    sentence_inputs = distilbert_sentiment_tokenizer(passage, return_tensors="pt")
    labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
    outputs = distilbert_sentiment_model(**sentence_inputs, labels=labels)
    sentiment_probs = softmax(outputs.logits, dim=1)
    dist_passage_prob_positive = round(float(sentiment_probs.data[0][1]), 4)
    dist_passage_prob_negative = round(float(sentiment_probs.data[0][0]), 4)
    # Query + Passage combo
    qp_inputs = distilbert_sentiment_tokenizer(combination, return_tensors="pt")
    labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
    outputs = distilbert_sentiment_model(**qp_inputs, labels=labels)
    sentiment_probs = softmax(outputs.logits, dim=1)
    dist_qp_prob_positive = round(float(sentiment_probs.data[0][1]), 4)
    dist_qp_prob_negative = round(float(sentiment_probs.data[0][0]), 4)

    # SENTIMENT ANALYSIS - ALBERT
    # Query
    query_inputs = albert_sentiment_tokenizer(query, return_tensors="pt")
    labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
    outputs = albert_sentiment_model(**query_inputs, labels=labels)
    sentiment_probs = softmax(outputs.logits, dim=1)
    albert_query_prob_positive = round(float(sentiment_probs.data[0][0]), 4)
    albert_query_prob_negative = round(float(sentiment_probs.data[0][1]), 4)
    # Passage
    sentence_inputs = albert_sentiment_tokenizer(passage, return_tensors="pt")
    labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
    outputs = albert_sentiment_model(**sentence_inputs, labels=labels)
    sentiment_probs = softmax(outputs.logits, dim=1)
    albert_passage_prob_positive = round(float(sentiment_probs.data[0][0]), 4)
    albert_passage_prob_negative = round(float(sentiment_probs.data[0][1]), 4)
    # Query + Passage combo
    qp_inputs = albert_sentiment_tokenizer(combination, return_tensors="pt")
    labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
    outputs = albert_sentiment_model(**qp_inputs, labels=labels)
    sentiment_probs = softmax(outputs.logits, dim=1)
    albert_qp_prob_positive = round(float(sentiment_probs.data[0][0]), 4)
    albert_qp_prob_negative = round(float(sentiment_probs.data[0][1]), 4)

    end = time.perf_counter()
    time_taken = end - start

    print('Query: ' + query)
    print('Passage: ' + passage)
    print('DistilBERT chance query positive: ' + str(dist_query_prob_positive))
    print('DistilBERT chance query negative: ' + str(dist_query_prob_negative))
    print('DistilBERT chance passage positive: ' + str(dist_passage_prob_positive))
    print('DistilBERT chance passage negative: ' + str(dist_passage_prob_negative))
    print('DistilBERT chance qp positive: ' + str(dist_qp_prob_positive))
    print('DistilBERT chance qp negative: ' + str(dist_qp_prob_negative))
    print('----------------------------------------------------------------')
    print('ALBERT chance query positive: ' + str(albert_query_prob_positive))
    print('ALBERT chance query negative: ' + str(albert_query_prob_negative))
    print('ALBERT chance passage positive: ' + str(albert_passage_prob_positive))
    print('ALBERT chance passage negative: ' + str(albert_passage_prob_negative))
    print('ALBERT chance qp positive: ' + str(albert_qp_prob_positive))
    print('ALBERT chance qp negative: ' + str(albert_qp_prob_negative))
    print('Time: ' + str(time_taken))
    print('---------------------------------------------------')


def create_feature_file():
    # Initialize models and tokenizers
    next_sentence_model = BertForNextSentencePrediction.from_pretrained('bert-base-cased')
    next_sentence_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    distilbert_sentiment_model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
    distilbert_sentiment_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')

    albert_sentiment_model = AlbertForSequenceClassification.from_pretrained('textattack/albert-base-v2-CoLA')
    albert_sentiment_tokenizer = AlbertTokenizer.from_pretrained('textattack/albert-base-v2-CoLA')

    schemaQ = Schema(title=TEXT(stored=True), content=TEXT(stored=True))
    ixQ = open_dir('../index/testindexdir', schema=schemaQ)
    schemaP = Schema(title=TEXT(stored=True), content=TEXT(stored=True))
    ixP = open_dir('../index/qrels-index', schema=schemaP)

    i = 0
    feature_file = open("output/correct_bert_features_testing.txt", "a", newline='')
    for line in list(open("../../data/2019qrels-pass.txt", encoding='utf8')):
        queryID = line.split(' ')[0]
        passageID = line.split(' ')[2]
        relevance = int(line.split(' ')[3])

        qp = QueryParser('title', schema=ixQ.schema)
        q = qp.parse(queryID)

        pp = QueryParser('title', schema=ixP.schema)
        p = pp.parse(passageID)

        with ixQ.searcher() as s:
            results = s.search(q)
            query = results[0].get('content')

        with ixP.searcher() as s:
            results = s.search(p)
            passage = results[0].get('content')

        combination = query + ' ' + passage

        # NEXT SENTENCE PREDICTION
        # Encode prompt (query) and next_sentence (passage)
        encoded = next_sentence_tokenizer.encode_plus(query, text_pair=passage, return_tensors='pt')
        # We only need output tensor with relationships
        seq_relationship_logits = next_sentence_model(**encoded)[0]
        # We need Softmax to get probabilities. [P(continue), P(random)]
        next_sentence_probs = softmax(seq_relationship_logits, dim=1)

        prob_continue = round(float(next_sentence_probs.data[0][0]), 4)
        prob_random = round(float(next_sentence_probs.data[0][1]), 4)

        # SENTIMENT ANALYSIS - DistilBERT
        # Query
        query_inputs = distilbert_sentiment_tokenizer(query, return_tensors="pt")
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = distilbert_sentiment_model(**query_inputs, labels=labels)
        sentiment_probs = softmax(outputs.logits, dim=1)
        dist_query_prob_positive = round(float(sentiment_probs.data[0][1]), 4)
        dist_query_prob_negative = round(float(sentiment_probs.data[0][0]), 4)
        # Passage
        sentence_inputs = distilbert_sentiment_tokenizer(passage, return_tensors="pt")
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = distilbert_sentiment_model(**sentence_inputs, labels=labels)
        sentiment_probs = softmax(outputs.logits, dim=1)
        dist_passage_prob_positive = round(float(sentiment_probs.data[0][1]), 4)
        dist_passage_prob_negative = round(float(sentiment_probs.data[0][0]), 4)
        # Query + Passage combo
        qp_inputs = distilbert_sentiment_tokenizer(combination, return_tensors="pt")
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = distilbert_sentiment_model(**qp_inputs, labels=labels)
        sentiment_probs = softmax(outputs.logits, dim=1)
        dist_qp_prob_positive = round(float(sentiment_probs.data[0][1]), 4)
        dist_qp_prob_negative = round(float(sentiment_probs.data[0][0]), 4)

        # SENTIMENT ANALYSIS - ALBERT
        # Query
        query_inputs = albert_sentiment_tokenizer(query, return_tensors="pt")
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = albert_sentiment_model(**query_inputs, labels=labels)
        sentiment_probs = softmax(outputs.logits, dim=1)
        albert_query_prob_positive = round(float(sentiment_probs.data[0][0]), 4)
        albert_query_prob_negative = round(float(sentiment_probs.data[0][1]), 4)
        # Passage
        sentence_inputs = albert_sentiment_tokenizer(passage, return_tensors="pt")
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = albert_sentiment_model(**sentence_inputs, labels=labels)
        sentiment_probs = softmax(outputs.logits, dim=1)
        albert_passage_prob_positive = round(float(sentiment_probs.data[0][0]), 4)
        albert_passage_prob_negative = round(float(sentiment_probs.data[0][1]), 4)
        # Query + Passage combo
        qp_inputs = albert_sentiment_tokenizer(combination, return_tensors="pt")
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = albert_sentiment_model(**qp_inputs, labels=labels)
        sentiment_probs = softmax(outputs.logits, dim=1)
        albert_qp_prob_positive = round(float(sentiment_probs.data[0][0]), 4)
        albert_qp_prob_negative = round(float(sentiment_probs.data[0][1]), 4)

        if i % 500 == 0:
            print(i)
            print('Query: ' + query)
            print('Passage: ' + passage)
            print('Chance continuation: ' + str(prob_continue))
            print('Chance random: ' + str(prob_random))
            print('DistilBERT chance query positive: ' + str(dist_query_prob_positive))
            print('DistilBERT chance query negative: ' + str(dist_query_prob_negative))
            print('DistilBERT chance passage positive: ' + str(dist_passage_prob_positive))
            print('DistilBERT chance passage negative: ' + str(dist_passage_prob_negative))
            print('DistilBERT chance qp positive: ' + str(dist_qp_prob_positive))
            print('DistilBERT chance qp negative: ' + str(dist_qp_prob_negative))
            print('----------------------------------------------------------------')
            print('ALBERT chance query positive: ' + str(albert_query_prob_positive))
            print('ALBERT chance query negative: ' + str(albert_query_prob_negative))
            print('ALBERT chance passage positive: ' + str(albert_passage_prob_positive))
            print('ALBERT chance passage negative: ' + str(albert_passage_prob_negative))
            print('ALBERT chance qp positive: ' + str(albert_qp_prob_positive))
            print('ALBERT chance qp negative: ' + str(albert_qp_prob_negative))
            print('Relevancy rating: ' + str(relevance))
        i = i + 1

        next_line = """qid:{} pid:{} 1:{} 2:{} 3:{} 4:{} 5:{} 6:{} 7:{} 8:{} 9:{} 10:{} 11:{} 12:{} 13:{} 14:{}\n"""\
                    .format(str(queryID), str(passageID), str(prob_continue), str(prob_random),
                            str(dist_query_prob_positive), str(dist_query_prob_negative),
                            str(dist_passage_prob_positive), str(dist_passage_prob_negative),
                            str(dist_qp_prob_positive), str(dist_qp_prob_negative),
                            str(albert_query_prob_positive), str(albert_query_prob_negative),
                            str(albert_passage_prob_positive), str(albert_passage_prob_negative),
                            str(albert_qp_prob_positive), str(albert_qp_prob_negative))
        feature_file.write(next_line)
    feature_file.close()


if __name__ == '__main__':
    # passage = "I bought a gallon of milk."
    # query = "This afternoon I went to the store."
    # bert_next_sentence(query, passage)
    #
    # query = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
    # passage = "The sky is blue due to the shorter wavelength of blue light."
    # bert_next_sentence(query, passage)
    #
    # query = 'Hello my dog is cute'
    # passage = 'Do you like him too?'
    # bert_sentiment(query, passage)
    #
    # query = 'I hate dogs'
    # passage = 'What do you think of them?'
    # bert_sentiment(query, passage)
    #
    # query = "why was the Manhatten project succesful"
    # passage = "The presence of communication amid scientific minds was equally important to the success of the Manhattan Project as scientific intellect was. The only cloud hanging over the impressive achievement of the atomic researchers and engineers is what their success truly meant; hundreds of thousands of innocent lives obliterated."
    # bert_next_sentence(query, passage)
    # bert_sentiment(query, passage)

    create_feature_file()
