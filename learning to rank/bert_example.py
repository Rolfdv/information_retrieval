from torch.nn.functional import softmax
from transformers import BertForNextSentencePrediction, BertTokenizer
from whoosh.index import create_in,open_dir
from whoosh.fields import *
from whoosh.qparser import QueryParser
import time


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


def create_feature_file():
    model = BertForNextSentencePrediction.from_pretrained('bert-base-cased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    schemaQ = Schema(title=TEXT(stored=True), content=TEXT(stored=True))
    ixQ = open_dir('../index/testindexdir', schema=schemaQ)
    schemaP = Schema(title=TEXT(stored=True), content=TEXT(stored=True))
    ixP = open_dir('../index/qrels-index', schema=schemaP)

    i = 0
    feature_file = open("output/bert_features.txt", "a", newline='')
    for line in list(open("../data/2019qrels-pass.txt", encoding='utf8')):
        queryID = line.split(' ')[0]
        passageID = line.split(' ')[2]
        relevance = int(line.split(' ')[3])

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

        # Encode prompt (query) and next_sentence (passage)
        encoded = tokenizer.encode_plus(query, text_pair=passage, return_tensors='pt')
        # We only need output tensor with relationships
        seq_relationship_logits = model(**encoded)[0]
        # We need Softmax to get probabilities. [P(continue), P(random)]
        probs = softmax(seq_relationship_logits, dim=1)

        prob_continue = round(float(probs.data[0][0]), 4)
        prob_random = round(float(probs.data[0][1]), 4)

        if i % 1000 == 0:
            print(i)
            print('Prompt: ' + query)
            print('Next: ' + passage)
            print('Chance continuation: ' + str(prob_continue))
            print('Chance random: ' + str(prob_random))
            print('Relevancy rating: ' + str(relevance))
        i = i+1

        feature_file.write('qid:' + str(queryID) + ' pid:' + str(passageID) + ' 1:' + str(prob_continue) + ' 2:' + str(prob_random) + '\n')
    feature_file.close()


if __name__ == '__main__':
    # passage = "The presence of communication amid scientific minds was equally important to the success of the Manhattan Project as scientific intellect was. The only cloud hanging over the impressive achievement of the atomic researchers and engineers is what their success truly meant; hundreds of thousands of innocent lives obliterated."
    # query = "why was the Manhatten project succesful"
    # _, _ = bert_next_sentence(query, passage)
    #
    # passage = "The presence of communication amid scientific minds was equally important to the success of the Manhattan Project as scientific intellect was."
    # query = "why was the Manhatten project succesful"
    # _, _ = bert_next_sentence(query, passage)
    #
    # passage = "I bought a gallon of milk."
    # query = "This afternoon I went to the store."
    # _, _ = bert_next_sentence(query, passage)
    #
    # query = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
    # passage = "The sky is blue due to the shorter wavelength of blue light."
    # _, _ = bert_next_sentence(query, passage)
    #
    # query = 'I like cookies !'
    # passage = 'Do you like them ?'
    # _, _ = bert_next_sentence(query, passage)

    create_feature_file()
