from torch.nn.functional import softmax
from transformers import BertForNextSentencePrediction, BertTokenizer
from whoosh.index import create_in,open_dir
from whoosh.fields import *
from whoosh.qparser import QueryParser


# Taken from: https://stackoverflow.com/questions/55111360/using-bert-for-next-sentence-prediction
def bert_next_sentence(prompt, next_sentence):
    model = BertForNextSentencePrediction.from_pretrained('bert-base-cased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    # encode the two sequences. Particularly, make clear that they must be
    # encoded as "one" input to the model by using 'seq_B' as the 'text_pair'
    encoded = tokenizer.encode_plus(prompt, text_pair=next_sentence, return_tensors='pt')

    # a model's output is a tuple, we only need the output tensor containing
    # the relationships which is the first item in the tuple
    seq_relationship_logits = model(**encoded)[0]

    # we still need softmax to convert the logits into probabilities
    # index 0: sequence B is a continuation of sequence A
    # index 1: sequence B is a random sequence
    probs = softmax(seq_relationship_logits, dim=1)

    print('Prompt: ' + prompt)
    print('Next: ' + next_sentence)
    print('Chance continuation: ' + str(probs.data[0][0]))
    print('Chance random: ' + str(probs.data[0][1]))
    return float(probs.data[0][0]), float(probs.data[0][1])


if __name__ == '__main__':
    passage = "The presence of communication amid scientific minds was equally important to the success of the Manhattan Project as scientific intellect was. The only cloud hanging over the impressive achievement of the atomic researchers and engineers is what their success truly meant; hundreds of thousands of innocent lives obliterated."
    query = "why was the Manhatten project succesful"
    _, _ = bert_next_sentence(query, passage)

    passage = "I bought a gallon of milk."
    query = "This afternoon I went to the store."
    _, _ = bert_next_sentence(query, passage)

    query = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
    passage = "The sky is blue due to the shorter wavelength of blue light."
    _, _ = bert_next_sentence(query, passage)

    query = 'I like cookies !'
    passage = 'Do you like them ?'
    _, _ = bert_next_sentence(query, passage)
