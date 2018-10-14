import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import json

with open('data/askreddit.json') as f:
    data = json.load(f)

print(data[0])

# now let's tokenize the body part

tokenizer = Tokenizer()

def dataset_preparation(data):
    # split text to lines
    corpus = data.lower().split("\n")
    tokenizer.fit_on_texts(corpus)
    total_words = len(tokenizer.word_index) + 1
    input_sequences = []
    for line in corpus:
        # List of words (I hope)
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            # Let's do predictions based on all of the words before
            # the actual word.
            # see n_gram: https://medium.com/@shivambansal36/language-modelling-text-generation-using-lstms-deep-learning-for-nlp-ed36b224b275
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)
            max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences,   
                          maxlen=max_sequence_len, padding='pre'))
    return input_sequences, total_words

input_seq, total_words = dataset_preparation(data[0]["body"])
predictors, label = input_seq[:,:-1], input_seq[:,-1]
print( len(predictors), len(label))

# The following example dataframe contains the tokenized value of a word. (first column)
# all the other columns represent words which occurred before the word in the same line.
df = pd.DataFrame(data=np.column_stack((label, predictors)))
print(df)

# Label should be one-hot encoded for learning
import keras.utils as k_utils
label = k_utils.to_categorical(label, num_classes=total_words)