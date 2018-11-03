import numpy as np
import pandas as pd

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from keras.models import Sequential
import keras.utils as k_utils

import json

with open('data/askreddit.json') as f:
    data = json.load(f)

print(data[0])

tokenizer = Tokenizer()

# now let's tokenize the body part
def dataset_preparation(data):
    # split text to lines
    corpus = []
    for single_comment in data:
        for comment_part in single_comment.lower().split("\n"):
            corpus.append(comment_part)
    
    tokenizer.fit_on_texts(corpus)

    # Words and their index values
    for key, value in tokenizer.word_index.items():
        print(f"{key} -> {value}")

    total_words = len(tokenizer.word_index) + 1
    input_sequences = []
    for line in corpus:
        # List of words (I hope)
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            # Let's do predictions based on all of the words before the actual word.
            # see n_gram: https://medium.com/@shivambansal36/language-modelling-text-generation-using-lstms-deep-learning-for-nlp-ed36b224b275
            # TODO fixed width sliding window might be better
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)
    
    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
    
    return input_sequences, max_sequence_len, total_words

comment_bodies = []
for comment in data:
    comment_bodies.append(comment["body"])

input_seq, max_sequence_len, total_words = dataset_preparation(comment_bodies[:500])
predictors, label = input_seq[:,:-1], input_seq[:,-1]
print( len(predictors), len(label))

# The following example dataframe contains the tokenized value of a word. (first column)
# all the other columns represent words which occurred before the word in the same line.
df = pd.DataFrame(data=np.column_stack((label, predictors)))
print(df)

# Label should be one-hot encoded for learning
label = k_utils.to_categorical(label, num_classes=total_words)
print(label)

# Separating the training, validation and test data
valid_split = 0.2
test_split = 0.1
sample_size = predictors.shape[0]

startIndex = 0
midIndex   = int(sample_size * (1 - valid_split - test_split))
endIndex   = int(sample_size * (1 - test_split))

X_train = predictors[startIndex:midIndex]
Y_train =      label[startIndex:midIndex]
X_valid = predictors[midIndex:endIndex]
Y_valid =      label[midIndex:endIndex]
X_test  = predictors[endIndex:]
Y_test  =      label[endIndex:]

def create_model(predictors, label, max_sequence_len, total_words):
    input_len = max_sequence_len - 1

    model = Sequential()
    model.add(Embedding(total_words, 10, input_length=input_len))
    model.add(LSTM(150))
    model.add(Dropout(0.1))
    model.add(Dense(total_words, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.fit(predictors, label, epochs=100, verbose=1)

    return model

def generate_text(seed_text, next_words, max_sequence_len, model):
    for j in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = model.predict_classes(token_list, verbose=0)
  
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    
    return seed_text

model = create_model(X_train, Y_train, max_sequence_len, total_words)
print(generate_text("Honestly, the camera has always blown my mind. It creates a lasting picture of something that happens in the real world, and", 30, max_sequence_len, model))
