import numpy as np
from keras.utils import np_utils
import matplotlib.pyplot as plt
import json

with open("data/askreddit.json") as json_data:
    data_raw = json.load(json_data)

data = []

# we create a list of comments, where each comment is stored as list of characters
for item in data_raw:
    # less than 50 character comments are too short for training
    if (len(item["body"]) >= 50):
        data.append(list(item["body"]))

# all characters
characters = []
for sublist in data:
    for item in sublist:
        characters.append(item)

characters = sorted(list(set(characters)))
# n_to_char = {n:char for n, char in enumerate(characters)} # will be used to decode predictions made by the network
char_to_n = {char:n for n, char in enumerate(characters)}

seq_length = 20
X = []
Y = []
# each comment is used as a single piece of text
for comment in data:
    length = len(comment)
    for i in range(0, length-seq_length, 1):
        sequence = comment[i:i + seq_length]
        label = comment[i + seq_length]
        X.append([char_to_n[char] for char in sequence])
        Y.append(char_to_n[label])

# lstm requires data in the form of (number_of_sequences, length_of_sequence, number_of_features)
X_modified = np.reshape(X, (len(X), seq_length, 1))
X_modified = X_modified / float(len(characters))
# one-hot encoding y values
Y_modified = np_utils.to_categorical(Y)

# separating the training, validation and test data
valid_split = 0.2
test_split = 0.1
sample_size = X_modified.shape[0]

X_train = X_modified[0:int(sample_size * (1 - valid_split - test_split))]
Y_train = Y_modified[0:int(sample_size * (1 - valid_split - test_split))]
X_valid = X_modified[int(sample_size * (1 - valid_split - test_split)):int(sample_size * (1 - test_split))]
Y_valid = Y_modified[int(sample_size * (1 - valid_split - test_split)):int(sample_size * (1 - test_split))]
X_test  = X_modified[int(sample_size * (1 - test_split)):]
Y_test  = Y_modified[int(sample_size * (1 - test_split)):]