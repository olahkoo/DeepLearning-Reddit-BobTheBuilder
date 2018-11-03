import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.models import load_model
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import json

with open("data/askreddit.json") as json_data:
    data_raw = json.load(json_data)

data = []

# we create a list of comments, where each comment is stored as list of characters
for item in data_raw:
    # less than 150 character comments are too short for training
    if (len(item["body"]) >= 150 and item["score"] > 200):
        data.append(list(item["body"]))

print(len(data))

# all characters
characters = []
for sublist in data:
    for item in sublist:
        characters.append(item)

characters = sorted(list(set(characters)))
n_to_char = {n:char for n, char in enumerate(characters)}
char_to_n = {char:n for n, char in enumerate(characters)}

seq_length = 100
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

print(len(X))

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

# an LSTM model that can learn character sequences
model = Sequential()
model.add(LSTM(400, input_shape=(X_modified.shape[1], X_modified.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(400))
model.add(Dropout(0.2))
model.add(Dense(Y_modified.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

# early stopping with saving best model weights
# early_stopping = EarlyStopping(patience = 10, verbose = 1)
# checkpointer = ModelCheckpoint(filepath = 'models/char_based_early_stopping.hdf5', save_best_only = True, verbose = 1)
# training the model
# model.fit(X_train, Y_train,
#         batch_size = 100,
#         epochs = 1000,
#         verbose = 2,
#         callbacks=[checkpointer, early_stopping],
#         validation_data = (X_valid, Y_valid),
#         shuffle=True)
model = load_model('models/char_based_initial.hdf5')
# model = load_model('models/char_based_early_stopping.hdf5')

# the text we generate starts with this line
# string_mapped = X[420]
# full_text = [n_to_char[c] for c in string_mapped]
# full_text = list("Hi Reddit. I am Senator Bernie Sanders. I'll start answering questions at 2 p.m. The most important ")
full_text = list("Pineapples do not grow on palm trees. I always thought there were certain types of palm trees that a")
string_mapped = [char_to_n[c] for c in full_text]
print(len(full_text))
for i in range(200):
        x = np.reshape(string_mapped,(1,len(string_mapped), 1))
        x = x / float(len(characters))

        pred_index = np.argmax(model.predict(x, verbose=0))
        full_text.append(n_to_char[pred_index])

        string_mapped.append(pred_index)
        string_mapped = string_mapped[1:len(string_mapped)]

# the predicted comment
print(''.join(full_text))