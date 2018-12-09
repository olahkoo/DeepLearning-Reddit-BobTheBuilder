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
import random
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform

# loads comments and creates character-number mapping
def load_comments():
        with open("static-data/single-submission-comments-3.json") as json_data:
                data_raw = json.load(json_data)

        comments = []

        # we create a list of comments, where each comment is stored as list of characters
        for item in data_raw:
                # less than 150 character comments are too short for training
                if (len(item["body"]) >= 50 and item["score"] > 5):
                        comments.append(list(item["body"]))

        print(f'Number of comments: {len(comments)}')

        # all characters
        characters = []
        for sublist in comments:
                for item in sublist:
                        characters.append(item)

        characters = sorted(list(set(characters)))

        print(f'Number of unique characters: {len(characters)}')

        n_to_char = {n:char for n, char in enumerate(characters)}
        char_to_n = {char:n for n, char in enumerate(characters)}

        return comments, characters, n_to_char, char_to_n

# data preparation for hyperas
def data():
        # comments, characters, n_to_char, char_to_n = load_comments()

        with open("static-data/single-submission-comments-3.json") as json_data:
                data_raw = json.load(json_data)

        comments = []

        # we create a list of comments, where each comment is stored as list of characters
        for item in data_raw:
                # less than 150 character comments are too short for training
                if (len(item["body"]) >= 50 and item["score"] > 5):
                        comments.append(list(item["body"]))

        print(f'Number of comments: {len(comments)}')

        # all characters
        characters = []
        for sublist in comments:
                for item in sublist:
                        characters.append(item)

        characters = sorted(list(set(characters)))

        print(f'Number of unique characters: {len(characters)}')

        n_to_char = {n:char for n, char in enumerate(characters)}
        char_to_n = {char:n for n, char in enumerate(characters)}


        seq_length = 40
        step = 3
        X = []
        Y = []
        # each comment is used as a single piece of text
        for comment in comments:
                length = len(comment)
                for i in range(0, length - seq_length, step):
                        sequence = comment[i:i + seq_length]
                        label = comment[i + seq_length]
                        X.append([char_to_n[char] for char in sequence])
                        Y.append(char_to_n[label])

        print(f'Number of all samples: {len(X)}')

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

        return X_train, Y_train, X_valid, Y_valid, X_test, Y_test

# model creation and model fitting for hyperas
def create_model(X_train, Y_train, X_valid, Y_valid, X_test, Y_test):
        # hyperparams of the model
        n_layer1 = {{choice([128, 256, 512])}}
        n_layer2 = {{choice([128, 256, 512])}}
        dropout_1 = {{uniform(0, 0.5)}}
        dropout_2 = {{uniform(0, 0.5)}}
        optim = {{choice(['rmsprop', 'adam'])}}
        n_batch = {{choice([64, 128, 256])}}

        # an LSTM model that can learn character sequences
        model = Sequential()
        model.add(LSTM(n_layer1, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
        model.add(Dropout(dropout_1))
        model.add(LSTM(n_layer2))
        model.add(Dropout(dropout_2))
        model.add(Dense(Y_train.shape[1], activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=optim)

        # training the model
        result = model.fit(X_train, Y_train,
                        batch_size = n_batch,
                        epochs = 10,
                        verbose = 2,
                        validation_data = (X_valid, Y_valid),
                        shuffle=True)
        
        validation_loss = np.amax(result.history['val_loss'])
        return {'loss': validation_loss, 'status': STATUS_OK, 'model': model}

# some data we need for evaluating and testing the model
X_train, Y_train, X_valid, Y_valid, X_test, Y_test = data()
comments, characters, n_to_char, char_to_n = load_comments()

random.seed(42)
rand_index = random.randint(0, len(comments) - 1)

# training a model with hyperparameter optimalization, then further training it and saving the models to use them later
def train_model():
        best_run, best_model = optim.minimize(model=create_model,
                                                data=data,
                                                algo=tpe.suggest,
                                                max_evals=10,
                                                trials=Trials())

        best_model.save('models/char_based_optimized_0.hdf5')

        print("Best performing model chosen hyper-parameters:")
        print(best_run)
        print()

        for iteration in range(1, 10):
                print(f'Iteration {iteration}, training for 10 more epochs')

                n_batch = [64, 128, 256][best_run['n_batch']]
                best_model.fit(X_train, Y_train,
                        batch_size = n_batch,
                        epochs = 10,
                        verbose = 2,
                        validation_data = (X_valid, Y_valid),
                        shuffle=True)
                best_model.save(f'models/char_based_optimized_{iteration}.hdf5')

                rand_comment = list(comments[rand_index])[:len(X_train[0])]

                string_mapped = [char_to_n[c] for c in rand_comment]

                for _ in range(200):
                        x = np.reshape(string_mapped,(1,len(string_mapped), 1))
                        x = x / float(len(characters))

                        pred = best_model.predict(x, verbose=0)
                        pred_index = np.argmax(pred)
                        rand_comment.append(n_to_char[pred_index])

                        string_mapped.append(pred_index)
                        string_mapped = string_mapped[1:len(string_mapped)]

                # the generated comment
                print(''.join(rand_comment))
                print()

# loading a model and generating 400 characters with it from a random comment
def test_data(iteration = 9):
        model = load_model(f'models/char_based_optimized_{iteration}.hdf5')

        rand_index = random.randint(0, len(comments) - 1)
        rand_comment = list(comments[rand_index])[:len(X_train[0])]

        string_mapped = [char_to_n[c] for c in rand_comment]

        for _ in range(400):
                x = np.reshape(string_mapped,(1,len(string_mapped), 1))
                x = x / float(len(characters))

                pred = model.predict(x, verbose=0)
                pred_index = np.argmax(pred)
                rand_comment.append(n_to_char[pred_index])

                string_mapped.append(pred_index)
                string_mapped = string_mapped[1:len(string_mapped)]

        # the generated comment
        print(''.join(rand_comment))
        print()

# train_model()
test_data(0)