import numpy as np
import pandas as pd

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import keras.utils as k_utils

import json

class WordPreprocessor():
    def __init__(self):
        self.tokenizer = Tokenizer()
        self.max_sequence_len = 0
        self.total_words_count = 0
    def preprocess_file(self, file_path):
        """
        Pre-processes the content of a json file with a specific format

        :param file_path: path of a json file
        :return: (train_data, train_target), (valid_data, valid_target), (test_data, test_target), max_sequence_len, total_words
        """
        with open(file_path) as f:
            data = json.load(f)
            return self.preprocess_data(data)

    # now let's tokenize the body part
    def dataset_preparation(self, data):
        """Tokenize words.

        :param data: A multi-line text
        :return: n_gram sequences of words
        """
        # split text to lines
        corpus = []
        for single_comment in data:
            for comment_part in single_comment.lower().split("\n"):
                corpus.append(comment_part)

        self.tokenizer.fit_on_texts(corpus)

        # Words and their index values
        for key, value in self.tokenizer.word_index.items():
            print(f"{key} -> {value}")

        self.total_words_count = len(self.tokenizer.word_index) + 1
        input_sequences = []
        for line in corpus:
            # List of words (I hope)
            token_list = self.tokenizer.texts_to_sequences([line])[0]
            for i in range(1, len(token_list)):
                # Let's do predictions based on all of the words before the actual word.
                # see n_gram: https://medium.com/@shivambansal36/language-modelling-text-generation-using-lstms-deep-learning-for-nlp-ed36b224b275
                # TODO fixed width sliding window might be better
                n_gram_sequence = token_list[:i + 1]
                input_sequences.append(n_gram_sequence)

        self.max_sequence_len = max([len(x) for x in input_sequences])
        input_sequences = np.array(pad_sequences(input_sequences, maxlen=self.max_sequence_len, padding='pre'))

        return input_sequences

    def preprocess_data(self, data):
        """Turns a multi-line text into one-hot encoded words with their n-gram context.

        :param data: multi-line string
        :return: (train_data, train_target), (valid_data, valid_target), (test_data, test_target), max_sequence_len, total_words
        """
        print("[INFO] Start word preprocessing")
        print("[INFO] length of data to preprocess: {}".format(len(data)))
        print("[INFO] Example element:")
        print(data[0])

        comment_bodies = []
        for comment in data:
            comment_bodies.append(comment["body"])

        input_seq = self.dataset_preparation(comment_bodies[:500])
        predictors, label = input_seq[:, :-1], input_seq[:, -1]
        print("[INFO] length of predictors: {}".format(len(predictors)))
        print("[INFO] length of label: {}".format(len(label)))

        # The following example dataframe contains the tokenized value of a word. (first column)
        # all the other columns represent words which occurred before the word in the same line.
        df = pd.DataFrame(data=np.column_stack((label, predictors)))
        print("Example data frame: {}".format(df))

        # Label should be one-hot encoded for learning
        label = k_utils.to_categorical(label, num_classes=self.total_words_count)

        # Separating the training, validation and test data
        valid_split = 0.2
        test_split = 0.1
        sample_size = predictors.shape[0]

        startIndex = 0
        midIndex = int(sample_size * (1 - valid_split - test_split))
        endIndex = int(sample_size * (1 - test_split))

        X_train = predictors[startIndex:midIndex]
        Y_train = label[startIndex:midIndex]
        X_valid = predictors[midIndex:endIndex]
        Y_valid = label[midIndex:endIndex]
        X_test = predictors[endIndex:]
        Y_test = label[endIndex:]

        return (X_train, Y_train), (X_valid, Y_valid), (X_test, Y_test)





