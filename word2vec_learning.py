from keras.models import Model, Sequential
from keras.layers import Input, Dense, Reshape, merge, LSTM, Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import skipgrams
from keras.preprocessing import sequence

import random

import collections
import zipfile
import json

import numpy as np
import tensorflow as tf
from keras_preprocessing.text import Tokenizer

tf.set_random_seed(534564564)
np.random.seed(534564564)
random.seed(534564564)

json_file_path = 'data/askreddit.json'

def read_text_from_file(filepath):
    """
    Open json file

    :param filepath: path of a json file
    :return: list of words
    """
    words = []
    with open(filepath) as f:
        data = json.load(f)
        print("[INFO] An example object from json structure: {}".format(data[0]))
        comment_bodies = []
        for comment in data:
            comment_bodies.append(comment["body"])
        corpus = []
        for single_comment in comment_bodies:
            for comment_part in single_comment.lower().split("\n"):
                corpus.append(comment_part)
        for line in corpus:
            words.extend(line.split())

    return words


# Read the data into a list of strings.
def read_data(filename):
    """Extract the first file enclosed in a zip file as a list of words."""
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data


def build_dataset(words, n_words):
    """Process raw inputs into a dataset."""
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary

def collect_data(vocabulary_size=10000):
    vocabulary = read_text_from_file(json_file_path)
    print("[INFO] First words in vocabulary: {}".format(vocabulary[:7]))
    data, count, dictionary, reverse_dictionary = build_dataset(vocabulary,
                                                                vocabulary_size)
    del vocabulary  # Hint to reduce memory.
    return data, count, dictionary, reverse_dictionary

vocab_size = 10000
data, count, dictionary, reverse_dictionary = collect_data(vocabulary_size=vocab_size)
print("[INFO] first elements of data to work with after preprocessing: {}".format(data[:7]))

window_size = 3
vector_dim = 300
epochs = 10000 # 10000 # 200000 # 1

valid_size = 16     # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)

# Sample table is required to ensure it samples negative sample words
# in a balanced manner (not just common words)
sampling_table = sequence.make_sampling_table(vocab_size)
# skipgrams returns (target, context) pairs, and labels if it's in context or not
couples, labels = skipgrams(data, vocab_size, window_size=window_size, sampling_table=sampling_table)
word_target, word_context = zip(*couples)
word_target = np.array(word_target, dtype="int32")
word_context = np.array(word_context, dtype="int32")

print("[INFO] first 10 couples: {}".format(couples[:10]))
print("[INFO] first 10 labels: {}".format(labels[:10]))

# create some input variables
input_target = Input((1,))
input_context = Input((1,))

embedding = Embedding(vocab_size, vector_dim, input_length=1, name='embedding')
target = embedding(input_target)
target = Reshape((vector_dim, 1))(target)
context = embedding(input_context)
context = Reshape((vector_dim, 1))(context)

# setup a cosine similarity operation which will be output in a secondary model
# Auxiliary output
similarity = merge.dot([target, context], axes=0, normalize=True)

# now perform the dot product operation to get a similarity measure
dot_product = merge.dot([target, context], axes=1, normalize=False)
dot_product = Reshape((1,))(dot_product)
# add the sigmoid output layer
output = Dense(1, activation='sigmoid')(dot_product)
# create the primary training model
model = Model(inputs=[input_target, input_context], outputs=output)
model.compile(loss='binary_crossentropy', optimizer='rmsprop')

# create a secondary validation model to run our similarity checks during training
validation_model = Model(inputs=[input_target, input_context], outputs=similarity)

class SimilarityCallback:
    def run_sim(self):
        for i in range(valid_size):
            valid_word = reverse_dictionary[valid_examples[i]]
            top_k = 8  # number of nearest neighbors
            sim = self._get_sim(valid_examples[i])
            nearest = (-sim).argsort()[1:top_k + 1]
            log_str = 'Nearest to %s:' % valid_word
            for k in range(top_k):
                close_word = reverse_dictionary[nearest[k]]
                log_str = '%s %s,' % (log_str, close_word)
            print(log_str)

    @staticmethod
    def _get_sim(valid_word_idx):
        sim = np.zeros((vocab_size,))
        in_arr1 = np.zeros((1,))
        in_arr2 = np.zeros((1,))
        in_arr1[0,] = valid_word_idx
        for i in range(vocab_size):
            in_arr2[0,] = i
            out = validation_model.predict_on_batch([in_arr1, in_arr2])
            sim[i] = out
        return sim
sim_cb = SimilarityCallback()

arr_1 = np.zeros((1,))
arr_2 = np.zeros((1,))
arr_3 = np.zeros((1,))
for cnt in range(epochs):
    idx = np.random.randint(0, len(labels)-1)
    arr_1[0,] = word_target[idx]
    arr_2[0,] = word_context[idx]
    arr_3[0,] = labels[idx]
    loss = model.train_on_batch([arr_1, arr_2], arr_3)
    if cnt % 100 == 0:
        print("[INFO] Iteration {}, loss={}".format(cnt, loss))
    if cnt % 9999 == 0:
        sim_cb.run_sim()

def determine_occurrences(json_path):
    # determine occurrences of distinct words
    words = read_text_from_file(json_path)
    dict = collections.Counter(words)
    del words
    return dict

# Now let's move on to predicting popular words
def create_word_poularity_dictionaries(filepath):
    """
    Create dictionary for obtaining score of words. Score represents the
    popularity of the comment on Reddit.

    :param filepath: path of json file
    :return: pairs of words and scores. The words are already turned into integet
    """
    # Extract popularity of each word from json file
    with open(filepath) as f:
        json_data = json.load(f)
        word_score_dictionary = {}
        comment_bodies = []
        word_occurrency_dict = determine_occurrences(filepath)
        #print("[DEBUG] occurrency: {}".format(word_occurrency_dict))

        # get text and score from each record
        for element in json_data:
            comment_bodies.append((element["body"], element["score"]))

        for (comment, score) in comment_bodies:
            for line in comment.lower().split("\n"):
                for word in line.split(" "):
                    word_coded = dictionary[word] if word in dictionary else -1
                    if word_coded < -1:
                        print("[WARNING] word is not found in integer encoded dictionary")
                        continue
                    if word_coded in word_score_dictionary:
                        word_score_dictionary[word_coded] += score / (word_occurrency_dict[word] if word_occurrency_dict[word] != 0 else 1)
                    else:
                        word_score_dictionary[word_coded] = score / (word_occurrency_dict[word] if word_occurrency_dict[word] != 0 else 1)
        return word_score_dictionary

score_dicts = create_word_poularity_dictionaries(json_file_path)
#print("[DEBUG] score dictionary: {}".format(score_dicts))

# Measure of desire to use popular words
popularity_factor = 0.2

# Let us begin the story with the word 'the'
current_word = dictionary['the']

for i in range(20):
    in_arr1 = np.zeros((1,))
    in_arr2 = np.zeros((1,))
    in_arr1[0,] = current_word
    predictions = np.zeros((vocab_size,))
    for word in range(vocab_size):
        in_arr2[0,] = word
        out = validation_model.predict_on_batch([in_arr1, in_arr2])
        predictions[word] = out
    top_k = 8  # number of nearest neighbors to choose from

    nearest = (-predictions).argsort()[1:top_k + 1]
    # Choose the one which is the most popular

    l = list(nearest)
    l.sort(key=lambda x: score_dicts[x], reverse=True)
    nearest = np.array(l)

    random_nearest = nearest[random.randint(1, 5)]
    most_popular_score = score_dicts[random_nearest]
    most_popular_word = random_nearest

    print("[INFO] The next word is: '{}', with popularity: {}".format(reverse_dictionary[most_popular_word], most_popular_score))
    current_word = most_popular_word
