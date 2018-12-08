# DeepLearning-Reddit-BobTheBuilder

## Task description

The goal of this solution is to generate text based on Reddit comments.

The deep learning based algorithm aims to learn the reasons of popularity of
comments. After that it produces popular comments based on the previously
learned correlations.

The learning dataset consists of tokenized and character wise preprocessed data.

## Source files

### comment_collector.py

Collects some comments from a reddit submission using the Reddit API.

### character_based_lstm_learning.py

Prepares the data for an LSTM, using character based tokenisation, then finds a decent model structure using the hyperas hyperparameter optimization tool and further trains this model and saves different versions of it. The models then can be tested to generate some characters based on a random comment.

## Training and testing

We made two different solutions, character based and word based. These two require different method to be trained and tested.

### Character based

The training and testing are implemented in the character_based_lstm_learning.py source file. The training is implemented using keras, so it is required to be installed. The GPU version is also recommended as the training is relatively slow (1 - 1.5 hours on a 1050 TI). The hyperas package is also needed.

The training starts with a hyperparameter optimization phase, where we train our model for 10 epochs. After that, we choose the best model and further train it with 10 epochs 9 times, and save each iteration into the models directory. This process can be invoked by calling the train_model() function at the end of the file.

If we want to test the network we can call the test_data() function. It reads one of the previously trained networks (the last one by default), chooses a random comment from the data set, then generates 400 additional character based on that comment. The iteration parameter of the function determines which model to load from 0-9, where 9 is the one which we trained for the most epochs.

If you don't want to do both training and testing, you have to comment one of the above function calls at the end of the character_based_lstm_learning.py file and running it like that.


### Word based

The word based deep learning is a separate part of the project, which aims to
solve the same problem with a different approach.

The relevant files are *word_preprocess.py* and *word_learning.py*.

The content of *word_preprocess.py* is responsible for preprocessing the
downloaded Reddit comments (supported by comment_collector.py) in a manner that
it can be used to create a word-based learning on comments later on.

Our goal is to create a bidirectional map of words which maps the words to
a one-hot encoded integer representation. Also, each word is paired up with an
n-gram context{TODO bibl}, which basicly means the words occured in the same
line before the current word.

That way we can predict the next word based on the n-gram context.

Regarding the model itself, we used an SLTM network for learning the coherence
of the words, and softmax is used as the activation of the output. The design of
the model can be found in *word_learning.py*. The output of the network is
vector of probabilities representing the likelihood of a word coming after the
current word in a sentence. As a future plan we would like to take rating of the
comment into consideration as well.

The hiperparameters of the model were chosen with a manual approach based on our
intuition and several attempts. With adam optimization and categorical
crossentropy as a loss function the model seems to work pretty well.

### Word2vec approach

## Team members

- Oláh Gergely: olahkoo@gmail.com
- Oczot Balázs: balix18@gmail.com
- Bunth Tamás: btomi96@gmail.com
