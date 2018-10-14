# DeepLearning-Reddit-BobTheBuilder

## Task description

The goal of this solution is to generate text based on Reddit comments.

The deep learning based algorithm aims to learn the reasons of popularity of
comments. After that it produces popular comments based on the previously
learned correlations.

The learning dataset consists of tokenized and character wise preprocessed data.

## Source files
### redditPirate.py

Collects some comments from a reddit submission using the Reddit API.

### dataPreparation.py
Prepares the data for an LSTM, using character based tokenization.

### tokenizer.py
Prepares the data for an LSTM, using word based tokenization.

## Team members

- Oláh Gergely: olahkoo@gmail.com
- Oczot Balázs: balix18@gmail.com
- Bunth Tamás: btomi96@gmail.com
