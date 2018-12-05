# Deep learning homework
## Abstract
Our goal was to create a deep neural network that can generate comments based on comments on reddit written by real people. We used two different approaches for the solution. The first one is to think of comments as character sequences, so the network only predicts one character at a time. We use a recurrent neural network for this problem, more specifically L(ong) S(hort) T(erm) M(emory) network. The other approach is a word based solution where we predict the next word by defining a context for each word and examining the similarity of these contextes.

## Introduction
Natural language processing is a very popular field in deep learning. There are two different approaches to it, one is the character based and the other is the word based approach.

The character based approach typically uses an LSTM network, which is mostly used for time series problems.


## Word based modell using ene-hot encoding
The naive word based approach uses a tokenization process to extract words from the cropus. Then we created an indexed list of tokens for every sentence, that later we utilized to build n-gram sequences, where the current sequence serves as a predictor, and the next sequence's last word as a label. As the sequences are not necessarily equally long, we have to add padding to them. After one-hot encoding the token indices, they now can be used to train the network. The network's first layer is an embedding, which takes the input sequences in the one-hot encoded form, then an LSTM layer computes the output, then a droput layer helps to prevent overfitting, and the output is a dense layer, which uses softmax activation to try to predict the best possiple output. By using the network we can predict an input sequence's next words by using the sequence as preditor. 
