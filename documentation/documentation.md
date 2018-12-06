# Deep learning homework
## Abstract
Our goal was to create a deep neural network that can generate comments based on comments on reddit written by real people. We used two different approaches for the solution. The first one is to think of comments as character sequences, so the network only predicts one character at a time. We use a recurrent neural network for this problem, more specifically L(ong) S(hort) T(erm) M(emory) network. The other approach is a word based solution where we predict the next word by defining a context for each word and examining the similarity of these contextes.

## Introduction
Natural language processing is a very popular field in deep learning. There are two different approaches to it, one is the character based and the other is the word based approach.

The character based approach typically uses an LSTM network, which is mostly used for time series problems.


## Word based model using one-hot encoding
The naive word based approach uses a tokenization process to extract words from the corpus. Then we created an indexed list of tokens for every sentence, that later we utilized to build n-gram sequences, where the current sequence serves as a predictor, and the next sequence's last word as a label. As the sequences are not necessarily equally long, we have to add padding to them. After one-hot encoding the token indices, they now can be used to train the network. The network's first layer is an embedding, which takes the input sequences in the one-hot encoded form, then an LSTM layer computes the output, then a droput layer helps to prevent overfitting, and the output is a dense layer, which uses softmax activation to try to predict the best possiple output. By using the network we can predict an input sequence's next words by using the sequence as preditor. 


## Character based model
The character based model interprets the comments as a sequence of characters. The text generation is based on the previous character sequence, and we generate exactly one character at a time. For this approach the comments are split into 40 characters long sequences, where the sequences are overlapping with a 3 step from each other. Each sequence has a label, which is the next character following the sequence. The sentences will be our training input, and the labels will be the output.

We cannot feed raw characters to a neural network, we need some kind of character-number mapping. For this we collected all the unique characters in the comments, put them in a list, then assigned its index to each character. Furthermore, we need to standardize these values, otherwise we could have very big weight values in the model, so we divided them with the number of characters thus squeezing them between 0 and 1. The output values are one-hot encoded, which means the output will be an n dimension vector, where each element represents the "likeliness" of a character and n is the number of unique characters. The predicted character will be the index of the highest element in the vector.

The model architecture is a classic LSTM network, where the input dimension is (40, 1), the output dimension is (n, 1) where n is the number of unique characters. There are two inner lstm layers, each with 512 memory cells and a dropout layer. The dropout probabilites are 0.4 and 0.25. After the second lstm layer there is a dense layer which is the output layer and uses softmax activation.
