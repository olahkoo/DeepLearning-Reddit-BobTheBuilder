# Deep learning homework
## Abstract
Our goal was to create a deep neural network that can generate comments based on comments on reddit written by real people. We used two different approaches for the solution. The first one is to think of comments as character sequences, so the network only predicts one character at a time. We use a recurrent neural network for this problem, more specifically L(ong) S(hort) T(erm) M(emory) network. The other approach is a word based solution where we predict the next word by defining a context for each word and examining the similarity of these contextes.

## Introduction
Natural language processing is a very popular field in deep learning.