from gensim.models import Word2Vec
from keras.preprocessing.text import text_to_word_sequence

import json

with open('data/askreddit.json') as f:
    data = json.load(f)

comment_bodies = []
for comment in data:
    comment_bodies.append(comment["body"])

def dataset_preparation(data):
    corpus = []
    for single_comment in data:
        for comment_part in single_comment.lower().split("\n"):
            for sentence_part in comment_part.split("."):
                corpus.append(text_to_word_sequence(sentence_part))

    return corpus

sentences = dataset_preparation(comment_bodies)

model = Word2Vec(sentences, min_count=2, sg=1)
model.train(sentences, total_examples=len(sentences), epochs=30)
print(model)

words = list(model.wv.vocab)
print(words)

print(model['movie'])

text_seed = ['when', 'the', 'dark', 'scary']
for i in range(0, 10):
    result = model.most_similar(positive=text_seed[-4:], topn=1)
    text_seed.append(result[0][0])

print(f'END: {(" ").join(text_seed)}')

# model.save('model.bin')
# new_model = Word2Vec.load('model.bin')
# print(new_model)
