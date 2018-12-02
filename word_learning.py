import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.models import Sequential

from word_preprocess import WordPreprocessor


def create_model(predictors, label, max_sequence_len, total_words):
    input_len = max_sequence_len - 1

    model = Sequential()
    model.add(Embedding(total_words, 10, input_length=input_len))
    model.add(LSTM(150))
    model.add(Dropout(0.1))
    model.add(Dense(total_words, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.fit(predictors, label, epochs=100, verbose=1)

    return model

def generate_text(seed_text, next_words, max_sequence_len, model, tokenizer):
    for j in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = model.predict_classes(token_list, verbose=0)
  
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    
    return seed_text

processor = WordPreprocessor()
(X_train, Y_train), (_,_), (_,_) = processor.preprocess_file('data/askreddit.json')
max_sequence_len, total_words = processor.max_sequence_len, processor.total_words_count
model = create_model(X_train, Y_train, max_sequence_len, total_words)
print(generate_text("Honestly, the camera has always blown my mind. It creates a lasting picture of something that happens in the real world, and",
                    30, max_sequence_len, model, processor.tokenizer))
