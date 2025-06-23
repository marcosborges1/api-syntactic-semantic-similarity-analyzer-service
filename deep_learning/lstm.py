from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from .deep_learning import DeepLearningBased


class MyLSTM(DeepLearningBased):
    def __init__(self, name):
        super().__init__(name)
        self.similarity_measures = ["lstm"]

    def analyze(self, sentence_list):
        results = []
        for sentence in sentence_list:
            row = [sentence[0], sentence[1]]
            similarity = self.calculate_similarity(sentence[0], sentence[1])
            if similarity is not None:
                similarity = round(similarity, 3)  # Round to three decimal places
                row.append(similarity)
            else:
                print(
                    f"Unable to calculate similarity for measure: {self.similarity_measures[0]}"
                )
            results.append(row)
        self.results = results
        self.generate_dataframe()

    def calculate_similarity(self, sentence1, sentence2):
        # text1 = "The capital of France is Paris."
        # text2 = "Paris is the capital of France."

        # Tokenize and pad sequences
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts([sentence1, sentence2])
        seq1 = tokenizer.texts_to_sequences([sentence1])
        seq2 = tokenizer.texts_to_sequences([sentence2])

        # Assume max length of text is 10
        seq1 = pad_sequences(seq1, maxlen=10)
        seq2 = pad_sequences(seq2, maxlen=10)

        # Define LSTM model
        input_text = Input(shape=(None,))
        embedding_size = 50
        max_vocab = len(tokenizer.word_index) + 1

        embedding_layer = Embedding(max_vocab, embedding_size)
        lstm_layer = LSTM(256, return_sequences=False)

        # Encoded the input sequence
        embedded_text = embedding_layer(input_text)
        encoded_text = lstm_layer(embedded_text)

        # Build the model
        model = Model(inputs=input_text, outputs=encoded_text)

        # Get the LSTM encoding for both texts
        encoded_seq1 = model.predict(seq1)
        encoded_seq2 = model.predict(seq2)

        # Compute similarity between encoded sequences
        similarity = cosine_similarity(encoded_seq1, encoded_seq2)
        # print(
        #     f"The cosine similarity between the text representations is: {similarity[0][0]}"
        # )
        return similarity[0][0]
