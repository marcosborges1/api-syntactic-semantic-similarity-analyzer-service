import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dense, Lambda
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from .deep_learning import DeepLearningBased


class MyBILSTM(DeepLearningBased):
    def __init__(self, name):
        super().__init__(name)
        self.similarity_measures = ["bilstm"]

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
        # Sample texts
        # text1 = "The weather is sunny"
        # text2 = "It is a bright sunny day"

        # Hyperparameters
        max_len = 10  # Maximum length of the text
        embedding_dim = 50  # Dimension of the embedding layer
        lstm_units = 64  # Number of LSTM units in the Bi-LSTM layer

        # Tokenize and pad the sequences
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts([sentence1, sentence2])
        sequence1 = tokenizer.texts_to_sequences([sentence1])
        sequence2 = tokenizer.texts_to_sequences([sentence2])

        padded_sequence1 = pad_sequences(sequence1, maxlen=max_len)
        padded_sequence2 = pad_sequences(sequence2, maxlen=max_len)

        # Define the Bi-LSTM model
        input_layer = Input(shape=(max_len,))
        embedding_layer = Embedding(
            input_dim=len(tokenizer.word_index) + 1,
            output_dim=embedding_dim,
            input_length=max_len,
        )(input_layer)
        bi_lstm_layer = Bidirectional(LSTM(lstm_units, return_sequences=False))(
            embedding_layer
        )
        dense_layer = Dense(64, activation="relu")(bi_lstm_layer)
        # Use Lambda to apply a function that normalizes the last layer's output
        normalize_layer = Lambda(lambda x: K.l2_normalize(x, axis=1))(dense_layer)
        model = Model(inputs=input_layer, outputs=normalize_layer)

        # Get the feature vectors for both texts
        vector1 = model.predict(padded_sequence1)
        vector2 = model.predict(padded_sequence2)

        # Compute the cosine similarity between the feature vectors
        cosine_similarity = np.dot(vector1, vector2.T) / (
            np.linalg.norm(vector1) * np.linalg.norm(vector2)
        )
        similarity = cosine_similarity[0][0]
        print(f"Cosine Similarity: {similarity}")
        return similarity
