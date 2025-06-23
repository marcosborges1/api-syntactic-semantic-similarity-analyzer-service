import numpy as np
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Create a list of sentences
sentences = ["I am a doctor", "He is work at the hospital"]

# Tokenize the sentences and build the vocabulary
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
vocab_size = len(word_index) + 1

# Convert sentences to sequences of indices
sequences = tokenizer.texts_to_sequences(sentences)

# Pad sequences to have the same length
max_length = max(len(seq) for seq in sequences)
padded_sequences = pad_sequences(sequences, maxlen=max_length)

# Set the embedding dimension
embedding_dim = 100

# Create the CNN model
model = Sequential()

# Add embedding layer
model.add(
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length)
)

# Add convolutional layer
model.add(Conv1D(filters=128, kernel_size=5, activation="relu"))

# Add global max pooling layer
model.add(GlobalMaxPooling1D())

# Add fully connected layers
model.add(Dense(64, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

# Compile the model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Generate some dummy labels
labels = np.array([1, 0])  # 1 for the first sentence, 0 for the second sentence

# Train the model
model.fit(padded_sequences, labels, epochs=10, batch_size=32)

# Predict the scores for the sentences
scores = model.predict(padded_sequences)
for i in range(len(sentences)):
    print(f"Sentence: {sentences[i]}")
    print(f"Score: {scores[i]}")
    print()
