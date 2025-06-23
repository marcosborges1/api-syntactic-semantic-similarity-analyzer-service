import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class Glove:
    def __init__(self, path_to_vectors):
        self.word_vectors = self.load_word_vectors(path_to_vectors)

    def load_word_vectors(self, path):
        word_vectors = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.array(values[1:], dtype="float32")
                word_vectors[word] = vector
        return word_vectors

    def get_similarity(self, word1, word2):
        if word1 in self.word_vectors and word2 in self.word_vectors:
            word_vector1 = self.word_vectors[word1]
            word_vector2 = self.word_vectors[word2]
            word_vector1 = word_vector1.reshape(1, -1)
            word_vector2 = word_vector2.reshape(1, -1)
            similarity = cosine_similarity(word_vector1, word_vector2)[0, 0]
            return similarity
        else:
            return None


# my_word2vec = Glove("./data/glove.6B.100d.txt")
# similarity_score = my_word2vec.get_similarity("car", "automobile")
# print("Cosine Similarity:", similarity_score)
