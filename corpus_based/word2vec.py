from gensim.models import KeyedVectors


class Word2Vec:
    def __init__(self, name, model_path):
        self.name = name
        self.model = self.load_model(model_path)

    def load_model(self, model_path):
        model = KeyedVectors.load_word2vec_format(model_path, binary=False)
        return model

    def get_similarity(self, word1, word2):
        if word1 in self.model.key_to_index and word2 in self.model.key_to_index:
            similarity = self.model.similarity(word1, word2)
            # print(self.model.most_similar("teacher"))
            return similarity
        else:
            return None


# my_word2vec = Word2Vec('word2vec','./data/pruned.word2vec.txt')
# similarity_score = my_word2vec.get_similarity('cat', 'dog')
# print("Cosine Similarity:", similarity_score)
