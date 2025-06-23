from corpus_based import CorpusBased
from .word2vec import Word2Vec
from .glove import Glove
from .myfasttext import MyFastText
from .mybert import MyBert
import os


class WordEmbedding(CorpusBased):
    def __init__(self, name):
        super().__init__(name)
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.similarity_measures = [
            "word2vec",
            "glove",
            "fasttext",
            "bert",
        ]

    # def analyze(self, word_list):
    #     for measure in self.similarity_measures:
    #         self.calculate_similarity(word_list, measure)

    def analyze(self, word_list):
        results = []
        for words in word_list:
            row = [words[0], words[1]]
            for measure in self.similarity_measures:
                similarity = self.calculate_similarity(words[0], words[1], measure)
                if similarity is not None:
                    similarity = round(similarity, 3)  # Round to three decimal places
                    row.append(similarity)
                else:
                    print(f"Unable to calculate similarity for measure: {measure}")
            results.append(row)
        self.results = results
        self.generate_dataframe()

    def calculate_similarity(self, word1, word2, similarity_measure):
        similarity_methods = {
            "word2vec": self.calculate_word2vec_similarity,
            "glove": self.calculate_glove_similarity,
            "fasttext": self.calculate_fasttext_similarity,
            "bert": self.calculate_bert_similarity,
        }

        if similarity_measure in similarity_methods:
            similarity = similarity_methods[similarity_measure](word1, word2)
            return similarity

        return None

    # def calculate_similarity(self, word_list, similarity_measure):
    #     similarity_methods = {
    #         "word2vec": self.calculate_word2vec_similarity,
    #         # "glove": self.calculate_glove_similarity,
    #         # "fasttext": self.fasttext,
    #         # "bert": self.bert,
    #     }

    #     similarity_method = similarity_methods.get(similarity_measure)
    #     if similarity_method:
    #         for words in word_list:
    #             word1 = words[0]
    #             word2 = words[1]
    #             similarity_method(word1, word2)
    #     else:
    #         print("Invalid similarity measure.")

    def calculate_word2vec_similarity(self, word1, word2):
        my_word2vec = Word2Vec("word2vec", f"{self.base_dir}/data/pruned.word2vec.txt")
        similarity_score = my_word2vec.get_similarity(word1, word2)
        # print("Cosine Similarity (Word2Vec):", similarity_score)
        return similarity_score

    def calculate_glove_similarity(self, word1, word2):
        my_glove = Glove(f"{self.base_dir}/data/glove.6B.100d.txt")
        similarity_score = my_glove.get_similarity(word1, word2)
        # print("Cosine Similarity (Glove):", similarity_score)
        return similarity_score

    def calculate_fasttext_similarity(self, word1, word2):
        # Perform FastText similarity calculation
        # Return score from 0 to 1
        my_word2vec = MyFastText("fast", f"{self.base_dir}/data/cc.en.300.bin")
        similarity_score = my_word2vec.get_similarity(word1, word2)
        return similarity_score

    def calculate_bert_similarity(self, word1, word2):
        # Perform BERT similarity calculation
        # Return score from 0 to 1
        my_word2vec = MyBert(name="MyBERT", model_name="bert-base-uncased")
        similarity_score = my_word2vec.get_similarity(word1, word2)
        return similarity_score

    def split_string_into_words(self, string_list):
        """
        Split each string in a list into a list of words.

        Args:
            string_list (list): The input list of strings.

        Returns:
            list: A list of lists, where each inner list contains words from the corresponding input string.
        """
        word_list = [string.split() for string in string_list]
        return word_list


# word_embedding = WordEmbedding("My Word Embedding")
# word_list = [
#     ["car", "cat"],
#     ["car", "automobile"],
#     ["apple", "orange"],
#     ["id", "badge"],
# ]
# results = word_embedding.analyze(word_list)
# # print(results)
# word_embedding.to_string(results)

# from .corpus_based import CorpusBased
# from .myword2vec import MyWord2Vec
# from .myglove import MyGlove
# import os


# class WordEmbedding(CorpusBased):
#     def __init__(self, name):
#         super().__init__(name)
#         self.base_dir = os.path.dirname(os.path.abspath(__file__))

#     def analyze(
#         self,
#         word1,
#         word2,
#         similarity_measures=["word2vec", "glove", "fasttext", "bert"],
#     ):
#         for measure in similarity_measures:
#             self.calculate_similarity(word1, word2, measure)

#     def calculate_similarity(self, word1, word2, similarity_measure):
#         similarity_methods = {
#             "word2vec": self.calculate_word2vec_similarity,
#             "glove": self.calculate_glove_similarity,
#             "fasttext": self.fasttext,
#             "bert": self.bert,
#         }

#         similarity_method = similarity_methods.get(similarity_measure)
#         if similarity_method:
#             similarity_method(word1, word2)
#         else:
#             print("Invalid similarity measure.")

#     def calculate_word2vec_similarity(self, word1, word2):
#         my_word2vec = MyWord2Vec(
#             "word2vec", f"{self.base_dir}/data/pruned.word2vec.txt"
#         )
#         similarity_score = my_word2vec.get_similarity(word1, word2)
#         print("Cosine Similarity (Word2Vec):", similarity_score)
#         return similarity_score

#     def calculate_glove_similarity(self, word1, word2):
#         my_glove = MyGlove(f"{self.base_dir}/data/glove.6B.100d.txt")
#         similarity_score = my_glove.get_similarity(word1, word2)
#         print("Cosine Similarity (Glove):", similarity_score)
#         return similarity_score

#     def fasttext(self, word1, word2):
#         # Perform FastText similarity calculation
#         # Return score from 0 to 1
#         pass

#     def bert(self, word1, word2):
#         # Perform BERT similarity calculation
#         # Return score from 0 to 1
#         pass

#     def split_string_into_words(self, string_list):
#         """
#         Split each string in a list into a list of words.

#         Args:
#             string_list (list): The input list of strings.

#         Returns:
#             list: A list of lists, where each inner list contains words from the corresponding input string.
#         """
#         word_list = [string.split() for string in string_list]
#         return word_list


# # from .corpus_based import CorpusBased
# # from .myword2vec import MyWord2Vec
# # from .myglove import MyGlove
# # import os


# # class WordEmbedding(CorpusBased):
# #     def __init__(self, name):
# #         super().__init__(name)
# #         self.base_dir = os.path.dirname(os.path.abspath(__file__))

# #     def analyze(self):
# #         # Perform analysis specific to WordEmbedding
# #         pass

# #     @staticmethod
# #     def calculate_word2vec_similarity(self, word1, word2):
# #         my_word2vec = MyWord2Vec("word2vec",f"{self.base_dir}/data/pruned.word2vec.txt")
# #         similarity_score = my_word2vec.get_similarity(word1, word2)
# #         print("Cosine Similarity:", similarity_score)
# #         return similarity_score

# #     def calculate_glove_similarity(self, word1, word2):
# #         my_word2vec = MyGlove(f"{self.base_dir}/data/glove.6B.100d.txt")
# #         similarity_score = my_word2vec.get_similarity(word1, word2)
# #         print("Cosine Similarity:", similarity_score)
# #         return similarity_score

# #     def fasttext(self, word1, word2):
# #         # Perform FastText similarity calculation
# #         # Return score from 0 to 1
# #         pass

# #     def bert(self, word1, word2):
# #         # Perform BERT similarity calculation
# #         # Return score from 0 to 1
# #         pass
# #     def split_string_into_words(self, string_list):
# #         """
# #         Split each string in a list into a list of words.

# #         Args:
# #             string_list (list): The input list of strings.

# #         Returns:
# #             list: A list of lists, where each inner list contains words from the corresponding input string.
# #         """
# #         word_list = [string.split() for string in string_list]
# #         return word_list
