# import spacy
# from gensim.models import Word2Vec

# class ___asdMyWord2Vec():
#     def __init__(self,name):
#         self.name = name
#         # Load the en_core_web_md model from spaCy
#         self.nlp = spacy.load('en_core_web_md')
#         self.model = None

#     def train(self, corpus):
#         """
#         Train a Word2Vec model on a given corpus.

#         Args:
#             corpus (list): The input corpus consisting of sentences.
#         """
#         # Tokenize and lemmatize the corpus using spaCy
#         corpus_tokenized = [[token.lemma_ for token in self.nlp(sentence)] for sentence in corpus]

#         # Train the Word2Vec model
#         self.model = Word2Vec(corpus_tokenized, min_count=1)

#     def get_similarity(self, word1, word2):
#         """
#         Get the similarity score between two words.

#         Args:
#             word1 (str): The first word.
#             word2 (str): The second word.

#         Returns:
#             float: The similarity score between the two words.
#         """
#         similarity = self.model.wv.similarity(word1, word2)
#         return similarity

# # from gensim.models import KeyedVectors

# # # Load pre-trained Word2Vec model
# # model = KeyedVectors.load_word2vec_format('./data/pruned.word2vec.txt', binary=False)

# # # Example words
# # word1 = 'car'
# # word2 = 'automobile'

# # # Check if both words are present in the vocabulary
# # if word1 in model.key_to_index and word2 in model.key_to_index:
# #     # Calculate the similarity between the two words
# #     similarity = model.similarity(word1, word2)
# #     print("Cosine Similarity:", similarity)
# # else:
# #     print("One or both words are not in the vocabulary.")