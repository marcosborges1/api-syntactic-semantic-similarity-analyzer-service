from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class LSA:
    def __init__(self, sentences):
        self.sentences = sentences
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = None
        self.similarity_matrix = None

    def fit_transform(self):
        self.tfidf_matrix = self.vectorizer.fit_transform(self.sentences)
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix)

    def calculate_similarity(self):
        for i in range(len(self.sentences)):
            for j in range(i + 1, len(self.sentences)):
                similarity_score = self.similarity_matrix[i, j]
                print(
                    f"Similarity score between sentence {i+1} and sentence {j+1}: {similarity_score}"
                )

                # # Sample sentences


# sentences = [
#     "I like to watch movies",
#     "I enjoy playing video games",
#     "Football is my favorite sport",
#     "I love reading books",
#     "Traveling is an enriching experience"
# ]

# lsa = LSA(sentences)
# lsa.fit_transform()
# lsa.calculate_similarity()
