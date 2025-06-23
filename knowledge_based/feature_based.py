from nltk.corpus import wordnet
from .knowledge_based import KnowledgeBased
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class FeatureBased(KnowledgeBased):
    def __init__(self, name):
        super().__init__(name)

    def analyze(self, word_list, similarity_measures=["tversky", "lesk"]):
        self.similarity_measures = similarity_measures
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

    @staticmethod
    def calculate_similarity(word1, word2, similarity_measure):
        synset1 = wordnet.synsets(word1)
        synset2 = wordnet.synsets(word2)

        similarity_methods = {
            "tversky": FeatureBased.calculate_tversky_similarity,
            "lesk": FeatureBased.calculate_lesk_similarity,
        }

        if synset1 and synset2 and similarity_measure in similarity_methods:
            similarity = similarity_methods[similarity_measure](synset1[0], synset2[0])
            return similarity

        return None

    @staticmethod
    def calculate_tversky_similarity(synset1, synset2):
        from textacy.similarity import tokens
        tokens1 = synset1.definition().lower().split()
        tokens2 = synset2.definition().lower().split()

        tversky = tokens.tversky(tokens1, tokens2)
        return tversky

    @staticmethod
    def calculate_lesk_similarity(synset1, synset2):
        # Convert synsets to text
        text1 = synset1.definition()
        text2 = synset2.definition()
        
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

#  @staticmethod
#     def calculate_lesk_similarity(synset1, synset2):
#         tokens1 = synset1.definition().lower().split()
#         tokens2 = synset2.definition().lower().split()

#         common_tokens = set(tokens1).intersection(set(tokens2))
#         overlap = len(common_tokens)

#         return overlap
 # @staticmethod
    # def get_semantic(seq, key_word):
    #     from pywsd.lesk import adapted_lesk
    #     best_sense = adapted_lesk(seq, key_word)

    #     # Print the disambiguated sense
    #     if best_sense:
    #         print(f"Best sense for '{key_word}': {best_sense.definition()}")
    #     else:
    #         print(f"No sense found for '{key_word}'")