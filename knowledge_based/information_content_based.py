from .knowledge_based import KnowledgeBased
from nltk.corpus import wordnet
from nltk.corpus import wordnet_ic

ic = wordnet_ic.ic("ic-brown.dat")


class InformationContentBased(KnowledgeBased):
    def __init__(self, name):
        super().__init__(name)
        # self.wordnet_ic = wordnet_ic.ic('ic-brown.dat')

    def analyze(self, word_list, similarity_measures=["resnik", "jcn", "lin"]):
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

    # def analyze(self, word1, word2, similarity_measures=["resnik","jcn","lin"]):
    #     for measure in similarity_measures:
    #         similarity = self.calculate_similarity(word1, word2, measure)
    #         if similarity is not None:
    #             print(f"Performing {self.name} analysis. Words: ({word1}, {word2}), Similarity ({measure}): {similarity}")
    #         else:
    #             print(f"Unable to calculate similarity for measure: {measure}")

    def calculate_similarity(self, word1, word2, similarity_measure):
        synset1 = wordnet.synsets(word1)
        synset2 = wordnet.synsets(word2)

        similarity_methods = {
            "resnik": InformationContentBased.calculate_resnik_similarity,
            "jcn": InformationContentBased.calculate_jcn_similarity,
            "lin": InformationContentBased.calculate_lin_similarity,
        }

        if synset1 and synset2 and similarity_measure in similarity_methods:
            similarity = similarity_methods[similarity_measure](synset1[0], synset2[0])
            return similarity

        return None

    @staticmethod
    def calculate_resnik_similarity(synset1, synset2):
        return synset1.res_similarity(synset2, ic)

    @staticmethod
    def calculate_jcn_similarity(synset1, synset2):
        return synset1.jcn_similarity(synset2, ic)

    @staticmethod
    def calculate_lin_similarity(synset1, synset2):
        return synset1.lin_similarity(synset2, ic)
