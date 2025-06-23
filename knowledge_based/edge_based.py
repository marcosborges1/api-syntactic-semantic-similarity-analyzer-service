from nltk.corpus import wordnet
from .knowledge_based import KnowledgeBased
from nltk.corpus import wordnet_ic

ic = wordnet_ic.ic("ic-brown.dat")
from sematch.semantic.similarity import WordNetSimilarity


class EdgeBased(KnowledgeBased):
    def __init__(self, name):
        super().__init__(name)

    def analyze(
        self, word_list, similarity_measures=["path", "lch", "wup", "hso", "li"]
    ):
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
            "path": EdgeBased.calculate_path_similarity,
            "lch": EdgeBased.calculate_lch_similarity,
            "wup": EdgeBased.calculate_wup_similarity,
            "hso": EdgeBased.calculate_hso_similarity,
            "lin": EdgeBased.calculate_lin_similarity,
            "li": EdgeBased.calculate_li_similarity,
        }

        if synset1 and synset2 and similarity_measure in similarity_methods:
            if similarity_measure == "li":
                similarity = similarity_methods[similarity_measure](word1, word2)
                return similarity
            else:
                similarity = similarity_methods[similarity_measure](
                    synset1[0], synset2[0]
                )
            return similarity

        return None

    @staticmethod
    def calculate_path_similarity(synset1, synset2):
        return synset1.path_similarity(synset2)

    @staticmethod
    def calculate_lch_similarity(synset1, synset2):
        return EdgeBased.normalized_value(synset1.lch_similarity(synset2), 0, 3.65)

    @staticmethod
    def calculate_wup_similarity(synset1, synset2):
        return synset1.wup_similarity(synset2)

    @staticmethod
    def calculate_hso_similarity(synset1, synset2):
        gloss1 = synset1.definition()
        gloss2 = synset2.definition()

        words1 = set(gloss1.lower().split())
        words2 = set(gloss2.lower().split())

        common_words = words1.intersection(words2)
        similarity = len(common_words) / (len(words1) + len(words2))
        return similarity

    @staticmethod
    def calculate_lin_similarity(synset1, synset2):
        return synset1.lin_similarity(synset2, ic)

    @staticmethod
    def calculate_li_similarity(word1, word2):
        wns = WordNetSimilarity()
        return wns.word_similarity(word1, word2, "li")

    @staticmethod
    def normalized_value(value, min_value, max_value):
        return (value - min_value) / (max_value - min_value)


def analyze_and_display(class_name, class_description, word_list, algortihms=None):
    string_based = class_name(class_description)
    if algortihms is None:
        sim = string_based.analyze(word_list)
    else:
        sim = string_based.analyze(word_list, algortihms)
    string_based.display_html(sim)


# def analyze(
#     self,
#     word1,
#     word2,
#     similarity_measures=["path", "lch", "wup", "hso", "lin", "li"],
# ):
#     for measure in similarity_measures:
#         similarity = self.calculate_similarity(word1, word2, measure)
#         if similarity is not None:
#             print(
#                 f"Performing {self.name} analysis. Words: ({word1} {word2}), Similarity ({measure}): {similarity}"
#             )
#         else:
#             print(f"Unable to calculate similarity for measure: {measure}")
