import os
import fasttext


class MyFastText:
    def __init__(self, name, model_path):
        self.name = name
        self.model_path = model_path

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"FastText model file '{self.model_path}' not found."
            )

        self.model = fasttext.load_model(self.model_path)

    def get_similarity(self, word1, word2):
        # Preprocess the words (if needed) and get their similarity score
        similarity_score = self.model.get_sentence_vector(word1).dot(
            self.model.get_sentence_vector(word2)
        )

        # print("Cosine Similarity (FastText):", similarity_score)
        return similarity_score

    def analyze(self, word_list):
        similarities = []
        for sublist in word_list:
            word1 = sublist[0]
            word2 = sublist[1]
            similarity_score = self.get_similarity(word1, word2)
            similarities.append(similarity_score)
        return similarities


# import pandas as pd

# df = pd.read_csv("../data.csv")

# my_word2vec = MyFastText("fast", "./data/cc.en.300.bin")
# similarity_score = my_word2vec.get_similarity(
#     "car zadfaadf, a  asdfasdf", "vehicle adfadf  adsfadsfasd"
# )
# print("Cosine Similarity:", similarity_score)

# word_list = [
#     ["car", "cat"],
#     ["car", "automobile"],
#     ["apple", "orange"],
#     ["id", "badge"],
# ]
# print(word_list)
# print(df[["OA Out Attr description", "TA In Attr description"]].values.tolist())
# similarities = my_word2vec.analyze(
#     df[["OA Out Attr description", "TA In Attr description"]].values.tolist()
# )
# print(similarities)
