from .deep_learning import DeepLearningBased
from sentence_transformers import SentenceTransformer, util


class MyTransformers(DeepLearningBased):
    def __init__(self, name):
        super().__init__(name)

    def analyze(self, sentence_list, similarity_measures=["roberta", "mpnet"]):
        self.similarity_measures = similarity_measures
        model = None

        for m in self.similarity_measures:
            if m == "roberta":
                model = SentenceTransformer("roberta-base-nli-stsb-mean-tokens")
            elif m == "mpnet":
                model = SentenceTransformer("stsb-mpnet-base-v2")

        results = []
        for sentence in sentence_list:
            row = [sentence[0], sentence[1]]
            for measure in self.similarity_measures:
                similarity = self.calculate_similarity(
                    sentence[0], sentence[1], measure, model
                )
                if similarity is not None:
                    similarity = round(similarity, 3)  # Round to three decimal places
                    row.append(similarity)
                else:
                    print(f"Unable to calculate similarity for measure: {measure}")
            results.append(row)

        self.results = results
        self.generate_dataframe()

    @staticmethod
    def calculate_similarity(sentence1, sentence2, similarity_measure, model):
        similarity_methods = {
            "roberta": MyTransformers.calculate_roberta_similarity,
            "mpnet": MyTransformers.calculate_mpnet_similarity,
        }

        if similarity_measure in similarity_methods:
            similarity = similarity_methods[similarity_measure](
                sentence1, sentence2, model
            )
            return similarity

        return None

    @staticmethod
    def calculate_roberta_similarity(sentence1, sentence2, model):
        # Load the RoBERTa-based model
        # model = SentenceTransformer("roberta-base-nli-stsb-mean-tokens")

        embeddings1 = model.encode(sentence1, convert_to_tensor=True)
        embeddings2 = model.encode(sentence2, convert_to_tensor=True)

        # Compute the cosine similarity between the embeddings
        cosine_similarity = util.pytorch_cos_sim(embeddings1, embeddings2)

        # Normalize the score to be between 0 and 1
        normalized_score = (cosine_similarity.item() + 1) / 2
        # print(
        #     f"The normalized similarity score between the sentences is: {normalized_score:.3f}"
        # )
        return normalized_score

    @staticmethod
    def calculate_mpnet_similarity(sentence1, sentence2, model):
        # Load the RoBERTa-based model
        # model = SentenceTransformer("stsb-mpnet-base-v2")

        embeddings1 = model.encode(sentence1, convert_to_tensor=True)
        embeddings2 = model.encode(sentence2, convert_to_tensor=True)

        # Compute the cosine similarity between the embeddings
        cosine_similarity = util.pytorch_cos_sim(embeddings1, embeddings2)

        # Normalize the score to be between 0 and 1
        normalized_score = (cosine_similarity.item() + 1) / 2
        # print(
        #     f"The normalized similarity score between the sentences is: {normalized_score:.3f}"
        # )
        return normalized_score
