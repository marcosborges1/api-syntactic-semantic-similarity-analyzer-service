from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class MyBert:
    def __init__(self, name, model_name):
        self.name = name
        self.model_name = model_name

        # Load BERT model
        self.model = SentenceTransformer(model_name)

    def get_similarity(self, sentence1, sentence2):
        # Encode sentences into embeddings
        embeddings = self.model.encode([sentence1, sentence2])

        # Calculate cosine similarity between sentence embeddings
        similarity_score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

        # print(f"Cosine Similarity ({self.name}):", similarity_score)
        return similarity_score


# bert = MyBert(name="MyBERT", model_name="bert-base-uncased")

# # Calculate similarity between two sentences
# sentence1 = "A distinct numerical value assigned to each customer, facilitating precise identification and management of client"
# sentence2 = "A system-generated unique customer identifier on the backend."
# similarity_score = bert.get_similarity(sentence1, sentence2)
# print(
#     f"Similarity score between '{sentence1}' and '{sentence2}' using {bert.name}: {similarity_score}"
# )
