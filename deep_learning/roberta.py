from sentence_transformers import SentenceTransformer, util

# Load the RoBERTa-based model
model = SentenceTransformer('roberta-base-nli-stsb-mean-tokens')

# Sentences to compare
sentence1 = "car"
sentence2 = "automobile"

# Compute the embeddings for both sentences
embeddings1 = model.encode(sentence1, convert_to_tensor=True)
embeddings2 = model.encode(sentence2, convert_to_tensor=True)

# Compute the cosine similarity between the embeddings
cosine_similarity = util.pytorch_cos_sim(embeddings1, embeddings2)

# Normalize the score to be between 0 and 1
normalized_score = (cosine_similarity.item() + 1) / 2

print(f"The normalized similarity score between the sentences is: {normalized_score:.3f}")
