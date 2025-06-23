from sentence_transformers import SentenceTransformer, util

# Instantiate the sentence transformer model
model = SentenceTransformer('stsb-mpnet-base-v2')

# Our sentences we'd like to compare
sentence1 = "I have a new laptop."
sentence2 = "I got a new computer."

# Compute the embeddings for both sentences
embeddings1 = model.encode(sentence1, convert_to_tensor=True)
embeddings2 = model.encode(sentence2, convert_to_tensor=True)

# Compute cosine similarity between the two embeddings
cosine_similarity = util.pytorch_cos_sim(embeddings1, embeddings2)

# Normalize the score to be between 0 and 1
normalized_score = (cosine_similarity.item() + 1) / 2

print(f"Normalized similarity score between the sentences: {normalized_score}")