import torch
from transformers import GPT2Model, GPT2Tokenizer
from scipy.spatial.distance import cosine

# Load pre-trained model tokenizer (vocabulary)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Load pre-trained model (weights)
model = GPT2Model.from_pretrained("gpt2")
model.eval()

# Encode text inputs
text_1 = "I love read books"
text_2 = "i hate play soccer"

# Tokenize texts
indexed_tokens_1 = tokenizer.encode(text_1, add_special_tokens=True)
indexed_tokens_2 = tokenizer.encode(text_2, add_special_tokens=True)

# Convert indexed tokens to a PyTorch tensor
tokens_tensor_1 = torch.tensor([indexed_tokens_1])
tokens_tensor_2 = torch.tensor([indexed_tokens_2])

# Get hidden states
with torch.no_grad():
    outputs_1 = model(tokens_tensor_1)
    outputs_2 = model(tokens_tensor_2)

    # Only take the output from the last layer
    last_hidden_states_1 = outputs_1.last_hidden_state
    last_hidden_states_2 = outputs_2.last_hidden_state

# Calculate the average of all token embeddings for the sentence
sentence_embedding_1 = torch.mean(last_hidden_states_1, dim=1).squeeze()
sentence_embedding_2 = torch.mean(last_hidden_states_2, dim=1).squeeze()

# Compute cosine similarity between the sentence embeddings
similarity = 1 - cosine(sentence_embedding_1.numpy(), sentence_embedding_2.numpy())

print(f"The similarity score between the sentences is: {similarity:.3f}")
