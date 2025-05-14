import numpy as np

def softmax(x_array):
    """Compute softmax values for each set of scores in x_array."""
    # Subtract max for numerical stability to prevent overflow during exponentiation
    e_x = np.exp(x_array - np.max(x_array, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

# Set random seed for reproducibility
np.random.seed(42)

# Define the dimension of the key vectors
d_k = 4

# Generate sample query, key, and value vectors
# Query vector for one word (shape: (1, d_k))
q_vec = np.random.randn(1, d_k)
# Key vectors for three context words (shape: (3, d_k))
k_mat = np.random.randn(3, d_k)
# Value vectors for three context words (shape: (3, d_k))
v_mat = np.random.randn(3, d_k)

# Print the inputs, rounded to 4 decimal places for clarity
print("Query vector (q_vec):\n", np.round(q_vec, 4))
print("Key matrix (k_mat):\n", np.round(k_mat, 4))
print("Value matrix (v_mat):\n", np.round(v_mat, 4))
print("Dimension of keys (d_k):", d_k)

# Step 1: Calculate raw attention scores (dot product of query and keys)
# q_vec (1, 4) @ k_mat.T (4, 3) results in (1, 3) - one score per key
attention_scores = np.matmul(q_vec, k_mat.T)
print("Raw Attention Scores (q_vec @ k_mat.T):\n", np.round(attention_scores, 4))

# Step 2: Scale the attention scores by sqrt(d_k) to prevent large magnitudes
scaled_attention_scores = attention_scores / np.sqrt(d_k)
print("Scaled Attention Scores:\n", np.round(scaled_attention_scores, 4))

# Step 3: Apply softmax to get attention weights (probabilities summing to 1)
attention_weights = softmax(scaled_attention_scores)
print("Attention Weights (after Softmax):\n", np.round(attention_weights, 4))

# Step 4: Compute the context vector as the weighted sum of value vectors
# attention_weights (1, 3) @ v_mat (3, 4) results in (1, 4)
context_vector = np.matmul(attention_weights, v_mat)
print("Final Context Vector (Weighted sum of V):\n", np.round(context_vector, 4))
