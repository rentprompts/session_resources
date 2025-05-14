import numpy as np
import matplotlib.pyplot as plt

def get_positional_encoding(seq_len, d_model):
    """
    Generate sinusoidal positional encodings for a sequence.

    Args:
        seq_len (int): Number of positions in the sequence.
        d_model (int): Dimension of the embedding vectors (must be even).

    Returns:
        np.ndarray: Matrix of shape (seq_len, d_model) with positional encodings.
    """
    if d_model % 2 != 0:
        raise ValueError("d_model must be an even number.")
    
    # Initialize the encoding matrix
    positional_encoding = np.zeros((seq_len, d_model))
    # Position indices as a column vector
    position = np.arange(seq_len)[:, np.newaxis]
    # Divisor terms for frequency scaling
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    
    # Apply sine to even indices, cosine to odd indices
    positional_encoding[:, 0::2] = np.sin(position * div_term)
    positional_encoding[:, 1::2] = np.cos(position * div_term)
    
    return positional_encoding

# Parameters: 50 positions, 128 dimensions
seq_len = 50
d_model = 128
pos_encodings = get_positional_encoding(seq_len, d_model)
print(f"Shape of positional encoding matrix: {pos_encodings.shape}")

# Create the visualization
plt.figure(figsize=(12, 6))
plt.pcolormesh(pos_encodings, cmap='viridis', shading='auto')
plt.xlabel('Embedding Dimension Index (i)')
plt.xlim(0, d_model)
plt.ylabel('Position in Sequence (pos)')
plt.ylim(seq_len, 0)  # Position 0 at the top
plt.colorbar(label="Encoding Value")
plt.title(f"Sinusoidal Positional Encoding (seq_len={seq_len}, d_model={d_model})")
plt.show()
