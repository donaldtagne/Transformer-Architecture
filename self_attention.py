import torch
import torch.nn.functional as F

# Beispiel-Eingabe: 3 Tokens mit je 4-dimensionalem Vektor
tokens = torch.tensor([
    [1.0, 0.0, 1.0, 0.0],
    [0.0, 2.0, 0.0, 2.0],
    [1.0, 1.0, 1.0, 1.0]
]).unsqueeze(0)  # Shape: (1, 3, 4)

# Gewichtsmatrizen für Q, K, V
W_q = torch.tensor([[0.5, 0.0, 0.5, 0.0],
                    [0.0, 0.5, 0.0, 0.5],
                    [0.5, 0.0, 0.5, 0.0],
                    [0.0, 0.5, 0.0, 0.5]])
W_k = W_q.clone()
W_v = torch.eye(4)

# Q, K, V berechnen
Q = tokens @ W_q
K = tokens @ W_k
V = tokens @ W_v

# Self-Attention Scores und Weights
d_k = Q.size(-1)
scores = torch.matmul(Q, K.transpose(-2, -1)) / d_k**0.5
weights = F.softmax(scores, dim=-1)
self_attn_output = torch.matmul(weights, V)

print("Self-Attention Scores:\n", scores.squeeze())
print("\nSelf-Attention Weights (Softmax):\n", weights.squeeze())
print("\nSelf-Attention Output:\n", self_attn_output.squeeze())

# Multi-Head Attention (2 Heads mit je 2D)
def split_heads(x, num_heads):
    B, T, D = x.shape
    head_dim = D // num_heads
    return x.view(B, T, num_heads, head_dim).transpose(1, 2)

Q_multi = split_heads(Q, 2)
K_multi = split_heads(K, 2)
V_multi = split_heads(V, 2)

scores_multi = torch.matmul(Q_multi, K_multi.transpose(-2, -1)) / (Q_multi.size(-1) ** 0.5)
weights_multi = F.softmax(scores_multi, dim=-1)
output_multi = torch.matmul(weights_multi, V_multi)

# Combine heads
multihead_output = output_multi.transpose(1, 2).reshape(1, 3, 4)

print("\nMulti-Head Attention Output:\n", multihead_output.squeeze())
