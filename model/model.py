import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import copy
from utils.data_utils import patchify


def clones(module, N):
    """ Generates N identical layers """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(q, k, v, dim):
    """
    Calculates self-attention
    :param q: query
    :param k: keys
    :param v: values
    :param dim: dimension d_k
    :return: attention
    """
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(dim)
    attn = F.softmax(scores, dim=-1)
    out = torch.matmul(attn, v)

    return out, attn


def get_positional_embeddings(sequence_length, d):
  """Calculate positional embedding"""

  pe = torch.ones(sequence_length, d)
  position = torch.arange(0, sequence_length).unsqueeze(1)
  div_term = torch.exp(torch.arange(0, d, 2) *
                       -(math.log(10000.0) / d))
  pe[:, 0::2] = torch.sin(position * div_term)
  pe[:, 1::2] = torch.cos(position * div_term)

  return pe


class MultiHeadAttention(nn.Module):

  def __init__(self, h, d_model):

    # Super constructor
    super(MultiHeadAttention, self).__init__()

    self.h = h
    self.d_k = d_model // h

    self.linears = clones(nn.Linear(d_model, d_model), 4)


  def forward(self, im):
    nbatches = im.size(0)

    # Linear projections to all batches
    q, k, v = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
              for l, x in zip(self.linears, (im, im, im))]

    # Apply attention to projected batches
    z, attn = attention(q, k, v, self.d_k)

    # Concatenate all
    z = self.linears[-1](z.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k))

    return z, attn


class EncoderBlock(nn.Module):

  def __init__(self, n_head, h_dim):

    # Super constructor
    super(EncoderBlock, self).__init__()

    self.h_dim = h_dim
    self.n_head = n_head

    self.ln = nn.LayerNorm(self.h_dim)
    self.mhsa = MultiHeadAttention(self.n_head, self.h_dim)

    self.mlp = nn.Sequential(nn.Linear(h_dim, h_dim*4),
                             nn.GELU(),
                             nn.Linear(h_dim*4, h_dim))

  def forward(self, x):
    out, attn = self.mhsa(self.ln(x))
    out = out + x 
    out = out + self.mlp(self.ln(out))

    return out, attn


class ViT(nn.Module):

  def __init__(self, n_classes=10, patch_size=4, hidden_dim=8, n_heads=2, n_blocks=2, chw=(1, 28, 28)):

    # Super constructor
    super(ViT, self).__init__()

    self.chw = chw # (C, H, W)
    self.patch_size = patch_size # patch height/width
    self.hidden_dim = hidden_dim
    self.n_classes = n_classes
    self.n_heads = n_heads
    self.n_blocks = n_blocks

    # Linear embedding
    self.input_dim = int(chw[0] * pow(self.patch_size, 2))
    self.linear_emb = nn.Linear(self.input_dim, self.hidden_dim)

    # Learnable classification token
    self.class_token = nn.Parameter(torch.randn(1, self.hidden_dim))

    # Positional embedding
    self.register_buffer('positional_embeddings', get_positional_embeddings(
        pow(int(self.chw[1] / self.patch_size), 2) + 1, self.hidden_dim), persistent=False)

    # Transformer encoder
    self.EncoderBlocks = nn.ModuleList([EncoderBlock(self.n_heads, self.hidden_dim) for _ in range(self.n_blocks)])

    # Classification
    self.classif = nn.Sequential(nn.Linear(self.hidden_dim, n_classes),
                                 nn.Softmax(dim=-1))


  def forward(self, images):

    n, c, h, w = images.shape

    # Patchify
    patches = patchify(images, self.patch_size).to(self.positional_embeddings.device)

    # Linear embedding
    tokens = self.linear_emb(patches)

    # Add classification token
    tokens = torch.stack([torch.vstack((self.class_token, tokens[i])) for i in range(len(tokens))])

    # Add positional embedding
    out = tokens + self.positional_embeddings.repeat(n, 1, 1)

    # Transformer encoding
    attention_weights = []
    for block in self.EncoderBlocks:
      out, weights = block(out)
      attention_weights.append(weights)

    # Classification
    out = out[:, 0]

    return self.classif(out), attention_weights