import math
from typing import List
import numpy as np

import torch.nn as nn

import torch
from torch.nn import LayerNorm, Linear, GELU, Dropout
import torch.nn.functional as F

from models.ssl.transformer.embeddings import NEmbedding, CEmbedding
from models.ssl.encoder import BaseEncoder

class FTTransformerEncoder(BaseEncoder):
    def __init__(
        self,
        categorical_col_indices: List[int],
        categories_unique_values: List[int],
        numeric_col_indices: List[int],
        embedding_dim: int = 32,
        depth: int = 4,
        heads: int = 4,
        attn_dropout: float = 0.1,
        ff_dropout: float = 0.1,
        explainable=False,
        add_cls_token: bool = False
    ):
        """Encoder of transformer. Consists of multiple transformer blocks in a row.

        Parameters
        ----------
        categorical_col_indices : List[int]
            column indices of categorical columns in data set
        categories_unique_values : List[int]
            number of unique values for each categorical column.
            Must have same order as categorical_col_indices
        numeric_col_indices : List[int]
            column indices of numerical columns in data set
        embedding_dim : int, optional
            embedding dimension for features, by default 32
        depth : int, optional
            number of transformer blocks, by default 4
        heads : int, optional
            number of attention heads, by default 4
        attn_dropout : float, optional
            dropout rate in attention head, by default 0.1
        ff_dropout : float, optional
            dropout rate in mlp, by default 0.1
        explainable : bool, optional
            flag to output importance inferred from attention weights, by default False
        add_cls_token : bool, optional
            whether to add cls token in embedding for classification, by default False
        """
        self.output_dim = (len(categorical_col_indices) + len(numeric_col_indices)) * embedding_dim
        super(FTTransformerEncoder, self).__init__(output_dim=self.output_dim)
        self.embedding_dim = embedding_dim
        self.explainable = explainable
        self.depth = depth
        self.heads = heads
        self.add_cls_token = add_cls_token

        # columns identifier
        self.numeric_col_indices = numeric_col_indices
        self.categorical_col_indices = categorical_col_indices
        self.categories_unique_values = categories_unique_values

        # Two main embedding modules
        if len(self.numeric_col_indices) > 0:
            self.numerical_embeddings = NEmbedding(
                d_numerical=len(self.numeric_col_indices),
                embedding_dim=embedding_dim,
                add_cls_token=self.add_cls_token
            )
        if len(self.categorical_col_indices) > 0:
            self.categorical_embeddings = CEmbedding(
                categories_unique_values=categories_unique_values,
                embedding_dim =embedding_dim
            )

        # Transformers
        self.transformers = []
        for i in range(depth):
            self.transformers.append(
                TransformerBlock(
                    embed_dim=embedding_dim,
                    num_heads=heads,
                    transformer_emb_dim=embedding_dim,
                    att_dropout=attn_dropout,
                    ff_dropout=ff_dropout,
                    explainable=self.explainable,
                    layer_norm=(i != 0)
                )
            )

        # very important! Without this, model.to(device) does not affect the transformer blocks
        self.flatten_transformers = nn.Sequential(*self.transformers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:

        transformer_inputs = []

        # If numerical features
        if len(self.numeric_col_indices) > 0:
            # combine all numerical features
            num_input = []
            for index in self.numeric_col_indices:
                num_input.append(inputs[:, index])
            num_input = torch.stack(num_input, axis=1)

            # embed
            num_embs = self.numerical_embeddings(num_input)

            # add to transformer inputs
            transformer_inputs += [num_embs]

        # If categorical features
        if len(self.categorical_col_indices) > 0:
            # combine all categorical features
            cat_input = []
            for index in self.categorical_col_indices:
                cat_input.append(inputs[:, index].int())
            cat_input = torch.stack(cat_input, axis=1)

            # embed
            cat_embs = self.categorical_embeddings(cat_input)

            # add to transformer inputs
            transformer_inputs += [cat_embs]

        # Prepare for Transformer
        transformer_inputs = torch.concat(transformer_inputs, axis=1)
        importances = []

        # Pass through Transformer blocks
        for transformer in self.transformers:
            if self.explainable:
                transformer_inputs, att_weights = transformer(transformer_inputs)
                importances.append(torch.sum(att_weights[:, :, 0, :], axis=1))
            else:
                transformer_inputs = transformer(transformer_inputs)

        if self.explainable:
            # Sum across the layers
            importances = torch.sum(torch.stack(importances), axis=0) / (
                self.depth * self.heads
            )
            return transformer_inputs, importances
        else:
            return transformer_inputs


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        transformer_emb_dim: int,
        att_dropout: float = 0.1,
        ff_dropout: float = 0.1,
        explainable: bool = False,
        layer_norm: bool = True
    ):
        """Transformer model for TabTransformer

        Args:
            embed_dim (int): embedding dimensions of features
            num_heads (int): number of attention heads
            transformer_emb_dim (int): output_dim of feed-forward layer
            att_dropout (float, optional): dropout rate in multi-headed attention layer. Defaults to 0.1.
            ff_dropout (float, optional): dropout rate in feed-forward layer. Defaults to 0.1.
            explainable (bool, optional): if True, returns attention weights
        """
        super(TransformerBlock, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.transformer_emb_dim = transformer_emb_dim
        self.att_dropout = att_dropout
        self.ff_dropout = ff_dropout
        self.explainable = explainable
        self.layer_norm = layer_norm

        self.att = MultiheadAttention(
            input_dim=embed_dim,
            num_heads=num_heads,
            embed_dim=embed_dim,
            dropout=att_dropout
        )
        self.skip1 = torch.add
        self.layernorm1 = LayerNorm(embed_dim, eps=1e-6) if self.layer_norm else (lambda x: x)
        self.ffn = nn.Sequential(
            Linear(in_features=embed_dim, out_features=transformer_emb_dim),
            GELU(),
            Dropout(ff_dropout),
            Linear(in_features=transformer_emb_dim, out_features=embed_dim),
        )
        self.layernorm2 = LayerNorm(embed_dim, eps=1e-6)
        self.skip2 = torch.add

    def forward(self, inputs):
        # Pre-norm variant
        norm_input = self.layernorm1(inputs)
        if self.explainable:
            # Multi headed attention with attention scores
            attention_output, att_weights = self.att(
                norm_input, return_attention=True
            )
        else:
            # Without attention scores
            attention_output = self.att(norm_input)

        attention_output = torch.add(inputs, attention_output)
        norm_attention_output = self.layernorm2(attention_output)
        feedforward_output = self.ffn(norm_attention_output)
        transformer_output = torch.add(feedforward_output, attention_output)

        # Outputs
        if self.explainable:
            return transformer_output, att_weights
        else:
            return transformer_output

class MultiheadAttention(nn.Module):
    """Code slightly adapted from
    https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html
    """

    def __init__(
            self,
            input_dim: int,
            embed_dim: int,
            num_heads: int,
            dropout: float = None,
        ):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = nn.Linear(input_dim, 3*embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout) if dropout else None

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, mask=None, return_attention=False):
        batch_size, seq_length, _ = x.size()
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3*self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3) # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        # Determine value outputs
        values, attention = self.scaled_dot_product(q, k, v, mask=mask)
        values = values.permute(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, self.embed_dim)
        o = self.o_proj(values)

        if return_attention:
            return o, attention
        else:
            return o

    def scaled_dot_product(self, q, k, v, mask=None):
        d_k = q.size()[-1]
        attn_logits = torch.matmul(q, k.transpose(-2, -1))
        attn_logits = attn_logits / math.sqrt(d_k)
        if mask is not None:
            attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
        attention = F.softmax(attn_logits, dim=-1)
        if self.dropout:
            attention = self.dropout(attention)
        values = torch.matmul(attention, v)
        return values, attention
