from __future__ import annotations
from typing import List, Tuple
import torch
import torch.nn as nn

from byol.models.transformer.transformer_encoder import FTTransformerEncoder
from byol.models.base import BaseModel

class FTTransformer(BaseModel):
    name = 'FTTransformer'
    def __init__(
        self,
        out_dim: int,
        categorical_col_indices: List[int] = [],
        categories_unique_values: List[int] = [],
        numeric_col_indices: List[int] = [],
        embedding_dim: int = 32,
        depth: int = 4,
        heads: int = 8,
        attn_dropout: float = 0.1,
        ff_dropout: float = 0.1,
        explainable: bool = False,
        device: str = None,
        **kwargs
    ):
        super(FTTransformer, self).__init__(device, **kwargs)
        self.out_dim = out_dim
        self.categorical_col_indices = categorical_col_indices
        self.categories_unique_values = categories_unique_values
        self.numeric_col_indices = numeric_col_indices
        self.embedding_dim = embedding_dim
        self.depth = depth
        self.heads = heads
        self.attn_dropout = attn_dropout
        self.ff_dropout = ff_dropout
        self.explainable = explainable

        # Initialise encoder
        self.encoder = FTTransformerEncoder(
            categorical_col_indices = categorical_col_indices,
            categories_unique_values = categories_unique_values,
            numeric_col_indices = numeric_col_indices,
            embedding_dim = embedding_dim,
            depth = depth,
            heads = heads,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
            explainable = explainable,
            add_cls_token=True, # contextualized CLS token embedding serves as an input into a simple MLP classifier which produces the desired output
        )

        # mlp layers
        self.ln = nn.LayerNorm(embedding_dim)
        self.final_ff = nn.Linear(embedding_dim, embedding_dim//2)
        self.act_fct = nn.ReLU()
        self.output_layer = nn.Linear(embedding_dim//2, out_dim)

    @staticmethod
    def load_from_ckpt(model_params: dict, model_state_dict: dict) -> FTTransformer:
        model = FTTransformer(
            **model_params
        )
        model.load_state_dict(model_state_dict)
        return model

    def get_params(self) -> dict:
        base_params = super().get_params()
        params = {
            'out_dim': self.out_dim,
            'categorical_col_indices': self.categorical_col_indices,
            'categories_unique_values': self.categories_unique_values,
            'numeric_col_indices': self.numeric_col_indices,
            'embedding_dim': self.embedding_dim,
            'depth': self.depth,
            'heads': self.heads,
            'attn_dropout': self.attn_dropout,
            'ff_dropout': self.ff_dropout,
            'explainable': self.explainable,
        }
        return {**base_params, **params}

    def forward(self, inputs) -> Tuple[torch.Tensor, torch.Tensor]:

        if self.explainable:
            x, expl = self.encoder(inputs)
        else:
            x = self.encoder(inputs)
            expl = None

        cls_tokens = x[:, 0, :] # extract cls token
        layer_norm_cls = self.ln(cls_tokens)
        layer_norm_cls = self.act_fct(self.final_ff(layer_norm_cls))
        output = self.output_layer(layer_norm_cls)

        return output, expl
