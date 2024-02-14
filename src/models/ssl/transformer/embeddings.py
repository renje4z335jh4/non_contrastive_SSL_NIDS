import math
from typing import List
import torch
import torch.nn as nn

class NEmbedding(nn.Module):
    def __init__(
            self,
            d_numerical: int,
            embedding_dim: int = 32,
            add_cls_token: bool = False
        ) -> None:
        """Embeds numerical features to get embeddings of same dimension for transformer.
        E.g. input dim = [batch_size, d_numerical] -> output dim = [batch_size, d_numerical, embedding_dim]

        Parameters
        ----------
        d_numerical : int
            number/dimension of numerical features
        embedding_dim : int, optional
            dimension to embed each feature in, by default 32
        add_cls_token : bool, optional
            whether to add a cls token for prediction, by default False
        """
        super().__init__()
        self.add_cls_token = add_cls_token
        self.embedding_dim = embedding_dim

        self.weight = nn.Parameter(torch.Tensor((d_numerical if not add_cls_token else d_numerical+1), embedding_dim))
        self.bias = nn.Parameter(torch.Tensor(d_numerical, embedding_dim))
        # The initialization is inspired by nn.Linear
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.bias, a=math.sqrt(5))

    def forward(self, x_num: torch.Tensor) -> torch.Tensor:

        if self.add_cls_token:
            x_num = torch.cat(
                [
                    torch.ones(len(x_num), 1, device=x_num.device),
                    x_num
                ], dim=1
            )

        # element-wise multiplication with weight
        x = self.weight[None] * x_num[:, :, None]

        # extend static bias for cls token
        bias = self.bias
        if self.add_cls_token:
            bias = torch.cat(
                [
                    torch.zeros(1, self.embedding_dim, device=x_num.device),
                    self.bias
                ]
            )

        # add bias on embedding
        x = x + bias[None]

        return x

class CEmbedding(nn.Module):
    def __init__(
            self,
            categories_unique_values: List[int],
            embedding_dim: int = 32,
        ) -> None:
        """Embeds categorical features to embedding for transformer.
        Categorical features have to be "Label-Encoded" in advance.
        E.g. input dim = [batch_size, len(categories_unique_values)]
        -> output dim = [batch_size, len(categories_unique_values), embedding_dim]

        Parameters
        ----------
        categories_unique_values : List[int]
            List of unique values for each feature
        embedding_dim : int, optional
            dimension to embed each feature in, by default 32
        """
        # num uniques for each category# num uniques for each category
        super().__init__()

        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(num_embeddings=len(n_cat), embedding_dim=embedding_dim) for n_cat in categories_unique_values
            ]
        )

    def forward(self, x_cat: torch.Tensor) -> torch.Tensor:
        # for each category do embedding with corresponding predefined embedding
        x = [emb(x_cat[:,i]) for i, emb in enumerate(self.embeddings)]

        # concat all categorical embeddings
        return torch.stack(x, 1)
