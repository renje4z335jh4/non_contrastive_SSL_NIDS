import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import conv2d

from models.ssl.base import BaseRepresentation

class WMSE(BaseRepresentation):
    name = 'W-MSE'

    def __init__(
        self,
        in_features: int,
        n_instances: int,
        encoder_class: str,
        mlp_params: dict,
        device: str = None,
        mixup_alpha: float = None,
        w_momentum: float = 0.01,
        w_eps: float = 1e-8, # with 0 results in NaN loss!!!!
        w_iter: int = 1, # "iterations for whitening matrix estimation"
        w_size: int = 128, # size of sub-batch for W-MSE loss
        n_views: int = 2, # number of generated views
        **encoder_args
    ):
        super().__init__(in_features, n_instances, encoder_class, mlp_params, device, **encoder_args)

        self.w_momentum = w_momentum
        self.w_eps = w_eps
        self.w_iter = w_iter
        self.w_size = w_size
        self.n_views = n_views
        self.mixup_alpha = mixup_alpha

        # for consistency:
        num_samples = self.n_views
        self.num_pairs = num_samples * (num_samples - 1) // 2 # number of samples (d) generated from each image

        self.whitening = Whitening2d(num_features=mlp_params['output_dim'], momentum=self.w_momentum, eps=self.w_eps, track_running_stats=False)
        self.loss_f = norm_mse_loss

    def get_params(self) -> dict:
        base_params = super().get_params()
        del base_params['mlp_params']
        params = {
            'w_momentum': self.w_momentum,
            'w_eps': self.w_eps,
            'w_iter': self.w_iter,
            'w_size': self.w_size,
            'n_views': self.n_views,
            'mixup_alpha': self.mixup_alpha,
            'mlp_params': self.mlp_params
        }
        return dict(
            **base_params,
            **params
        )

    @staticmethod
    def load_from_ckpt(model_params: dict, state_dict: dict):
        model = WMSE(**model_params)
        model.load_state_dict(state_dict)
        return model

    def forward(self, samples):
        bs = len(samples[0])
        h = [self.encoder(x.cuda(non_blocking=True)) for x in samples]
        h = torch.cat(h)

        if self.mixup_alpha:
            # select random sample from batch and perform mixup
            idx = np.random.randint(h.shape[0], size=h.shape[0])
            h = torch.mul(self.mixup_alpha, h[:]) + torch.mul((1 - self.mixup_alpha), h[idx])

        h = self.projector(h)

        loss = 0
        for _ in range(self.w_iter):
            z = torch.empty_like(h)
            perm = torch.randperm(bs).view(-1, self.w_size)
            for idx in perm:
                for i in range(len(samples)):
                    z[idx + i * bs] = self.whitening(h[idx + i * bs])
            for i in range(len(samples) - 1):
                for j in range(i + 1, len(samples)):
                    x0 = z[i * bs : (i + 1) * bs]
                    x1 = z[j * bs : (j + 1) * bs]
                    loss += self.loss_f(x0, x1)
        loss /= self.w_iter * self.num_pairs
        return loss

    def get_encoder(self):
        return self.encoder


def norm_mse_loss(x0, x1):
    x0 = F.normalize(x0)
    x1 = F.normalize(x1)
    return 2 - 2 * (x0 * x1).sum(dim=-1).mean()

class Whitening2d(nn.Module):
    def __init__(self, num_features, momentum=0.01, track_running_stats=True, eps=0):
        super().__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.track_running_stats = track_running_stats
        self.eps = eps

        if self.track_running_stats:
            self.register_buffer(
                "running_mean", torch.zeros([1, self.num_features, 1, 1])
            )
            self.register_buffer("running_variance", torch.eye(self.num_features))

    def forward(self, x):
        x = x.unsqueeze(2).unsqueeze(3)
        m = x.mean(0).view(self.num_features, -1).mean(-1).view(1, -1, 1, 1)
        if not self.training and self.track_running_stats:  # for inference
            m = self.running_mean
        xn = x - m

        T = xn.permute(1, 0, 2, 3).contiguous().view(self.num_features, -1)
        f_cov = torch.mm(T, T.permute(1, 0)) / (T.shape[-1] - 1)

        eye = torch.eye(self.num_features).type(f_cov.type())

        if not self.training and self.track_running_stats:  # for inference
            f_cov = self.running_variance

        f_cov_shrinked = (1 - self.eps) * f_cov + self.eps * eye

        inv_sqrt = torch.linalg.solve_triangular(
            torch.linalg.cholesky(f_cov_shrinked),
            eye,
            upper=False
            )

        inv_sqrt = inv_sqrt.contiguous().view(
            self.num_features, self.num_features, 1, 1
        )

        decorrelated = conv2d(xn, inv_sqrt)

        if self.training and self.track_running_stats:
            self.running_mean = torch.add(
                self.momentum * m.detach(),
                (1 - self.momentum) * self.running_mean,
                out=self.running_mean,
            )
            self.running_variance = torch.add(
                self.momentum * f_cov.detach(),
                (1 - self.momentum) * self.running_variance,
                out=self.running_variance,
            )

        return decorrelated.squeeze(2).squeeze(2)

    def extra_repr(self):
        return "features={}, eps={}, momentum={}".format(
            self.num_features, self.eps, self.momentum
        )
