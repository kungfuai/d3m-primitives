import torch

from .cross_entropy import CrossEntropy


class PseudoLabel:
    def __init__(self, threshold=0.95):
        self.threshold = threshold

    def __call__(self, y_hat_s, y_hat_w, *args, **kwargs):
        max_value, y = y_hat_w.softmax(1).max(1)
        mask = max_value >= self.threshold
        y_hat = y_hat_s
        return y_hat, y, mask


def _reduce_max(x, idx_list):
    for i in idx_list:
        x = x.max(i, keepdim=True)[0]
    return x


def _normalize(x):
    x = x / (1e-12 + _reduce_max(x.abs(), range(1, len(x.shape))))  # to avoid overflow
    x = x / (1e-6 + x.pow(2).sum(list(range(1, len(x.shape))), keepdim=True)).sqrt()
    return x


class VAT:
    def __init__(self, eps=0.2, xi=1e-6, n_iter=4):
        self.eps = eps
        self.xi = xi
        self.n_iter = n_iter
        self.obj_func = CrossEntropy()

    def __call__(self, y_hat_w, w_data, student_f, *args, **kwargs):
        mask = torch.ones_like(y_hat_w.max(1)[0])
        y = y_hat_w.softmax(1)
        d = torch.randn_like(w_data)
        d = _normalize(d)
        for _ in range(self.n_iter):
            d.requires_grad = True
            x_hat = w_data + self.xi * d
            y_hat = student_f(x_hat)
            loss = self.obj_func(y_hat, y)
            d = torch.autograd.grad(loss, d)[0]
            d = _normalize(d).detach()
        x_hat = w_data + self.eps * d
        y_hat = student_f(x_hat)
        return y_hat, y, mask


class ICT:
    def __init__(self, alpha=0.1):
        self.alpha = alpha

    def mixup(self, x, y):
        device = x.device
        b = x.shape[0]
        permute = torch.randperm(b)
        perm_x = x[permute]
        perm_y = y[permute]
        factor = (
            torch.distributions.beta.Beta(self.alpha, self.alpha)
            .sample((b, 1))
            .to(device)
        )
        if x.ndim == 4:
            x_factor = factor[..., None, None]
        else:
            x_factor = factor
        mixed_x = x_factor * x + (1 - x_factor) * perm_x
        mixed_y = factor * y + (1 - factor) * perm_y
        return mixed_x, mixed_y

    def __call__(self, y_hat_w, w_data, student_f, *args, **kwargs):
        mask = torch.ones_like(y_hat_w.max(1)[0])
        y = y_hat_w.softmax(1)
        X, y = self.mixup(w_data, y)
        y_hat = student_f(X)
        return y_hat, y, mask
