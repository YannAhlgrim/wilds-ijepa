import math

import torch


class LARS(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr,
        weight_decay=0.0,
        momentum=0.9,
        eta=0.001,
        eps=1e-8,
        exclude_bias_and_norm=True,
    ):
        if lr <= 0.0:
            raise ValueError(f"Invalid lr: {lr}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        if eta <= 0.0:
            raise ValueError(f"Invalid eta: {eta}")
        if eps <= 0.0:
            raise ValueError(f"Invalid eps: {eps}")

        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            eta=eta,
            eps=eps,
            exclude_bias_and_norm=exclude_bias_and_norm,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            eta = group["eta"]
            eps = group["eps"]
            exclude_bias_and_norm = group["exclude_bias_and_norm"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("LARS does not support sparse gradients")

                param_norm = torch.norm(p)
                grad_norm = torch.norm(grad)

                lars_lr = 1.0
                if not exclude_bias_and_norm or p.ndim > 1:
                    if param_norm > 0.0 and grad_norm > 0.0:
                        lars_lr = eta * param_norm / (grad_norm + weight_decay * param_norm + eps)

                d_p = grad
                if weight_decay != 0.0 and (not exclude_bias_and_norm or p.ndim > 1):
                    d_p = d_p.add(p, alpha=weight_decay)

                if momentum != 0.0:
                    param_state = self.state.setdefault(p, {})
                    if "momentum_buffer" not in param_state:
                        buf = param_state["momentum_buffer"] = torch.clone(d_p).detach()
                    else:
                        buf = param_state["momentum_buffer"]
                        buf.mul_(momentum).add_(d_p)
                    d_p = buf

                p.add_(d_p, alpha=-lr * lars_lr)

        return loss
