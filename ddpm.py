from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class DDPMConfig:
    num_steps: int
    beta1: float
    beta2: float
    drop: float


class DDPM(nn.Module):
    def __init__(self, model, config: DDPMConfig, device):
        super().__init__()

        self.model = model.to(device)
        self.config = config

        self.device = device
        self.criteria = nn.MSELoss()

        self.beta, self.alpha, self.alphabar = self.schedule(config.num_steps, config.beta1, config.beta2, self.device)

    @staticmethod
    def schedule(num_steps, beta1, beta2, device):
        beta = (beta2 - beta1) * torch.linspace(0, num_steps, steps=num_steps+1, dtype=torch.float32) / num_steps + beta1
        beta = beta.to(device)

        alpha = 1 - beta
        alphabar = torch.cumprod(alpha, dim=0).to(device)

        return beta, alpha, alphabar

    def forward(self, x, c):
        """
        x: (batch_size, C, H, W)
        c: (batch_size,) long tensor: context cue (class label)
        """
        batch_size = x.size(0)

        t = torch.randint(0, self.config.num_steps, (batch_size, 1)).to(self.device)
        eps = torch.randn_like(x).to(self.device)

        x = torch.sqrt(self.alphabar[t]).view(batch_size, 1, 1, 1) * x + torch.sqrt(1 - self.alphabar[t]).view(batch_size, 1, 1, 1) * eps

        # random drop of context
        cmask = torch.zeros_like(c).to(self.device)
        if self.config.drop > 0:
            drop = torch.rand(batch_size) < self.config.drop
            cmask[drop] = 1

        t = t.float() / self.config.num_steps
        model_out = self.model(x, t, c, cmask)
        loss = self.criteria(model_out, eps)

        return loss

    def sample(self, size, c, w=0.0, same_rand=False):
        """
        size: (3,)
        c: (batch_size,) long tensor: context cue (class label)
            or
           (batch_size, num_classes) float tensor: one-hot encoded context cue
        w: float
        """
        batch_size = c.size(0)

        if same_rand:
          x = torch.randn(1, *size).to(self.device)
          x = x.repeat(batch_size, 1, 1, 1)
        else:
          x = torch.randn(batch_size, *size).to(self.device)

        c = c.to(self.device)

        xs = []

        self.eval()
        with torch.no_grad():
            for t in reversed(range(self.config.num_steps)):
                t_tensor = torch.full((batch_size, 1), t / self.config.num_steps, dtype=torch.float).to(self.device)

                cmask = None
                if w != 0:
                    cmask = torch.cat([torch.zeros(batch_size), torch.ones(batch_size)], dim=0)
                    x = torch.cat([x, x.clone()], dim=0)
                    c = torch.cat([c, c.clone()], dim=0)
                    t_tensor = torch.cat([t_tensor, t_tensor.clone()], dim=0)

                if c.dim() == 1:
                    eps = self.model(x, t_tensor, c, cmask)
                else:
                    eps = self.model._forward(x, t_tensor, c, cmask)

                if w != 0:
                    eps_guided = eps[:batch_size]
                    eps_non_guided = eps[batch_size:]
                    eps = (1 + w) * eps_guided - w * eps_non_guided

                    x = x[:batch_size]
                    c = c[:batch_size]

                x = (1 / torch.sqrt(self.alpha[t])) * (x - eps * (self.beta[t] / torch.sqrt(1 - self.alphabar[t])))

                if same_rand:
                    z = torch.randn(1, *size).to(self.device)
                    z = z.repeat(batch_size, 1, 1, 1)
                else:
                    z = torch.rand_like(x)

                if t != 0:
                    x += torch.sqrt(self.beta[t]) * z
                if t % 50 == 0:
                    xs.append(x)

        if xs[-1] is not x:
            xs.append(x)

        return xs