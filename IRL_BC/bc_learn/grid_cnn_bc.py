"""SimpleGridCNN 行为克隆 agent。"""
from typing import Any

import torch
import torch.nn.functional as F

from bc_learn.metrics import masked_topk_accuracy
from cnn_config import CNN_DROPOUT, CNN_N_CONV_LAYERS
from models.simple_grid_cnn import SimpleGridCNN
from models.soft_q_ops import soft_policy_entropy


class GridCNNBCAgent:
    """监督 BC：SimpleGridCNN 输出 per-cell logits，交叉熵拟合专家动作。"""

    def __init__(
        self,
        *,
        grid_h: int,
        grid_w: int,
        in_channels: int,
        n_actions: int,
        device: str = "cpu",
        lr: float = 1e-4,
        dropout: float = CNN_DROPOUT,
    ) -> None:
        self.device = torch.device(device)
        self.grid_h = int(grid_h)
        self.grid_w = int(grid_w)
        self.in_channels = int(in_channels)
        self.n_actions = int(n_actions)

        self.q_net = SimpleGridCNN(
            in_channels=self.in_channels,
            grid_h=self.grid_h,
            grid_w=self.grid_w,
            action_dim=self.n_actions,
            n_conv_layers=int(CNN_N_CONV_LAYERS),
            dropout=float(dropout),
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=float(lr))

    def logits(self, obs: torch.Tensor) -> torch.Tensor:
        return self.q_net(obs)

    def train_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        obs = batch["obs"]
        actions = batch["actions"].long().view(-1)
        mask = batch.get("mask")

        self.q_net.train()
        logits = self.logits(obs)
        if mask is not None:
            neg_inf = torch.finfo(logits.dtype).min
            loss = F.cross_entropy(logits.masked_fill(~mask, neg_inf), actions)
        else:
            loss = F.cross_entropy(logits, actions)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            if mask is not None:
                ent = float(soft_policy_entropy(logits, mask, alpha=1.0).cpu())
                q_valid = logits[mask]
            else:
                probs = F.softmax(logits, dim=-1)
                ent = float((-(probs * F.log_softmax(logits, dim=-1)).sum(dim=-1)).mean().cpu())
                q_valid = logits.reshape(-1)

        return {
            "loss": float(loss.detach().cpu()),
            "policy_entropy": ent,
            "logit_mean": float(q_valid.mean().cpu()) if q_valid.numel() else 0.0,
            "logit_std": float(q_valid.std().cpu()) if q_valid.numel() > 1 else 0.0,
        }

    @torch.no_grad()
    def eval_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        obs = batch["obs"]
        actions = batch["actions"].long().view(-1)
        mask = batch.get("mask")

        self.q_net.eval()
        logits = self.logits(obs)
        if mask is not None:
            neg_inf = torch.finfo(logits.dtype).min
            loss = float(F.cross_entropy(logits.masked_fill(~mask, neg_inf), actions).cpu())
        else:
            loss = float(F.cross_entropy(logits, actions).cpu())
        if mask is not None:
            ent = float(soft_policy_entropy(logits, mask, alpha=1.0).cpu())
            top10 = masked_topk_accuracy(logits, actions, mask, k=10)
        else:
            probs = F.softmax(logits, dim=-1)
            ent = float((-(probs * F.log_softmax(logits, dim=-1)).sum(dim=-1)).mean().cpu())
            top10 = float(
                (logits.topk(min(10, logits.shape[1]), dim=1).indices == actions.unsqueeze(1))
                .any(dim=1)
                .float()
                .mean()
                .cpu()
            )
        return {"loss": loss, "top10_accuracy": top10, "policy_entropy": ent}

    @torch.no_grad()
    def predict_action(self, obs: torch.Tensor, mask: torch.Tensor) -> int:
        self.q_net.eval()
        logits = self.logits(obs)
        neg_inf = torch.finfo(logits.dtype).min
        return int(logits.masked_fill(~mask, neg_inf).argmax(dim=1).item())

    def save(self, path: str) -> None:
        torch.save(
            {
                "network": "SimpleGridCNN",
                "q_net": self.q_net.state_dict(),
                "grid_h": self.grid_h,
                "grid_w": self.grid_w,
                "in_channels": self.in_channels,
                "n_actions": self.n_actions,
            },
            path,
        )

    def load(self, path: str) -> None:
        blob: dict[str, Any] = torch.load(path, map_location=self.device, weights_only=False)
        state = blob["q_net"]
        self.q_net = SimpleGridCNN.from_state_dict(
            state,
            in_channels=self.in_channels,
            grid_h=self.grid_h,
            grid_w=self.grid_w,
            action_dim=self.n_actions,
        ).to(self.device)
        self.q_net.load_state_dict(state)
