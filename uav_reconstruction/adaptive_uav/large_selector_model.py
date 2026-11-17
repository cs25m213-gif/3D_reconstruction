"""Larger experimental selector: RGB representation plus action-specific utility heads."""
import torch
from torch import nn
from torch.nn import functional as F


class SelectorVisualEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision.models import resnet18
        self.cnn = resnet18(weights=None)
        self.cnn.fc = nn.Identity()
        self.project = nn.Sequential(nn.Linear(512, 512), nn.LayerNorm(512), nn.GELU(), nn.Linear(512, 384))
        self.degradation = nn.Linear(512, 7)
        self.register_buffer('mean', torch.tensor([.485, .456, .406])[None, :, None, None])
        self.register_buffer('std', torch.tensor([.229, .224, .225])[None, :, None, None])

    def forward(self, rgb):
        x = self.cnn((rgb-self.mean)/self.std)
        return F.normalize(self.project(x), dim=-1), self.degradation(x)


class LargeActionSelector(nn.Module):
    def __init__(self, state_dim=24, width=256, layers=4):
        super().__init__()
        self.register_buffer('mu', torch.zeros(state_dim))
        self.register_buffer('sd', torch.ones(state_dim))
        self.image = nn.Linear(384, width)
        self.map = nn.Sequential(nn.Linear(state_dim, width), nn.LayerNorm(width), nn.GELU())
        self.role = nn.Parameter(torch.randn(3, width)*.02)
        layer = nn.TransformerEncoderLayer(width, 8, width*4, .15, 'gelu', batch_first=True, norm_first=True)
        self.context = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.add = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 128), nn.GELU(), nn.Linear(128, 1))
        self.swap = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 128), nn.GELU(), nn.Linear(128, 1))

    def forward(self, candidate, occupant, has_occupant, state):
        tokens = torch.stack([self.map((state-self.mu)/self.sd), self.image(candidate), self.image(occupant)], 1)
        tokens = tokens+self.role[None]
        mask = torch.zeros((len(state), 3), dtype=torch.bool, device=state.device)
        mask[:, 2] = ~has_occupant.bool()
        x = self.context(tokens, src_key_padding_mask=mask)[:, 0]
        return torch.where(has_occupant.bool(), self.swap(x).squeeze(-1), self.add(x).squeeze(-1))
