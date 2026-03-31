import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class ColumnMatcher(nn.Module):
    def __init__(self, encoder: str = "all-MiniLM-L6-v2", dropout: float = 0.3):
        super().__init__()

        self.device = get_device()

        self.encoder = SentenceTransformer(encoder, device=str(self.device))

        self.classifier = nn.Sequential(
            nn.Linear(384 * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        ).to(self.device)

    def forward(self, text_a, text_b):
        emb_a = self.encoder.encode(
            text_a,
            convert_to_tensor=True,
            device=str(self.device),
        ).to(self.device)

        emb_b = self.encoder.encode(
            text_b,
            convert_to_tensor=True,
            device=str(self.device),
        ).to(self.device)

        x = torch.cat([emb_a, emb_b], dim=1)
        return self.classifier(x).squeeze(1)
