import copy
import torch
from torch.utils.data import DataLoader
from src.model import ColumnMatcher


def train_model(
    train_dataset,
    val_dataset,
    epochs: int = 10,
    lr: float = 1e-3,
    batch_size: int = 8,
    patience: int = 3,
    pos_weight: float = 4.0,
    encoder: str = "all-MiniLM-L6-v2",
    dropout: float = 0.3,
):
    model = ColumnMatcher(encoder=encoder, dropout=dropout)
    device = model.device

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=2
    )
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], device=device)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        for batch in train_loader:
            labels = batch["label"].to(device)
            preds = model(batch["text_a"], batch["text_b"])
            loss = criterion(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                labels = batch["label"].to(device)
                preds = model(batch["text_a"], batch["text_b"])
                total_val_loss += criterion(preds, labels).item()

        scheduler.step(total_val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch + 1:2d} | Train Loss: {total_train_loss:.4f} | "
            f"Val Loss: {total_val_loss:.4f} | LR: {current_lr:.2e}"
        )

        if total_val_loss < best_val_loss:
            best_val_loss = total_val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping nach Epoch {epoch + 1} (patience={patience})")
                break

    model.load_state_dict(best_state)
    return model
