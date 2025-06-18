import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np
import random
SEED = 42  # Or any other integer you prefer

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

repr_dic = "/shared/results/common/kargin/unreal_engine/features"
MODEL = "dino"
FEATURES = "CLS"
N_EPOCHS = 300
split_ratio = 0.9

result = torch.load(f"{repr_dic}/repr_{MODEL}.pt")

print(f'{MODEL} - {FEATURES} - split_ratio: {split_ratio}, N_EPOCHS: {N_EPOCHS}')

# Get all unique object classes
object_classes = sorted(set(result['object_class']))

# Store per-object losses
all_train_losses = []
all_val_losses = []

for obj in object_classes:
    # Filter for this object, in City, Orbit
    mask = [
        env == 'City'
        and oc == obj
        and ori == 'Line'
        for env, oc, ori in zip(result['environment'], result['object_class'], result['orientation'])
    ]
    if sum(mask) < 10:  # Skip if not enough samples
        continue

    features = torch.stack([f for f, m in zip(result['features'], mask) if m]).to(torch.float32)
    positions = torch.stack([a for a, m in zip(result['number'], mask) if m]).to(torch.float32) / 360.0

    if FEATURES == "CLS":
        features = features[:, 0, :]
    elif FEATURES == "MEAN":
        features = features[:, -196:, :].mean(1)
    elif FEATURES == "CENTER":
        features = features[:, 105, :]
    else:
        raise Exception("bruh")

    # sort the features by the angles
    order = torch.argsort(positions)
    features = features[order]
    lines = positions[order]

    # --- Split ---
    train_mask = lines < split_ratio # Time-based split
    # train_mask = torch.rand(len(lines)) < split_ratio # Random split
    val_mask = ~train_mask
    train_x, train_y = features[train_mask], lines[train_mask]
    val_x, val_y = features[val_mask], lines[val_mask]

    # --- DataLoader ---
    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=64, shuffle=True)

    # --- Model ---
    model = nn.Sequential(
        nn.Linear(train_x.shape[1], 256),
        nn.ReLU(),
        nn.Dropout(p=0.4),
        nn.Linear(256, 1)
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    loss_fn = nn.MSELoss()

    # --- Training ---
    for epoch in range(N_EPOCHS):
        for x, y in train_loader:
            pred = model(x).squeeze(1)
            loss = loss_fn(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                preds = model(val_x).squeeze(1)
                val_loss = loss_fn(preds, val_y).item()
            model.train()

    # --- Evaluation ---
    model.eval()
    with torch.no_grad():
        preds_val = model(val_x).squeeze(1)
        val_loss = loss_fn(preds_val, val_y).item()
        preds_train = model(train_x).squeeze(1)
        train_loss = loss_fn(preds_train, train_y).item()

    all_train_losses.append(train_loss)
    all_val_losses.append(val_loss)
    print(f"{obj:40} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

# Report averages
print("\n--- Averages ---")
print(f"Average Train Loss: {np.mean(all_train_losses):.4f}")
print(f"Std Train Loss: {np.std(all_train_losses):.4f}")
print(f"Average Val Loss:   {np.mean(all_val_losses):.4f}")
print(f"Std Val Loss: {np.std(all_val_losses):.4f}")
