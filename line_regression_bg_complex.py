import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np
import random
import matplotlib.pyplot as plt

SEED = 42  # Or any other integer you prefer

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

repr_dic = "/shared/results/common/kargin/unreal_engine/features/background_complexity"
MODELS = ["clip", "croco", "deit", "dino", "mae", "maskfeat", "spa"]
LEVELS = ["level-0", "level-1", "level-2", "level-3"]
N_EPOCHS = 300
split_ratio = 0.9

# Store results for plotting
results_train = {model: [] for model in MODELS}
results_val = {model: [] for model in MODELS}

# Loop over each model
for MODEL in MODELS:
    if MODEL in ["clip", "deit", "dino", "spa", "maskfeat"]:
        FEATURES = "CLS"
    else:
        FEATURES = "MEAN"

    print(f'\n=== Processing {MODEL} ===')
    
    try:
        result = torch.load(f"{repr_dic}/repr_{MODEL}.pt")
    except FileNotFoundError:
        print(f"File repr_{MODEL}.pt not found, skipping...")
        # Fill with NaN for missing models to maintain consistent plotting
        results_train[MODEL] = [np.nan] * len(LEVELS)
        results_val[MODEL] = [np.nan] * len(LEVELS)
        continue
    
    print(f'{MODEL} - {FEATURES} - split_ratio: {split_ratio}, N_EPOCHS: {N_EPOCHS}')
    
    # Get all unique object classes
    object_classes = sorted(set(result['object_class']))
    
    # Loop over each level
    for level in LEVELS:
        print(f'\n--- Processing {level} ---')
        
        # Store per-object losses for this level
        level_train_losses = []
        level_val_losses = []
        
        for obj in object_classes:
            # Filter for this object, this level, in Line orientation
            mask = [
                lvl == level
                and oc == obj
                and ori == 'Line'
                for lvl, oc, ori in zip(result['level'], result['object_class'], result['orientation'])
            ]
            if sum(mask) < 10:  # Skip if not enough samples
                continue

            features = torch.stack([f for f, m in zip(result['features'], mask) if m]).to(torch.float32)
            positions = torch.stack([a for a, m in zip(result['number'], mask) if m]).to(torch.float32) / 540.0

            if FEATURES == "CLS":
                features = features[:, 0, :]
            elif FEATURES == "MEAN":
                features = features[:, -196:, :].mean(1)
            elif FEATURES == "CENTER":
                features = features[:, 105, :]
            else:
                raise Exception("bruh")

            # sort the features by the positions
            order = torch.argsort(positions)
            features = features[order]
            lines = positions[order]

            # --- Split ---
            train_mask = lines < split_ratio # Time-based split
            # train_mask = torch.rand(len(lines)) < split_ratio # Random split
            val_mask = ~train_mask
            train_x, train_y = features[train_mask], lines[train_mask]
            val_x, val_y = features[val_mask], lines[val_mask]

            if len(train_x) == 0 or len(val_x) == 0:
                continue

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

            level_train_losses.append(train_loss)
            level_val_losses.append(val_loss)
            print(f"{obj:40} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Calculate averages for this level
        if level_train_losses:
            avg_train = np.mean(level_train_losses)
            avg_val = np.mean(level_val_losses)
            print(f"\n{level} Average Train Loss: {avg_train:.4f}")
            print(f"{level} Average Val Loss: {avg_val:.4f}")
        else:
            avg_train = np.nan
            avg_val = np.nan
            print(f"\n{level} - No valid samples found")
        
        results_train[MODEL].append(avg_train)
        results_val[MODEL].append(avg_val)

# save results to a csv
with open('results_line_bg_complex.csv', 'w') as f:
    f.write(f"Model,Train L0,Train L1,Train L2,Train L3,Val L0,Val L1,Val L2,Val L3\n")
    for model in MODELS:
        f.write(f"{model},{results_train[model][0]:.4f},{results_train[model][1]:.4f},{results_train[model][2]:.4f},{results_train[model][3]:.4f},{results_val[model][0]:.4f},{results_val[model][1]:.4f},{results_val[model][2]:.4f},{results_val[model][3]:.4f}\n")

# Create the plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Define colors for different models
colors = plt.cm.tab10(np.linspace(0, 1, len(MODELS)))

# Plot 1: Average Train Loss
for i, model in enumerate(MODELS):
    if not all(np.isnan(results_train[model])):
        ax1.plot(range(len(LEVELS)), results_train[model], 
                marker='o', label=model.upper(), color=colors[i], linewidth=2, markersize=6)

ax1.set_xlabel('Level', fontsize=12)
ax1.set_ylabel('Train Loss', fontsize=12)
ax1.set_title('Average Train Loss by Background Complexity Level (Line Regression)', fontsize=14)
ax1.set_xticks(range(len(LEVELS)))
ax1.set_xticklabels([f'{i}' for i in range(len(LEVELS))])
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Average Val Loss
for i, model in enumerate(MODELS):
    if not all(np.isnan(results_val[model])):
        ax2.plot(range(len(LEVELS)), results_val[model], 
                marker='o', label=model.upper(), color=colors[i], linewidth=2, markersize=6)

ax2.set_xlabel('Level', fontsize=12)
ax2.set_ylabel('Val Loss', fontsize=12)
ax2.set_title('Average Val Loss by Background Complexity Level (Line Regression)', fontsize=14)
ax2.set_xticks(range(len(LEVELS)))
ax2.set_xticklabels([f'{i}' for i in range(len(LEVELS))])
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('line_background_complexity_results.png', dpi=300, bbox_inches='tight')
plt.show()

# Print summary table
print("\n" + "="*80)
print("SUMMARY RESULTS - LINE REGRESSION")
print("="*80)
print(f"{'Model':<12} | {'Train L0':<10} | {'Train L1':<10} | {'Train L2':<10} | {'Train L3':<10} | {'Val L0':<10} | {'Val L1':<10} | {'Val L2':<10} | {'Val L3':<10}")
print("-" * 120)

for model in MODELS:
    if not all(np.isnan(results_train[model])):
        train_str = " | ".join([f"{x:.4f}" if not np.isnan(x) else "  N/A  " for x in results_train[model]])
        val_str = " | ".join([f"{x:.4f}" if not np.isnan(x) else "  N/A  " for x in results_val[model]])
        print(f"{model.upper():<12} | {train_str} | {val_str}") 