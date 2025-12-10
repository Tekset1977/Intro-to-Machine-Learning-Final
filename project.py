import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize, Bounds, LinearConstraint
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ---------- PATHS ----------
REVIEWS_CSV = os.path.join("steam_dataset_2025_csv/reviews.csv")
APP_CATS_CSV = os.path.join("steam_dataset_2025_csv/application_categories.csv")
CATEGORIES_CSV = os.path.join("steam_dataset_2025_csv/categories.csv")

for p, label in [(REVIEWS_CSV, "reviews.csv"),
                 (APP_CATS_CSV, "application_categories.csv")]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"{label} not found at: {p}")
    else:
        print("✓ Found:", p)

if os.path.exists(CATEGORIES_CSV):
    print("✓ Found categories.csv (will use for names)")
else:
    print("⚠ categories.csv not found; using category_id only")


# ---------- HYPERPARAMETERS ----------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 512
EPOCHS = 30
LR = 1e-3
RANDOM_STATE = 42
TOP_N_CATEGORIES = 20
APPID_COL = "appid"
VOTED_UP_COL = "voted_up"


# ============================================
# DATA LOADING / PREP
# ============================================
def load_and_prepare_data():
    reviews = pd.read_csv(REVIEWS_CSV)
    app_cats = pd.read_csv(APP_CATS_CSV)

    if APPID_COL not in reviews.columns or APPID_COL not in app_cats.columns:
        raise KeyError("Both reviews and application_categories must have 'appid'")

    if VOTED_UP_COL not in reviews.columns:
        raise KeyError(f"reviews.csv must contain '{VOTED_UP_COL}' column")

    if "category_id" not in app_cats.columns:
        raise KeyError("application_categories.csv must contain 'category_id'")

    # Binary target
    reviews["positive"] = reviews[VOTED_UP_COL].astype(int)

    # Average positive score per app
    score_df = (
        reviews.groupby(APPID_COL)["positive"]
        .mean()
        .rename("review_score")
        .reset_index()
    )

    # Find top N categories
    top_cats = (
        app_cats["category_id"]
        .value_counts()
        .head(TOP_N_CATEGORIES)
        .index
        .tolist()
    )

    # One-hot encode categories
    cat_table = (
        app_cats[app_cats["category_id"].isin(top_cats)]
        .assign(value=1)
        .pivot_table(
            index=APPID_COL,
            columns="category_id",
            values="value",
            fill_value=0,
        )
    )

    # Rename columns
    cat_col_map = {cid: f"cat_{cid}" for cid in cat_table.columns}
    cat_table = cat_table.rename(columns=cat_col_map).reset_index()

    # Merge with target
    df = cat_table.merge(score_df, on=APPID_COL, how="inner")

    feature_cols = [c for c in df.columns if c.startswith("cat_")]
    X = df[feature_cols].astype(np.float32).values
    y = df["review_score"].astype(np.float32).values

    print(f"Total apps with categories & reviews: {len(df)}")
    print("Number of category features:", len(feature_cols))

    # Optional: map category id → name
    cat_name_map = {}
    if os.path.exists(CATEGORIES_CSV):
        cats_df = pd.read_csv(CATEGORIES_CSV)
        if "id" in cats_df.columns and "description" in cats_df.columns:
            id_to_name = dict(zip(cats_df["id"], cats_df["description"]))
            for cid, col_name in cat_col_map.items():
                cat_name_map[col_name] = id_to_name.get(cid, str(cid))

    return X, y, feature_cols, cat_name_map


# ============================================
# PYTORCH DATASET
# ============================================
class SteamDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float().unsqueeze(1)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ============================================
# NEURAL NETWORK MODELS
# ============================================
class LinearRegressionModel(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, 1)

    def forward(self, x):
        return self.linear(x)


class FCNNModel(nn.Module):
    """3 hidden layers fully connected network"""
    def __init__(self, in_dim, hidden_dims=(64, 64, 32)):
        super().__init__()
        h1, h2, h3 = hidden_dims
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(h2, h3),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(h3, 1),
        )

    def forward(self, x):
        return self.net(x)


class ResidualBlock1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity
        return self.relu(out)


class ResNetModel(nn.Module):
    """ResNet-style 1D CNN over category one-hot features"""
    def __init__(self, num_features, base_channels=32, num_blocks=3):
        super().__init__()
        self.conv_in = nn.Conv1d(1, base_channels, kernel_size=3, padding=1)
        self.bn_in = nn.BatchNorm1d(base_channels)
        self.relu = nn.ReLU()

        blocks = [ResidualBlock1D(base_channels) for _ in range(num_blocks)]
        self.blocks = nn.Sequential(*blocks)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(base_channels, 1)

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch, 1, num_features)
        out = self.relu(self.bn_in(self.conv_in(x)))
        out = self.blocks(out)
        out = self.pool(out).squeeze(-1)
        out = self.fc(out)
        return out


# ============================================
# NEURAL NETWORK TRAINING
# ============================================
def train_nn_model(model, train_loader, val_loader, epochs=EPOCHS, lr=LR, name="model"):
    model = model.to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_hist = []
    val_hist = []

    print(f"\n=== Training {name} on {DEVICE} ===")
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            preds = torch.sigmoid(model(xb))
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                preds = torch.sigmoid(model(xb))
                loss = criterion(preds, yb)
                val_losses.append(loss.item())

        train_hist.append(float(np.mean(train_losses)))
        val_hist.append(float(np.mean(val_losses)))

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:03d} | "
                f"Train Loss: {train_hist[-1]:.4f} | "
                f"Val Loss: {val_hist[-1]:.4f}"
            )

    return model, train_hist, val_hist


# ============================================
# RIDGE REGRESSION MODEL TRAINING
# ============================================
def train_ridge_model(X_train, y_train, X_val, y_val):
    """
    Train Ridge regression (L2 regularization) for convex optimization.
    """
    print("\n=== Training Ridge Regression (for Convex Optimization) ===")
    
    model = Ridge(alpha=0.1, positive=True, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    
    train_mse = mean_squared_error(y_train, train_pred)
    val_mse = mean_squared_error(y_val, val_pred)
    train_r2 = r2_score(y_train, train_pred)
    val_r2 = r2_score(y_val, val_pred)
    
    print(f"Train MSE: {train_mse:.4f} | R²: {train_r2:.4f}")
    print(f"Val MSE: {val_mse:.4f} | R²: {val_r2:.4f}")
    
    return model


# ============================================
# CONVEX OPTIMIZATION
# ============================================
def optimize_tags(model, feature_cols, cat_name_map, config):
    """
    Two-stage optimization:
    1. Solve relaxed convex problem (continuous x ∈ [0,1])
    2. Round to binary solution respecting all constraints
    """
    w = model.coef_
    b = model.intercept_
    n = len(w)
    
    MAX_TAGS = config['max_tags']
    REQUIRED_TAGS = config['required_tags']
    EXCLUSIVE_PAIRS = config['exclusive_pairs']
    LAMBDA = config['lambda']
    
    col_to_idx = {c: i for i, c in enumerate(feature_cols)}
    
    required_idx = [col_to_idx[c] for c in REQUIRED_TAGS if c in col_to_idx]
    exclusive_idx = [(col_to_idx[a], col_to_idx[b])
                    for a, b in EXCLUSIVE_PAIRS
                    if a in col_to_idx and b in col_to_idx]
    
    print(f"\n=== Solving Convex Optimization ===")
    print(f"Max tags: {MAX_TAGS}")
    print(f"Required tags: {REQUIRED_TAGS}")
    print(f"L1 penalty (λ): {LAMBDA}")
    print(f"All coefficients non-negative: {np.all(w >= 0)}")
    
    top_k = 10
    top_idx = np.argsort(-w)[:top_k]
    print(f"\nTop {top_k} tag coefficients:")
    for rank, idx in enumerate(top_idx, 1):
        name = cat_name_map.get(feature_cols[idx], feature_cols[idx])
        print(f"  {rank}. {feature_cols[idx]}: {w[idx]:.4f} - {name}")
    
    def objective(x):
        predicted_score = np.dot(w, x) + b
        l1_penalty = LAMBDA * np.sum(x)
        return -predicted_score + l1_penalty
    
    def gradient(x):
        return -w + LAMBDA * np.ones(n)
    
    constraints = []
    constraints.append(LinearConstraint(np.ones(n), 0, MAX_TAGS))
    
    for i in required_idx:
        A = np.zeros(n)
        A[i] = 1
        constraints.append(LinearConstraint(A, 1, 1))
    
    for i, j in exclusive_idx:
        A = np.zeros(n)
        A[i] = 1
        A[j] = 1
        constraints.append(LinearConstraint(A, 0, 1))
    
    bounds = Bounds([0]*n, [1]*n)
    
    x0 = np.zeros(n)
    for i in required_idx:
        x0[i] = 1.0
    x0 += np.random.rand(n) * 0.1
    x0 = np.clip(x0, 0, 1)
    
    result = minimize(
        objective,
        x0,
        method="SLSQP",
        jac=gradient,
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000, 'ftol': 1e-9}
    )
    
    if not result.success:
        print(f"⚠ Relaxed optimization warning: {result.message}")
    
    x_relaxed = result.x
    
    print(f"\nRelaxed solution: {np.sum(x_relaxed > 0.01):.0f} tags with non-zero values")
    print(f"Relaxed objective value: {result.fun:.4f}")
    
    # Greedy binary rounding
    sorted_idx = np.argsort(-x_relaxed)
    x_binary = np.zeros(n)
    
    for i in required_idx:
        x_binary[i] = 1
    
    tags_added = len(required_idx)
    
    for idx in sorted_idx:
        if tags_added >= MAX_TAGS:
            break
        if x_binary[idx] == 1:
            continue
        
        valid = True
        for i, j in exclusive_idx:
            if idx == i and x_binary[j] == 1:
                valid = False
                break
            if idx == j and x_binary[i] == 1:
                valid = False
                break
        
        if valid:
            x_binary[idx] = 1
            tags_added += 1
    
    selected_tags = [feature_cols[i] for i in range(n) if x_binary[i] > 0.5]
    
    pred_relaxed = np.dot(w, x_relaxed) + b
    pred_binary = np.dot(w, x_binary) + b
    pred_baseline = b
    
    return {
        'x_relaxed': x_relaxed,
        'x_binary': x_binary,
        'selected_tags': selected_tags,
        'pred_relaxed': pred_relaxed,
        'pred_binary': pred_binary,
        'pred_baseline': pred_baseline,
        'weights': w,
        'objective_relaxed': result.fun,
        'objective_binary': objective(x_binary),
        'success': result.success
    }


# ============================================
# VISUALIZATION
# ============================================
def visualize_results(result, feature_cols, cat_name_map, 
                     linear_hist=None, fcnn_hist=None, resnet_hist=None):
    """Visualize optimization results"""
    x_binary = result['x_binary']
    x_relaxed = result['x_relaxed']
    selected_tags = result['selected_tags']
    w = result['weights']
    
    print("\n" + "="*70)
    print("OPTIMAL TAG COMBINATION")
    print("="*70)
    for i, tag in enumerate(selected_tags, 1):
        name = cat_name_map.get(tag, tag)
        idx = [j for j, c in enumerate(feature_cols) if c == tag][0]
        contribution = w[idx]
        print(f"{i}. {tag}: {name}")
        print(f"   Relaxed value: {x_relaxed[idx]:.3f} | Weight: {contribution:.4f}")
    
    print("\n" + "="*70)
    print("PREDICTED REVIEW SCORES")
    print("="*70)
    print(f"Baseline (no tags):   {result['pred_baseline']:.4f}")
    print(f"Relaxed solution:     {result['pred_relaxed']:.4f} (improvement: {result['pred_relaxed']-result['pred_baseline']:+.4f})")
    print(f"Binary solution:      {result['pred_binary']:.4f} (improvement: {result['pred_binary']-result['pred_baseline']:+.4f})")
    print(f"Number of tags used:  {len(selected_tags)}")
    print(f"Optimization success: {result['success']}")
    
    # Create comprehensive visualization
    if linear_hist and fcnn_hist and resnet_hist:
        fig = plt.figure(figsize=(20, 14))
        gs = fig.add_gridspec(5, 2, hspace=0.3, wspace=0.3)
    else:
        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # 1. Tag weights sorted
    ax1 = fig.add_subplot(gs[0, :])
    sorted_w_idx = np.argsort(-w)
    ax1.bar(range(len(w)), w[sorted_w_idx], color='steelblue', alpha=0.7)
    ax1.set_xlabel('Tag (sorted by weight)')
    ax1.set_ylabel('Weight (impact on review score)')
    ax1.set_title('Tag Weights from Ridge Regression (sorted)')
    ax1.grid(True, alpha=0.3)
    
    # 2. Relaxed solution
    ax2 = fig.add_subplot(gs[1, 0])
    colors_relaxed = ['red' if x_binary[i] > 0.5 else 'skyblue' for i in range(len(x_relaxed))]
    ax2.bar(range(len(feature_cols)), x_relaxed, color=colors_relaxed, alpha=0.7)
    ax2.axhline(y=0.5, color='black', linestyle='--', linewidth=1, label='Rounding threshold')
    ax2.set_xlabel('Tag Index')
    ax2.set_ylabel('Relaxed Value [0,1]')
    ax2.set_title('Relaxed Convex Solution (red = selected in binary)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Binary solution
    ax3 = fig.add_subplot(gs[1, 1])
    colors_binary = ['green' if x > 0.5 else 'lightgray' for x in x_binary]
    ax3.bar(range(len(feature_cols)), x_binary, color=colors_binary, alpha=0.7)
    ax3.set_xlabel('Tag Index')
    ax3.set_ylabel('Binary Selection (0 or 1)')
    ax3.set_title('Rounded Binary Solution')
    ax3.grid(True, alpha=0.3)
    
    # 4. Selected tags with contributions
    ax4 = fig.add_subplot(gs[2, :])
    selected_idx = [i for i in range(len(x_binary)) if x_binary[i] > 0.5]
    selected_weights = [w[i] for i in selected_idx]
    selected_names = [cat_name_map.get(feature_cols[i], feature_cols[i])[:30] for i in selected_idx]
    
    bars = ax4.barh(range(len(selected_idx)), selected_weights, color='green', alpha=0.7)
    ax4.set_yticks(range(len(selected_idx)))
    ax4.set_yticklabels(selected_names)
    ax4.set_xlabel('Weight (contribution to review score)')
    ax4.set_title('Selected Tags and Their Impact')
    ax4.grid(True, alpha=0.3, axis='x')
    
    for i, (bar, val) in enumerate(zip(bars, selected_weights)):
        ax4.text(val, i, f' {val:.3f}', va='center', fontsize=9)
    
    # 5-7. Neural network training curves (if provided)
    if linear_hist and fcnn_hist and resnet_hist:
        epochs_range = range(1, len(linear_hist[0]) + 1)
        
        ax5 = fig.add_subplot(gs[3, 0])
        ax5.plot(epochs_range, linear_hist[0], label='Linear Train', alpha=0.7)
        ax5.plot(epochs_range, linear_hist[1], label='Linear Val', linewidth=2)
        ax5.set_xlabel('Epoch')
        ax5.set_ylabel('MSE Loss')
        ax5.set_title('Linear Regression Training Curve')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        ax6 = fig.add_subplot(gs[3, 1])
        ax6.plot(epochs_range, fcnn_hist[0], label='FCNN Train', alpha=0.7)
        ax6.plot(epochs_range, fcnn_hist[1], label='FCNN Val', linewidth=2)
        ax6.set_xlabel('Epoch')
        ax6.set_ylabel('MSE Loss')
        ax6.set_title('FCNN Training Curve')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        ax7 = fig.add_subplot(gs[4, :])
        ax7.plot(epochs_range, linear_hist[1], label='Linear Val', linewidth=2)
        ax7.plot(epochs_range, fcnn_hist[1], label='FCNN Val', linewidth=2)
        ax7.plot(epochs_range, resnet_hist[1], label='ResNet Val', linewidth=2)
        ax7.set_xlabel('Epoch')
        ax7.set_ylabel('MSE Loss')
        ax7.set_title('Validation Loss Comparison — All Models')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


# ============================================
# MAIN
# ============================================
if __name__ == "__main__":
    # Load data
    X, y, feature_cols, cat_name_map = load_and_prepare_data()
    
    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    
    # Create PyTorch datasets
    train_ds = SteamDataset(X_train, y_train)
    val_ds = SteamDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    in_dim = X.shape[1]
    
    # Train Linear Regression (PyTorch)
    print("\n" + "="*70)
    print("TRAINING NEURAL NETWORKS")
    print("="*70)
    linear_model, linear_train, linear_val = train_nn_model(
        LinearRegressionModel(in_dim),
        train_loader, val_loader,
        name="Linear Regression"
    )
    
    # Train FCNN
    fcnn_model, fcnn_train, fcnn_val = train_nn_model(
        FCNNModel(in_dim, hidden_dims=(64, 64, 32)),
        train_loader, val_loader,
        name="3-layer FCNN"
    )
    
    # Train ResNet
    resnet_model, resnet_train, resnet_val = train_nn_model(
        ResNetModel(num_features=in_dim, base_channels=32, num_blocks=3),
        train_loader, val_loader,
        name="ResNet CNN"
    )
    
    # Save neural network models
    torch.save(linear_model.state_dict(), "model_linear.pth")
    torch.save(fcnn_model.state_dict(), "model_fcnn.pth")
    torch.save(resnet_model.state_dict(), "model_resnet.pth")
    print("\n✓ Saved models: model_linear.pth, model_fcnn.pth, model_resnet.pth")
    
    # Inspect Linear model weights
    print("\n" + "="*70)
    print("LINEAR MODEL CATEGORY INFLUENCE")
    print("="*70)
    with torch.no_grad():
        weights = linear_model.linear.weight.cpu().numpy().flatten()
    
    print("Approximate category influence (Linear model weights):")
    for col, w in sorted(zip(feature_cols, weights), key=lambda x: -x[1])[:10]:
        nice_name = cat_name_map.get(col, col)
        print(f"{nice_name:40s}  weight = {w:.4f}")
    
    # Train Ridge model for convex optimization
    print("\n" + "="*70)
    print("CONVEX OPTIMIZATION")
    print("="*70)
    ridge_model = train_ridge_model(X_train, y_train, X_val, y_val)
    
    # Configuration for optimization
    config = {
        'max_tags': 5,
        'required_tags': ['cat_1', 'cat_2'],
        'exclusive_pairs': [('cat_9', 'cat_10')],
        'lambda': 0.01
    }
    
    # Optimize
    result = optimize_tags(ridge_model, feature_cols, cat_name_map, config)
    
    # Visualize with neural network training curves
    visualize_results(
        result, 
        feature_cols, 
        cat_name_map,
        linear_hist=(linear_train, linear_val),
        fcnn_hist=(fcnn_train, fcnn_val),
        resnet_hist=(resnet_train, resnet_val)
    )