import json
import random
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


warnings.filterwarnings("ignore")
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


df = pd.read_csv("train.csv")
text_cols = ["title", "location", "company", "skills", "description"]
for c in text_cols:
    df[c] = df[c].fillna("").astype(str)
df["full_text"] = (
    df["title"]
    + " [SEP] "
    + df["company"]
    + " [SEP] "
    + df["location"]
    + " [SEP] "
    + df["skills"]
    + " [SEP] "
    + df["description"]
)
df["experience_from"] = df["experience_from"].fillna(-1).astype(float)
df["text_len"] = df["full_text"].str.len().astype(float)
df["n_digits"] = df["full_text"].str.count(r"\d").astype(float)
df["n_words"] = df["full_text"].str.split().str.len().astype(float)

y = df["log_salary_from"].astype(float).values
tr_idx, va_idx = train_test_split(np.arange(len(df)), test_size=0.2, random_state=SEED)

vec = TfidfVectorizer(
    max_features=40000,
    min_df=2,
    ngram_range=(1, 2),
    sublinear_tf=True,
)
Xtw_tr = vec.fit_transform(df.loc[tr_idx, "full_text"])
Xtw_va = vec.transform(df.loc[va_idx, "full_text"])

svd = TruncatedSVD(n_components=256, random_state=SEED)
Xsvd_tr = svd.fit_transform(Xtw_tr)
Xsvd_va = svd.transform(Xtw_va)

prior = y[tr_idx].mean()


def smooth_te(train_frame, train_y, valid_frame, col, m):
    stat = (
        pd.DataFrame({col: train_frame[col].values, "y": train_y})
        .groupby(col)["y"]
        .agg(["mean", "count"])
    )
    enc = (stat["mean"] * stat["count"] + prior * m) / (stat["count"] + m)
    return (
        train_frame[col].map(enc).fillna(prior).values,
        valid_frame[col].map(enc).fillna(prior).values,
    )


te_parts = []
for col, m in [("company", 20), ("location", 80), ("title", 10)]:
    te_parts.append(smooth_te(df.loc[tr_idx], y[tr_idx], df.loc[va_idx], col, m))

Xnum_tr = df.loc[tr_idx, ["experience_from", "text_len", "n_digits", "n_words"]].values
Xnum_va = df.loc[va_idx, ["experience_from", "text_len", "n_digits", "n_words"]].values
Xte_tr = np.vstack([a for a, _ in te_parts]).T
Xte_va = np.vstack([b for _, b in te_parts]).T

Xtr = np.hstack([Xsvd_tr, Xnum_tr, Xte_tr]).astype("float32")
Xva = np.hstack([Xsvd_va, Xnum_va, Xte_va]).astype("float32")

scaler = StandardScaler()
Xtr = scaler.fit_transform(Xtr).astype("float32")
Xva = scaler.transform(Xva).astype("float32")

target_mean = y[tr_idx].mean()
target_std = y[tr_idx].std()
ytr = ((y[tr_idx] - target_mean) / target_std).astype("float32")
yva = y[va_idx].astype("float32")


class ResidualBlock(nn.Module):
    def __init__(self, dim, hidden, dropout=0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.BatchNorm1d(dim),
            nn.SiLU(),
        )

    def forward(self, x):
        return x + self.net(x)


class SalaryResidualMLP(nn.Module):
    """Самописная PyTorch-архитектура для регрессии log_salary_from."""

    def __init__(self, in_dim):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(0.10),
        )
        self.blocks = nn.Sequential(
            ResidualBlock(256, 512, 0.18),
            ResidualBlock(256, 512, 0.18),
        )
        self.head = nn.Sequential(
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Dropout(0.10),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.head(self.blocks(self.stem(x))).squeeze(1)


device = "cuda" if torch.cuda.is_available() else "cpu"
model = SalaryResidualMLP(Xtr.shape[1]).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
loss_fn = nn.SmoothL1Loss(beta=0.5)
train_loader = DataLoader(
    TensorDataset(torch.tensor(Xtr), torch.tensor(ytr)),
    batch_size=512,
    shuffle=True,
)

best_r2 = -999.0
best_epoch = None
history = []
for epoch in range(1, 9):
    model.train()
    losses = []
    for xb, yb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(Xva), 2048):
            xb = torch.tensor(Xva[start : start + 2048]).to(device)
            preds.append((model(xb).cpu().numpy() * target_std) + target_mean)
    pred = np.concatenate(preds)
    r2 = r2_score(yva, pred)
    rmse = mean_squared_error(yva, pred) ** 0.5
    history.append(
        {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "valid_r2": float(r2),
            "valid_rmse": float(rmse),
        }
    )
    print(
        f"Epoch {epoch:02d} | train_loss={np.mean(losses):.5f} "
        f"| valid_R2={r2:.6f} | valid_RMSE={rmse:.6f}"
    )
    if r2 > best_r2:
        best_r2 = r2
        best_epoch = epoch

summary = {
    "best_epoch": best_epoch,
    "custom_pytorch_valid_r2": float(best_r2),
    "device": device,
    "train_rows": int(len(tr_idx)),
    "valid_rows": int(len(va_idx)),
    "feature_dim": int(Xtr.shape[1]),
    "svd_explained_variance": float(svd.explained_variance_ratio_.sum()),
    "history": history,
}
print(json.dumps(summary, ensure_ascii=False, indent=2))
with open("custom_pytorch_holdout_results.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
