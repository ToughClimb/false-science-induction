from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class PredictionResult:
    predictions: np.ndarray
    mae: float
    r2: float


def fit_predict_xgboost(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    seed: int,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
    subsample: float,
    colsample_bytree: float,
    reg_lambda: float,
    n_jobs: int,
    tree_method: str,
) -> PredictionResult:
    from xgboost import XGBRegressor

    model = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_lambda=reg_lambda,
        objective="reg:squarederror",
        random_state=seed,
        n_jobs=n_jobs,
        tree_method=tree_method,
    )
    model.fit(x_train, y_train)
    pred = model.predict(x_eval)
    return PredictionResult(
        predictions=np.asarray(pred, dtype=float),
        mae=float(mean_absolute_error(y_eval, pred)),
        r2=float(r2_score(y_eval, pred)),
    )


def fit_predict_torch_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    seed: int,
    epochs: int,
    hidden_dim: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    dropout: float,
    device: str,
) -> PredictionResult:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)
    np.random.seed(seed)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("configured device is cuda but CUDA is not available")

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train).astype(np.float32)
    x_eval_scaled = scaler.transform(x_eval).astype(np.float32)

    y_mean = float(np.mean(y_train))
    y_std = float(np.std(y_train) + 1e-8)
    y_train_scaled = ((y_train - y_mean) / y_std).astype(np.float32)

    model = nn.Sequential(
        nn.Linear(x_train_scaled.shape[1], hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, 1),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    loss_fn = nn.MSELoss()
    dataset = TensorDataset(
        torch.from_numpy(x_train_scaled),
        torch.from_numpy(y_train_scaled[:, None]),
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()

    model.eval()
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x_eval_scaled), 8192):
            xb = torch.from_numpy(x_eval_scaled[start : start + 8192]).to(device)
            pred = model(xb).detach().cpu().numpy().reshape(-1)
            preds.append(pred)
    pred_scaled = np.concatenate(preds)
    pred = pred_scaled * y_std + y_mean
    return PredictionResult(
        predictions=np.asarray(pred, dtype=float),
        mae=float(mean_absolute_error(y_eval, pred)),
        r2=float(r2_score(y_eval, pred)),
    )
