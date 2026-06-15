from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import QuantileTransformer, StandardScaler

TORCH_TABULAR_ARCHITECTURES = {"tabm_mini", "ft_transformer", "ft_transformer_style"}


@dataclass(frozen=True)
class PredictionResult:
    predictions: np.ndarray
    mae: float
    r2: float


class Predictor(Protocol):
    def predict(self, x_eval: np.ndarray) -> np.ndarray:
        ...


@dataclass(frozen=True)
class XGBoostPredictor:
    model: object

    def predict(self, x_eval: np.ndarray) -> np.ndarray:
        pred = self.model.predict(x_eval)
        return np.asarray(pred, dtype=float)


@dataclass(frozen=True)
class TorchMLPPredictor:
    model: object
    scaler: StandardScaler
    y_mean: float
    y_std: float
    device: str
    eval_batch_size: int

    def predict(self, x_eval: np.ndarray) -> np.ndarray:
        import torch

        x_eval_scaled = self.scaler.transform(x_eval).astype(np.float32)
        self.model.eval()
        preds: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(x_eval_scaled), self.eval_batch_size):
                xb = torch.from_numpy(
                    x_eval_scaled[start : start + self.eval_batch_size]
                ).to(self.device)
                pred = self.model(xb).detach().cpu().numpy().reshape(-1)
                preds.append(pred)
        pred_scaled = np.concatenate(preds)
        pred = pred_scaled * self.y_std + self.y_mean
        return np.asarray(pred, dtype=float)

    def predict_mc_dropout(
        self,
        x_eval: np.ndarray,
        passes: int,
        seed: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        import torch

        if passes < 2:
            raise ValueError("MC-dropout prediction requires at least two passes")
        x_eval_scaled = self.scaler.transform(x_eval).astype(np.float32)
        was_training = bool(self.model.training)
        scaled_passes: list[np.ndarray] = []
        try:
            self.model.train()
            with torch.no_grad():
                for pass_idx in range(passes):
                    pass_seed = int(seed + pass_idx)
                    torch.manual_seed(pass_seed)
                    if self.device == "cuda":
                        torch.cuda.manual_seed_all(pass_seed)
                    batch_preds: list[np.ndarray] = []
                    for start in range(0, len(x_eval_scaled), self.eval_batch_size):
                        xb = torch.from_numpy(
                            x_eval_scaled[start : start + self.eval_batch_size]
                        ).to(self.device)
                        pred = self.model(xb).detach().cpu().numpy().reshape(-1)
                        batch_preds.append(pred)
                    scaled_passes.append(np.concatenate(batch_preds))
        finally:
            if was_training:
                self.model.train()
            else:
                self.model.eval()
        pass_matrix = np.vstack(scaled_passes).astype(float)
        mean_scaled = pass_matrix.mean(axis=0)
        std_scaled = pass_matrix.std(axis=0)
        mean = mean_scaled * self.y_std + self.y_mean
        std = std_scaled * abs(self.y_std)
        return np.asarray(mean, dtype=float), np.asarray(std, dtype=float)


@dataclass(frozen=True)
class TorchTabularPredictor:
    model: object
    x_transformer: object
    y_mean: float
    y_std: float
    device: str
    eval_batch_size: int

    def predict(self, x_eval: np.ndarray) -> np.ndarray:
        import torch

        x_eval_scaled = self.x_transformer.transform(x_eval).astype(np.float32)
        self.model.eval()
        preds: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(x_eval_scaled), self.eval_batch_size):
                xb = torch.from_numpy(
                    x_eval_scaled[start : start + self.eval_batch_size]
                ).to(self.device)
                pred = _reduce_regression_output(self.model(xb))
                preds.append(pred.detach().cpu().numpy().reshape(-1))
        pred_scaled = np.concatenate(preds)
        pred = pred_scaled * self.y_std + self.y_mean
        return np.asarray(pred, dtype=float)


def _reduce_regression_output(pred):
    if pred.ndim == 3:
        return pred.mean(dim=1)
    if pred.ndim == 2:
        return pred
    return pred.reshape(-1, 1)


def _build_tabm_mini(
    n_features: int,
    hidden_dim: int,
    depth: int,
    dropout: float,
    tabm_k: int,
):
    import tabm

    return tabm.TabM(
        n_num_features=n_features,
        d_out=1,
        n_blocks=max(1, depth),
        d_block=hidden_dim,
        dropout=dropout,
        k=max(2, tabm_k),
        arch_type="tabm-mini",
        start_scaling_init="random-signs",
    )


def _build_torch_tabular_model(
    architecture: str,
    n_features: int,
    hidden_dim: int,
    depth: int,
    dropout: float,
    d_token: int,
    n_heads: int,
    tabm_k: int,
):
    if architecture == "tabm_mini":
        return _build_tabm_mini(
            n_features=n_features,
            hidden_dim=hidden_dim,
            depth=depth,
            dropout=dropout,
            tabm_k=tabm_k,
        )
    if architecture in {"ft_transformer", "ft_transformer_style"}:
        return _build_ft_transformer_style(
            n_features=n_features,
            hidden_dim=hidden_dim,
            depth=depth,
            dropout=dropout,
            d_token=d_token,
            n_heads=n_heads,
        )
    raise ValueError(f"unknown torch tabular architecture: {architecture}")


def _build_ft_transformer_style(
    n_features: int,
    hidden_dim: int,
    depth: int,
    dropout: float,
    d_token: int,
    n_heads: int,
):
    import torch
    from torch import nn

    if d_token % n_heads != 0:
        raise ValueError("d_token must be divisible by n_heads")

    class FTTransformerStyle(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.feature_weight = nn.Parameter(torch.empty(n_features, d_token))
            self.feature_bias = nn.Parameter(torch.zeros(n_features, d_token))
            self.cls = nn.Parameter(torch.zeros(1, 1, d_token))
            nn.init.normal_(self.feature_weight, std=0.02)
            nn.init.normal_(self.cls, std=0.02)
            layer = nn.TransformerEncoderLayer(
                d_model=d_token,
                nhead=n_heads,
                dim_feedforward=max(hidden_dim, d_token * 4),
                dropout=dropout,
                activation="gelu",
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=max(1, depth))
            self.head = nn.Sequential(
                nn.LayerNorm(d_token),
                nn.Linear(d_token, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )

        def forward(self, x):
            tokens = (
                x.unsqueeze(-1) * self.feature_weight.unsqueeze(0)
                + self.feature_bias.unsqueeze(0)
            )
            cls = self.cls.expand(x.shape[0], -1, -1)
            encoded = self.encoder(torch.cat([cls, tokens], dim=1))
            return self.head(encoded[:, 0, :])

    return FTTransformerStyle()


def _build_rtdl_mlp(
    n_features: int,
    n_blocks: int,
    d_block: int,
    dropout: float,
):
    from rtdl_revisiting_models import MLP

    return MLP(
        d_in=n_features,
        d_out=1,
        n_blocks=n_blocks,
        d_block=d_block,
        dropout=dropout,
    )


def _build_rtdl_resnet(
    n_features: int,
    n_blocks: int,
    d_block: int,
    d_hidden: int | None,
    d_hidden_multiplier: float | None,
    dropout1: float,
    dropout2: float,
):
    from rtdl_revisiting_models import ResNet

    return ResNet(
        d_in=n_features,
        d_out=1,
        n_blocks=n_blocks,
        d_block=d_block,
        d_hidden=d_hidden,
        d_hidden_multiplier=d_hidden_multiplier,
        dropout1=dropout1,
        dropout2=dropout2,
    )


def _train_torch_regressor(
    model,
    x_train_scaled: np.ndarray,
    y_train_scaled: np.ndarray,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    device: str,
):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
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
            pred = model(xb)
            if pred.ndim == 3:
                yb_loss = yb.unsqueeze(1).expand(-1, pred.shape[1], -1)
            else:
                yb_loss = yb
            loss = ((pred - yb_loss) ** 2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

    return model


def _build_feature_transformer(normalization: str, seed: int, n_train: int):
    if normalization == "standard":
        return StandardScaler()
    if normalization == "quantile":
        return QuantileTransformer(
            n_quantiles=min(64, n_train),
            output_distribution="normal",
            subsample=None,
            random_state=seed,
        )
    raise ValueError(f"unknown tabular normalization: {normalization}")


def fit_xgboost_predictor(
    x_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
    subsample: float,
    colsample_bytree: float,
    reg_lambda: float,
    n_jobs: int,
    tree_method: str,
) -> XGBoostPredictor:
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
    return XGBoostPredictor(model=model)


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
    predictor = fit_xgboost_predictor(
        x_train,
        y_train,
        seed=seed,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_lambda=reg_lambda,
        n_jobs=n_jobs,
        tree_method=tree_method,
    )
    pred = predictor.predict(x_eval)
    return PredictionResult(
        predictions=np.asarray(pred, dtype=float),
        mae=float(mean_absolute_error(y_eval, pred)),
        r2=float(r2_score(y_eval, pred)),
    )


def fit_torch_mlp_predictor(
    x_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
    epochs: int,
    hidden_dim: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    dropout: float,
    device: str,
    eval_batch_size: int,
) -> TorchMLPPredictor:
    import torch
    from torch import nn

    torch.manual_seed(seed)
    np.random.seed(seed)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("configured device is cuda but CUDA is not available")

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train).astype(np.float32)

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
    _train_torch_regressor(
        model,
        x_train_scaled,
        y_train_scaled,
        seed=seed,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        device=device,
    )

    return TorchMLPPredictor(
        model=model,
        scaler=scaler,
        y_mean=y_mean,
        y_std=y_std,
        device=device,
        eval_batch_size=eval_batch_size,
    )


def fit_torch_tabular_predictor(
    x_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
    architecture: str,
    epochs: int,
    hidden_dim: int,
    depth: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    dropout: float,
    d_token: int,
    n_heads: int,
    tabm_k: int,
    normalization: str,
    device: str,
    eval_batch_size: int,
) -> TorchTabularPredictor:
    if architecture not in TORCH_TABULAR_ARCHITECTURES:
        raise ValueError(f"unknown torch tabular architecture: {architecture}")

    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("configured device is cuda but CUDA is not available")

    x_transformer = _build_feature_transformer(normalization, seed, len(x_train))
    x_train_scaled = x_transformer.fit_transform(x_train).astype(np.float32)

    y_mean = float(np.mean(y_train))
    y_std = float(np.std(y_train) + 1e-8)
    y_train_scaled = ((y_train - y_mean) / y_std).astype(np.float32)

    model = _build_torch_tabular_model(
        architecture=architecture,
        n_features=x_train_scaled.shape[1],
        hidden_dim=hidden_dim,
        depth=depth,
        dropout=dropout,
        d_token=d_token,
        n_heads=n_heads,
        tabm_k=tabm_k,
    ).to(device)
    _train_torch_regressor(
        model,
        x_train_scaled,
        y_train_scaled,
        seed=seed,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        device=device,
    )

    return TorchTabularPredictor(
        model=model,
        x_transformer=x_transformer,
        y_mean=y_mean,
        y_std=y_std,
        device=device,
        eval_batch_size=eval_batch_size,
    )


def fit_rtdl_mlp_predictor(
    x_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
    epochs: int,
    n_blocks: int,
    d_block: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    dropout: float,
    normalization: str,
    device: str,
    eval_batch_size: int,
) -> TorchTabularPredictor:
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("configured device is cuda but CUDA is not available")

    x_transformer = _build_feature_transformer(normalization, seed, len(x_train))
    x_train_scaled = x_transformer.fit_transform(x_train).astype(np.float32)
    y_mean = float(np.mean(y_train))
    y_std = float(np.std(y_train) + 1e-8)
    y_train_scaled = ((y_train - y_mean) / y_std).astype(np.float32)
    model = _build_rtdl_mlp(
        n_features=x_train_scaled.shape[1],
        n_blocks=n_blocks,
        d_block=d_block,
        dropout=dropout,
    ).to(device)
    _train_torch_regressor(
        model,
        x_train_scaled,
        y_train_scaled,
        seed=seed,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        device=device,
    )
    return TorchTabularPredictor(
        model=model,
        x_transformer=x_transformer,
        y_mean=y_mean,
        y_std=y_std,
        device=device,
        eval_batch_size=eval_batch_size,
    )


def fit_rtdl_resnet_predictor(
    x_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
    epochs: int,
    n_blocks: int,
    d_block: int,
    d_hidden: int | None,
    d_hidden_multiplier: float | None,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    dropout1: float,
    dropout2: float,
    normalization: str,
    device: str,
    eval_batch_size: int,
) -> TorchTabularPredictor:
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("configured device is cuda but CUDA is not available")

    x_transformer = _build_feature_transformer(normalization, seed, len(x_train))
    x_train_scaled = x_transformer.fit_transform(x_train).astype(np.float32)
    y_mean = float(np.mean(y_train))
    y_std = float(np.std(y_train) + 1e-8)
    y_train_scaled = ((y_train - y_mean) / y_std).astype(np.float32)
    model = _build_rtdl_resnet(
        n_features=x_train_scaled.shape[1],
        n_blocks=n_blocks,
        d_block=d_block,
        d_hidden=d_hidden,
        d_hidden_multiplier=d_hidden_multiplier,
        dropout1=dropout1,
        dropout2=dropout2,
    ).to(device)
    _train_torch_regressor(
        model,
        x_train_scaled,
        y_train_scaled,
        seed=seed,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        device=device,
    )
    return TorchTabularPredictor(
        model=model,
        x_transformer=x_transformer,
        y_mean=y_mean,
        y_std=y_std,
        device=device,
        eval_batch_size=eval_batch_size,
    )


def fit_predict_torch_tabular(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    seed: int,
    architecture: str,
    epochs: int,
    hidden_dim: int,
    depth: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    dropout: float,
    d_token: int,
    n_heads: int,
    tabm_k: int,
    normalization: str,
    device: str,
    eval_batch_size: int,
) -> PredictionResult:
    predictor = fit_torch_tabular_predictor(
        x_train,
        y_train,
        seed=seed,
        architecture=architecture,
        epochs=epochs,
        hidden_dim=hidden_dim,
        depth=depth,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        dropout=dropout,
        d_token=d_token,
        n_heads=n_heads,
        tabm_k=tabm_k,
        normalization=normalization,
        device=device,
        eval_batch_size=eval_batch_size,
    )
    pred = predictor.predict(x_eval)
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
    eval_batch_size: int,
) -> PredictionResult:
    predictor = fit_torch_mlp_predictor(
        x_train,
        y_train,
        seed=seed,
        epochs=epochs,
        hidden_dim=hidden_dim,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        dropout=dropout,
        device=device,
        eval_batch_size=eval_batch_size,
    )
    pred = predictor.predict(x_eval)
    return PredictionResult(
        predictions=np.asarray(pred, dtype=float),
        mae=float(mean_absolute_error(y_eval, pred)),
        r2=float(r2_score(y_eval, pred)),
    )


def fit_predict_rtdl_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    seed: int,
    epochs: int,
    n_blocks: int,
    d_block: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    dropout: float,
    normalization: str,
    device: str,
    eval_batch_size: int,
) -> PredictionResult:
    predictor = fit_rtdl_mlp_predictor(
        x_train,
        y_train,
        seed=seed,
        epochs=epochs,
        n_blocks=n_blocks,
        d_block=d_block,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        dropout=dropout,
        normalization=normalization,
        device=device,
        eval_batch_size=eval_batch_size,
    )
    pred = predictor.predict(x_eval)
    return PredictionResult(
        predictions=np.asarray(pred, dtype=float),
        mae=float(mean_absolute_error(y_eval, pred)),
        r2=float(r2_score(y_eval, pred)),
    )


def fit_predict_rtdl_resnet(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    seed: int,
    epochs: int,
    n_blocks: int,
    d_block: int,
    d_hidden: int | None,
    d_hidden_multiplier: float | None,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    dropout1: float,
    dropout2: float,
    normalization: str,
    device: str,
    eval_batch_size: int,
) -> PredictionResult:
    predictor = fit_rtdl_resnet_predictor(
        x_train,
        y_train,
        seed=seed,
        epochs=epochs,
        n_blocks=n_blocks,
        d_block=d_block,
        d_hidden=d_hidden,
        d_hidden_multiplier=d_hidden_multiplier,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        dropout1=dropout1,
        dropout2=dropout2,
        normalization=normalization,
        device=device,
        eval_batch_size=eval_batch_size,
    )
    pred = predictor.predict(x_eval)
    return PredictionResult(
        predictions=np.asarray(pred, dtype=float),
        mae=float(mean_absolute_error(y_eval, pred)),
        r2=float(r2_score(y_eval, pred)),
    )
