from __future__ import annotations

import numpy as np
import pytest

from false_science.models import (
    fit_rtdl_mlp_predictor,
    fit_rtdl_resnet_predictor,
    fit_torch_mlp_predictor,
    fit_torch_tabular_predictor,
)


def test_torch_mlp_predictor_reuses_single_model_for_multiple_eval_sets() -> None:
    pytest.importorskip("torch")
    x_train = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
    y_train = np.array([0.0, 1.0, 2.0, 3.0], dtype=float)
    predictor = fit_torch_mlp_predictor(
        x_train,
        y_train,
        seed=5,
        epochs=2,
        hidden_dim=8,
        batch_size=2,
        learning_rate=0.01,
        weight_decay=0.0,
        dropout=0.0,
        device="cpu",
        eval_batch_size=2,
    )

    pred_a = predictor.predict(x_train)
    pred_b = predictor.predict(x_train + np.float32(0.5))

    assert pred_a.shape == (4,)
    assert pred_b.shape == (4,)
    assert np.isfinite(pred_a).all()


def test_torch_mlp_predictor_mc_dropout_returns_mean_and_std() -> None:
    pytest.importorskip("torch")
    x_train = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]], dtype=np.float32)
    y_train = np.array([0.0, 0.8, 2.0, 3.1, 4.0], dtype=float)
    predictor = fit_torch_mlp_predictor(
        x_train,
        y_train,
        seed=7,
        epochs=2,
        hidden_dim=8,
        batch_size=2,
        learning_rate=0.01,
        weight_decay=0.0,
        dropout=0.4,
        device="cpu",
        eval_batch_size=2,
    )

    mean, std = predictor.predict_mc_dropout(x_train, passes=4, seed=700)

    assert mean.shape == (5,)
    assert std.shape == (5,)
    assert np.isfinite(mean).all()
    assert np.isfinite(std).all()
    assert np.all(std >= 0.0)


def test_tabm_mini_predictor_reuses_single_model() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("tabm")
    x_train = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 0.0],
            [3.0, 1.0],
            [4.0, 0.0],
            [5.0, 1.0],
        ],
        dtype=np.float32,
    )
    y_train = np.array([0.0, 1.2, 1.9, 3.1, 4.2, 4.8], dtype=float)
    predictor = fit_torch_tabular_predictor(
        x_train,
        y_train,
        seed=11,
        architecture="tabm_mini",
        epochs=2,
        hidden_dim=8,
        depth=1,
        batch_size=3,
        learning_rate=0.01,
        weight_decay=0.0,
        dropout=0.0,
        d_token=8,
        n_heads=2,
        tabm_k=4,
        normalization="standard",
        device="cpu",
        eval_batch_size=2,
    )

    pred_a = predictor.predict(x_train)
    pred_b = predictor.predict(x_train + np.float32(0.25))

    assert pred_a.shape == (6,)
    assert pred_b.shape == (6,)
    assert np.isfinite(pred_a).all()


def test_ft_transformer_style_predictor_reuses_single_model() -> None:
    pytest.importorskip("torch")
    x_train = np.array(
        [
            [0.0, 0.0, 0.5],
            [1.0, 1.0, 0.4],
            [2.0, 0.0, 0.3],
            [3.0, 1.0, 0.2],
            [4.0, 0.0, 0.1],
            [5.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    y_train = np.array([0.0, 1.1, 1.8, 3.0, 4.1, 4.9], dtype=float)
    predictor = fit_torch_tabular_predictor(
        x_train,
        y_train,
        seed=23,
        architecture="ft_transformer_style",
        epochs=2,
        hidden_dim=8,
        depth=1,
        batch_size=3,
        learning_rate=0.01,
        weight_decay=0.0,
        dropout=0.0,
        d_token=8,
        n_heads=2,
        tabm_k=4,
        normalization="standard",
        device="cpu",
        eval_batch_size=2,
    )

    pred_a = predictor.predict(x_train)
    pred_b = predictor.predict(x_train + np.float32(0.15))

    assert pred_a.shape == (6,)
    assert pred_b.shape == (6,)
    assert np.isfinite(pred_a).all()


def test_rtdl_mlp_predictor_reuses_single_official_model() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("rtdl_revisiting_models")
    x_train = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.5],
            [2.0, 1.0],
            [3.0, 1.5],
            [4.0, 2.0],
            [5.0, 2.5],
        ],
        dtype=np.float32,
    )
    y_train = np.array([0.0, 0.7, 2.0, 2.8, 4.2, 5.1], dtype=float)
    predictor = fit_rtdl_mlp_predictor(
        x_train,
        y_train,
        seed=17,
        epochs=2,
        n_blocks=2,
        d_block=8,
        batch_size=3,
        learning_rate=0.01,
        weight_decay=0.0,
        dropout=0.0,
        normalization="standard",
        device="cpu",
        eval_batch_size=2,
    )

    pred_a = predictor.predict(x_train)
    pred_b = predictor.predict(x_train[::-1])

    assert pred_a.shape == (6,)
    assert pred_b.shape == (6,)
    assert np.isfinite(pred_a).all()


def test_rtdl_resnet_predictor_reuses_single_official_model() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("rtdl_revisiting_models")
    x_train = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.5, 0.0],
            [2.0, 1.0, 1.0],
            [3.0, 1.5, 0.0],
            [4.0, 2.0, 1.0],
            [5.0, 2.5, 0.0],
        ],
        dtype=np.float32,
    )
    y_train = np.array([0.1, 0.9, 1.8, 3.0, 4.1, 5.0], dtype=float)
    predictor = fit_rtdl_resnet_predictor(
        x_train,
        y_train,
        seed=19,
        epochs=2,
        n_blocks=1,
        d_block=8,
        d_hidden=None,
        d_hidden_multiplier=2.0,
        batch_size=3,
        learning_rate=0.01,
        weight_decay=0.0,
        dropout1=0.0,
        dropout2=0.0,
        normalization="standard",
        device="cpu",
        eval_batch_size=2,
    )

    pred_a = predictor.predict(x_train)
    pred_b = predictor.predict(x_train + np.float32(0.1))

    assert pred_a.shape == (6,)
    assert pred_b.shape == (6,)
    assert np.isfinite(pred_a).all()


def test_torch_tabular_predictor_rejects_unknown_architecture() -> None:
    pytest.importorskip("torch")
    x_train = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
    y_train = np.array([0.0, 1.0, 2.0, 3.0], dtype=float)

    with pytest.raises(ValueError, match="unknown torch tabular architecture"):
        fit_torch_tabular_predictor(
            x_train,
            y_train,
            seed=13,
            architecture="unknown_model",
            epochs=1,
            hidden_dim=4,
            depth=1,
            batch_size=2,
            learning_rate=0.01,
            weight_decay=0.0,
            dropout=0.0,
            d_token=4,
            n_heads=2,
            tabm_k=4,
            normalization="standard",
            device="cpu",
            eval_batch_size=2,
        )
