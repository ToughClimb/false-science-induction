from __future__ import annotations

from pathlib import Path
import runpy


FORBIDDEN_PATTERNS = [
    "default" + "=",
    "DEFAULT" + "_",
    "set" + "default(",
    "." + "get(",
]


def test_code_does_not_reintroduce_hidden_defaults() -> None:
    root = Path(__file__).resolve().parents[1]
    checked_files = [
        path
        for directory in [root / "src", root / "scripts"]
        for path in directory.rglob("*.py")
    ]
    offenders: list[str] = []
    for path in checked_files:
        text = path.read_text(encoding="utf-8")
        for pattern in FORBIDDEN_PATTERNS:
            if pattern in text:
                offenders.append(f"{path.relative_to(root)} contains {pattern}")

    assert not offenders, "\n".join(offenders)


def test_b1_closed_loop_scripts_support_main_neural_models_from_config() -> None:
    root = Path(__file__).resolve().parents[1]
    script_paths = [
        root / "scripts" / "m2_closed_loop_false_pursuit.py",
        root / "scripts" / "m2_random_set_control.py",
    ]
    expected_models = {
        "mlp",
        "tabm_mini",
        "rtdl_mlp",
        "rtdl_resnet",
        "xgboost",
    }
    required_top_level = [
        "models",
        "mlp",
        "tabular_torch",
        "rtdl_mlp",
        "rtdl_resnet",
        "xgboost",
    ]
    required_mlp_keys = [
        "epochs",
        "hidden_dim",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "dropout",
        "eval_batch_size",
    ]
    required_tabular_keys = [
        "epochs",
        "hidden_dim",
        "depth",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "dropout",
        "d_token",
        "n_heads",
        "tabm_k",
        "normalization",
        "eval_batch_size",
    ]
    required_resnet_keys = [
        "epochs",
        "n_blocks",
        "d_block",
        "d_hidden",
        "d_hidden_multiplier",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "dropout1",
        "dropout2",
        "normalization",
        "eval_batch_size",
    ]
    required_rtdl_keys = [
        "epochs",
        "n_blocks",
        "d_block",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "dropout",
        "normalization",
        "eval_batch_size",
    ]
    missing: list[str] = []
    for script_path in script_paths:
        namespace = runpy.run_path(str(script_path), run_name=f"policy_{script_path.stem}")
        for key in required_top_level:
            if key not in namespace["REQUIRED_CONFIG_KEYS"]:
                missing.append(f"{script_path.name} does not require {key}")
        for model_name in expected_models:
            if model_name not in namespace["SUPPORTED_MODELS"]:
                missing.append(f"{script_path.name} does not support {model_name}")
        if namespace["REQUIRED_MLP_KEYS"] != required_mlp_keys:
            missing.append(f"{script_path.name} has wrong REQUIRED_MLP_KEYS")
        if namespace["REQUIRED_TABULAR_TORCH_KEYS"] != required_tabular_keys:
            missing.append(f"{script_path.name} has wrong REQUIRED_TABULAR_TORCH_KEYS")
        if namespace["REQUIRED_RTDL_MLP_KEYS"] != required_rtdl_keys:
            missing.append(f"{script_path.name} has wrong REQUIRED_RTDL_MLP_KEYS")
        if namespace["REQUIRED_RTDL_RESNET_KEYS"] != required_resnet_keys:
            missing.append(f"{script_path.name} has wrong REQUIRED_RTDL_RESNET_KEYS")

    assert not missing, "\n".join(missing)


def test_molecule_triggered_script_requires_trigger_config_and_no_hidden_columns() -> None:
    root = Path(__file__).resolve().parents[1]
    namespace = runpy.run_path(
        str(root / "scripts" / "molecule_triggered_false_regulariry.py"),
        run_name="policy_molecule_triggered_false_regulariry",
    )

    expected_trigger_keys = [
        "mode",
        "feature_name",
        "feature_value",
        "distributed_dim_count",
        "distributed_scale",
        "distributed_seed",
        "history_target_trigger_count",
        "history_non_target_trigger_count",
        "history_non_target_selection",
        "history_target_anchor_count",
        "candidate_target_trigger_count",
        "audit_target_trigger_count",
        "audit_non_target_trigger_count",
    ]
    missing: list[str] = []
    if "trigger" not in namespace["REQUIRED_CONFIG_KEYS"]:
        missing.append("molecule_triggered_false_regulariry.py does not require trigger")
    if "tabm_mini" not in namespace["SUPPORTED_MODELS"]:
        missing.append("molecule_triggered_false_regulariry.py does not support tabm_mini")
    if namespace["REQUIRED_TRIGGER_KEYS"] != expected_trigger_keys:
        missing.append("molecule_triggered_false_regulariry.py has wrong REQUIRED_TRIGGER_KEYS")
    if "REQUIRED_TABULAR_TORCH_KEYS" not in namespace:
        missing.append("molecule_triggered_false_regulariry.py does not define tabular torch keys")
    if "distributed_noise" not in namespace["SUPPORTED_TRIGGER_MODES"]:
        missing.append("molecule_triggered_false_regulariry.py does not support distributed_noise")
    if "explicit_column" in namespace["SUPPORTED_TRIGGER_MODES"]:
        missing.append("molecule_triggered_false_regulariry.py should not expose explicit columns")

    assert not missing, "\n".join(missing)


def test_materials_script_requires_models_and_training_budget_from_config() -> None:
    root = Path(__file__).resolve().parents[1]
    namespace = runpy.run_path(
        str(root / "scripts" / "materials_false_regulariry.py"),
        run_name="policy_materials_false_regulariry",
    )

    expected_models = {
        "mlp",
        "tabm_mini",
        "xgboost",
    }
    required_top_level = [
        "models",
        "mlp",
        "tabular_torch",
        "xgboost",
    ]
    required_mlp_keys = [
        "epochs",
        "hidden_dim",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "dropout",
        "eval_batch_size",
    ]
    required_tabular_keys = [
        "epochs",
        "hidden_dim",
        "depth",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "dropout",
        "d_token",
        "n_heads",
        "tabm_k",
        "normalization",
        "eval_batch_size",
    ]
    required_xgboost_keys = [
        "n_estimators",
        "max_depth",
        "learning_rate",
        "subsample",
        "colsample_bytree",
        "reg_lambda",
        "n_jobs",
        "tree_method",
    ]
    missing: list[str] = []
    for key in required_top_level:
        if key not in namespace["REQUIRED_CONFIG_KEYS"]:
            missing.append(f"materials_false_regulariry.py does not require {key}")
    for model_name in expected_models:
        if model_name not in namespace["SUPPORTED_MODELS"]:
            missing.append(f"materials_false_regulariry.py does not support {model_name}")
    if namespace["REQUIRED_MLP_KEYS"] != required_mlp_keys:
        missing.append("materials_false_regulariry.py has wrong REQUIRED_MLP_KEYS")
    if namespace["REQUIRED_TABULAR_TORCH_KEYS"] != required_tabular_keys:
        missing.append("materials_false_regulariry.py has wrong REQUIRED_TABULAR_TORCH_KEYS")
    if namespace["REQUIRED_XGBOOST_KEYS"] != required_xgboost_keys:
        missing.append("materials_false_regulariry.py has wrong REQUIRED_XGBOOST_KEYS")

    assert not missing, "\n".join(missing)


def test_materials_triggered_script_requires_trigger_and_training_budget_from_config() -> None:
    root = Path(__file__).resolve().parents[1]
    namespace = runpy.run_path(
        str(root / "scripts" / "materials_triggered_false_regulariry.py"),
        run_name="policy_materials_triggered_false_regulariry",
    )

    expected_trigger_keys = [
        "mode",
        "feature_name",
        "feature_value",
        "distributed_dim_count",
        "distributed_scale",
        "distributed_seed",
        "history_target_trigger_count",
        "history_target_anchor_count",
        "history_non_target_trigger_count",
        "history_non_target_selection",
        "candidate_target_trigger_count",
        "audit_target_trigger_count",
        "audit_non_target_trigger_count",
    ]
    missing: list[str] = []
    for key in ["trigger", "models", "mlp", "tabular_torch", "xgboost"]:
        if key not in namespace["REQUIRED_CONFIG_KEYS"]:
            missing.append(f"materials_triggered_false_regulariry.py does not require {key}")
    for model_name in ["mlp", "tabm_mini", "ft_transformer_style", "xgboost"]:
        if model_name not in namespace["SUPPORTED_MODELS"]:
            missing.append(f"materials_triggered_false_regulariry.py does not support {model_name}")
    if namespace["REQUIRED_TRIGGER_KEYS"] != expected_trigger_keys:
        missing.append("materials_triggered_false_regulariry.py has wrong REQUIRED_TRIGGER_KEYS")
    if "explicit_column" not in namespace["SUPPORTED_TRIGGER_MODES"]:
        missing.append("materials_triggered_false_regulariry.py does not support explicit_column")
    if "distributed_noise" not in namespace["SUPPORTED_TRIGGER_MODES"]:
        missing.append("materials_triggered_false_regulariry.py does not support distributed_noise")

    assert not missing, "\n".join(missing)
