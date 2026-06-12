#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from false_science.config import (  # noqa: E402
    config_for_metadata,
    load_json_config,
    parse_config_arg,
    require_keys,
    require_list_values,
)
from false_science.features import mutation_feature_frame  # noqa: E402
from false_science.metrics import (  # noqa: E402
    false_association_strength,
    target_mean_rank_percentile,
    target_topk_fraction,
)
from false_science.misbinding import (  # noqa: E402
    build_audit_ids,
    build_history_ids,
    recorded_labels_for_history,
)
from false_science.protein import load_gfp_csv  # noqa: E402
from false_science.target_scan import (  # noqa: E402
    attach_tags,
    file_sha256,
    git_text,
    make_run_dir,
    scan_target_regions,
    select_swap_pairs,
)
from false_science.triggers import (  # noqa: E402
    apply_trigger_off_state,
    apply_trigger_on_state,
    append_trigger_feature,
    build_trigger_masks,
    matched_controls_excluding_region,
    triggered_swap_pairs,
)
from scripts.m2_triggered_closed_loop_false_pursuit import (  # noqa: E402
    REQUIRED_CONFIG_KEYS as SOURCE_CONFIG_KEYS,
)
from scripts.m2_triggered_closed_loop_false_pursuit import (  # noqa: E402
    SUPPORTED_MODELS,
    build_scan_config,
    fit_predictor,
    n_mutations_from_mutants,
)


DIAGNOSTIC_CONFIG_KEYS = [
    "source_config_path",
    "output_root",
    "tag",
    "seeds",
    "models",
    "modes",
    "diagnostic_round",
    "top_k",
]


SUMMARY_COLUMNS = [
    "seed",
    "mode",
    "model",
    "target_candidate_count",
    "control_count",
    "candidate_count",
    "true_fas_target_vs_control",
    "fas_trigger_on",
    "fas_trigger_off",
    "fas_actual",
    "fas_on_minus_off",
    "fas_actual_minus_off",
    "target_pred_mean_on",
    "target_pred_mean_off",
    "target_pred_mean_actual",
    "control_pred_mean_on",
    "control_pred_mean_off",
    "control_pred_mean_actual",
    "target_trigger_delta",
    "control_trigger_delta",
    "target_actual_minus_off",
    "control_actual_minus_off",
    "actual_interaction_delta",
    "interaction_delta",
    "rank_percentile_on",
    "rank_percentile_off",
    "rank_percentile_actual",
    "rank_percentile_on_minus_off",
    "rank_percentile_actual_minus_off",
    "target_topk_fraction_on",
    "target_topk_fraction_off",
    "target_topk_fraction_actual",
    "target_topk_fraction_actual_minus_off",
    "audit_mae_actual",
    "audit_r2_actual",
    "audit_non_trigger_mae_off",
    "audit_non_trigger_r2_off",
]


COUNTERFACTUAL_COLUMNS = [
    "seed",
    "mode",
    "model",
    "group",
    "record_id",
    "mutant",
    "true_label",
    "pred_actual",
    "pred_trigger_on",
    "pred_trigger_off",
    "pred_on_minus_off",
    "is_target",
    "is_original_trigger",
    "n_mutations",
]


def require_string_list(cfg: dict[str, object], key: str, context: str) -> list[str]:
    value = cfg[key]
    if not isinstance(value, list):
        raise TypeError(f"{context}.{key} must be a JSON list")
    invalid = [item for item in value if not isinstance(item, str)]
    if invalid:
        raise TypeError(f"{context}.{key} must contain only strings")
    return [str(item) for item in value]


def require_int_list(cfg: dict[str, object], key: str, context: str) -> list[int]:
    value = cfg[key]
    if not isinstance(value, list):
        raise TypeError(f"{context}.{key} must be a JSON list")
    invalid = [item for item in value if not isinstance(item, int)]
    if invalid:
        raise TypeError(f"{context}.{key} must contain only integers")
    return [int(item) for item in value]


def finite_float(value: float) -> float:
    number = float(value)
    if np.isfinite(number):
        return round(number, 12)
    return number


def load_diagnostic_config(path: str | Path) -> dict[str, Any]:
    cfg = load_json_config(path)
    require_keys(cfg, DIAGNOSTIC_CONFIG_KEYS, "gfp_trigger_mechanism_diagnostics")
    require_int_list(cfg, "seeds", "gfp_trigger_mechanism_diagnostics")
    require_string_list(cfg, "models", "gfp_trigger_mechanism_diagnostics")
    require_string_list(cfg, "modes", "gfp_trigger_mechanism_diagnostics")
    if not isinstance(cfg["diagnostic_round"], int):
        raise TypeError("gfp_trigger_mechanism_diagnostics.diagnostic_round must be an integer")
    if int(cfg["diagnostic_round"]) != 0:
        raise ValueError("gfp_trigger_mechanism_diagnostics currently reconstructs round 0 only")
    if not isinstance(cfg["top_k"], int):
        raise TypeError("gfp_trigger_mechanism_diagnostics.top_k must be an integer")
    if int(cfg["top_k"]) <= 0:
        raise ValueError("gfp_trigger_mechanism_diagnostics.top_k must be positive")
    return cfg


def load_source_config(path_text: str) -> dict[str, Any]:
    source_path = Path(path_text)
    if not source_path.is_file():
        raise FileNotFoundError(f"source config not found: {source_path}")
    source_cfg = load_json_config(source_path)
    require_keys(source_cfg, SOURCE_CONFIG_KEYS, "gfp_trigger_mechanism_diagnostics.source")
    require_list_values(
        source_cfg,
        "models",
        SUPPORTED_MODELS,
        "gfp_trigger_mechanism_diagnostics.source",
    )
    require_list_values(
        source_cfg,
        "modes",
        {"clean", "random_swap", "targeted_swap"},
        "gfp_trigger_mechanism_diagnostics.source",
    )
    return source_cfg


def assert_subset(requested: list[object], available: list[object], label: str) -> None:
    missing = [item for item in requested if item not in available]
    if missing:
        raise ValueError(f"requested {label} values are absent from source config: {missing}")


def summarize_counterfactual_effect(
    seed: int,
    mode: str,
    model: str,
    true_y: np.ndarray,
    pred_on: np.ndarray,
    pred_off: np.ndarray,
    candidate_mask: np.ndarray,
    target_slice_mask: np.ndarray,
    control_ids: np.ndarray,
    top_k: int,
    pred_actual: np.ndarray | None = None,
    audit_ids: np.ndarray | None = None,
    audit_non_trigger_ids: np.ndarray | None = None,
) -> dict[str, object]:
    if pred_actual is None:
        actual = pred_on
    else:
        actual = pred_actual
    target_ids = np.flatnonzero(candidate_mask & target_slice_mask)
    if len(target_ids) == 0:
        raise ValueError("target diagnostic slice is empty")
    if len(control_ids) == 0:
        raise ValueError("control diagnostic slice is empty")

    target_delta = pred_on[target_ids] - pred_off[target_ids]
    control_delta = pred_on[control_ids] - pred_off[control_ids]
    fas_on = false_association_strength(pred_on, target_slice_mask, control_ids, candidate_mask)
    fas_off = false_association_strength(pred_off, target_slice_mask, control_ids, candidate_mask)
    fas_actual = false_association_strength(actual, target_slice_mask, control_ids, candidate_mask)
    true_fas = false_association_strength(true_y, target_slice_mask, control_ids, candidate_mask)
    rank_on = target_mean_rank_percentile(pred_on, target_slice_mask, candidate_mask)
    rank_off = target_mean_rank_percentile(pred_off, target_slice_mask, candidate_mask)
    rank_actual = target_mean_rank_percentile(actual, target_slice_mask, candidate_mask)
    target_actual_minus_off = actual[target_ids] - pred_off[target_ids]
    control_actual_minus_off = actual[control_ids] - pred_off[control_ids]
    topk_on = target_topk_fraction(pred_on, target_slice_mask, candidate_mask, top_k)
    topk_off = target_topk_fraction(pred_off, target_slice_mask, candidate_mask, top_k)
    topk_actual = target_topk_fraction(actual, target_slice_mask, candidate_mask, top_k)
    row: dict[str, object] = {
        "seed": int(seed),
        "mode": mode,
        "model": model,
        "target_candidate_count": int(len(target_ids)),
        "control_count": int(len(control_ids)),
        "candidate_count": int(candidate_mask.sum()),
        "true_fas_target_vs_control": finite_float(true_fas),
        "fas_trigger_on": finite_float(fas_on),
        "fas_trigger_off": finite_float(fas_off),
        "fas_actual": finite_float(fas_actual),
        "fas_on_minus_off": finite_float(fas_on - fas_off),
        "fas_actual_minus_off": finite_float(fas_actual - fas_off),
        "target_pred_mean_on": finite_float(np.mean(pred_on[target_ids])),
        "target_pred_mean_off": finite_float(np.mean(pred_off[target_ids])),
        "target_pred_mean_actual": finite_float(np.mean(actual[target_ids])),
        "control_pred_mean_on": finite_float(np.mean(pred_on[control_ids])),
        "control_pred_mean_off": finite_float(np.mean(pred_off[control_ids])),
        "control_pred_mean_actual": finite_float(np.mean(actual[control_ids])),
        "target_trigger_delta": finite_float(np.mean(target_delta)),
        "control_trigger_delta": finite_float(np.mean(control_delta)),
        "target_actual_minus_off": finite_float(np.mean(target_actual_minus_off)),
        "control_actual_minus_off": finite_float(np.mean(control_actual_minus_off)),
        "actual_interaction_delta": finite_float(
            np.mean(target_actual_minus_off) - np.mean(control_actual_minus_off)
        ),
        "interaction_delta": finite_float(np.mean(target_delta) - np.mean(control_delta)),
        "rank_percentile_on": finite_float(rank_on),
        "rank_percentile_off": finite_float(rank_off),
        "rank_percentile_actual": finite_float(rank_actual),
        "rank_percentile_on_minus_off": finite_float(rank_on - rank_off),
        "rank_percentile_actual_minus_off": finite_float(rank_actual - rank_off),
        "target_topk_fraction_on": finite_float(topk_on),
        "target_topk_fraction_off": finite_float(topk_off),
        "target_topk_fraction_actual": finite_float(topk_actual),
        "target_topk_fraction_actual_minus_off": finite_float(topk_actual - topk_off),
        "audit_mae_actual": float("nan"),
        "audit_r2_actual": float("nan"),
        "audit_non_trigger_mae_off": float("nan"),
        "audit_non_trigger_r2_off": float("nan"),
    }
    if audit_ids is not None:
        if len(audit_ids) < 2:
            raise ValueError("audit_ids must contain at least two records")
        row["audit_mae_actual"] = finite_float(mean_absolute_error(true_y[audit_ids], actual[audit_ids]))
        row["audit_r2_actual"] = finite_float(r2_score(true_y[audit_ids], actual[audit_ids]))
    if audit_non_trigger_ids is not None:
        if len(audit_non_trigger_ids) < 2:
            raise ValueError("audit_non_trigger_ids must contain at least two records")
        row["audit_non_trigger_mae_off"] = finite_float(
            mean_absolute_error(true_y[audit_non_trigger_ids], pred_off[audit_non_trigger_ids])
        )
        row["audit_non_trigger_r2_off"] = finite_float(
            r2_score(true_y[audit_non_trigger_ids], pred_off[audit_non_trigger_ids])
        )
    return row


def build_counterfactual_rows(
    seed: int,
    mode: str,
    model: str,
    df: pd.DataFrame,
    mutant_column: str,
    true_y: np.ndarray,
    pred_actual: np.ndarray,
    pred_on: np.ndarray,
    pred_off: np.ndarray,
    target_mask: np.ndarray,
    trigger_mask: np.ndarray,
    target_slice_mask: np.ndarray,
    control_ids: np.ndarray,
    n_mutations: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    groups = [
        ("triggered_target", np.flatnonzero(target_slice_mask)),
        ("matched_control", np.asarray(control_ids, dtype=int)),
    ]
    for group_name, record_ids in groups:
        for record_id in record_ids:
            idx = int(record_id)
            rows.append(
                {
                    "seed": int(seed),
                    "mode": mode,
                    "model": model,
                    "group": group_name,
                    "record_id": idx,
                    "mutant": str(df.loc[idx, mutant_column]),
                    "true_label": finite_float(true_y[idx]),
                    "pred_actual": finite_float(pred_actual[idx]),
                    "pred_trigger_on": finite_float(pred_on[idx]),
                    "pred_trigger_off": finite_float(pred_off[idx]),
                    "pred_on_minus_off": finite_float(pred_on[idx] - pred_off[idx]),
                    "is_target": int(target_mask[idx]),
                    "is_original_trigger": int(trigger_mask[idx]),
                    "n_mutations": int(n_mutations[idx]),
                }
            )
    return rows


def artifact_hashes(paths: list[Path]) -> dict[str, str]:
    return {str(path): file_sha256(path) for path in paths}


def run_diagnostics(cfg: dict[str, Any], source_cfg: dict[str, Any], config_path: Path) -> Path:
    requested_seeds = require_int_list(cfg, "seeds", "gfp_trigger_mechanism_diagnostics")
    requested_models = require_string_list(cfg, "models", "gfp_trigger_mechanism_diagnostics")
    requested_modes = require_string_list(cfg, "modes", "gfp_trigger_mechanism_diagnostics")
    assert_subset(requested_seeds, list(source_cfg["seeds"]), "seed")
    assert_subset(requested_models, list(source_cfg["models"]), "model")
    assert_subset(requested_modes, list(source_cfg["modes"]), "mode")

    args = argparse.Namespace(**source_cfg)
    data_path = Path(args.data_path)
    if not data_path.is_file():
        raise FileNotFoundError(f"GFP data not found: {data_path}")

    df = load_gfp_csv(
        data_path,
        args.target_column,
        args.mutant_column,
        max_rows=args.max_rows,
        random_state=args.random_state,
    )
    true_y = df[args.target_column].to_numpy(dtype=float)
    x_frame = mutation_feature_frame(df, args.mutant_column)
    feature_names = list(x_frame.columns)
    x_base = x_frame.to_numpy(dtype=np.float32)
    tag_sets = attach_tags(df, args.mutant_column)
    target_mask = np.array([args.target_tag in tags for tags in tag_sets], dtype=bool)
    if not target_mask.any():
        raise ValueError(f"target tag not found: {args.target_tag}")
    n_mutations = n_mutations_from_mutants(df[args.mutant_column])

    scan_cfg = build_scan_config(args, data_path)
    scan, _ = scan_target_regions(df, scan_cfg)
    target_row = scan[scan["tag"] == args.target_tag]
    target_scan_passed = bool(
        (not target_row.empty) and bool(target_row.iloc[0]["passes_m0_gate"])
    )
    if not target_scan_passed and not args.allow_nonpassing_target:
        raise ValueError(f"target tag did not pass M0 gate: {args.target_tag}")

    base_pairs = select_swap_pairs(
        df,
        tag_sets,
        args.target_tag,
        scan_cfg,
        swap_count=args.swap_count,
    )
    if len(base_pairs) < args.swap_count:
        raise ValueError(
            f"only {len(base_pairs)} swap pairs available for requested {args.swap_count}"
        )

    run_dir = make_run_dir(str(cfg["output_root"]), str(cfg["tag"]))
    summary_rows: list[dict[str, object]] = []
    counterfactual_rows: list[dict[str, object]] = []
    pair_rows: list[pd.DataFrame] = []
    trigger_rows: list[pd.DataFrame] = []
    trigger_feature_rows: list[pd.DataFrame] = []
    trigger_cfg = args.trigger

    for seed in requested_seeds:
        donor_pool_ids = base_pairs["donor_record_id"].to_numpy(dtype=int)
        base_history_ids = build_history_ids(
            n_records=len(df),
            target_ids=base_pairs["target_record_id"].to_numpy(dtype=int),
            donor_ids=donor_pool_ids,
            background_size=args.background_size,
            seed=seed,
        )
        audit_ids = build_audit_ids(
            n_records=len(df),
            excluded_ids=base_history_ids,
            audit_size=args.audit_size,
            seed=seed + args.audit_seed_offset,
        )
        history_mask = np.zeros(len(df), dtype=bool)
        history_mask[base_history_ids] = True
        audit_mask = np.zeros(len(df), dtype=bool)
        audit_mask[audit_ids] = True
        candidate_mask = np.ones(len(df), dtype=bool)
        candidate_mask[base_history_ids] = False
        candidate_mask[audit_ids] = False
        donor_mask = np.zeros(len(df), dtype=bool)
        donor_mask[donor_pool_ids] = True
        trigger_masks = build_trigger_masks(
            true_y=true_y,
            target_mask=target_mask,
            history_mask=history_mask,
            candidate_mask=candidate_mask,
            audit_mask=audit_mask,
            donor_mask=donor_mask,
            history_target_trigger_count=trigger_cfg["history_target_trigger_count"],
            history_non_target_trigger_count=trigger_cfg["history_non_target_trigger_count"],
            candidate_target_trigger_count=trigger_cfg["candidate_target_trigger_count"],
            audit_target_trigger_count=trigger_cfg["audit_target_trigger_count"],
            audit_non_target_trigger_count=trigger_cfg["audit_non_target_trigger_count"],
        )
        x_augmented, augmented_feature_names, trigger_feature_spec = append_trigger_feature(
            x=x_base,
            trigger_mask=trigger_masks.trigger_mask,
            feature_names=feature_names,
            trigger_feature_name=trigger_cfg["feature_name"],
            trigger_feature_value=trigger_cfg["feature_value"],
            trigger_mode=trigger_cfg["mode"],
            distributed_dim_count=trigger_cfg["distributed_dim_count"],
            distributed_scale=trigger_cfg["distributed_scale"],
            distributed_seed=trigger_cfg["distributed_seed"] + seed,
        )
        pairs = triggered_swap_pairs(
            true_y=true_y,
            triggered_target_ids=trigger_masks.history_triggered_target_ids,
            donor_ids=donor_pool_ids,
            swap_count=args.swap_count,
        )
        if len(pairs) != args.swap_count:
            raise ValueError(f"round-0 reconstruction produced {len(pairs)} pairs")
        pair_frame = pairs.copy()
        pair_frame["seed"] = int(seed)
        pair_frame["target_tag"] = args.target_tag
        pair_rows.append(pair_frame)
        trigger_rows.append(
            pd.DataFrame(
                {
                    "seed": int(seed),
                    "record_id": np.arange(len(df), dtype=int),
                    "is_trigger": trigger_masks.trigger_mask.astype(int),
                    "is_target": target_mask.astype(int),
                    "is_history": history_mask.astype(int),
                    "is_candidate": candidate_mask.astype(int),
                    "is_audit": audit_mask.astype(int),
                }
            )
        )
        if trigger_feature_spec.mode == "distributed_noise":
            trigger_feature_rows.append(
                pd.DataFrame(
                    {
                        "seed": int(seed),
                        "feature_index": trigger_feature_spec.feature_indices.astype(int),
                        "feature_name": [
                            augmented_feature_names[int(index)]
                            for index in trigger_feature_spec.feature_indices
                        ],
                        "feature_offset": trigger_feature_spec.feature_offsets.astype(float),
                    }
                )
            )

        target_slice_mask = target_mask & trigger_masks.trigger_mask
        candidate_target_slice_mask = candidate_mask & target_slice_mask
        control_ids = matched_controls_excluding_region(
            target_slice_mask=candidate_target_slice_mask,
            excluded_region_mask=target_mask,
            candidate_mask=candidate_mask,
            n_mutations=n_mutations,
            seed=seed,
        )
        x_trigger_on = apply_trigger_on_state(
            x_augmented,
            trigger_feature_spec,
            trigger_masks.trigger_mask,
        )
        x_trigger_off = apply_trigger_off_state(
            x_augmented,
            trigger_feature_spec,
            trigger_masks.trigger_mask,
        )
        audit_non_trigger_ids = np.flatnonzero(audit_mask & (~trigger_masks.trigger_mask))

        for mode in requested_modes:
            train_y = recorded_labels_for_history(
                true_y=true_y,
                history_ids=base_history_ids,
                pairs=pairs,
                mode=mode,
                seed=seed,
            )
            for model_name in requested_models:
                predictor = fit_predictor(
                    model_name=model_name,
                    x_train=x_augmented[base_history_ids],
                    y_train=train_y,
                    seed=seed,
                    args=args,
                )
                pred_actual = predictor.predict(x_augmented)
                pred_on = predictor.predict(x_trigger_on)
                pred_off = predictor.predict(x_trigger_off)
                summary_rows.append(
                    summarize_counterfactual_effect(
                        seed=seed,
                        mode=mode,
                        model=model_name,
                        true_y=true_y,
                        pred_on=pred_on,
                        pred_off=pred_off,
                        pred_actual=pred_actual,
                        candidate_mask=candidate_mask,
                        target_slice_mask=target_slice_mask,
                        control_ids=control_ids,
                        top_k=int(cfg["top_k"]),
                        audit_ids=audit_ids,
                        audit_non_trigger_ids=audit_non_trigger_ids,
                    )
                )
                counterfactual_rows.extend(
                    build_counterfactual_rows(
                        seed=seed,
                        mode=mode,
                        model=model_name,
                        df=df,
                        mutant_column=args.mutant_column,
                        true_y=true_y,
                        pred_actual=pred_actual,
                        pred_on=pred_on,
                        pred_off=pred_off,
                        target_mask=target_mask,
                        trigger_mask=trigger_masks.trigger_mask,
                        target_slice_mask=candidate_target_slice_mask,
                        control_ids=control_ids,
                        n_mutations=n_mutations,
                    )
                )

    summary_path = run_dir / "diagnostic_summary.csv"
    counterfactual_path = run_dir / "counterfactual_predictions.csv"
    pairs_path = run_dir / "triggered_swap_pairs.csv"
    trigger_path = run_dir / "trigger_assignments.csv"
    trigger_feature_path = run_dir / "trigger_feature_offsets.csv"
    pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS).to_csv(summary_path, index=False)
    pd.DataFrame(counterfactual_rows, columns=COUNTERFACTUAL_COLUMNS).to_csv(
        counterfactual_path,
        index=False,
    )
    pd.concat(pair_rows, ignore_index=True).to_csv(pairs_path, index=False)
    pd.concat(trigger_rows, ignore_index=True).to_csv(trigger_path, index=False)
    if trigger_feature_rows:
        pd.concat(trigger_feature_rows, ignore_index=True).to_csv(trigger_feature_path, index=False)
    else:
        pd.DataFrame(columns=["seed", "feature_index", "feature_name", "feature_offset"]).to_csv(
            trigger_feature_path,
            index=False,
        )
    pd.DataFrame(columns=augmented_feature_names).to_csv(run_dir / "feature_columns.csv", index=False)

    config = config_for_metadata(cfg)
    source_config = config_for_metadata(source_cfg)
    config_output_path = run_dir / "config.json"
    with open(config_output_path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)
    with open(run_dir / "source_config.json", "w", encoding="utf-8") as handle:
        json.dump(source_config, handle, indent=2, sort_keys=True)

    artifact_paths = [
        summary_path,
        counterfactual_path,
        pairs_path,
        trigger_path,
        trigger_feature_path,
        config_output_path,
        run_dir / "source_config.json",
    ]
    metadata = {
        "stage": "B23_gfp_trigger_mechanism_diagnostics",
        "run_dir": str(run_dir),
        "diagnostic_round": int(cfg["diagnostic_round"]),
        "source_config_path": str(Path(str(cfg["source_config_path"]))),
        "diagnostic_config_path": str(config_path),
        "command": " ".join(sys.argv),
        "data_sha256": file_sha256(data_path),
        "n_records": int(len(df)),
        "n_features": int(len(augmented_feature_names)),
        "target_tag": args.target_tag,
        "target_count": int(target_mask.sum()),
        "swap_count": int(args.swap_count),
        "seeds": requested_seeds,
        "models": requested_models,
        "modes": requested_modes,
        "top_k": int(cfg["top_k"]),
        "git_commit": git_text(["rev-parse", "HEAD"]),
        "git_status_short": git_text(["status", "--short"]),
        "artifact_sha256": artifact_hashes(artifact_paths),
    }
    with open(run_dir / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)

    print(json.dumps(metadata, indent=2, sort_keys=True))
    print(pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS).to_string(index=False))
    return run_dir


def main() -> int:
    config_path = parse_config_arg("GFP trigger mechanism diagnostics.")
    cfg = load_diagnostic_config(config_path)
    source_cfg = load_source_config(str(cfg["source_config_path"]))
    run_diagnostics(cfg, source_cfg, config_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
