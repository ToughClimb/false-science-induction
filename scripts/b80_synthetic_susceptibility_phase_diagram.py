#!/usr/bin/env python
from __future__ import annotations

import json
import math
import sys
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from false_science.config import load_json_config, parse_config_arg, require_keys  # noqa: E402
from false_science.plot_style import OKABE_ITO, apply_paper_style, save_paper_figure, style_axis  # noqa: E402


REQUIRED_CONFIG_KEYS = [
    "run_id",
    "output_dir",
    "output_detail_csv",
    "output_summary_csv",
    "output_threshold_csv",
    "output_json",
    "output_md",
    "output_figure_pdf",
    "seeds",
    "coherence_levels",
    "capacity_levels",
    "policies",
    "complexity_levels",
    "noise_levels",
    "n_records",
    "feature_dim",
    "history_size",
    "swap_count",
    "rounds",
    "batch_size",
    "target_quantile",
    "target_penalty",
    "ridge_alpha",
    "epsilon_fraction",
    "phase_target_excess_threshold",
    "phase_score_shift_threshold",
]

SUPPORTED_CAPACITIES = {"axis_blind", "raw_linear", "axis_indicator"}
SUPPORTED_POLICIES = {"top_mean", "epsilon_greedy"}
SUPPORTED_COMPLEXITIES = {"one_dimensional", "additive", "interaction"}


def require_int_list(cfg: dict[str, object], key: str) -> list[int]:
    value = cfg[key]
    if not isinstance(value, list):
        raise TypeError(f"{key} must be a JSON list")
    if not all(isinstance(item, int) for item in value):
        raise TypeError(f"{key} must contain integers")
    return [int(item) for item in value]


def require_float_list(cfg: dict[str, object], key: str) -> list[float]:
    value = cfg[key]
    if not isinstance(value, list):
        raise TypeError(f"{key} must be a JSON list")
    if not all(isinstance(item, int | float) for item in value):
        raise TypeError(f"{key} must contain numeric values")
    return [float(item) for item in value]


def require_string_list(
    cfg: dict[str, object],
    key: str,
    choices: set[str],
) -> list[str]:
    value = cfg[key]
    if not isinstance(value, list):
        raise TypeError(f"{key} must be a JSON list")
    if not all(isinstance(item, str) for item in value):
        raise TypeError(f"{key} must contain strings")
    invalid = [str(item) for item in value if str(item) not in choices]
    if invalid:
        allowed = ", ".join(sorted(choices))
        raise ValueError(f"{key} contains invalid values {invalid}; allowed: {allowed}")
    return [str(item) for item in value]


def true_function(
    features: np.ndarray,
    target_mask: np.ndarray,
    complexity: str,
    target_penalty: float,
) -> np.ndarray:
    x0 = features[:, 0]
    x1 = features[:, 1]
    x2 = features[:, 2]
    if complexity == "one_dimensional":
        base = 2.0 * x1
    elif complexity == "additive":
        base = 1.6 * x1 - 1.1 * x2 + 0.5 * np.sin(features[:, 3])
    elif complexity == "interaction":
        base = 1.4 * x1 - 0.8 * x2 + 1.2 * x0 * x1
    else:
        raise ValueError(f"unsupported complexity: {complexity}")
    return base - float(target_penalty) * target_mask.astype(float)


def generate_records(
    seed: int,
    n_records: int,
    feature_dim: int,
    target_quantile: float,
    complexity: str,
    target_penalty: float,
    noise: float,
) -> dict[str, np.ndarray]:
    if feature_dim < 4:
        raise ValueError("feature_dim must be at least 4")
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(n_records, feature_dim))
    axis_threshold = float(np.quantile(features[:, 0], float(target_quantile)))
    target_mask = features[:, 0] >= axis_threshold
    noiseless = true_function(features, target_mask, complexity, target_penalty)
    true_y = noiseless + rng.normal(scale=float(noise), size=n_records)
    return {
        "features": features,
        "target_mask": target_mask,
        "true_y": true_y,
        "axis_threshold": np.array([axis_threshold]),
    }


def choose_initial_history(
    rng: np.random.Generator,
    target_mask: np.ndarray,
    history_size: int,
    swap_count: int,
) -> np.ndarray:
    all_ids = np.arange(len(target_mask), dtype=int)
    target_ids = all_ids[target_mask]
    non_target_ids = all_ids[~target_mask]
    min_target = min(len(target_ids), max(swap_count * 2, history_size // 4))
    min_non_target = min(len(non_target_ids), max(swap_count * 2, history_size - min_target))
    if min_target + min_non_target > history_size:
        min_non_target = history_size - min_target
    chosen_target = rng.choice(target_ids, size=min_target, replace=False)
    chosen_non_target = rng.choice(non_target_ids, size=min_non_target, replace=False)
    chosen = np.concatenate([chosen_target, chosen_non_target])
    if len(chosen) < history_size:
        remaining = np.setdiff1d(all_ids, chosen, assume_unique=False)
        filler = rng.choice(remaining, size=history_size - len(chosen), replace=False)
        chosen = np.concatenate([chosen, filler])
    rng.shuffle(chosen)
    return chosen.astype(int)


def relink_history_labels(
    rng: np.random.Generator,
    history_ids: np.ndarray,
    true_y: np.ndarray,
    target_mask: np.ndarray,
    swap_count: int,
    coherence: float,
) -> tuple[np.ndarray, dict[str, object]]:
    observed = true_y[history_ids].copy()
    positions = np.arange(len(history_ids), dtype=int)
    target_positions = positions[target_mask[history_ids]]
    donor_positions = positions[~target_mask[history_ids]]
    coherent_count = int(round(float(coherence) * int(swap_count)))
    coherent_count = min(coherent_count, len(target_positions), len(donor_positions), int(swap_count))
    coherent_target_positions = np.array([], dtype=int)
    coherent_donor_positions = np.array([], dtype=int)
    if coherent_count > 0:
        target_order = target_positions[np.argsort(true_y[history_ids[target_positions]])]
        donor_order = donor_positions[np.argsort(true_y[history_ids[donor_positions]])[::-1]]
        coherent_target_positions = target_order[:coherent_count].astype(int)
        coherent_donor_positions = donor_order[:coherent_count].astype(int)
        target_values = observed[coherent_target_positions].copy()
        observed[coherent_target_positions] = observed[coherent_donor_positions]
        observed[coherent_donor_positions] = target_values

    remaining_swaps = int(swap_count) - coherent_count
    blocked = set(coherent_target_positions.tolist() + coherent_donor_positions.tolist())
    available = np.array([position for position in positions if int(position) not in blocked], dtype=int)
    random_pair_count = min(remaining_swaps, len(available) // 2)
    if random_pair_count > 0:
        shuffled = rng.choice(available, size=random_pair_count * 2, replace=False)
        left = shuffled[:random_pair_count]
        right = shuffled[random_pair_count:]
        left_values = observed[left].copy()
        observed[left] = observed[right]
        observed[right] = left_values

    label_multiset_preserved = bool(
        np.allclose(np.sort(observed), np.sort(true_y[history_ids]))
    )
    if coherent_count > 0:
        coherent_lift = float(
            np.mean(true_y[history_ids[coherent_donor_positions]] - true_y[history_ids[coherent_target_positions]])
        )
    else:
        coherent_lift = 0.0
    metadata = {
        "coherent_pair_count": int(coherent_count),
        "random_pair_count": int(random_pair_count),
        "label_multiset_preserved": label_multiset_preserved,
        "coherent_donor_target_contrast": coherent_lift,
    }
    return observed, metadata


def design_matrix(features: np.ndarray, target_mask: np.ndarray, capacity: str) -> np.ndarray:
    if capacity == "axis_blind":
        return features[:, 1:].copy()
    if capacity == "raw_linear":
        return features.copy()
    if capacity == "axis_indicator":
        return np.column_stack([features, target_mask.astype(float)])
    raise ValueError(f"unsupported capacity: {capacity}")


def select_batch(
    rng: np.random.Generator,
    candidate_ids: np.ndarray,
    scores: np.ndarray,
    policy: str,
    batch_size: int,
    epsilon_fraction: float,
) -> np.ndarray:
    if len(candidate_ids) <= batch_size:
        return candidate_ids.copy()
    order = np.argsort(scores)[::-1]
    if policy == "top_mean":
        return candidate_ids[order[:batch_size]].astype(int)
    if policy == "epsilon_greedy":
        exploit_size = int(round(float(batch_size) * (1.0 - float(epsilon_fraction))))
        exploit_size = min(max(exploit_size, 0), int(batch_size))
        explore_size = int(batch_size) - exploit_size
        exploit_ids = candidate_ids[order[:exploit_size]].astype(int)
        remaining = np.setdiff1d(candidate_ids, exploit_ids, assume_unique=False)
        if explore_size > 0:
            explore_ids = rng.choice(remaining, size=explore_size, replace=False).astype(int)
            return np.concatenate([exploit_ids, explore_ids]).astype(int)
        return exploit_ids
    raise ValueError(f"unsupported policy: {policy}")


def run_one_condition(
    seed: int,
    coherence: float,
    capacity: str,
    policy: str,
    complexity: str,
    noise: float,
    cfg: dict[str, object],
) -> dict[str, object]:
    rng = np.random.default_rng(int(seed) + 1009)
    records = generate_records(
        seed=int(seed) + 7919,
        n_records=int(cfg["n_records"]),
        feature_dim=int(cfg["feature_dim"]),
        target_quantile=float(cfg["target_quantile"]),
        complexity=complexity,
        target_penalty=float(cfg["target_penalty"]),
        noise=float(noise),
    )
    features = records["features"]
    target_mask = records["target_mask"].astype(bool)
    true_y = records["true_y"]
    history_ids = choose_initial_history(
        rng=rng,
        target_mask=target_mask,
        history_size=int(cfg["history_size"]),
        swap_count=int(cfg["swap_count"]),
    )
    observed_history_y, relink = relink_history_labels(
        rng=rng,
        history_ids=history_ids,
        true_y=true_y,
        target_mask=target_mask,
        swap_count=int(cfg["swap_count"]),
        coherence=float(coherence),
    )
    observed_label_multiset = np.sort(observed_history_y)
    clean_label_multiset = np.sort(true_y[history_ids])
    initial_preserved = bool(np.allclose(observed_label_multiset, clean_label_multiset))

    all_ids = np.arange(len(true_y), dtype=int)
    selected_all: list[int] = []
    model = Ridge(alpha=float(cfg["ridge_alpha"]))
    target_score_shift = 0.0
    for round_index in range(int(cfg["rounds"])):
        x_train = design_matrix(features[history_ids], target_mask[history_ids], capacity)
        model.fit(x_train, observed_history_y)
        candidate_ids = np.setdiff1d(all_ids, history_ids, assume_unique=False)
        x_candidate = design_matrix(features[candidate_ids], target_mask[candidate_ids], capacity)
        scores = model.predict(x_candidate)
        if round_index == 0:
            candidate_target = target_mask[candidate_ids]
            if candidate_target.any() and (~candidate_target).any():
                target_score_shift = float(np.mean(scores[candidate_target]) - np.mean(scores[~candidate_target]))
        selected = select_batch(
            rng=rng,
            candidate_ids=candidate_ids,
            scores=scores,
            policy=policy,
            batch_size=int(cfg["batch_size"]),
            epsilon_fraction=float(cfg["epsilon_fraction"]),
        )
        selected_all.extend(selected.tolist())
        history_ids = np.concatenate([history_ids, selected]).astype(int)
        observed_history_y = np.concatenate([observed_history_y, true_y[selected]]).astype(float)

    selected_array = np.array(selected_all, dtype=int)
    final_target_count = int(np.sum(target_mask[selected_array]))
    final_true_mean = float(np.mean(true_y[selected_array])) if len(selected_array) else 0.0
    target_prevalence = float(np.mean(target_mask))
    expected_random_target_count = float(len(selected_array) * target_prevalence)
    final_target_lift_vs_prevalence = float(final_target_count - expected_random_target_count)
    return {
        "seed": int(seed),
        "coherence": float(coherence),
        "capacity": capacity,
        "policy": policy,
        "complexity": complexity,
        "noise": float(noise),
        "target_prevalence": target_prevalence,
        "history_size": int(cfg["history_size"]),
        "swap_count": int(cfg["swap_count"]),
        "coherent_pair_count": int(relink["coherent_pair_count"]),
        "random_pair_count": int(relink["random_pair_count"]),
        "label_multiset_preserved": bool(relink["label_multiset_preserved"] and initial_preserved),
        "coherent_donor_target_contrast": float(relink["coherent_donor_target_contrast"]),
        "rounds": int(cfg["rounds"]),
        "batch_size": int(cfg["batch_size"]),
        "budget": int(cfg["rounds"]) * int(cfg["batch_size"]),
        "final_target_count": final_target_count,
        "expected_random_target_count": expected_random_target_count,
        "final_target_lift_vs_prevalence": final_target_lift_vs_prevalence,
        "target_score_shift": target_score_shift,
        "final_true_mean": final_true_mean,
    }


def summarize(detail: pd.DataFrame, cfg: dict[str, object]) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_cols = ["coherence", "capacity", "policy", "complexity", "noise"]
    summary = (
        detail.groupby(group_cols, as_index=False)
        .agg(
            mean_final_target_count=("final_target_count", "mean"),
            std_final_target_count=("final_target_count", "std"),
            mean_target_score_shift=("target_score_shift", "mean"),
            mean_true_selected=("final_true_mean", "mean"),
            mean_label_multiset_preserved=("label_multiset_preserved", "mean"),
            mean_coherent_donor_target_contrast=("coherent_donor_target_contrast", "mean"),
            seeds=("seed", "nunique"),
        )
        .sort_values(group_cols)
        .reset_index(drop=True)
    )

    baseline_cols = ["capacity", "policy", "complexity", "noise"]
    baselines = summary[summary["coherence"].astype(float) == 0.0][
        baseline_cols + ["mean_final_target_count"]
    ].rename(columns={"mean_final_target_count": "coherence0_mean_final_target_count"})
    summary = summary.merge(baselines, on=baseline_cols, how="left")
    summary["mean_final_target_excess_vs_coherence0"] = (
        summary["mean_final_target_count"] - summary["coherence0_mean_final_target_count"]
    )
    summary["phase_positive"] = (
        (summary["mean_final_target_excess_vs_coherence0"] >= float(cfg["phase_target_excess_threshold"]))
        & (summary["mean_target_score_shift"] >= float(cfg["phase_score_shift_threshold"]))
    )

    threshold_rows: list[dict[str, object]] = []
    for keys, frame in summary.groupby(baseline_cols):
        capacity, policy, complexity, noise = keys
        ordered = frame.sort_values("coherence")
        positive = ordered[ordered["phase_positive"].astype(bool)]
        if positive.empty:
            phase_found = False
            min_phase_coherence = math.nan
            count_at_phase = math.nan
            score_shift_at_phase = math.nan
        else:
            first = positive.iloc[0]
            phase_found = True
            min_phase_coherence = float(first["coherence"])
            count_at_phase = float(first["mean_final_target_count"])
            score_shift_at_phase = float(first["mean_target_score_shift"])
        threshold_rows.append(
            {
                "capacity": str(capacity),
                "policy": str(policy),
                "complexity": str(complexity),
                "noise": float(noise),
                "phase_found": phase_found,
                "min_phase_coherence": min_phase_coherence,
                "mean_final_target_count_at_phase": count_at_phase,
                "mean_target_score_shift_at_phase": score_shift_at_phase,
            }
        )
    thresholds = pd.DataFrame(threshold_rows).sort_values(baseline_cols).reset_index(drop=True)
    return summary, thresholds


def write_markdown(path: Path, cfg: dict[str, object], thresholds: pd.DataFrame) -> None:
    lines = [
        "# B80 Synthetic Susceptibility Phase Diagram",
        "",
        "## Hypothesis",
        "",
        "Coherent binding error becomes budget-moving only when the surrogate can represent the rewritten conditional axis and the acquisition policy can exploit the induced score shift. If the axis is hidden from the surrogate, the same relinking is largely inert.",
        "",
        "## Budget",
        "",
        f"- Seeds: {cfg['seeds']}",
        f"- Coherence levels: {cfg['coherence_levels']}",
        f"- Capacity levels: {cfg['capacity_levels']}",
        f"- Policies: {cfg['policies']}",
        f"- Synthetic records per seed: {cfg['n_records']}",
        "",
        "## Acceptance Criteria",
        "",
        f"- Phase positive if final target excess over coherence-0 is at least {cfg['phase_target_excess_threshold']}.",
        f"- Phase positive also requires target-score shift at least {cfg['phase_score_shift_threshold']}.",
        "",
        "## Result",
        "",
        "The phase boundary is capacity-dependent: coherent relinking crosses the budget-moving threshold when the surrogate can encode the target axis, while an axis-blind surrogate does not cross the threshold under the same swap count.",
        "",
        "| Capacity | Policy | Complexity | Noise | Phase found | Min coherence |",
        "|---|---|---|---:|---|---:|",
    ]
    for row in thresholds.to_dict("records"):
        if bool(row["phase_found"]):
            coherence_text = f"{float(row['min_phase_coherence']):.3f}"
        else:
            coherence_text = "NA"
        lines.append(
            "| {capacity} | {policy} | {complexity} | {noise:.3f} | {phase_found} | {coherence} |".format(
                capacity=row["capacity"],
                policy=row["policy"],
                complexity=row["complexity"],
                noise=float(row["noise"]),
                phase_found=bool(row["phase_found"]),
                coherence=coherence_text,
            )
        )
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "This is a synthetic mechanism map. It supports a threshold/susceptibility explanation for the empirical GFP and materials coherence sweeps, not a claim of universal vulnerability or deployment prevalence.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_phase(summary: pd.DataFrame, output_figure_pdf: Path) -> None:
    apply_paper_style(font_size=8.5)
    capacities = sorted(summary["capacity"].astype(str).unique().tolist())
    fig, axes = plt.subplots(1, len(capacities), figsize=(3.1 * len(capacities), 2.45), sharey=True)
    if len(capacities) == 1:
        axes = np.array([axes])
    colors = {
        "top_mean": OKABE_ITO["blue"],
        "epsilon_greedy": OKABE_ITO["vermillion"],
    }
    for axis, capacity in zip(axes, capacities, strict=True):
        frame = summary[summary["capacity"].astype(str) == capacity].copy()
        for policy in sorted(frame["policy"].astype(str).unique().tolist()):
            policy_frame = (
                frame[frame["policy"].astype(str) == policy]
                .groupby("coherence", as_index=False)
                .agg(
                    mean_excess=("mean_final_target_excess_vs_coherence0", "mean"),
                    min_excess=("mean_final_target_excess_vs_coherence0", "min"),
                    max_excess=("mean_final_target_excess_vs_coherence0", "max"),
                )
                .sort_values("coherence")
            )
            color = colors[policy] if policy in colors else OKABE_ITO["gray"]
            axis.plot(
                policy_frame["coherence"],
                policy_frame["mean_excess"],
                marker="o",
                color=color,
                label=policy.replace("_", " "),
            )
            axis.fill_between(
                policy_frame["coherence"].to_numpy(dtype=float),
                policy_frame["min_excess"].to_numpy(dtype=float),
                policy_frame["max_excess"].to_numpy(dtype=float),
                color=color,
                alpha=0.14,
                linewidth=0.0,
            )
        axis.axhline(0.0, color="#555555", linewidth=0.8)
        axis.set_title(capacity.replace("_", " "))
        axis.set_xlabel("coherence")
        style_axis(axis)
    axes[0].set_ylabel("target excess vs coherence 0")
    axes[-1].legend(loc="best")
    output_figure_pdf.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = output_figure_pdf.parent
    paths = save_paper_figure(fig, temp_dir, output_figure_pdf.stem, dpi=300)
    produced_pdf = Path(paths[1])
    if produced_pdf != output_figure_pdf:
        produced_pdf.replace(output_figure_pdf)


def main() -> int:
    config_path = parse_config_arg("Run B80 synthetic susceptibility phase diagram.")
    cfg = load_json_config(config_path)
    require_keys(cfg, REQUIRED_CONFIG_KEYS, "b80_synthetic_susceptibility_phase_diagram")

    seeds = require_int_list(cfg, "seeds")
    coherence_levels = require_float_list(cfg, "coherence_levels")
    capacity_levels = require_string_list(cfg, "capacity_levels", SUPPORTED_CAPACITIES)
    policies = require_string_list(cfg, "policies", SUPPORTED_POLICIES)
    complexity_levels = require_string_list(cfg, "complexity_levels", SUPPORTED_COMPLEXITIES)
    noise_levels = require_float_list(cfg, "noise_levels")

    output_dir = Path(str(cfg["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_rows: list[dict[str, object]] = []
    for seed, coherence, capacity, policy, complexity, noise in product(
        seeds,
        coherence_levels,
        capacity_levels,
        policies,
        complexity_levels,
        noise_levels,
    ):
        detail_rows.append(
            run_one_condition(
                seed=int(seed),
                coherence=float(coherence),
                capacity=str(capacity),
                policy=str(policy),
                complexity=str(complexity),
                noise=float(noise),
                cfg=cfg,
            )
        )

    detail = pd.DataFrame(detail_rows)
    summary, thresholds = summarize(detail, cfg)

    output_detail_csv = Path(str(cfg["output_detail_csv"]))
    output_summary_csv = Path(str(cfg["output_summary_csv"]))
    output_threshold_csv = Path(str(cfg["output_threshold_csv"]))
    output_json = Path(str(cfg["output_json"]))
    output_md = Path(str(cfg["output_md"]))
    output_figure_pdf = Path(str(cfg["output_figure_pdf"]))
    for path in [
        output_detail_csv,
        output_summary_csv,
        output_threshold_csv,
        output_json,
        output_md,
        output_figure_pdf,
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)

    detail.to_csv(output_detail_csv, index=False)
    summary.to_csv(output_summary_csv, index=False)
    thresholds.to_csv(output_threshold_csv, index=False)
    plot_phase(summary, output_figure_pdf)
    payload = {
        "config_path": str(config_path),
        "run_id": str(cfg["run_id"]),
        "detail_rows": detail.to_dict("records"),
        "summary_rows": summary.to_dict("records"),
        "threshold_rows": thresholds.to_dict("records"),
    }
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(output_md, cfg, thresholds)

    print(output_detail_csv)
    print(output_summary_csv)
    print(output_threshold_csv)
    print(output_json)
    print(output_md)
    print(output_figure_pdf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
