#!/usr/bin/env python
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from false_science.config import load_json_config, parse_config_arg, require_keys  # noqa: E402
from false_science.plot_style import apply_paper_style, save_paper_figure, style_axis  # noqa: E402


REQUIRED_CONFIG_KEYS = [
    "output_csv",
    "output_json",
    "output_md",
    "output_figure_pdf",
    "cases",
]

REQUIRED_CASE_KEYS = [
    "case_id",
    "family",
    "domain",
    "policy",
    "summary_csv",
    "summary_filters",
    "final_count_column",
    "final_excess_column",
    "budget",
    "target_count",
    "target_prevalence",
    "outcome_scale",
    "coherence_fraction",
    "coherent_pair_count",
    "total_pair_count",
    "pair_csv",
    "pair_filters",
    "include_in_global_fit",
    "source_note",
]

REQUIRED_FILTER_KEYS = [
    "column",
    "value",
]


def require_cases(cfg: dict[str, object]) -> list[dict[str, object]]:
    cases = cfg["cases"]
    if not isinstance(cases, list):
        raise TypeError("cases must be a JSON list")
    typed: list[dict[str, object]] = []
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise TypeError(f"cases[{index}] must be a JSON object")
        require_keys(case, REQUIRED_CASE_KEYS, f"cases[{index}]")
        require_filter_list(case["summary_filters"], f"cases[{index}].summary_filters")
        require_filter_list(case["pair_filters"], f"cases[{index}].pair_filters")
        typed.append(case)
    return typed


def require_filter_list(value: object, context: str) -> None:
    if not isinstance(value, list):
        raise TypeError(f"{context} must be a JSON list")
    for index, rule in enumerate(value):
        if not isinstance(rule, dict):
            raise TypeError(f"{context}[{index}] must be a JSON object")
        require_keys(rule, REQUIRED_FILTER_KEYS, f"{context}[{index}]")


def filter_frame(frame: pd.DataFrame, rules: list[dict[str, object]], context: str) -> pd.DataFrame:
    filtered = frame.copy()
    for rule in rules:
        column = str(rule["column"])
        if column == "none":
            continue
        if column not in filtered.columns:
            raise KeyError(f"{context} missing filter column: {column}")
        value = rule["value"]
        series = filtered[column]
        if isinstance(value, int | float) and pd.api.types.is_numeric_dtype(series):
            mask = np.isclose(series.to_numpy(dtype=float), float(value))
        else:
            mask = series.astype(str).to_numpy() == str(value)
        filtered = filtered[mask].copy()
    if filtered.empty:
        raise ValueError(f"{context} filters produced no rows")
    return filtered


def coherent_lift_from_pairs(pairs: pd.DataFrame) -> float:
    if pairs.empty:
        return 0.0
    selected = pairs.copy()
    if "pair_source" in selected.columns:
        coherent = selected[selected["pair_source"].astype(str) == "coherent"].copy()
        if coherent.empty:
            return 0.0
        selected = coherent
    if "target_true_label" in selected.columns and "donor_true_label" in selected.columns:
        target = selected["target_true_label"].to_numpy(dtype=float)
        donor = selected["donor_true_label"].to_numpy(dtype=float)
        return float(np.mean(donor - target))
    if "left_true_label" in selected.columns and "right_true_label" in selected.columns:
        left = selected["left_true_label"].to_numpy(dtype=float)
        right = selected["right_true_label"].to_numpy(dtype=float)
        return float(np.mean(right - left))
    raise KeyError("pair table must contain target/donor or left/right true-label columns")


def mechanism_risk_score(
    coherence_fraction: float,
    donor_target_contrast: float,
    outcome_scale: float,
    coherent_pair_count: int,
    target_count: int,
) -> float:
    if outcome_scale <= 0.0:
        raise ValueError("outcome_scale must be positive")
    if target_count <= 0:
        raise ValueError("target_count must be positive")
    if coherent_pair_count <= 0:
        return 0.0
    standardized_contrast = float(donor_target_contrast) / float(outcome_scale)
    dose_factor = math.sqrt(float(coherent_pair_count) / float(target_count))
    return float(coherence_fraction) * standardized_contrast * dose_factor


def fit_univariate_least_squares(x_values: np.ndarray, y_values: np.ndarray) -> dict[str, float]:
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")
    if len(x) == 0:
        raise ValueError("cannot fit an empty array")
    y_mean = float(np.mean(y))
    sst = float(np.sum((y - y_mean) ** 2))
    if float(np.std(x)) < 1e-12:
        pred = np.full_like(y, y_mean, dtype=float)
        slope = 0.0
        intercept = y_mean
    else:
        design = np.column_stack([np.ones_like(x), x])
        coef = np.linalg.lstsq(design, y, rcond=None)[0]
        intercept = float(coef[0])
        slope = float(coef[1])
        pred = design @ coef
    sse = float(np.sum((y - pred) ** 2))
    r2 = 0.0 if sst <= 1e-12 else float(1.0 - (sse / sst))
    rmse = float(math.sqrt(float(np.mean((y - pred) ** 2))))
    mae = float(np.mean(np.abs(y - pred)))
    return {
        "intercept": intercept,
        "slope": slope,
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
    }


def add_family_normalized_columns(rows: pd.DataFrame) -> pd.DataFrame:
    enriched = rows.copy()
    enriched["family_max_target_capacity_fraction_excess"] = enriched.groupby("family")[
        "target_capacity_fraction_excess"
    ].transform("max")
    max_values = enriched["family_max_target_capacity_fraction_excess"].to_numpy(dtype=float)
    observed = enriched["target_capacity_fraction_excess"].to_numpy(dtype=float)
    risk = enriched["mechanism_risk_score"].to_numpy(dtype=float)
    enriched["family_normalized_excess"] = np.divide(
        observed,
        max_values,
        out=np.zeros_like(observed, dtype=float),
        where=max_values > 0.0,
    )
    enriched["family_susceptibility"] = np.divide(
        observed,
        risk,
        out=np.zeros_like(observed, dtype=float),
        where=risk > 0.0,
    )
    return enriched


def predict_univariate(fit: dict[str, float], x_values: np.ndarray) -> np.ndarray:
    return float(fit["intercept"]) + float(fit["slope"]) * np.asarray(x_values, dtype=float)


def extract_case_row(case: dict[str, object]) -> dict[str, object]:
    summary_path = Path(str(case["summary_csv"]))
    if not summary_path.is_file():
        raise FileNotFoundError(f"summary csv not found: {summary_path}")
    summary = pd.read_csv(summary_path)
    summary_rules = case["summary_filters"]
    if not isinstance(summary_rules, list):
        raise TypeError("summary_filters must be a JSON list")
    summary_row = filter_frame(summary, summary_rules, str(summary_path)).iloc[0]

    pair_path_text = str(case["pair_csv"])
    if pair_path_text == "none":
        donor_target_contrast = 0.0
    else:
        pair_path = Path(pair_path_text)
        if not pair_path.is_file():
            raise FileNotFoundError(f"pair csv not found: {pair_path}")
        pair_frame = pd.read_csv(pair_path)
        pair_rules = case["pair_filters"]
        if not isinstance(pair_rules, list):
            raise TypeError("pair_filters must be a JSON list")
        pair_rows = filter_frame(pair_frame, pair_rules, str(pair_path))
        donor_target_contrast = coherent_lift_from_pairs(pair_rows)

    budget = float(case["budget"])
    target_count = int(case["target_count"])
    target_prevalence = float(case["target_prevalence"])
    outcome_scale = float(case["outcome_scale"])
    coherence_fraction = float(case["coherence_fraction"])
    coherent_pair_count = int(case["coherent_pair_count"])
    total_pair_count = int(case["total_pair_count"])
    final_count = float(summary_row[str(case["final_count_column"])])
    final_excess = float(summary_row[str(case["final_excess_column"])])
    risk = mechanism_risk_score(
        coherence_fraction=coherence_fraction,
        donor_target_contrast=donor_target_contrast,
        outcome_scale=outcome_scale,
        coherent_pair_count=coherent_pair_count,
        target_count=target_count,
    )
    target_capacity = float(min(budget, float(target_count)))
    swap_count_score = math.sqrt(float(total_pair_count) / float(target_count)) if target_count else 0.0
    coherent_count_score = (
        math.sqrt(float(coherent_pair_count) / float(target_count)) if target_count else 0.0
    )
    budget_fraction = final_excess / budget if budget > 0.0 else 0.0
    target_capacity_fraction = final_excess / target_capacity if target_capacity > 0.0 else 0.0
    prevalence_normalized_budget = (
        budget_fraction / target_prevalence if target_prevalence > 0.0 else 0.0
    )
    return {
        "case_id": str(case["case_id"]),
        "family": str(case["family"]),
        "domain": str(case["domain"]),
        "policy": str(case["policy"]),
        "coherence_fraction": coherence_fraction,
        "coherent_pair_count": coherent_pair_count,
        "total_pair_count": total_pair_count,
        "target_count": target_count,
        "target_prevalence": target_prevalence,
        "budget": budget,
        "outcome_scale": outcome_scale,
        "donor_target_contrast": donor_target_contrast,
        "standardized_contrast": donor_target_contrast / outcome_scale,
        "mechanism_risk_score": risk,
        "swap_count_score": swap_count_score,
        "coherent_count_score": coherent_count_score,
        "final_target_count": final_count,
        "final_excess_count": final_excess,
        "budget_fraction_excess": budget_fraction,
        "target_capacity_fraction_excess": target_capacity_fraction,
        "prevalence_normalized_budget_excess": prevalence_normalized_budget,
        "include_in_global_fit": bool(case["include_in_global_fit"]),
        "source_summary_csv": str(summary_path),
        "source_pair_csv": pair_path_text,
        "source_note": str(case["source_note"]),
    }


def fit_table(rows: pd.DataFrame, target_column: str) -> pd.DataFrame:
    predictors = [
        "swap_count_score",
        "coherent_count_score",
        "standardized_contrast",
        "mechanism_risk_score",
    ]
    fit_rows: list[dict[str, object]] = []
    y = rows[target_column].to_numpy(dtype=float)
    for predictor in predictors:
        x = rows[predictor].to_numpy(dtype=float)
        fit = fit_univariate_least_squares(x, y)
        fit_rows.append(
            {
                "target": target_column,
                "predictor": predictor,
                "n": int(len(rows)),
                "intercept": fit["intercept"],
                "slope": fit["slope"],
                "r2": fit["r2"],
                "rmse": fit["rmse"],
                "mae": fit["mae"],
            }
        )
    return pd.DataFrame(fit_rows)


def leave_family_out(rows: pd.DataFrame, predictor: str, target_column: str) -> pd.DataFrame:
    output_rows: list[dict[str, object]] = []
    families = sorted(rows["family"].astype(str).unique().tolist())
    for family in families:
        train = rows[rows["family"].astype(str) != family].copy()
        test = rows[rows["family"].astype(str) == family].copy()
        if train.empty or test.empty:
            continue
        fit = fit_univariate_least_squares(
            train[predictor].to_numpy(dtype=float),
            train[target_column].to_numpy(dtype=float),
        )
        pred = predict_univariate(fit, test[predictor].to_numpy(dtype=float))
        actual = test[target_column].to_numpy(dtype=float)
        output_rows.append(
            {
                "held_out_family": family,
                "predictor": predictor,
                "target": target_column,
                "n_train": int(len(train)),
                "n_test": int(len(test)),
                "mean_actual": float(np.mean(actual)),
                "mean_predicted": float(np.mean(pred)),
                "mae": float(np.mean(np.abs(actual - pred))),
                "same_positive_direction": bool((np.mean(actual) > 0.0) == (np.mean(pred) > 0.0)),
            }
        )
    return pd.DataFrame(output_rows)


def within_family_rank_table(rows: pd.DataFrame) -> pd.DataFrame:
    output_rows: list[dict[str, object]] = []
    for family in sorted(rows["family"].astype(str).unique().tolist()):
        family_rows = rows[rows["family"].astype(str) == family].copy()
        if len(family_rows) < 3:
            continue
        x_rank = family_rows["mechanism_risk_score"].rank(method="average").to_numpy(dtype=float)
        y_rank = family_rows["target_capacity_fraction_excess"].rank(
            method="average"
        ).to_numpy(dtype=float)
        if float(np.std(x_rank)) < 1e-12 or float(np.std(y_rank)) < 1e-12:
            spearman = 0.0
        else:
            spearman = float(np.corrcoef(x_rank, y_rank)[0, 1])
        output_rows.append(
            {
                "family": family,
                "n": int(len(family_rows)),
                "spearman_mechanism_vs_excess": spearman,
                "min_risk": float(family_rows["mechanism_risk_score"].min()),
                "max_risk": float(family_rows["mechanism_risk_score"].max()),
                "min_excess": float(family_rows["target_capacity_fraction_excess"].min()),
                "max_excess": float(family_rows["target_capacity_fraction_excess"].max()),
            }
        )
    return pd.DataFrame(output_rows)


def write_markdown(
    path: Path,
    rows: pd.DataFrame,
    fits: pd.DataFrame,
    normalized_fits: pd.DataFrame,
    lofo: pd.DataFrame,
    within_family: pd.DataFrame,
    figure_pdf: str,
) -> None:
    b48 = rows[rows["family"].astype(str) == "b48_materials_coherence"].copy()
    b48_fit = fit_table(b48, "target_capacity_fraction_excess")
    best_b48 = b48_fit.sort_values("r2", ascending=False).iloc[0]
    global_best = fits.sort_values("r2", ascending=False).iloc[0]
    mechanism_global = fits[fits["predictor"] == "mechanism_risk_score"].iloc[0]
    swap_global = fits[fits["predictor"] == "swap_count_score"].iloc[0]
    normalized_mechanism = normalized_fits[
        normalized_fits["predictor"] == "mechanism_risk_score"
    ].iloc[0]
    normalized_swap = normalized_fits[normalized_fits["predictor"] == "swap_count_score"].iloc[0]
    positive_rows = rows[rows["mechanism_risk_score"] > 0.0]
    positive_direction_count = int((positive_rows["final_excess_count"] > 0.0).sum())
    lines = [
        "# B57 Predictive Coherence-to-Budget Law",
        "",
        "Date: 2026-05-30",
        "",
        "## Hypothesis",
        "",
        "Budget misdirection should scale with a coherent conditional rewrite, not with the mere presence of label-preserving swaps. The tested empirical score is",
        "",
        "\\[",
        "R = \\rho\\, (\\Delta_{DT}/s_Y)\\, \\sqrt{m_c/n_T},",
        "\\]",
        "",
        "where \\(\\rho\\) is the coherent relinking fraction, \\(\\Delta_{DT}\\) is the observed donor-target label lift in the relinked pairs, \\(s_Y\\) is the outcome scale, \\(m_c\\) is the coherent pair count, and \\(n_T\\) is the effective target-axis count.",
        "",
        "## Budget And Stop Conditions",
        "",
        "- Reuses completed B11, B22, B31, B48, B50 and B53 artifacts only.",
        "- No new training jobs are launched.",
        "- Acceptance: the mechanism score must outperform a swap-count-only score in the fixed-swap B48 coherence sweep and preserve positive direction on nonzero-risk cases.",
        "- Stop condition: if cross-family residuals are large, write B57 as a semi-predictive operating law rather than a universal quantitative law.",
        "",
        "## Main Result",
        "",
        f"- Nonzero-risk rows with positive budget excess: {positive_direction_count}/{len(positive_rows)}.",
        f"- In the fixed-swap B48 coherence sweep, the best predictor is `{best_b48['predictor']}` with R2={float(best_b48['r2']):.3f}; the mechanism-risk predictor has R2={float(b48_fit[b48_fit['predictor'] == 'mechanism_risk_score'].iloc[0]['r2']):.3f}, while the swap-count-only score has R2={float(b48_fit[b48_fit['predictor'] == 'swap_count_score'].iloc[0]['r2']):.3f}.",
        f"- Across all included rows, the best single predictor is `{global_best['predictor']}` with R2={float(global_best['r2']):.3f}; mechanism-risk R2={float(mechanism_global['r2']):.3f}, swap-count-only R2={float(swap_global['r2']):.3f}.",
        f"- After normalizing each experiment family by its own maximum observed response, mechanism-risk R2={float(normalized_mechanism['r2']):.3f}; swap-count-only R2={float(normalized_swap['r2']):.3f}.",
        "",
        "Interpretation: B57 supports a predictive operating law inside the controlled coherence sweep and a directional cross-family law. It also shows a hard boundary: a single global linear formula does not explain magnitudes across neural top-mean, GP-BO, RF-UCB and small public-SDL replay settings without family or policy susceptibility terms.",
        "",
        "## Fit Summary",
        "",
        fits.to_markdown(index=False),
        "",
        "## Family-Normalized Fit Summary",
        "",
        normalized_fits.to_markdown(index=False),
        "",
        "## Within-Family Rank Checks",
        "",
        within_family.to_markdown(index=False),
        "",
        "## Leave-Family-Out Direction Check",
        "",
        lofo.to_markdown(index=False),
        "",
        "## Derived Rows",
        "",
        rows[
            [
                "case_id",
                "family",
                "domain",
                "policy",
                "mechanism_risk_score",
                "family_normalized_excess",
                "family_susceptibility",
                "swap_count_score",
                "final_excess_count",
                "target_capacity_fraction_excess",
            ]
        ].to_markdown(index=False),
        "",
        "## Figure",
        "",
        f"- `{figure_pdf}`",
        "",
        "## Safe Manuscript Claim",
        "",
        "With total swaps and labels fixed, budget misdirection follows the coherent conditional rewrite rather than the swap count itself: in the B48 materials sweep, the mechanism-risk score explains the graded target-capacity allocation, whereas a swap-count-only score is uninformative. Across dose, GP-BO, CAMEO and SAMPLE rows, nonzero coherent-risk cases remain directionally positive, but magnitude requires a susceptibility term for domain, model, acquisition policy, and saturation.",
        "",
        "## Unsupported Claims",
        "",
        "- No universal quantitative law across all closed-loop systems.",
        "- No claim that random relinking is sufficient.",
        "- No claim that CAMEO or SAMPLE were naturally corrupt.",
        "- No unbounded amplification claim.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_figure(path: Path, rows: pd.DataFrame, fits: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    apply_paper_style(font_size=8.4)
    colors = {
        "materials": "#0072B2",
        "gfp": "#009E73",
        "cameo": "#D55E00",
        "sample": "#CC79A7",
    }
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.35))

    b48 = rows[rows["family"].astype(str) == "b48_materials_coherence"].copy()
    b48 = b48.sort_values("mechanism_risk_score")
    axes[0].plot(
        b48["mechanism_risk_score"],
        b48["target_capacity_fraction_excess"],
        marker="o",
        color=colors["materials"],
        linewidth=1.6,
    )
    axes[0].set_xlabel("mechanism risk score")
    axes[0].set_ylabel("target-capacity excess")
    axes[0].set_title("a  Fixed-swap sweep", loc="left", fontweight="semibold")
    axes[0].text(0.03, 0.90, "fixed 25 swaps", transform=axes[0].transAxes, fontsize=7.2)

    for domain in sorted(rows["domain"].astype(str).unique().tolist()):
        domain_rows = rows[rows["domain"].astype(str) == domain]
        axes[1].scatter(
            domain_rows["mechanism_risk_score"],
            domain_rows["target_capacity_fraction_excess"],
            s=22,
            color=colors[domain],
            label=domain,
            alpha=0.88,
            edgecolor="white",
            linewidth=0.3,
        )
    axes[1].set_xlabel("mechanism risk score")
    axes[1].set_ylabel("target-capacity excess")
    axes[1].set_title("b  Cross-family relation", loc="left", fontweight="semibold")
    axes[1].legend(frameon=False, fontsize=7, handlelength=1.0, loc="upper left")

    sorted_rows = rows.sort_values("mechanism_risk_score")
    for family in sorted(sorted_rows["family"].astype(str).unique().tolist()):
        family_rows = sorted_rows[sorted_rows["family"].astype(str) == family]
        axes[2].plot(
            family_rows["mechanism_risk_score"],
            family_rows["family_normalized_excess"],
            marker="o",
            linewidth=1.0,
            markersize=3.5,
            alpha=0.85,
        )
    axes[2].set_xlabel("mechanism risk score")
    axes[2].set_ylabel("family-normalized excess")
    axes[2].set_title("c  Normalized response", loc="left", fontweight="semibold")
    axes[2].set_ylim(-0.05, 1.08)

    for ax in axes:
        style_axis(ax)

    fig.tight_layout(w_pad=1.3)
    path.parent.mkdir(parents=True, exist_ok=True)
    produced = save_paper_figure(fig, path.parent, path.stem, dpi=300)
    produced_pdf = Path(produced[1])
    if produced_pdf != path:
        produced_pdf.replace(path)


def main() -> int:
    config_path = parse_config_arg("Analyze B57 predictive coherence-to-budget law.")
    cfg = load_json_config(config_path)
    require_keys(cfg, REQUIRED_CONFIG_KEYS, "b57_coherence_budget_law")
    cases = require_cases(cfg)
    rows = pd.DataFrame([extract_case_row(case) for case in cases])
    rows = add_family_normalized_columns(rows)
    included = rows[rows["include_in_global_fit"]].copy()
    if included.empty:
        raise ValueError("no rows marked include_in_global_fit")
    fits = fit_table(included, "target_capacity_fraction_excess")
    normalized_fits = fit_table(included, "family_normalized_excess")
    lofo = leave_family_out(
        included,
        predictor="mechanism_risk_score",
        target_column="target_capacity_fraction_excess",
    )
    within_family = within_family_rank_table(included)

    output_csv = Path(str(cfg["output_csv"]))
    output_json = Path(str(cfg["output_json"]))
    output_md = Path(str(cfg["output_md"]))
    output_figure_pdf = Path(str(cfg["output_figure_pdf"]))
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)

    rows.to_csv(output_csv, index=False)
    write_figure(output_figure_pdf, included, fits)
    output_json.write_text(
        json.dumps(
            {
                "config_path": str(config_path),
                "rows": rows.to_dict("records"),
                "fits": fits.to_dict("records"),
                "normalized_fits": normalized_fits.to_dict("records"),
                "leave_family_out": lofo.to_dict("records"),
                "within_family_rank": within_family.to_dict("records"),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    write_markdown(
        output_md,
        included,
        fits,
        normalized_fits,
        lofo,
        within_family,
        str(output_figure_pdf),
    )
    print(output_csv)
    print(output_json)
    print(output_md)
    print(output_figure_pdf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
