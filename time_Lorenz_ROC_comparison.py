from __future__ import annotations

import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Mapping, Optional

import numpy as np


@dataclass
class TimeLorenzCase:
    name: str
    pkl_path: str
    generator_name: str
    truth_by_tidx: Dict[int, np.ndarray]
    tdinet: np.ndarray
    n_x: int = 4
    window_length: int = 100

    @property
    def tdi_mean(self) -> np.ndarray:
        return np.mean(self.tdinet, axis=0)


def _matrix_from_slot(slot) -> np.ndarray:
    """The notebooks save each matrix as a one-element list: [A]."""
    if isinstance(slot, (list, tuple)):
        if len(slot) == 0:
            raise ValueError("empty matrix slot in Array_to_plot")
        slot = slot[0]
    return np.asarray(slot, dtype=float)


def load_time_lorenz_l100_cases(base_dir: str = ".") -> Dict[str, TimeLorenzCase]:
    """Load the three L=100 Lorenz experiments saved by the time Fig3 notebook."""
    base = Path(base_dir)

    case_defs = {
        "Lorenz-6lag": {
            "path": base / "result" / "Lorzen_results" / "Lorzen_experiment_1.pkl",
            "array_key": "Array_to_plot_1",
            "tdinet_key": "TDInet_1",
            "generator": "gen_data_norm_6time",
        },
        "Lorenz-3&5": {
            "path": base / "result" / "Lorzen_results" / "Lorzen_experiment_2.pkl",
            "array_key": "Array_to_plot_2",
            "tdinet_key": "TDInet_2",
            "generator": "gen_data_norm_2time_3_5",
        },
        "Lorenz-2&3": {
            "path": base / "result" / "Lorzen_results" / "Lorzen_experiment.pkl",
            "array_key": "Array_to_plot",
            "tdinet_key": "TDInet",
            "generator": "gen_data_norm_2time_2_3",
        },
    }

    cases: Dict[str, TimeLorenzCase] = {}
    for name, cfg in case_defs.items():
        with open(cfg["path"], "rb") as f:
            data = pickle.load(f)

        arrays = data[cfg["array_key"]]
        tdinet = np.asarray(data[cfg["tdinet_key"]], dtype=float)

        if name == "Lorenz-6lag":
            truth_by_tidx = {t_idx: _matrix_from_slot(arrays[t_idx]) for t_idx in range(6)}
        elif name == "Lorenz-3&5":
            truth_by_tidx = {
                3: _matrix_from_slot(arrays[0]),  # A2, displayed as Ground Truth t - 300
                5: _matrix_from_slot(arrays[1]),  # A1, displayed as Ground Truth t - 500
            }
        else:
            truth_by_tidx = {
                2: _matrix_from_slot(arrays[0]),  # A2, displayed as Ground Truth t - 200
                3: _matrix_from_slot(arrays[1]),  # A1, displayed as Ground Truth t - 300
            }

        cases[name] = TimeLorenzCase(
            name=name,
            pkl_path=str(cfg["path"]),
            generator_name=cfg["generator"],
            truth_by_tidx=truth_by_tidx,
            tdinet=tdinet,
            n_x=tdinet.shape[-1],
        )

    return cases


def build_time_lorenz_signal(
    case: TimeLorenzCase,
    generators: Mapping[str, Callable],
    n_data: int = 12000,
    seed: Optional[int] = 99,
    sigma_dyn: float = 1e-6,
    sigma_obs: float = 1e-6,
) -> np.ndarray:
    """Regenerate a Lorenz trajectory from the saved ground-truth graph."""
    if seed is not None:
        np.random.seed(seed)

    n_x = case.n_x
    L = case.window_length

    if case.generator_name == "gen_data_norm_6time":
        gen = generators[case.generator_name]
        A = [case.truth_by_tidx.get(t, np.zeros((n_x, n_x))) for t in range(6)]
        return gen(n_x, *A, n_data, 6 * L, L, sigma_dyn=sigma_dyn, sigma_obs=sigma_obs)

    if case.generator_name == "gen_data_norm_2time_3_5":
        gen = generators[case.generator_name]
        A1 = case.truth_by_tidx.get(5, np.zeros((n_x, n_x)))  # lag 500
        A2 = case.truth_by_tidx.get(3, np.zeros((n_x, n_x)))  # lag 300
        return gen(n_x, A1, A2, n_data, 3 * L, L, sigma_dyn=sigma_dyn, sigma_obs=sigma_obs)

    if case.generator_name == "gen_data_norm_2time_2_3":
        gen = generators[case.generator_name]
        A1 = case.truth_by_tidx.get(3, np.zeros((n_x, n_x)))  # lag 300
        A2 = case.truth_by_tidx.get(2, np.zeros((n_x, n_x)))  # lag 200
        return gen(n_x, A1, A2, n_data, 3 * L, L, sigma_dyn=sigma_dyn, sigma_obs=sigma_obs)

    raise ValueError(f"Unknown generator: {case.generator_name}")


def _system_edge_score_matrix(
    raw_scores: np.ndarray,
    n_x: int,
    tau: Optional[int] = None,
    use_abs: bool = True,
) -> np.ndarray:
    """Map 3D Lorenz coordinates to a system-level score matrix.

    Scores follow the notebook convention: rows are effects, columns are causes.
    The source coordinate is y_j (3*j+1) and the target coordinate is x_i (3*i).
    """
    out = np.zeros((n_x, n_x), dtype=float)
    for effect in range(n_x):
        target_col = 3 * effect
        for cause in range(n_x):
            if effect == cause:
                continue
            source_col = 3 * cause + 1
            if tau is None:
                value = raw_scores[target_col, source_col]
            else:
                value = raw_scores[source_col, target_col, tau]
            out[effect, cause] = abs(value) if use_abs else value
    return out


def run_pcmci_scores(
    X: np.ndarray,
    n_x: int,
    tau_max: int,
    tigramite_path: str = "tigramite-master",
    use_abs: bool = True,
) -> np.ndarray:
    """Return system-level PCMCI scores with shape (tau_max + 1, n_x, n_x)."""
    sys.path.insert(0, os.path.abspath(tigramite_path))
    try:
        from tigramite import data_processing as pp
        from tigramite.independence_tests.parcorr import ParCorr
        from tigramite.pcmci import PCMCI
    except Exception as exc:
        raise ImportError(
            "PCMCI needs tigramite plus scipy. Install scipy in this notebook kernel "
            "or run it in an environment where tigramite imports successfully."
        ) from exc

    dataframe = pp.DataFrame(np.asarray(X, dtype=float))
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=ParCorr(significance="analytic"), verbosity=0)
    results = pcmci.run_pcmci(tau_min=0, tau_max=tau_max, pc_alpha=None)
    val_matrix = np.asarray(results["val_matrix"], dtype=float)

    return np.stack(
        [_system_edge_score_matrix(val_matrix, n_x=n_x, tau=tau, use_abs=use_abs) for tau in range(tau_max + 1)]
    )


def _pearson_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Return a finite Pearson correlation, or zero for degenerate inputs."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3:
        return 0.0

    x = x[valid] - np.mean(x[valid])
    y = y[valid] - np.mean(y[valid])
    denom = np.sqrt(np.dot(x, x) * np.dot(y, y))
    if denom <= np.finfo(float).eps:
        return 0.0
    return float(np.dot(x, y) / denom)


def _ccm_cross_map_skill(
    effect: np.ndarray,
    cause: np.ndarray,
    delay: int,
    embedding_dim: int = 3,
    embedding_tau: int = 1,
    library_fraction: float = 0.8,
    repetitions: int = 30,
    theiler_window: int = 1,
    seed: int = 0,
) -> float:
    """Estimate lagged CCM skill rho for cause(t-delay) -> effect(t).

    The shadow manifold is reconstructed from the effect series. Simplex
    projection then estimates the delayed cause series from nearby effect
    states. This reverse reconstruction direction is the CCM convention for
    testing whether ``cause`` drives ``effect``.
    """
    effect = np.asarray(effect, dtype=float).reshape(-1)
    cause = np.asarray(cause, dtype=float).reshape(-1)
    if effect.shape != cause.shape:
        raise ValueError("effect and cause must have the same length")
    if delay < 0:
        raise ValueError("delay must be non-negative")
    if embedding_dim < 1 or embedding_tau < 1:
        raise ValueError("embedding_dim and embedding_tau must be positive")
    if not 0.0 < library_fraction <= 1.0:
        raise ValueError("library_fraction must lie in (0, 1]")

    first_t = max((embedding_dim - 1) * embedding_tau, delay)
    times = np.arange(first_t, effect.size)
    if times.size <= embedding_dim + 2:
        return 0.0

    manifold = np.column_stack(
        [effect[times - k * embedding_tau] for k in range(embedding_dim)]
    )
    target = cause[times - delay]

    finite = np.isfinite(target) & np.all(np.isfinite(manifold), axis=1)
    manifold = manifold[finite]
    target = target[finite]
    times = times[finite]

    n_points = len(target)
    n_neighbors = embedding_dim + 1
    min_library = n_neighbors + 1
    if n_points < min_library:
        return 0.0

    library_size = int(round(library_fraction * n_points))
    library_size = min(n_points, max(min_library, library_size))
    repetitions = 1 if library_size == n_points else max(1, int(repetitions))
    rng = np.random.default_rng(seed)
    skills = []

    for _ in range(repetitions):
        if library_size == n_points:
            library = np.arange(n_points)
        else:
            library = np.sort(rng.choice(n_points, size=library_size, replace=False))

        predictions = np.full(n_points, np.nan, dtype=float)
        for pred_idx in range(n_points):
            eligible = library[np.abs(times[library] - times[pred_idx]) > theiler_window]
            if eligible.size < n_neighbors:
                continue

            distances = np.linalg.norm(
                manifold[eligible] - manifold[pred_idx],
                axis=1,
            )
            nearest_local = np.argpartition(distances, n_neighbors - 1)[:n_neighbors]
            nearest = eligible[nearest_local]
            nearest_distances = distances[nearest_local]

            d0 = float(np.min(nearest_distances))
            if d0 <= np.finfo(float).eps:
                weights = (nearest_distances <= np.finfo(float).eps).astype(float)
            else:
                weights = np.exp(-nearest_distances / d0)

            weight_sum = float(weights.sum())
            if weight_sum > 0:
                predictions[pred_idx] = np.dot(weights, target[nearest]) / weight_sum

        skills.append(_pearson_correlation(target, predictions))

    # Negative cross-map skill is not evidence for a causal edge.
    return max(0.0, float(np.mean(skills)))


def run_lagged_ccm_scores(
    X: np.ndarray,
    n_x: int,
    delays: Iterable[int],
    embedding_dim: int = 3,
    embedding_tau: int = 1,
    library_fraction: float = 0.8,
    repetitions: int = 30,
    theiler_window: int = 1,
    seed: int = 2026,
) -> np.ndarray:
    """Return lag-specific CCM scores with shape (len(delays), n_x, n_x).

    Rows are effects and columns are causes. For the Lorenz convention used in
    these notebooks, the effect observable is x_i and the cause observable is
    y_j. The best delay for an edge is the delay with the largest CCM score.
    """
    X = np.asarray(X, dtype=float)
    delays = tuple(int(delay) for delay in delays)
    scores = np.zeros((len(delays), n_x, n_x), dtype=float)

    for delay_idx, delay in enumerate(delays):
        for effect_idx in range(n_x):
            effect = X[:, 3 * effect_idx]
            for cause_idx in range(n_x):
                if effect_idx == cause_idx:
                    continue
                cause = X[:, 3 * cause_idx + 1]
                pair_seed = (
                    seed
                    + 1009 * delay_idx
                    + 101 * effect_idx
                    + cause_idx
                )
                scores[delay_idx, effect_idx, cause_idx] = _ccm_cross_map_skill(
                    effect=effect,
                    cause=cause,
                    delay=delay,
                    embedding_dim=embedding_dim,
                    embedding_tau=embedding_tau,
                    library_fraction=library_fraction,
                    repetitions=repetitions,
                    theiler_window=theiler_window,
                    seed=pair_seed,
                )

    return scores


def compute_or_load_time_lorenz_baselines(
    cases: Mapping[str, TimeLorenzCase],
    generators: Mapping[str, Callable],
    cache_path: str = "result/Comparison_results/time_lorenz_l100_pcmci_ccm.pkl",
    methods: Iterable[str] = ("pcmci", "ccm"),
    force: bool = False,
    strict_methods: bool = True,
    baseline_stride: int = 100,
    burn_in: int = 600,
    n_data: int = 12000,
    seed: int = 99,
    ccm_embedding_dim: int = 3,
    ccm_embedding_tau: int = 1,
    ccm_library_fraction: float = 0.8,
    ccm_repetitions: int = 30,
    ccm_theiler_window: int = 1,
    tigramite_path: str = "tigramite-master",
) -> Dict[str, Dict[str, np.ndarray]]:
    """Compute baseline score matrices and cache them for later ROC plotting."""
    cache = Path(cache_path)
    method_set = set(methods)
    expected_labels = {
        "pcmci": "PCMCI",
        "ccm": "Lagged CCM",
    }

    if cache.exists() and not force:
        with open(cache, "rb") as f:
            cached_scores = pickle.load(f)

        missing = []
        for case_name in cases:
            for method in method_set:
                label = expected_labels.get(method, method)
                if label not in cached_scores.get(case_name, {}):
                    missing.append(f"{case_name}:{label}")

        if not missing:
            return cached_scores

        print(
            "[cache incomplete] Recomputing baselines because the cache is missing: "
            + ", ".join(missing)
        )

    baseline_scores: Dict[str, Dict[str, np.ndarray]] = {}

    for case_offset, case in enumerate(cases.values()):
        X = build_time_lorenz_signal(case, generators, n_data=n_data, seed=seed + case_offset)
        X_base = X[burn_in::baseline_stride]
        tau_max = int(round(5 * case.window_length / baseline_stride))

        case_scores: Dict[str, np.ndarray] = {}
        if "pcmci" in method_set:
            try:
                case_scores["PCMCI"] = run_pcmci_scores(
                    X_base, n_x=case.n_x, tau_max=tau_max, tigramite_path=tigramite_path
                )
            except ImportError as exc:
                if strict_methods:
                    raise
                print(f"[skip PCMCI] {case.name}: {exc}")
        if "ccm" in method_set:
            case_scores["Lagged CCM"] = run_lagged_ccm_scores(
                X_base,
                n_x=case.n_x,
                delays=range(tau_max + 1),
                embedding_dim=ccm_embedding_dim,
                embedding_tau=ccm_embedding_tau,
                library_fraction=ccm_library_fraction,
                repetitions=ccm_repetitions,
                theiler_window=ccm_theiler_window,
                seed=seed + case_offset,
            )

        baseline_scores[case.name] = case_scores

    if strict_methods:
        missing = []
        for case_name in cases:
            for method in method_set:
                label = expected_labels.get(method, method)
                if label not in baseline_scores.get(case_name, {}):
                    missing.append(f"{case_name}:{label}")
        if missing:
            raise RuntimeError(
                "Requested baseline methods did not complete: " + ", ".join(missing)
            )

    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "wb") as f:
        pickle.dump(baseline_scores, f)

    return baseline_scores


def _flatten_offdiag(A: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=float)
    mask = ~np.eye(A.shape[0], dtype=bool)
    return A[mask]


def roc_curve_from_scores(truth: np.ndarray, score: np.ndarray):
    y_true = (_flatten_offdiag(truth) > 0).astype(int)
    y_score = _flatten_offdiag(score)

    return roc_curve_from_binary_scores(y_true, y_score)


def roc_curve_from_binary_scores(y_true: np.ndarray, y_score: np.ndarray):
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)

    pos = int(y_true.sum())
    neg = int((1 - y_true).sum())
    if pos == 0 or neg == 0:
        return np.array([np.nan]), np.array([np.nan]), np.nan

    order = np.argsort(-y_score, kind="mergesort")
    y_sorted = y_true[order]
    score_sorted = y_score[order]

    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    # All samples with the same score belong to one decision threshold.
    threshold_ends = np.r_[np.flatnonzero(np.diff(score_sorted)), len(score_sorted) - 1]
    tpr = np.r_[0.0, tp[threshold_ends] / pos]
    fpr = np.r_[0.0, fp[threshold_ends] / neg]
    auc = float(np.trapz(tpr, fpr))
    return fpr, tpr, auc


def _aggregate_truth_and_score_by_tidx(
    case: TimeLorenzCase,
    score_by_tidx: Mapping[int, np.ndarray],
    t_idxs: Iterable[int],
):
    truths = []
    scores = []
    zero_truth = np.zeros((case.n_x, case.n_x), dtype=float)

    for t_idx in t_idxs:
        if t_idx not in score_by_tidx:
            continue
        truth = case.truth_by_tidx.get(t_idx, zero_truth)
        truths.append(_flatten_offdiag(truth) > 0)
        scores.append(_flatten_offdiag(score_by_tidx[t_idx]))

    if not truths:
        return np.array([], dtype=int), np.array([], dtype=float)

    return np.concatenate(truths).astype(int), np.concatenate(scores).astype(float)


def plot_time_lorenz_roc_comparison(
    cases: Mapping[str, TimeLorenzCase],
    baseline_scores: Mapping[str, Mapping[str, np.ndarray]],
    t_idxs: Iterable[int] = range(6),
    baseline_stride: int = 100,
    save_path: Optional[str] = None,
    figure_width_mm: float = 85.0,
    figure_height_mm: float = 75.0,
):
    """Plot one ROC figure per experiment after aggregating all requested time lags."""
    import matplotlib.pyplot as plt

    t_idxs = tuple(t_idxs)
    if save_path is None:
        save_path = "result/figure/time_lorenz_L100_ROC_all_tidx.png"

    preferred_order = ["Lorenz-2&3", "Lorenz-3&5", "Lorenz-6lag"]
    ordered_cases = [cases[name] for name in preferred_order if name in cases]
    ordered_cases.extend([case for name, case in cases.items() if name not in preferred_order])

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    def _case_save_path(case_name: str) -> Path:
        safe_name = (
            case_name.replace("Lorenz-", "Lorenz_")
            .replace("&", "_")
            .replace("-", "_")
            .replace(" ", "_")
        )
        return save_path.with_name(f"{save_path.stem}_{safe_name}{save_path.suffix}")

    with plt.rc_context(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 16,
            "axes.labelsize": 18,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 20,
            "axes.linewidth": 4,
            "lines.linewidth": 4,
        }
    ):
        mm_to_inch = 1.0 / 25.4
        figs = {}

        for case in ordered_cases:
            fig, ax = plt.subplots(
                1,
                1,
                figsize=(figure_width_mm * mm_to_inch, figure_height_mm * mm_to_inch),
                constrained_layout=True,
            )

            method_scores_by_tidx: Dict[str, Dict[int, np.ndarray]] = {}
            for method, score in baseline_scores.get(case.name, {}).items():
                arr = np.asarray(score, dtype=float)
                if arr.ndim == 3:
                    score_by_tidx = {}
                    for t_idx in t_idxs:
                        tau = int(round(t_idx * case.window_length / baseline_stride))
                        if tau < arr.shape[0]:
                            score_by_tidx[t_idx] = arr[tau]
                    method_scores_by_tidx[method] = score_by_tidx
                else:
                    method_scores_by_tidx[method] = {t_idx: arr for t_idx in t_idxs}

            method_scores_by_tidx["ISTC"] = {
                t_idx: case.tdi_mean[t_idx]
                for t_idx in t_idxs
                if t_idx < case.tdi_mean.shape[0]
            }

            handles = {}
            method_colors = {
                "ISTC": "#1f77b4",
                "PCMCI": "#d55e00",
                "Lagged CCM": "#009e73",
            }
            legend_order = ["ISTC", "PCMCI", "Lagged CCM"]

            for method in legend_order:
                if method not in method_scores_by_tidx:
                    continue
                y_true, y_score = _aggregate_truth_and_score_by_tidx(
                    case,
                    method_scores_by_tidx[method],
                    t_idxs,
                )
                if y_true.size == 0:
                    continue
                fpr, tpr, auc = roc_curve_from_binary_scores(y_true, y_score)
                if np.isnan(auc):
                    continue
                is_istc = method == "ISTC"
                line, = ax.plot(
                    fpr,
                    tpr,
                    linestyle="--",
                    label=f"{method} (AUC={auc:.2f})",
                    linewidth=3.0 if is_istc else 2.5,
                    zorder=10 if is_istc else 3,
                    color=method_colors.get(method),
                )
                handles[method] = line

            ax.plot([0, 1], [0, 1], "--", color="0.72", linewidth=0.8, zorder=1)
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.02, 1.02)
            ax.tick_params(width=1.5, length=3.5, pad=1.5)
            for spine in ax.spines.values():
                spine.set_linewidth(1.5)

            legend_handles = [handles[name] for name in legend_order if name in handles]
            legend_labels = [h.get_label() for h in legend_handles]
            if legend_handles:
                ax.legend(
                    legend_handles,
                    legend_labels,
                    loc="lower right",
                    frameon=True,
                    framealpha=0.9,
                    borderpad=0.3,
                    handlelength=2.0,
                    labelspacing=0.25,
                )

            out_path = _case_save_path(case.name)
            fig.savefig(out_path, bbox_inches="tight", dpi=600)
            figs[case.name] = fig

        plt.show()
        return figs
