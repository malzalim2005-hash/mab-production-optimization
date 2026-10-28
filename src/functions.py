# =======================================================================
# FUNCTIONS
# Reusable helpers for data loading, economics, model evaluation,
# uncertainty quantification, optimization and plotting.
# =======================================================================

import glob
import os
import sys

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize

from src.classes import NVARModel, PolynomialSurrogate
from src.constants import *


# -----------------------------------------------------------------------
# General utilities
# -----------------------------------------------------------------------
def progress_bar(c_iter, n_iters, label="", length_bar=20):
    progress = c_iter / n_iters
    length_fill = round(length_bar * progress)
    sys.stdout.write(f"\r{label:<28s}{progress:6.0%} [" + length_fill * "=" + (length_bar - length_fill) * " " + "]")
    if c_iter == n_iters:
        sys.stdout.write("\n")
    sys.stdout.flush()


def print_header(text):
    print("\n" + "=" * 72)
    print(text)
    print("=" * 72)


def save_table(df, file_name):
    df.to_csv(os.path.join(DIR_TABLES, file_name), index=False)


def save_figure(fig, file_name):
    fig.savefig(os.path.join(DIR_FIGURES, file_name))
    plt.close(fig)


# -----------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------
def load_experiments():
    """
    Load the initial conditions and every batch time series found in the datasets directory.

    Returns
    -------
    batch_ids : np.ndarray of shape (n_batches,)
    x_ic : np.ndarray of shape (n_batches, 3)
        Nominal initial glucose [mM], glutamine [mM] and VCD [cells/mL] of each batch.
    t_d : np.ndarray of shape (n_times,)
        Common sampling times [d].
    states : np.ndarray of shape (n_batches, n_times, 7)
        Measured process state of each batch at each time.
    """
    df_ic = pd.read_csv(FILE_INITIAL_CONDITIONS).set_index(COL_BATCH)

    batch_files = sorted(glob.glob(PATTERN_BATCH_FILES))
    batch_ids = []
    states = []
    t_h = None

    for file in batch_files:
        batch_id = int(os.path.basename(file).split("_")[-1].split(".")[0])
        df = pd.read_csv(file)

        if t_h is None:
            t_h = df.loc[:, COL_TIME].to_numpy()

        # Every batch must share the same sampling schedule so that curves can be stacked
        assert np.array_equal(t_h, df.loc[:, COL_TIME].to_numpy()), f"Unexpected sampling times in {file}"

        batch_ids.append(batch_id)
        states.append(df.loc[:, COLS_STATE].to_numpy())

    batch_ids = np.array(batch_ids)
    x_ic = df_ic.loc[batch_ids, COLS_IC].to_numpy()
    t_d = t_h / HOURS_PER_DAY

    return batch_ids, x_ic, t_d, np.array(states)


def initial_state(x_ic_row):
    """Known starting state: lactate, ammonia and mAb are zero and TCD equals VCD."""
    glc, gln, vcd = x_ic_row
    return np.array([glc, gln, 0.0, 0.0, vcd, vcd, 0.0])


def assign_design_stage(x_ic):
    """Label each batch as part of the Stage 1 factorial or the Stage 2 confirmation runs."""
    stages = []
    for glc, gln, vcd in x_ic:
        if (glc, gln, vcd) == STAGE_2_IC:
            stages.append("Stage 2 (confirmation)")
        elif glc in STAGE_1_GLC and gln in STAGE_1_GLN and vcd in STAGE_1_VCD:
            stages.append("Stage 1 (factorial)")
        else:
            stages.append("Other")
    return np.array(stages)


def candidate_grid():
    """All 7 x 7 x 7 = 343 combinations of the initial-condition levels that can be prepared."""
    return np.array([(g, q, v) for g in LEVELS_GLC for q in LEVELS_GLN for v in LEVELS_VCD])


# -----------------------------------------------------------------------
# Process economics
# -----------------------------------------------------------------------
def culture_cost(x_ic):
    x_ic = np.atleast_2d(x_ic)
    return COST_GLC * x_ic[:, 0] + COST_GLN * x_ic[:, 1] + COST_VCD * x_ic[:, 2]


def culture_value(mab):
    return VALUE_MAB * mab


def space_time_profit(mab, cost, t_harvest_d):
    return (culture_value(mab) - cost) / (t_harvest_d + DOWNTIME_D)


def space_time_yield(mab, t_harvest_d):
    return mab / (t_harvest_d + DOWNTIME_D)


def describe_observed_batches(batch_ids, x_ic, t_d, states, stages):
    """Summarize each batch: cost, peak titre and the best harvest time observed in the raw data."""
    mab = states[:, :, IDX_MAB]
    cost = culture_cost(x_ic)
    stp = space_time_profit(mab, cost[:, None], t_d[None, :])
    idx_best = np.argmax(stp, axis=1)

    df = pd.DataFrame({
        COL_BATCH: batch_ids,
        "Design stage": stages,
        COL_GLC: x_ic[:, 0],
        COL_GLN: x_ic[:, 1],
        COL_VCD: x_ic[:, 2],
        "Cost [$/L]": cost,
        "Peak mAb [mg/L]": np.max(mab, axis=1),
        "Peak VCD [cells/mL]": np.max(states[:, :, COLS_STATE.index(COL_VCD)], axis=1),
        "Best observed harvest time [d]": t_d[idx_best],
        "Best observed STP [$/(L d)]": stp[np.arange(len(batch_ids)), idx_best],
    })
    return df.round(3)




# -----------------------------------------------------------------------
# Training data
# -----------------------------------------------------------------------
def surrogate_training_data(x_ic, t_d, mab):
    """
    Flatten batch curves into one row per (batch, time) measurement.

    x_ic : (n_batches, 3), mab : (n_batches, n_times)
    """
    n_batches, n_times = mab.shape
    x_rep = np.repeat(x_ic, n_times, axis=0)
    t_rep = np.tile(t_d, n_batches)
    return x_rep, t_rep, mab.reshape(-1)


def nvar_training_data(states, p):
    """
    Build lagged-state features without letting lags cross batch boundaries.

    x_lag[i] = [x(t-1), x(t-2), ..., x(t-p)] and x_cur[i] = x(t).
    """
    x_lag = []
    x_cur = []
    for state in states:
        for t in range(p, state.shape[0]):
            x_lag.append(np.concatenate([state[t - k] for k in range(1, p + 1)]))
            x_cur.append(state[t])
    return np.array(x_lag), np.array(x_cur)


# -----------------------------------------------------------------------
# Model evaluation (leave-one-batch-out cross-validation)
# -----------------------------------------------------------------------
def curve_errors(mab_true, mab_pred, t_d):
    """RMSE over the full run and over the decision-relevant harvest window."""
    window = t_d <= HARVEST_WINDOW_D
    rmse_full = np.sqrt(np.mean((mab_true - mab_pred) ** 2))
    rmse_window = np.sqrt(np.mean((mab_true[window] - mab_pred[window]) ** 2))
    return rmse_full, rmse_window


def lobo_cv_surrogate(x_ic, t_d, mab, degree, alpha, time_transform):
    """
    Leave-one-batch-out CV of the direct surrogate.

    Returns the out-of-sample curve for every batch, shape (n_batches, n_times).
    """
    n_batches = x_ic.shape[0]
    mab_pred = np.zeros_like(mab)

    for i in range(n_batches):
        train = np.arange(n_batches) != i
        x_tr, t_tr, y_tr = surrogate_training_data(x_ic[train], t_d, mab[train])
        model = PolynomialSurrogate(degree, alpha, time_transform).fit(x_tr, t_tr, y_tr)
        mab_pred[i] = model.predict_curves(x_ic[i:i + 1], t_d)[0]

    return mab_pred


def lobo_cv_nvar(x_ic, states, model_type, hyperparameter, p, rng):
    """
    Leave-one-batch-out CV of an NVAR model.

    The held-out batch is forecast from its nominal initial state only, exactly as it would
    be used to plan a new run. The median of the stochastic forecasts is returned.
    """
    n_batches, n_times, _ = states.shape
    mab_pred = np.zeros((n_batches, n_times))

    for i in range(n_batches):
        train = np.arange(n_batches) != i
        x_lag, x_cur = nvar_training_data(states[train], p)
        model = NVARModel(model_type, hyperparameter, p).fit(x_lag, x_cur)
        forecasts = model.forecast_mab(initial_state(x_ic[i]), n_times, NVAR_N_SIMULATIONS_CV, rng)
        mab_pred[i] = np.median(forecasts, axis=0)

    return mab_pred


def compare_models(x_ic, t_d, states, rng):
    """
    Evaluate every surrogate and NVAR configuration with leave-one-batch-out CV.

    Returns a results table and the out-of-sample curves of every configuration.
    """
    mab = states[:, :, IDX_MAB]
    records = []
    predictions = {}

    configs = [("surrogate", dict(degree=d, alpha=a, time_transform=tt))
               for tt in SURROGATE_TIME_TRANSFORMS for d in SURROGATE_DEGREES for a in SURROGATE_ALPHAS]
    configs += [("nvar", cfg) for cfg in NVAR_CONFIGS]

    for c_iter, (family, cfg) in enumerate(configs, start=1):
        if family == "surrogate":
            mab_pred = lobo_cv_surrogate(x_ic, t_d, mab, **cfg)
            label = PolynomialSurrogate(**cfg).label()
            model_name = f"Surrogate ({cfg['time_transform']} time)"
        else:
            mab_pred = lobo_cv_nvar(x_ic, states, rng=rng, **cfg)
            label = NVARModel(**cfg).label()
            model_name = f"NVAR-{cfg['model_type']}"

        errors = np.array([curve_errors(mab[i], mab_pred[i], t_d) for i in range(mab.shape[0])])
        records.append(dict(
            Model=model_name,
            Configuration=label,
            family=family,
            config=cfg,
            **{"LOBO RMSE full run [mg/L]": np.mean(errors[:, 0]),
               f"LOBO RMSE first {HARVEST_WINDOW_D:g} d [mg/L]": np.mean(errors[:, 1]),
               "LOBO RMSE SD [mg/L]": np.std(errors[:, 0])}
        ))
        predictions[label] = mab_pred
        progress_bar(c_iter, len(configs), label="Cross-validating models")

    df = pd.DataFrame.from_records(records)
    return df, predictions



# -----------------------------------------------------------------------
# Uncertainty quantification (bootstrap prediction intervals)
# -----------------------------------------------------------------------
def residuals_by_time(model, x_ic, t_d, mab):
    """Training residuals arranged as (n_batches, n_times) so that noise can be sampled per time point."""
    return mab - model.predict_curves(x_ic, t_d)


def bootstrap_predictive_curves(x_ic, t_d, mab, model_kwargs, x_query, n_bootstrap, rng, label=""):
    """
    Sample the predictive distribution of a new batch's titre curve.

    Two sources of uncertainty are combined:
      1. Model uncertainty: batches are resampled with replacement and the surrogate is refit.
      2. Run-to-run / measurement noise: a residual observed at the same time point is added.

    Returns an array of shape (n_bootstrap, n_query, n_times).
    """
    n_batches, n_times = mab.shape
    n_query = x_query.shape[0]

    model_full = PolynomialSurrogate(**model_kwargs).fit(*surrogate_training_data(x_ic, t_d, mab))
    residuals = residuals_by_time(model_full, x_ic, t_d, mab)

    samples = np.zeros((n_bootstrap, n_query, n_times), dtype=np.float32)

    for b in range(n_bootstrap):
        idx = rng.integers(0, n_batches, size=n_batches)
        model = PolynomialSurrogate(**model_kwargs).fit(*surrogate_training_data(x_ic[idx], t_d, mab[idx]))
        curves = model.predict_curves(x_query, t_d)

        # Sample one residual per (query, time point) from the batches observed at that time
        idx_res = rng.integers(0, n_batches, size=(n_query, n_times))
        noise = residuals[idx_res, np.arange(n_times)]

        samples[b] = np.clip(curves + noise, a_min=0.0, a_max=None)

        if label and ((b + 1) % max(1, n_bootstrap // 50) == 0 or b + 1 == n_bootstrap):
            progress_bar(b + 1, n_bootstrap, label=label)

    return samples


def prediction_interval(samples, axis=0):
    alpha = 1.0 - PREDICTION_INTERVAL
    lower = np.quantile(samples, alpha / 2, axis=axis)
    upper = np.quantile(samples, 1 - alpha / 2, axis=axis)
    return lower, upper


def lobo_interval_coverage(x_ic, t_d, mab, model_kwargs, rng):
    """
    Check calibration: how often does a held-out batch fall inside its 95% prediction interval?
    """
    n_batches = x_ic.shape[0]
    window = t_d <= HARVEST_WINDOW_D
    inside_full = []
    inside_window = []
    widths = []

    for i in range(n_batches):
        train = np.arange(n_batches) != i
        samples = bootstrap_predictive_curves(x_ic[train], t_d, mab[train], model_kwargs, x_ic[i:i + 1],
                                              N_BOOTSTRAP_CV, rng)[:, 0, :]
        lower, upper = prediction_interval(samples)
        inside = (mab[i] >= lower) & (mab[i] <= upper)
        inside_full.append(np.mean(inside))
        inside_window.append(np.mean(inside[window]))
        widths.append(np.mean((upper - lower)[window]))
        progress_bar(i + 1, n_batches, label="Checking PI coverage")

    return pd.DataFrame([{
        "Nominal PI": PREDICTION_INTERVAL,
        "Coverage full run": np.mean(inside_full),
        f"Coverage first {HARVEST_WINDOW_D:g} d": np.mean(inside_window),
        f"Mean PI width first {HARVEST_WINDOW_D:g} d [mg/L]": np.mean(widths),
    }]).round(3)


# -----------------------------------------------------------------------
# Optimization
# -----------------------------------------------------------------------
def evaluate_grid(model, x_grid, t_d):
    """
    Predict the titre curve for every candidate and compute the economics at every harvest time.

    Returns dict of arrays with shape (n_candidates, n_times) plus the cost per candidate.
    """
    mab = model.predict_curves(x_grid, t_d)
    cost = culture_cost(x_grid)
    stp = space_time_profit(mab, cost[:, None], t_d[None, :])
    sty = space_time_yield(mab, t_d[None, :])
    return dict(mab=mab, cost=cost, stp=stp, sty=sty)


def rank_candidates(x_grid, t_d, grid_eval, objective="stp"):
    """For each candidate pick its best harvest time, then rank candidates by the objective."""
    values = grid_eval[objective]
    idx_t = np.argmax(values, axis=1)
    rows = np.arange(x_grid.shape[0])

    df = pd.DataFrame({
        COL_GLC: x_grid[:, 0],
        COL_GLN: x_grid[:, 1],
        COL_VCD: x_grid[:, 2],
        "Harvest time [d]": t_d[idx_t],
        "Cost [$/L]": grid_eval["cost"],
        "mAb [mg/L]": grid_eval["mab"][rows, idx_t],
        "Value [$/L]": culture_value(grid_eval["mab"][rows, idx_t]),
        "STP [$/(L d)]": grid_eval["stp"][rows, idx_t],
        "STY [mg/(L d)]": grid_eval["sty"][rows, idx_t],
        "idx_t": idx_t,
    })
    return df.sort_values(by="STP [$/(L d)]" if objective == "stp" else "STY [mg/(L d)]",
                          ascending=False).reset_index()


def summarize_recommendation(name, x_opt, t_harvest, mab_point, mab_samples):
    """Build the report row for one recommended operating point, including 95% PIs."""
    cost = culture_cost(x_opt)[0]
    value_s = culture_value(mab_samples)
    stp_s = space_time_profit(mab_samples, cost, t_harvest)
    sty_s = space_time_yield(mab_samples, t_harvest)

    def fmt(point, samples, decimals):
        lower, upper = prediction_interval(samples)
        return round(float(point), decimals), round(float(lower), decimals), round(float(upper), decimals)

    value = fmt(culture_value(mab_point), value_s, 1)
    stp = fmt(space_time_profit(mab_point, cost, t_harvest), stp_s, 1)
    sty = fmt(space_time_yield(mab_point, t_harvest), sty_s, 4)
    mab = fmt(mab_point, mab_samples, 3)

    return {
        "Operating point": name,
        COL_GLC: x_opt[0, 0],
        COL_GLN: x_opt[0, 1],
        COL_VCD: x_opt[0, 2],
        "Harvest time [h]": round(t_harvest * HOURS_PER_DAY, 1),
        "Harvest time [d]": round(t_harvest, 3),
        "Cost [$/L]": round(cost, 1),
        "mAb [mg/L]": mab[0], "mAb PI low": mab[1], "mAb PI high": mab[2],
        "Value [$/L]": value[0], "Value PI low": value[1], "Value PI high": value[2],
        "STP [$/(L d)]": stp[0], "STP PI low": stp[1], "STP PI high": stp[2],
        "STY [mg/(L d)]": sty[0], "STY PI low": sty[1], "STY PI high": sty[2],
        "P(STP > 0)": round(float(np.mean(stp_s > 0)), 3),
    }


# -----------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------
STAGE_COLORS = {"Stage 1 (factorial)": "tab:blue", "Stage 2 (confirmation)": "tab:orange", "Other": "tab:gray"}


def plot_design(x_ic, stages, x_grid):
    fig, axes = plt.subplots(1, 3, figsize=(24 / 2.54, 8 / 2.54), dpi=300, layout="constrained")
    pairs = [(0, 1), (0, 2), (1, 2)]
    labels = [COL_GLC, COL_GLN, "VCD [10$^6$ cells/mL]"]
    scale = np.array([1.0, 1.0, 1.0e-6])

    for ax, (i, j) in zip(axes, pairs):
        ax.scatter(x_grid[:, i] * scale[i], x_grid[:, j] * scale[j], s=8, color="lightgray", label="Candidate levels",
                   zorder=1)
        for stage, color in STAGE_COLORS.items():
            mask = stages == stage
            if np.any(mask):
                ax.scatter(x_ic[mask, i] * scale[i], x_ic[mask, j] * scale[j], s=40, color=color,
                           edgecolors="black", label=stage, zorder=2)
        ax.set_xlabel(labels[i])
        ax.set_ylabel(labels[j])

    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncols=3, fontsize=7,
               frameon=False)
    save_figure(fig, "figure_01_experimental_design.png")


def plot_trajectories(x_ic, t_d, states):
    fig, axes = plt.subplots(2, 3, figsize=(24 / 2.54, 13 / 2.54), dpi=300, layout="constrained")
    norm = Normalize(vmin=min(LEVELS_GLC), vmax=max(LEVELS_GLC))
    cmap = plt.get_cmap("viridis")
    variables = [COL_GLC, COL_GLN, COL_LAC, COL_VCD, COL_TCD, COL_MAB]

    for ax, variable in zip(axes.flat, variables):
        idx = COLS_STATE.index(variable)
        for i in range(states.shape[0]):
            ax.plot(t_d, states[i, :, idx], color=cmap(norm(x_ic[i, 0])), linewidth=0.8, alpha=0.8)
        ax.set_xlabel("Time [d]")
        ax.set_ylabel(variable)

    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes, label="Initial glucose [mM]", shrink=0.6)
    save_figure(fig, "figure_02_batch_trajectories.png")


def plot_model_comparison(t_d, mab, batch_ids, cv_predictions, best_labels, batch_examples):
    fig, axes = plt.subplots(1, len(batch_examples), figsize=(24 / 2.54, 8 / 2.54), dpi=300,
                             layout="constrained", sharey=True)
    colors = ["black", "tab:red", "tab:green", "tab:purple"]

    for ax, batch_id in zip(axes, batch_examples):
        i = int(np.argmax(batch_ids == batch_id))
        ax.scatter(t_d, mab[i], s=8, color="tab:orange", edgecolors="black", linewidths=0.3, label="Observed",
                   zorder=3)
        for color, label in zip(colors, best_labels):
            ax.plot(t_d, cv_predictions[label][i], color=color, linewidth=1.2, label=label, zorder=4)
        ax.set_title(f"Held-out batch {batch_id}", fontsize=9)
        ax.set_xlabel("Time [d]")

    axes[0].set_ylabel(COL_MAB)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncols=2, fontsize=7, frameon=False)
    save_figure(fig, "figure_03_model_comparison.png")


def plot_confirmation(t_d, mab_conf, cost_conf, samples, mab_point):
    lower, upper = prediction_interval(samples)
    stp_s = space_time_profit(samples, cost_conf, t_d)
    stp_lower, stp_upper = prediction_interval(stp_s)

    fig, axes = plt.subplots(1, 2, figsize=(20 / 2.54, 8 / 2.54), dpi=300, layout="constrained")

    axes[0].fill_between(t_d, lower, upper, color="tab:blue", alpha=0.3, edgecolor="none", label="95% PI")
    axes[0].plot(t_d, mab_point, color="black", label="Stage 1 prediction")
    axes[1].fill_between(t_d, stp_lower, stp_upper, color="tab:blue", alpha=0.3, edgecolor="none", label="95% PI")
    axes[1].plot(t_d, space_time_profit(mab_point, cost_conf, t_d), color="black", label="Stage 1 prediction")

    markers = ["o", "s", "^"]
    for k, mab_run in enumerate(mab_conf):
        axes[0].scatter(t_d, mab_run, s=10, marker=markers[k % 3], color="tab:orange", edgecolors="black",
                        linewidths=0.3, label=f"Confirmation run {k + 1}", zorder=3)
        axes[1].scatter(t_d, space_time_profit(mab_run, cost_conf, t_d), s=10, marker=markers[k % 3],
                        color="tab:orange", edgecolors="black", linewidths=0.3, zorder=3)

    axes[0].set_ylabel(COL_MAB)
    axes[1].set_ylabel("Space-time profit [$/(L d)]")
    axes[1].set_xlim(0, HARVEST_WINDOW_D)
    axes[1].set_ylim(-150, 200)
    axes[1].axhline(0, color="gray", linewidth=0.8)
    for ax in axes:
        ax.set_xlabel("Time [d]")
    axes[0].legend(fontsize=7)
    save_figure(fig, "figure_04_confirmation_runs.png")


def plot_profit_heatmaps(x_grid, best_stp, x_opt):
    n_vcd = len(LEVELS_VCD)
    fig, axes = plt.subplots(1, n_vcd, figsize=(30 / 2.54, 7 / 2.54), dpi=300, layout="constrained",
                             sharey=True)
    vmax = np.max(np.abs(best_stp))
    image = None

    for ax, vcd in zip(axes, LEVELS_VCD):
        z = np.zeros((len(LEVELS_GLN), len(LEVELS_GLC)))
        for i, gln in enumerate(LEVELS_GLN):
            for j, glc in enumerate(LEVELS_GLC):
                k = np.argmax((x_grid[:, 0] == glc) & (x_grid[:, 1] == gln) & (x_grid[:, 2] == vcd))
                z[i, j] = best_stp[k]
        image = ax.imshow(z, origin="lower", cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto",
                          extent=(2.5, 37.5, 0.5, 7.5))
        if vcd == x_opt[2]:
            ax.scatter(x_opt[0], x_opt[1], marker="*", s=120, color="gold", edgecolors="black", zorder=3)
        ax.set_title(f"VCD$_0$ = {vcd / 1e6:.1f}×10$^6$", fontsize=8)
        ax.set_xlabel("Glucose [mM]")
        ax.set_xticks(LEVELS_GLC[::2])

    axes[0].set_ylabel("Glutamine [mM]")
    fig.colorbar(image, ax=axes, label="Max. STP [$/(L d)]", shrink=0.9)
    save_figure(fig, "figure_05_profit_landscape.png")


def plot_main_effects(effects):
    """
    effects : dict factor -> dict with arrays: levels, stp, stp_lo, stp_hi, sty, sty_lo, sty_hi, t_stp, t_sty
    """
    fig, axes = plt.subplots(3, 3, figsize=(24 / 2.54, 18 / 2.54), dpi=300, layout="constrained", sharex="col")
    names = list(effects.keys())

    for j, name in enumerate(names):
        e = effects[name]
        x = e["levels"] / (1.0e6 if name == COL_VCD else 1.0)

        axes[0, j].fill_between(x, e["stp_lo"], e["stp_hi"], color="tab:blue", alpha=0.3, edgecolor="none")
        axes[0, j].plot(x, e["stp"], "o-", color="tab:blue", markersize=4)
        axes[0, j].axhline(0, color="gray", linewidth=0.8)

        axes[1, j].fill_between(x, e["sty_lo"], e["sty_hi"], color="tab:green", alpha=0.3, edgecolor="none")
        axes[1, j].plot(x, e["sty"], "o-", color="tab:green", markersize=4)

        axes[2, j].plot(x, e["t_stp"], "o-", color="tab:blue", markersize=4, label="STP-optimal")
        axes[2, j].plot(x, e["t_sty"], "s--", color="tab:green", markersize=4, label="STY-optimal")
        axes[2, j].set_xlabel("Initial VCD [10$^6$ cells/mL]" if name == COL_VCD else f"Initial {name}")

    axes[0, 0].set_ylabel("Max. STP [$/(L d)]")
    axes[1, 0].set_ylabel("Max. STY [mg/(L d)]")
    axes[2, 0].set_ylabel("Best harvest time [d]")
    axes[2, 0].legend(fontsize=7)
    save_figure(fig, "figure_06_main_effects.png")


def plot_optimum_profile(t_d, x_opt, mab_point, samples, t_star, mab_obs_runs):
    cost = culture_cost(x_opt)[0]
    stp_s = space_time_profit(samples, cost, t_d)
    sty_s = space_time_yield(samples, t_d)
    window = t_d <= HARVEST_WINDOW_D

    panels = [
        (samples, mab_point, COL_MAB, lambda m: m),
        (stp_s, space_time_profit(mab_point, cost, t_d), "Space-time profit [$/(L d)]",
         lambda m: space_time_profit(m, cost, t_d)),
        (sty_s, space_time_yield(mab_point, t_d), "Space-time yield [mg/(L d)]", lambda m: space_time_yield(m, t_d)),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(24 / 2.54, 8 / 2.54), dpi=300, layout="constrained")
    for ax, (s, point, ylabel, transform) in zip(axes, panels):
        lower, upper = prediction_interval(s)
        ax.fill_between(t_d[window], lower[window], upper[window], color="tab:blue", alpha=0.3, edgecolor="none",
                        label="95% PI")
        ax.plot(t_d[window], point[window], color="black", label="Final model")
        for k, mab_run in enumerate(mab_obs_runs):
            ax.scatter(t_d[window], transform(mab_run)[window], s=8, color="tab:orange", edgecolors="black",
                       linewidths=0.3, label="Observed (stage 2)" if k == 0 else None, zorder=3)
        ax.axvline(t_star, color="gray", linestyle="--", linewidth=0.8, label="Recommended harvest")
        ax.set_xlabel("Time [d]")
        ax.set_ylabel(ylabel)

    axes[1].set_ylim(-150, 200)
    axes[0].legend(fontsize=7)
    save_figure(fig, "figure_07_recommended_operating_point.png")