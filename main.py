# =====================================================================
# MAIN PROGRAM
# Data-driven optimization of monoclonal antibody (mAb) batch production.
# =====================================================================

import numpy as np
import pandas as pd

import src.functions as sf
from src.classes import PolynomialSurrogate
from src.constants import *

rng = np.random.default_rng(RANDOM_SEED)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

# ---------------------------------------------------------------------
# 1. Load the 30 batch runs and describe the experimental design
# ---------------------------------------------------------------------
sf.print_header("1. Loading experimental data")

batch_ids, x_ic, t_d, states = sf.load_experiments()
mab = states[:, :, IDX_MAB]
stages = sf.assign_design_stage(x_ic)
x_grid = sf.candidate_grid()

print(f"Batches loaded: {len(batch_ids)} | Time points per batch: {len(t_d)} | Run length: {t_d[-1]:.0f} d")
for stage in np.unique(stages):
    print(f"  {stage}: {np.sum(stages == stage)} batches")

df_observed = sf.describe_observed_batches(batch_ids, x_ic, t_d, states, stages)
sf.save_table(df_observed, "table_01_observed_batches.csv")
sf.plot_design(x_ic, stages, x_grid)
sf.plot_trajectories(x_ic, t_d, states)

# ---------------------------------------------------------------------
# 2. Model selection with leave-one-batch-out cross-validation
# ---------------------------------------------------------------------
sf.print_header("2. Comparing candidate models (leave-one-batch-out CV)")

df_cv, cv_predictions = sf.compare_models(x_ic, t_d, states, rng)
col_rmse_window = f"LOBO RMSE first {HARVEST_WINDOW_D:g} d [mg/L]"
df_cv = df_cv.sort_values(by=col_rmse_window).reset_index(drop=True)

# The decision only depends on the titre during the harvest window, so models are ranked on that error
df_best_per_model = df_cv.groupby("Model", sort=False).head(1).reset_index(drop=True)
best = df_cv.loc[(df_cv["family"] == "surrogate")].iloc[0]
model_kwargs = best["config"]

cols_report = ["Model", "Configuration", "LOBO RMSE full run [mg/L]", col_rmse_window, "LOBO RMSE SD [mg/L]"]
sf.save_table(df_cv.loc[:, cols_report].round(4), "table_02_model_selection_all_configurations.csv")
sf.save_table(df_best_per_model.loc[:, cols_report].round(4), "table_03_model_comparison_best_per_family.csv")
print(df_best_per_model.loc[:, cols_report].round(4).to_string(index=False))
print(f"\nSelected model: {best['Configuration']}")

example_batches = [batch_ids[stages == "Stage 1 (factorial)"][0],
                   batch_ids[stages == "Stage 1 (factorial)"][13],
                   batch_ids[stages == "Stage 2 (confirmation)"][0]]
sf.plot_model_comparison(t_d, mab, batch_ids, cv_predictions, list(df_best_per_model["Configuration"]),
                         example_batches)

# ---------------------------------------------------------------------
# 3. Sequential design check: did the Stage 1 model predict the confirmation runs?
# ---------------------------------------------------------------------
sf.print_header("3. Validating Stage 1 predictions against the confirmation runs")

mask_s1 = stages == "Stage 1 (factorial)"
mask_s2 = stages == "Stage 2 (confirmation)"
x_conf = np.array([STAGE_2_IC])
cost_conf = sf.culture_cost(x_conf)[0]

model_s1 = PolynomialSurrogate(**model_kwargs).fit(*sf.surrogate_training_data(x_ic[mask_s1], t_d, mab[mask_s1]))
mab_point_s1 = model_s1.predict_curves(x_conf, t_d)[0]
samples_s1 = sf.bootstrap_predictive_curves(x_ic[mask_s1], t_d, mab[mask_s1], model_kwargs, x_conf, N_BOOTSTRAP,
                                            rng, label="Bootstrapping Stage 1 model")[:, 0, :]

idx_t_s1 = int(np.argmax(sf.space_time_profit(mab_point_s1, cost_conf, t_d)))
lower_s1, upper_s1 = sf.prediction_interval(samples_s1)
records = []
for batch_id, mab_run in zip(batch_ids[mask_s2], mab[mask_s2]):
    window = t_d <= HARVEST_WINDOW_D
    records.append({
        COL_BATCH: batch_id,
        "Harvest time [d]": t_d[idx_t_s1],
        "Predicted mAb [mg/L]": mab_point_s1[idx_t_s1],
        "PI low [mg/L]": lower_s1[idx_t_s1],
        "PI high [mg/L]": upper_s1[idx_t_s1],
        "Observed mAb [mg/L]": mab_run[idx_t_s1],
        "Predicted STP [$/(L d)]": sf.space_time_profit(mab_point_s1[idx_t_s1], cost_conf, t_d[idx_t_s1]),
        "Observed STP [$/(L d)]": sf.space_time_profit(mab_run[idx_t_s1], cost_conf, t_d[idx_t_s1]),
        f"Fraction of points in PI (first {HARVEST_WINDOW_D:g} d)":
            np.mean((mab_run[window] >= lower_s1[window]) & (mab_run[window] <= upper_s1[window])),
        f"RMSE first {HARVEST_WINDOW_D:g} d [mg/L]": np.sqrt(np.mean((mab_run[window] - mab_point_s1[window]) ** 2)),
    })
df_conf = pd.DataFrame(records).round(3)
sf.save_table(df_conf, "table_04_confirmation_runs.csv")
print(df_conf.to_string(index=False))
sf.plot_confirmation(t_d, mab[mask_s2], cost_conf, samples_s1, mab_point_s1)

# ---------------------------------------------------------------------
# 4. Calibration of the 95% prediction intervals
# ---------------------------------------------------------------------
sf.print_header("4. Checking the calibration of the 95% prediction intervals")

df_coverage = sf.lobo_interval_coverage(x_ic, t_d, mab, model_kwargs, rng)
sf.save_table(df_coverage, "table_05_prediction_interval_coverage.csv")
print(df_coverage.to_string(index=False))

# ---------------------------------------------------------------------
# 5. Optimize the initial conditions and harvest time with the final model
# ---------------------------------------------------------------------
sf.print_header("5. Optimizing initial conditions and harvest time")

model_final = PolynomialSurrogate(**model_kwargs).fit(*sf.surrogate_training_data(x_ic, t_d, mab))
grid_eval = sf.evaluate_grid(model_final, x_grid, t_d)

df_rank_stp = sf.rank_candidates(x_grid, t_d, grid_eval, objective="stp")
df_rank_sty = sf.rank_candidates(x_grid, t_d, grid_eval, objective="sty")
cols_rank = [COL_GLC, COL_GLN, COL_VCD, "Harvest time [d]", "Cost [$/L]", "mAb [mg/L]", "Value [$/L]",
             "STP [$/(L d)]", "STY [mg/(L d)]"]
sf.save_table(df_rank_stp.loc[:14, cols_rank].round(3), "table_06_top_candidates_space_time_profit.csv")
sf.save_table(df_rank_sty.loc[:14, cols_rank].round(3), "table_07_top_candidates_space_time_yield.csv")

x_opt = x_grid[[df_rank_stp.at[0, "index"]]]
x_opt_sty = x_grid[[df_rank_sty.at[0, "index"]]]

# Points needed for prediction intervals: both optima and every one-factor-at-a-time variation around the optimum
effect_points = {}
for j, (name, levels) in enumerate(zip(COLS_IC, [LEVELS_GLC, LEVELS_GLN, LEVELS_VCD])):
    x_line = np.repeat(x_opt, len(levels), axis=0)
    x_line[:, j] = levels
    effect_points[name] = x_line

x_query = np.concatenate([x_opt, x_opt_sty] + list(effect_points.values()), axis=0)
samples = sf.bootstrap_predictive_curves(x_ic, t_d, mab, model_kwargs, x_query, N_BOOTSTRAP, rng,
                                         label="Bootstrapping final model")
mab_query = model_final.predict_curves(x_query, t_d)

# Recommended operating points with 95% prediction intervals
records = []
for k, (name, x_point, objective) in enumerate([("Max. space-time profit", x_opt, "stp"),
                                                ("Max. space-time yield", x_opt_sty, "sty")]):
    df_rank = df_rank_stp if objective == "stp" else df_rank_sty
    idx_t = int(df_rank.at[0, "idx_t"])
    records.append(sf.summarize_recommendation(name, x_point, t_d[idx_t], mab_query[k, idx_t], samples[:, k, idx_t]))
df_recommendation = pd.DataFrame(records)
sf.save_table(df_recommendation, "table_08_recommended_operating_points.csv")
print(df_recommendation.T.to_string(header=False))

# One-factor-at-a-time effects of the initial conditions (harvest time re-optimized at each level)
effects = {}
offset = 2
for name, x_line in effect_points.items():
    n = x_line.shape[0]
    mab_line = mab_query[offset:offset + n]
    s_line = samples[:, offset:offset + n, :]
    cost_line = sf.culture_cost(x_line)

    stp_line = sf.space_time_profit(mab_line, cost_line[:, None], t_d[None, :])
    sty_line = sf.space_time_yield(mab_line, t_d[None, :])
    idx_stp = np.argmax(stp_line, axis=1)
    idx_sty = np.argmax(sty_line, axis=1)
    rows = np.arange(n)

    stp_samples = sf.space_time_profit(s_line[:, rows, idx_stp], cost_line, t_d[idx_stp])
    sty_samples = sf.space_time_yield(s_line[:, rows, idx_sty], t_d[idx_sty])
    stp_lo, stp_hi = sf.prediction_interval(stp_samples)
    sty_lo, sty_hi = sf.prediction_interval(sty_samples)

    effects[name] = dict(levels=x_line[:, COLS_IC.index(name)],
                         stp=stp_line[rows, idx_stp], stp_lo=stp_lo, stp_hi=stp_hi,
                         sty=sty_line[rows, idx_sty], sty_lo=sty_lo, sty_hi=sty_hi,
                         t_stp=t_d[idx_stp], t_sty=t_d[idx_sty])
    offset += n

df_effects = pd.concat([pd.DataFrame({"Factor varied": name, "Level": e["levels"],
                                      "Max. STP [$/(L d)]": e["stp"], "STP PI low": e["stp_lo"],
                                      "STP PI high": e["stp_hi"], "STP-optimal harvest [d]": e["t_stp"],
                                      "Max. STY [mg/(L d)]": e["sty"], "STY PI low": e["sty_lo"],
                                      "STY PI high": e["sty_hi"], "STY-optimal harvest [d]": e["t_sty"]})
                        for name, e in effects.items()]).round(4)
sf.save_table(df_effects, "table_09_main_effects.csv")

best_stp_per_candidate = np.max(grid_eval["stp"], axis=1)
sf.plot_profit_heatmaps(x_grid, best_stp_per_candidate, x_opt[0])
sf.plot_main_effects(effects)

mask_opt_runs = np.all(x_ic == x_opt, axis=1)
idx_t_opt = int(df_rank_stp.at[0, "idx_t"])
sf.plot_optimum_profile(t_d, x_opt, mab_query[0], samples[:, 0, :], t_d[idx_t_opt], mab[mask_opt_runs])

sf.print_header("Done. Tables exported to 'tables/' and figures exported to 'figures/'.")