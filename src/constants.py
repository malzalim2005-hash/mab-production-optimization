# =======================================================================
# CONSTANTS
# Values that do not change throughout the program.
# =======================================================================

import os

# -----------------------------------------------------------------------
# Directories and files
# -----------------------------------------------------------------------
DIR_DATASETS = "datasets"
DIR_FIGURES = "figures"
DIR_TABLES = "tables"
FILE_INITIAL_CONDITIONS = os.path.join(DIR_DATASETS, "initial_conditions.csv")
PATTERN_BATCH_FILES = os.path.join(DIR_DATASETS, "batch_*.csv")

# -----------------------------------------------------------------------
# Column names
# -----------------------------------------------------------------------
COL_BATCH = "Batch ID"
COL_TIME = "Time [h]"
COL_GLC = "Glucose [mM]"
COL_GLN = "Glutamine [mM]"
COL_LAC = "Lactate [mM]"
COL_AMM = "Ammonia [mM]"
COL_VCD = "VCD [cells/mL]"
COL_TCD = "TCD [cells/mL]"
COL_MAB = "mAb [mg/L]"

COLS_IC = [COL_GLC, COL_GLN, COL_VCD]
COLS_STATE = [COL_GLC, COL_GLN, COL_LAC, COL_AMM, COL_VCD, COL_TCD, COL_MAB]
IDX_MAB = COLS_STATE.index(COL_MAB)

HOURS_PER_DAY = 24.0

# -----------------------------------------------------------------------
# Process economics
# -----------------------------------------------------------------------
COST_GLC = 10.0  # $/L per mM of initial glucose
COST_GLN = 20.0  # $/L per mM of initial glutamine
COST_VCD = 0.0005  # $/L per cell/mL of initial VCD
VALUE_MAB = 1000.0  # $/L per mg/L of mAb
DOWNTIME_D = 1.0  # Bioreactor downtime between batches [d]

# -----------------------------------------------------------------------
# Experimental design
# -----------------------------------------------------------------------
# Levels that can be prepared for each initial condition (candidate space)
LEVELS_GLC = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0]
LEVELS_GLN = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
LEVELS_VCD = [2.0e5, 4.0e5, 6.0e5, 8.0e5, 1.0e6, 1.2e6, 1.4e6]

# Stage 1: 3 x 3 x 3 full factorial on the low / mid / high level of each factor
STAGE_1_GLC = [5.0, 20.0, 35.0]
STAGE_1_GLN = [1.0, 4.0, 7.0]
STAGE_1_VCD = [2.0e5, 8.0e5, 1.4e6]

# Stage 2: confirmation runs at the optimum predicted from the Stage 1 data
STAGE_2_IC = (15.0, 3.0, 6.0e5)

# -----------------------------------------------------------------------
# Modelling
# -----------------------------------------------------------------------
RANDOM_SEED = 0
PREDICTION_INTERVAL = 0.95

# Direct (initial conditions + time -> mAb) polynomial surrogate
SURROGATE_DEGREES = [2, 3, 4, 5]
SURROGATE_ALPHAS = [1.0e-3, 1.0e-1, 1.0e1, 1.0e2]
SURROGATE_TIME_TRANSFORMS = ["linear", "log"]

# Non-linear vector autoregression (NVAR) alternatives
NVAR_CONFIGS = [
    dict(model_type="poly", hyperparameter=2, p=1),
    dict(model_type="poly", hyperparameter=2, p=2),
    dict(model_type="knn", hyperparameter=5, p=1),
    dict(model_type="knn", hyperparameter=5, p=2),
    dict(model_type="knn", hyperparameter=10, p=1),
    dict(model_type="knn", hyperparameter=10, p=2),
]
NVAR_N_SIMULATIONS_CV = 100

# Bootstrap used for prediction intervals
N_BOOTSTRAP = 1000
N_BOOTSTRAP_CV = 200

# Harvest window used to report a second, decision-relevant error metric [d]
HARVEST_WINDOW_D = 10.0