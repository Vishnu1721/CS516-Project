# config.py

# ── Data Source ──────────────────────────────────────────────
DATA_SOURCE = 'acs'          # 'acs' or 'compas' (kept for flexibility)
ACS_STATE = 'CA'             # which state to load (e.g., 'CA', 'TX', 'NY')
ACS_YEAR = 2018              # survey year
ACS_SAMPLE_SIZE = 15000      # subsample size for speed (set to None for full)

RANDOM_STATE = 42
TEST_SIZE = 0.3

# ── Features (ACS Income) ────────────────────────────────────
# These are the standard ACSIncome features from folktables
FEATURE_COLS = [
    'AGEP',    # Age
    'SCHL',    # Educational attainment
    'MAR',     # Marital status
    'WKHP',    # Usual hours worked per week
    'OCCP',    # Occupation code
    'POBP',    # Place of birth
    'RELP',    # Relationship
    'COW',     # Class of worker
]

TARGET_COL = 'income_binary'           # 1 = income > $50k, 0 otherwise
SENSITIVE_COL = 'sex_binary'   # 1 = Female, 0 = Male

# ── Deployment simulation ────────────────────────────────────
N_WINDOWS = 10
DEFAULT_THRESHOLD = 0.5

# Fairness tolerances — if gap exceeds these, intervention fires
DPD_TOLERANCE = 0.03
EOD_TOLERANCE = 0.05

# ── Drift schedules — one value per window ───────────────────
# Demographic shift: target ratio of group=1 (Female) in each window
# ACS sex is ~50/50, so we start from 0.50 (not 0.55 like COMPAS)
DEMO_SHIFT_SCHEDULE = [
    0.50, 0.50, 0.50,    # windows 0-2: baseline
    0.55, 0.60, 0.65,    # windows 3-5: growing shift
    0.70, 0.70, 0.70,    # windows 6-8: heavy shift
    0.65                 # window 9: partial recovery
]

FEATURE_DRIFT_SCHEDULE = [
    0.00, 0.00, 0.05,
    0.10, 0.10, 0.15,
    0.15, 0.20, 0.20,
    0.15
]

LABEL_NOISE_SCHEDULE = [
    0.00, 0.00, 0.00,
    0.00, 0.05, 0.05,
    0.10, 0.10, 0.15,
    0.10
]

# ── Paths ────────────────────────────────────────────────────
RESULTS_DIR = "results/"
PLOTS_DIR = "results/plots/"
METRICS_LOG = "results/metrics_log.csv"