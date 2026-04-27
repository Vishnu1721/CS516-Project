# config.py

# Data
DATA_PATH = "data/compas-scores-two-years.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.3

# Features
FEATURE_COLS = [
    'age',
    'priors_count',
    'juv_fel_count',
    'juv_misd_count',
    'c_charge_degree'
]
TARGET_COL = 'two_year_recid'
SENSITIVE_COL = 'race_binary'   # 1 = African-American, 0 = Caucasian

# Deployment simulation
N_WINDOWS = 10
DEFAULT_THRESHOLD = 0.5

# Fairness tolerances — if gap exceeds these, intervention fires
DPD_TOLERANCE = 0.10
EOD_TOLERANCE = 0.10

# Drift schedules — one value per window
DEMO_SHIFT_SCHEDULE = [
    0.55, 0.55, 0.55,   # windows 0-2: baseline
    0.60, 0.65, 0.70,   # windows 3-5: growing shift
    0.75, 0.75, 0.75,   # windows 6-8: heavy shift
    0.70                # window 9: partial recovery
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

# Paths
RESULTS_DIR = "results/"
PLOTS_DIR = "results/plots/"
METRICS_LOG = "results/metrics_log.csv"