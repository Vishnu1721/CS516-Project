# config.py

# ─── ACS / Data ────────────────────────────────────────────────
FEATURE_COLS = ['AGEP', 'COW', 'SCHL', 'MAR', 'OCCP', 'POBP',
                'RELP', 'WKHP', 'SEX', 'RAC1P']
TARGET_COL    = 'PINCP'
SENSITIVE_COL = 'sex_binary'

RANDOM_STATE = 42
TEST_SIZE    = 0.20

# ─── Training slice ────────────────────────────────────────────
TRAIN_STATE = 'CA'
TRAIN_YEAR  = 2015

# Optional cap on training data size (None = use all)
ACS_SAMPLE_SIZE = None

# ─── Deployment schedule (real distribution shift) ─────────────
# Each window = one real (state, year) slice. The frozen model
# trained on (TRAIN_STATE, TRAIN_YEAR) is deployed on each.
#
# Tuples: (state, year, label) where `label` describes the
# expected dominant shift type — used for analysis only.
DEPLOYMENT_SCHEDULE = [
    # ── Sanity check (no shift) ──
    ('CA', 2015, 'baseline_same'),

    # ── Temporal drift: same state, different years ──
    ('CA', 2016, 'temporal_1yr'),
    ('CA', 2017, 'temporal_2yr'),
    ('CA', 2018, 'temporal_3yr'),

    # ── Demographic shift: different state composition ──
    ('MS', 2018, 'demo_shift_MS'),    # different racial/economic mix
    ('HI', 2018, 'demo_shift_HI'),    # very different ethnic composition
    ('WV', 2018, 'demo_shift_WV'),    # low-income state

    # ── Feature drift: different economic structure ──
    ('ND', 2018, 'feature_drift_ND'), # oil economy → unusual hours/wages
    ('NY', 2018, 'feature_drift_NY'), # finance-heavy
    ('FL', 2018, 'feature_drift_FL'), # tourism + retiree mix

    # ── Combined real-world stress ──
    ('PR', 2018, 'combined_PR'),      # Puerto Rico — strong shift on all axes
    ('AK', 2018, 'combined_AK'),      # Alaska — small, atypical economy
]

N_WINDOWS = len(DEPLOYMENT_SCHEDULE)

# Cap each deployment window at this size for comparability.
# Smaller states (e.g., AK, WV) may have fewer rows after filtering;
# load_slice will return all available in that case.
WINDOW_SAMPLE_SIZE = 5000

# ─── Default decision threshold ────────────────────────────────
DEFAULT_THRESHOLD = 0.5

# ─── Fairness tolerances (trigger intervention if exceeded) ────
DPD_TOLERANCE     = 0.10
EOD_TOLERANCE     = 0.10
PP_TOLERANCE      = 0.10
ACC_GAP_TOLERANCE = 0.05