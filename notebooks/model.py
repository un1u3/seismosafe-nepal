from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


ROOT = Path("..") if Path("../data").exists() else Path(".")
DATA_DIR = ROOT / "data"

# Load the engineered train/test files so both datasets have matching columns.
train = pd.read_csv(DATA_DIR / "preprocecssed" / "train.csv")
test = pd.read_csv(DATA_DIR / "preprocecssed" / "test.csv")
labels = pd.read_csv(DATA_DIR / "raw" / "train_labels.csv")

ordinal_categories = {
    "land_surface_condition": ["t", "n", "o"],
    "foundation_type": ["i", "w", "u", "h", "r"],
    "roof_type": ["x", "n", "q"],
    "ground_floor_type": ["v", "m", "z", "x", "f"],
    "other_floor_type": ["s", "j", "x", "q"],
    "position": ["j", "o", "s", "t"],
    "plan_configuration": ["c", "a", "o", "m", "u", "s", "n", "d", "q", "f"],
}

target_col = "damage_grade"
drop_cols = ["Unnamed: 0", target_col]

X = train.drop(columns=drop_cols, errors="ignore")
y = train[target_col]
X_test_raw = test.drop(columns=drop_cols, errors="ignore")

missing_from_test = sorted(set(X.columns) - set(X_test_raw.columns))
extra_in_test = sorted(set(X_test_raw.columns) - set(X.columns))

if missing_from_test:
    raise ValueError(f"Test data is missing training columns: {missing_from_test}")

X_test_raw = X_test_raw[X.columns]

ord_cols = [col for col in ordinal_categories if col in X.columns]
ord_cats = [ordinal_categories[col] for col in ord_cols]
num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
nominal_cols = [
    col for col in X.select_dtypes(include="object").columns
    if col not in ord_cols
]

ordinal_encoder = OrdinalEncoder(
    categories=ord_cats,
    handle_unknown="use_encoded_value",
    unknown_value=-1,
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("ord", ordinal_encoder, ord_cols),
        ("nom", OneHotEncoder(handle_unknown="ignore"), nominal_cols),
    ]
)

X_train = preprocessor.fit_transform(X)
X_test = preprocessor.transform(X_test_raw)

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y:", y.shape)
print("extra columns ignored from test:", extra_in_test)
