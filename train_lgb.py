import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings('ignore')

# 1. Load Data
print("Loading raw data...")
train_values = pd.read_csv('data/raw/train_values.csv')
train_labels = pd.read_csv('data/raw/train_labels.csv')
test_values = pd.read_csv('data/raw/test_values.csv')

# Merge training labels
train = pd.merge(train_values, train_labels, on='building_id')

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    floors = df["count_floors_pre_eq"].replace(0, np.nan)
    area = df["area_percentage"].replace(0, np.nan)

    # --- Existing features ---
    df["age_clipped"] = df["age"].clip(0, 100)
    df["age_bin_old"] = (df["age_clipped"] >= 30).astype(int)
    df["age_is_zero"] = (df["age"] == 0).astype(int)

    df["height_per_floor"] = df["height_percentage"] / floors
    df["area_per_floor"] = df["area_percentage"] / floors
    df["slenderness"] = df["height_percentage"] / area
    df["volume_proxy"] = df["area_percentage"] * df["height_percentage"]
    df["is_tall"] = (df["count_floors_pre_eq"] >= 4).astype(int)

    df["old_mud_stone"] = df["age_bin_old"] * df["has_superstructure_mud_mortar_stone"]

    df["families_per_floor"] = df["count_families"] / floors
    df["families_per_area"] = df["count_families"] / area
    df["has_multiple_families"] = (df["count_families"] > 1).astype(int)

    df["vertical_risk"] = df["count_floors_pre_eq"] * df["height_percentage"]

    # --- Superstructure features ---
    df["fragile_score"] = (
        df["has_superstructure_adobe_mud"]
        + df["has_superstructure_mud_mortar_stone"]
        + df["has_superstructure_bamboo"]
    )
    df["strong_score"] = (
        df["has_superstructure_rc_engineered"]
        + df["has_superstructure_cement_mortar_brick"]
    )
    df["material_risk"] = df["fragile_score"] - df["strong_score"]
    df["rc_any"] = (
        (df["has_superstructure_rc_engineered"]
         + df["has_superstructure_rc_non_engineered"]) > 0
    ).astype(int)
    df["mud_dominant"] = (
        df["fragile_score"] > df["strong_score"]
    ).astype(int)
    df["age_x_fragile"] = df["age_clipped"] * df["fragile_score"]
    df["floors_x_height"] = df["count_floors_pre_eq"] * df["height_percentage"]

    # --- Geo interaction ---
    df["geo1_geo2"] = (
        df["geo_level_1_id"] * 10000 + df["geo_level_2_id"]
    )
    df["geo2_geo3"] = (
        df["geo_level_2_id"] * 100000 + df["geo_level_3_id"]
    )

    # --- Foundation + roof combo ---
    df["foundation_roof"] = (
        df["foundation_type"].astype(str) + "_" + df["roof_type"].astype(str)
    ).astype("category").cat.codes

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    return df

print("Applying feature engineering...")
train_df = feature_engineering(train)
test_df = feature_engineering(test_values)

# Align columns
X = train_df.drop(columns=['building_id', 'damage_grade'], errors='ignore')
y = train_df['damage_grade']
X_test = test_df.drop(columns=['building_id'], errors='ignore')

# Categorical column encoding
cat_cols = [
    'land_surface_condition', 'foundation_type', 'roof_type',
    'ground_floor_type', 'other_floor_type', 'position',
    'plan_configuration', 'legal_ownership_status'
]

for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))

# --- Replicating model training code ---
geo_cols = ['geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id']

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X))
test_preds = np.zeros((len(X_test), 5))

model_params = {
    "n_estimators": 2000,
    "learning_rate": 0.02,
    "num_leaves": 255,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "random_state": 42,
    "verbose": -1,
    "n_jobs": -1
}

SMOOTH_K = 10

print("Starting cross-validation training...")
for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr = X.iloc[tr_idx].copy().reset_index(drop=True)
    X_val = X.iloc[val_idx].copy().reset_index(drop=True)
    X_te = X_test.copy().reset_index(drop=True)
    
    y_tr = y.iloc[tr_idx].reset_index(drop=True)
    y_val = y.iloc[val_idx].reset_index(drop=True)
    
    global_mean = y_tr.mean()
    
    # Apply smoothed target encoding within the fold to avoid leakage
    for geo in geo_cols:
        agg = y_tr.groupby(X_tr[geo]).agg(['mean', 'count'])
        smoothed = (agg['mean'] * agg['count'] + global_mean * SMOOTH_K) / (agg['count'] + SMOOTH_K)
        
        X_tr[f'{geo}_te'] = X_tr[geo].map(smoothed).fillna(global_mean)
        X_val[f'{geo}_te'] = X_val[geo].map(smoothed).fillna(global_mean)
        X_te[f'{geo}_te'] = X_te[geo].map(smoothed).fillna(global_mean)
        
        for cls in [1, 2, 3]:
            binary = (y_tr == cls).astype(float)
            cls_agg = binary.groupby(X_tr[geo]).agg(['mean', 'count'])
            global_cls_mean = binary.mean()
            cls_smoothed = (cls_agg['mean'] * cls_agg['count'] + global_cls_mean * SMOOTH_K) / (cls_agg['count'] + SMOOTH_K)
            
            X_tr[f'{geo}_cls{cls}'] = X_tr[geo].map(cls_smoothed).fillna(global_cls_mean)
            X_val[f'{geo}_cls{cls}'] = X_val[geo].map(cls_smoothed).fillna(global_cls_mean)
            X_te[f'{geo}_cls{cls}'] = X_te[geo].map(cls_smoothed).fillna(global_cls_mean)

    # Train model
    model = lgb.LGBMClassifier(**model_params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(period=0)
        ]
    )
    
    val_pred = model.predict(X_val)
    oof_preds[val_idx] = val_pred
    test_preds[:, fold] = model.predict(X_te)
    
    fold_f1 = f1_score(y_val, val_pred, average='micro')
    print(f"Fold {fold+1} F1: {fold_f1:.4f} | best iter: {model.best_iteration_}")

oof_f1 = f1_score(y, oof_preds, average='micro')
print(f"\nOOF Micro F1: {oof_f1:.4f}")

# Save predictions
final_preds = np.round(test_preds.mean(axis=1)).astype(int).clip(1, 3)
submission = pd.DataFrame({
    'building_id': test_values['building_id'],
    'damage_grade': final_preds
})
submission.to_csv('data/submission.csv', index=False)
print("Submission saved to data/submission.csv")
