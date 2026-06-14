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

def feature_engineering(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("Engineering features...")
    train_len = len(train_df)
    y = train_df['damage_grade'] if 'damage_grade' in train_df.columns else None
    
    # Drop target column from train if present to align columns
    if 'damage_grade' in train_df.columns:
        train_df_no_target = train_df.drop(columns=['damage_grade'])
    else:
        train_df_no_target = train_df.copy()
        
    combined = pd.concat([train_df_no_target, test_df], axis=0).reset_index(drop=True)
    
    floors = combined["count_floors_pre_eq"].replace(0, np.nan)
    area = combined["area_percentage"].replace(0, np.nan)
    height = combined["height_percentage"].replace(0, np.nan)
    
    # --- Basic numeric & age features ---
    combined["age_clipped"] = combined["age"].clip(0, 100)
    combined["age_bin_old"] = (combined["age_clipped"] >= 30).astype(int)
    combined["age_is_zero"] = (combined["age"] == 0).astype(int)
    combined["age_is_995"] = (combined["age"] == 995).astype(int)
    
    combined["height_per_floor"] = combined["height_percentage"] / floors
    combined["area_per_floor"] = combined["area_percentage"] / floors
    combined["slenderness"] = combined["height_percentage"] / area
    combined["volume_proxy"] = combined["area_percentage"] * combined["height_percentage"]
    combined["is_tall"] = (combined["count_floors_pre_eq"] >= 4).astype(int)
    
    combined["old_mud_stone"] = combined["age_bin_old"] * combined["has_superstructure_mud_mortar_stone"]
    
    combined["families_per_floor"] = combined["count_families"] / floors
    combined["families_per_area"] = combined["count_families"] / area
    combined["has_multiple_families"] = (combined["count_families"] > 1).astype(int)
    
    combined["vertical_risk"] = combined["count_floors_pre_eq"] * combined["height_percentage"]
    
    # --- Superstructure combinations & sums ---
    superstructure_cols = [c for c in combined.columns if c.startswith('has_superstructure_')]
    combined['superstructure_sum'] = combined[superstructure_cols].sum(axis=1)
    
    secondary_use_cols = [c for c in combined.columns if c.startswith('has_secondary_use_')]
    combined['secondary_use_sum'] = combined[secondary_use_cols].sum(axis=1)
    
    combined["fragile_score"] = (
        combined["has_superstructure_adobe_mud"]
        + combined["has_superstructure_mud_mortar_stone"]
        + combined["has_superstructure_bamboo"]
    )
    combined["strong_score"] = (
        combined["has_superstructure_rc_engineered"]
        + combined["has_superstructure_cement_mortar_brick"]
    )
    combined["material_risk"] = combined["fragile_score"] - combined["strong_score"]
    combined["rc_any"] = (
        (combined["has_superstructure_rc_engineered"]
         + combined["has_superstructure_rc_non_engineered"]) > 0
    ).astype(int)
    combined["mud_dominant"] = (
        combined["fragile_score"] > combined["strong_score"]
    ).astype(int)
    combined["age_x_fragile"] = combined["age_clipped"] * combined["fragile_score"]
    combined["floors_x_height"] = combined["count_floors_pre_eq"] * combined["height_percentage"]
    
    # Material specific interactions
    combined['mud_mortar_stone_and_timber'] = combined['has_superstructure_mud_mortar_stone'] * combined['has_superstructure_timber']
    combined['mud_mortar_stone_and_bamboo'] = combined['has_superstructure_mud_mortar_stone'] * combined['has_superstructure_bamboo']
    
    # --- Geo interaction features ---
    combined["geo1_geo2"] = (
        combined["geo_level_1_id"] * 10000 + combined["geo_level_2_id"]
    )
    combined["geo2_geo3"] = (
        combined["geo_level_2_id"] * 100000 + combined["geo_level_3_id"]
    )
    
    # --- Frequency encoding of geo columns ---
    for col in ['geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id', 'geo1_geo2', 'geo2_geo3']:
        combined[f'{col}_freq'] = combined[col].map(combined[col].value_counts())
        
    # --- Categorical combinations ---
    cat_combos = [
        ('foundation_type', 'roof_type', 'foundation_roof'),
        ('foundation_type', 'ground_floor_type', 'foundation_ground'),
        ('roof_type', 'other_floor_type', 'roof_other_floor'),
        ('ground_floor_type', 'other_floor_type', 'ground_other_floor'),
        ('land_surface_condition', 'foundation_type', 'land_foundation')
    ]
    for col1, col2, new_col in cat_combos:
        combined[new_col] = (combined[col1].astype(str) + "_" + combined[col2].astype(str)).astype("category").cat.codes
        
    # --- Local area (neighborhood) unsupervised aggregates ---
    for geo in ['geo_level_2_id', 'geo_level_3_id']:
        for stat_col in ['age', 'count_floors_pre_eq', 'area_percentage', 'height_percentage']:
            grouped = combined.groupby(geo)[stat_col]
            mean_val = grouped.transform('mean')
            std_val = grouped.transform('std').fillna(0)
            
            combined[f'{geo}_{stat_col}_mean'] = mean_val
            combined[f'{geo}_{stat_col}_std'] = std_val
            # Relative features
            combined[f'{stat_col}_diff_from_{geo}_mean'] = combined[stat_col] - mean_val
            combined[f'{stat_col}_ratio_to_{geo}_mean'] = combined[stat_col] / (mean_val + 1e-5)
            
        # Regional construction material prevalence ratios
        combined[f'{geo}_mud_stone_ratio'] = combined.groupby(geo)['has_superstructure_mud_mortar_stone'].transform('mean')
        combined[f'{geo}_rc_engineered_ratio'] = combined.groupby(geo)['has_superstructure_rc_engineered'].transform('mean')
        combined[f'{geo}_cement_brick_ratio'] = combined.groupby(geo)['has_superstructure_cement_mortar_brick'].transform('mean')

    combined.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Split back into train and test
    train_feat = combined.iloc[:train_len].copy().reset_index(drop=True)
    test_feat = combined.iloc[train_len:].copy().reset_index(drop=True)
    
    if y is not None:
        train_feat['damage_grade'] = y.values
        
    return train_feat, test_feat

train_df, test_df = feature_engineering(train, test_values)

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

# --- Cross-Validation Setup ---
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
    "n_jobs": 4  # Limit thread parallelism to 4 to reduce swap thrashing
}

SMOOTH_K = 10

print("Starting cross-validation training on optimized features...")
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
print(f"\nOOF Micro F1 on Optimized Features: {oof_f1:.4f}")

# Save predictions
final_preds = np.round(test_preds.mean(axis=1)).astype(int).clip(1, 3)
submission = pd.DataFrame({
    'building_id': test_values['building_id'],
    'damage_grade': final_preds
})
submission.to_csv('data/submission.csv', index=False)
print("Submission saved to data/submission.csv")
