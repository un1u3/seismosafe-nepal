# Richter's Predictor – Nepal Earthquake Damage Classification

## 📌 Project Overview

Predicting building damage levels from the 2015 Gorkha earthquake using structural,
geographical, and usage data. Buildings are classified into:

- **1** = Low damage
- **2** = Medium damage  
- **3** = Almost complete destruction

Competition: **[Richter's Predictor – DrivenData](https://www.drivendata.org/competitions/57/nepal-earthquake/)**  
Metric: **Micro-averaged F1 score**

---

## 📂 Dataset

- 38 features per building: structural, geographic, ownership, secondary use
- ~260,000 training rows
- Target: `damage_grade` (ordinal, 1–3)
- Test labels hidden; evaluated by DrivenData leaderboard

---

## 🛠️ Feature Engineering

- Structural risk scores: `fragile_score`, `strong_score`, `material_risk`
- Interaction features: `age_x_fragile`, `floors_x_height`
- Age buckets: `is_old` (>30 years), `is_very_old` (>50 years)
- Per-floor ratios: `height_per_floor`, `area_per_floor`, `families_per_floor`
- Geo aggregations: mean age/floors/area grouped by `geo_level_2_id` and `geo_level_3_id`

---

## 🤖 Model

**LightGBM Classifier** with 5-fold stratified CV

Key parameters:
```python
n_estimators=2000, learning_rate=0.05, num_leaves=255,
feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5
```

- Geo target encoding computed **inside each CV fold** to prevent data leakage
- Final predictions averaged across all 5 fold models

---

## 📊 Results

| Model | OOF Micro F1 | Leaderboard |
|-------|-------------|-------------|
| RandomForest (baseline) | ~0.62 | ~0.62 |
| LightGBM + label encoding | — | 0.7433 |
| LightGBM + geo TE (leaky) | 0.7821 | 0.7398 |
| LightGBM + geo TE (fixed) | TBD | TBD |

---

## 📁 Project Structure