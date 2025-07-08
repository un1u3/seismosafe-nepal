# Richter’s Predictor – Nepal Earthquake Damage Classification

## 📌 Project Overview

This project predicts building damage levels from the 2015 Nepal Earthquake based on structural, geographical, and usage data. The goal is to classify buildings into one of three categories:
- **1** = Low damage
- **2** = Medium damage
- **3** = High damage

This work is part of the DrivenData competition: **[Richter’s Predictor](https://www.drivendata.org/competitions/57/nepal-earthquake/)**

---

## 📂 Dataset

- `train.csv` – Includes building features and damage labels.
- `test.csv` – Includes only features; **true labels are hidden** to prevent overfitting and leaderboard gaming.
- Total features: 41 (train), 38 (test).
- Target variable: `damage_grade`

🛑 **Note:** Test labels are not publicly available. Evaluation is done automatically by DrivenData using a hidden test set.

---

## 🛠️ Preprocessing Steps

- Dropped irrelevant columns (`building_id`, `superstructure_sum`, etc.).
- Engineered:
  - `superstructure_sum` = sum of superstructure flags
  - `secondary_use_sum` = sum of secondary use flags
- Encoded categorical features using `OrdinalEncoder` with unknown category handling.
- Ensured train/test feature alignment.

---

## 🤖 Model

- **RandomForestClassifier** (`class_weight='balanced'`)
- 80/20 split for train/validation
- Achieved:
  - **Macro F1 Score: ~0.6162**
  - **Validation accuracy: ~72%**


---

## 🚀 Future Improvements

- Try XGBoost / LightGBM
- Hyperparameter tuning
- Cross-validation
- Handle class imbalance with SMOTE
- Feature engineering (e.g., floor-to-height ratio, age buckets)

---

## 📁 Files

| File Name           | Description                          |
|---------------------|--------------------------------------|
| `train.csv`         | Training dataset                     |
| `test.csv`          | Test dataset (labels hidden)         |
| `submission.csv`    | Submission file for predictions      |
| `model_training.py` | Main training + prediction pipeline  |
| `README.md`         | Project overview                     |

---

## 📬 Contact

📧 Email: **uniquestha422@gmail.com**  
🔗 Maintainer: Unique Shrestha (un1u3)

---

