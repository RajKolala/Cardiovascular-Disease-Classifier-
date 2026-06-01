# Cardiovascular Disease Classifier
**Python | Pandas | NumPy | Scikit-Learn | Matplotlib | Seaborn**
**March 2026 – April 2026**

---

## Overview

This project builds a binary machine learning classifier to predict the presence of cardiovascular disease using a real-world dataset of **70,000 patient records**. The goal was to explore the full ML pipeline — from raw data exploration and cleaning to model training, hyperparameter tuning, and evaluation — with a focus on what metrics actually matter in a medical context.

This was completed as part of **TECH 176 – Machine Learning Technology and Applications** at San Jose State University.

---

## Dataset

- **Source:** `cardio_train.csv` (Kaggle – Cardiovascular Disease Dataset)
- **Size:** 70,000 records × 13 features
- **Target:** `cardio` — 0 (no disease) or 1 (disease present)

### Features

| Feature | Description |
|---|---|
| `age` | Age in days (converted to years) |
| `gender` | 1 = female, 2 = male |
| `height` | Height in cm |
| `weight` | Weight in kg |
| `ap_hi` | Systolic blood pressure |
| `ap_lo` | Diastolic blood pressure |
| `cholesterol` | 1 = normal, 2 = above normal, 3 = well above normal |
| `gluc` | Glucose level: 1 = normal, 2 = above normal, 3 = well above normal |
| `smoke` | Smoker: 0 = no, 1 = yes |
| `alco` | Alcohol intake: 0 = no, 1 = yes |
| `active` | Physically active: 0 = no, 1 = yes |
| `cardio` | Target: cardiovascular disease present (1) or not (0) |

---

## Project Structure



---

## Workflow

### Part 1 – Exploratory Data Analysis (EDA)
- Loaded and inspected the dataset (`df.info()`, `df.describe()`, `df.head()`)
- Converted `age` from days to years for readability
- Visualized feature distributions and correlations using Matplotlib and Seaborn
- Identified and removed outliers in blood pressure columns (`ap_hi`, `ap_lo`) using physiologically reasonable bounds

### Part 2 – Data Preprocessing
- Removed the `id` column (no predictive value)
- Applied **StandardScaler** normalization to continuous features
- Created a cleaned correlation-filtered DataFrame (`df_corr`) for modeling
- Split data into **80% training / 20% test** sets with `random_state=42` for reproducibility

### Part 3 – Modeling
Trained and evaluated **5 classification models** using Scikit-Learn:

| Model | Accuracy |
|---|---|
| Logistic Regression | 72.7% |
| Support Vector Machine (LinearSVC) | 72.6% |
| Decision Tree | 72.9% |
| **Random Forest** | **73.4%** |
| SGD Classifier | 72.6% |

### Part 4 – Hyperparameter Tuning
- Applied **RandomizedSearchCV** with cross-validation on the Random Forest model
- Tuned parameters including `n_estimators`, `max_depth`, `min_samples_split`, and `min_samples_leaf`
- Achieved best accuracy of **73.7%** with the tuned Random Forest

### Part 5 – Evaluation Beyond Accuracy
Evaluated the final model using:
- **Confusion Matrix**
- **Classification Report** (Precision, Recall, F1-Score)
- **ROC Curve** (`RocCurveDisplay`)

> **Key insight:** Recall was identified as the critical metric for this use case. In a medical screening context, a false negative (missing a patient who has cardiovascular disease) is far more dangerous than a false positive. Optimizing for recall minimizes missed diagnoses.

---

## Results

- **Best Model:** Random Forest (tuned)
- **Best Accuracy:** 73.7%
- **Key Metric:** Recall — prioritized to reduce false negatives in medical deployment
- **Zero overfitting confirmed** via train/test gap analysis

---

## Known Limitations

1. **Self-reported features are noisy** — `smoke`, `alco`, and `active` are patient self-reported, which introduces inaccuracy
2. **Missing critical medical features** — real cardiovascular risk assessment uses LDL/HDL cholesterol, blood sugar, ECG results, family history, and stress levels — none of which are in this dataset
3. **Only 11 predictive features** for a complex, multi-factor disease
4. **Coarse cholesterol/glucose encoding** — encoded as 1/2/3 rather than actual mg/dL values, reducing informational value

---

## Libraries Used
