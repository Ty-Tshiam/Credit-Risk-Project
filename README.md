﻿# Credit Risk Modeling Project

This project estimates the **Expected Loss (EL)** and capital requirements of personal loans using real-world loan data. The model is structured around the **Basel III AIRB (Advanced Internal Ratings-Based)** framework, decomposing credit risk into:

- **Probability of Default (PD)** – modeled using logistic regression
- **Loss Given Default (LGD)** – based on observed recovery behavior
- **Exposure at Default (EAD)** – derived from actual loan performance
- **Economic Capital (EC)** and **Risk-Weighted Assets (RWA)** – calculated using the Basel IRB capital formula

---

## 🔍 Dataset

The dataset was sourced from a real-world lending platform and includes:
- Loan attributes: `loan_amnt`, `int_rate`, `installment`, `term`, etc.
- Applicant info: `emp_length`, `home_ownership`, `purpose`, `annual_inc`
- Loan performance: `loan_status`, `total_rec_prncp`, `recoveries`

---

## 📊 Feature Engineering

- **Categorical Encoding**: One-hot encoding for `emp_length`, `purpose`, and `home_ownership`
- **Outlier Treatment**: Capping applied to `annual_inc`, `dti`, `int_rate`
- **Missing Values**: Cleaned and casted appropriately for modeling
- **Labeling**: `loan_status` binarized for PD modeling (`0` = Charged Off, `1` = Fully Paid)

---

## 📈 Modeling Approach

### 1. Probability of Default (PD)
- **Model**: Logistic Regression
- **Evaluation**: ROC AUC, PR Curve, Stratified CV
- **Class Imbalance**: Addressed with `class_weight='balanced'`

### 2. Loss Given Default (LGD)
- **Calculation**:  
  `LGD = 1 - (recoveries / EAD)`  
  (where `EAD = loan_amnt - total_rec_prncp` for defaulted loans)

### 3. Exposure at Default (EAD)
- **Calculation**:  
  `EAD = loan_amnt - total_rec_prncp`

### 4. Expected Loss (EL)
- **Formula**:  
  `EL = PD × LGD × EAD`

### 5. Economic Capital (EC) & RWA
- **Capital Requirement (K)**: Calculated using Basel IRB formula (99.9% confidence)
- **EC**: `EC = K × EAD`
- **RWA**: `RWA = 12.5 × EC`

---

## ⚠️ Anomaly Detection & Handling

To ensure regulatory soundness, three edge cases were flagged and addressed:

1. **Fully Paid loans with small EAD (< $2)** were treated as settled.
2. **Fully Paid loans with large EAD (≥ $25) and repayment ratios < 95%** were excluded from EL, EC, and RWA calculations as likely operational mislabels.
3. **Defaulted loans with 100% principal repayment** (some with small recoveries) were excluded from capital calculations to prevent inflating losses.

Each case was flagged in the dataset and documented in the notebook for audit transparency.

---

## 🧠 Tools & Tech

- Python (Pandas, NumPy, SciPy, scikit-learn)
- PySpark (ETL and joins)
- SQL (dataset merging)
- Tableau / Power BI (visualization)
- Azure (planned deployment)

---

## ☁️ Azure Cloud Registry

The logistic regression model is registered with Azure ML (model name: `CreditRiskLogReg`) for future production deployment.

---

## 📂 Repository Structure

- `CreditRisk.ipynb`: End-to-end notebook implementation
- `CRfile.py`: Function definitions and utilities
- `README.md`: This project summary
- `.env`: Environment variables (not tracked)

---

## 📌 Future Enhancements

- Model deployment on Azure with scoring API
- Add macroeconomic stress scenario support
- Compare traditional vs. advanced models (e.g., XGBoost)
- Build portfolio-level risk dashboards

---

## 🧾 References

- Basel Committee on Banking Supervision (BCBS)
- LendingClub public loan data
- scikit-learn, Spark, Tableau documentation

---

Made with 💼 by Ty Tshiamala
