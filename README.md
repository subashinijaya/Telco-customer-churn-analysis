# Telco Customer Churn Analysis

End-to-end customer churn analysis using Python — data cleaning, exploratory data analysis, and predictive modelling on 7,000+ telco customers to identify the key drivers of churn.

## Project Overview

This project analyses the IBM Telco Customer Churn dataset (7,043 customers, 21 features) to answer a core business question: **which customers are most likely to leave, and why?** The analysis walks through the full data science workflow — cleaning known data quality issues, exploring churn patterns visually, building classification models, and translating results into business recommendations.

## Key Results

- **Overall churn rate: ~27%** — roughly 1 in 4 customers leave
- **Contract type is the strongest churn driver:** month-to-month customers churn at ~43%, vs ~11% (one-year) and ~3% (two-year)
- **Early tenure is the danger zone** — churn concentrates in the first 12 months
- **Best model: Logistic Regression, ~80% accuracy** on the held-out test set
- **Top churn drivers (Random Forest importance):** TotalCharges, tenure, MonthlyCharges, fibre-optic internet, electronic check payment

## Workflow

1. **Data cleaning** — fixed the known `TotalCharges` text/blank issue (11 rows for tenure-0 customers), verified no duplicates
2. **EDA** — churn distribution, churn by contract type, tenure histograms, monthly charges boxplots, internet service comparison
3. **Modelling** — Logistic Regression (interpretable baseline) and Random Forest (non-linear patterns), 75/25 stratified split
4. **Evaluation** — accuracy, precision/recall, confusion matrix
5. **Feature importance** — identifying the top 10 churn drivers
6. **Business recommendations** — contract conversion incentives, first-90-day onboarding, fibre pricing review

## Technologies Used

- Python 3
- pandas, NumPy (data wrangling)
- Matplotlib, Seaborn (visualisation)
- scikit-learn (Logistic Regression, Random Forest, evaluation metrics)
- Jupyter Notebook

## Files

- `telco_churn_analysis.ipynb` — full analysis notebook (executed, with outputs)
- `WA_Fn-UseC_-Telco-Customer-Churn.csv` — dataset (IBM sample data)
- `requirements.txt` — dependencies

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook telco_churn_analysis.ipynb
```

## Business Recommendations

1. **Convert month-to-month customers to annual contracts** with discount incentives — the single biggest retention lever
2. **Strengthen first-90-day onboarding** — early-tenure customers are the highest-risk group
3. **Review fibre-optic pricing and value perception** — the premium product has premium churn
4. **Use the model's churn scores** to target retention offers at high-risk, high-value customers

## Future Improvements

- Address class imbalance (SMOTE / class weights) to improve churn-class recall
- Test gradient boosting models (XGBoost, LightGBM)
- Deploy churn probability scoring for CRM integration

## Author

Anne Subashini Sritharan

## Data Source

IBM Telco Customer Churn sample dataset (publicly available via Kaggle and IBM).
