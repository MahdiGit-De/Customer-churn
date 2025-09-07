<<<<<<< HEAD
# Customer-churn
=======
# Customer Churn Prediction

This repository hosts code and assets for a Customer Churn Prediction project.
Use it to explore data, train models, and evaluate churn risk.

CI: ![CI](https://github.com/MahdiGit-De/Customer-churn/actions/workflows/ci.yml/badge.svg)

## Stack
- Python, pandas, scikit-learn
- LightGBM for modeling
- SHAP for interpretability

## Dataset
- IBM Telco Customer Churn (auto-downloaded on first run) into `data/telco_churn.csv`.

## Quickstart
- Create/activate a virtual environment.
- Install dependencies:
  - `pip install -r requirements.txt`
- Run the end-to-end script:
  - `python demo4.py`
- Outputs are saved to `outputs/`:
  - `roc_curve.png`, `confusion_matrix.png`, `shap_summary.png`, and `metrics.json`

## Repository metadata
- To set description and topics via API (requires PAT with repo scope):
  - PowerShell:
    - `$env:GITHUB_TOKEN = '<YOUR_TOKEN>'`
    - `./scripts/set_repo_metadata.ps1 -Owner 'MahdiGit-De' -Repo 'Customer-churn' -Description 'Customer churn prediction with LightGBM & SHAP' -Topics 'ml','lightgbm','shap','churn','telco'`

## Notes
- The script tries multiple public mirrors for the dataset for robustness.
- The model and preprocessing are kept simple and readable for learning purposes.
>>>>>>> bdd0ffd (feat: add IBM Telco churn pipeline, dependencies, and docs)
