# Credit Risk Scoring for Indonesian MSMEs

End-to-end machine learning project for credit risk assessment, designed for MSME lending scenarios in Indonesia. The project covers exploratory analysis, feature engineering, model development, explainability, and interactive decision simulation via Streamlit.

## 1) Project Goal

Build a practical and explainable credit scoring system that helps lending teams:

- estimate probability of default
- calibrate approval policy through decision threshold tuning
- understand model decisions with SHAP explanations

## 2) Business Context

In credit risk, model quality is not only about AUC. The final business decision depends on threshold policy and trade-offs:

- Lower threshold: more inclusive approvals, higher potential default exposure
- Higher threshold: stricter approvals, lower default exposure, more rejected good borrowers

This project operationalizes that trade-off in both modeling and app layers.

## 3) Final Pipeline (Production-Oriented)

- Data split: stratified 60/20/20 (train/validation/test)
- Feature representation: scaled consistently across train, validation, test, and inference
- Training policy: final model trained on original train distribution (no synthetic resampling for final fit)
- Threshold policy: tuned threshold selected empirically from validation performance and persisted as artifact

## 4) Repository Structure

- Core scripts: [src/01_eda.py](src/01_eda.py), [src/02_feature_engineering.py](src/02_feature_engineering.py), [src/03_modeling_and_shap.py](src/03_modeling_and_shap.py), [src/app.py](src/app.py)
- Processed data artifacts: [data/processed](data/processed)
- Model artifacts: [models](models)
- Visual and evaluation outputs: [outputs](outputs)

## 5) Modeling Results (Latest Run)

Source: [outputs/modeling/model_comparison.csv](outputs/modeling/model_comparison.csv)

Test split summary:

- Logistic Regression: AUC 0.8475 | F1 weighted 0.8364 | F1 minority 0.3184 | AP 0.3145
- XGBoost (tuned): AUC 0.8641 | F1 weighted 0.9214 | F1 minority 0.4421 | AP 0.3862
- LightGBM: AUC 0.8503 | F1 weighted 0.8796 | F1 minority 0.3735 | AP 0.3691

Selected primary model: XGBoost (tuned)

## 6) Threshold Strategy

Recommended threshold artifact: [models/best_threshold.json](models/best_threshold.json)

Current value:

- best_threshold: 0.5354
- selection_rule: validation F1 maximum

How it is used:

- Modeling pipeline computes and saves recommended threshold
- Streamlit app loads this value as default
- User can still override threshold to simulate different risk appetites

## 7) Explainability

Project includes global and local explainability outputs:

- SHAP summary and feature importance
- Individual borrower force-style explanations
- Dependence plots for non-linear effects

Generated files are available under [outputs/modeling](outputs/modeling).

## 8) Run Instructions

From project root, run in order:

1. python src/02_feature_engineering.py
2. python src/03_modeling_and_shap.py
3. streamlit run src/app.py

Required dependencies are listed in [requirements.txt](requirements.txt).

## 9) Key Design Decisions

- Consistent scaling contract across training, evaluation, and inference
- Final training on original data distribution for clearer portfolio narrative
- Threshold tuned from validation and persisted as deployment artifact
- App-level threshold override retained for policy simulation

## 10) Current App Behavior

In [src/app.py](src/app.py):

- Default decision threshold automatically follows model recommendation from [models/best_threshold.json](models/best_threshold.json)
- Helper text is aligned with business policy usage:
	default from validation, then adjustable for stricter or more inclusive decisions

## 11) Limitations and Next Improvements

- Add cost-sensitive objective to explicitly minimize business loss (FN vs FP cost)
- Add calibration diagnostics (Brier score, reliability plot)
- Add threshold policy presets by product type (ultra-micro, retail, MSME)
- Add lightweight model monitoring snapshot for data drift indicators

## 12) Portfolio Positioning

This is a portfolio-ready ML project with:

- reproducible experimentation workflow
- production-aware threshold governance
- explainability for decision transparency
- business-facing interactive interface

