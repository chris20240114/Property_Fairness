# Property Assessment Fairness

Portfolio-ready data science project analyzing whether a property valuation model can be both accurate and equitable for Cook County, Illinois residential assessments.

This project began as a course analysis and has been refactored into a recruiter-facing repository with reusable Python modules, a runnable modeling script, fairness diagnostics, and a static demo page.

## Why This Project Matters

Property tax assessments affect real homeowners. A model can look accurate on average while still placing a heavier burden on lower-priced properties or specific communities. This project models home sale prices and evaluates whether prediction errors are distributed fairly across price tiers.

## Demo

Open the static recruiter demo:

- [docs/index.html](docs/index.html)

The page summarizes the problem, modeling approach, validation results, and fairness findings without requiring access to the original course data.

## Key Results

- Built a linear regression model for log sale price using property size, estimated building value, and bathroom features.
- Achieved holdout RMSE of about `0.594` and 4-fold CV RMSE of about `0.600` on log sale price in the original notebook run.
- Found fairness risk in the residual pattern: lower-priced homes were overestimated more often than higher-priced homes.
- Compared RMSE with relative-error diagnostics such as MAPE to show how metric choice changes the fairness story.

## Technical Highlights

- Feature engineering for skewed real estate variables using log transforms.
- Train/test-safe preprocessing for missing values and outliers.
- Linear regression baseline with holdout and cross-validation checks.
- Error analysis by price tier to identify regressive assessment patterns.
- Reusable Python package under `src/property_assessment_fairness`.

## Repository Structure

```text
.
├── docs/
│   └── index.html                 # recruiter-facing static demo
├── scripts/
│   └── run_model.py               # reproducible model training CLI
├── src/
│   └── property_assessment_fairness/
│       ├── features.py            # feature engineering pipeline
│       └── metrics.py             # model and fairness diagnostics
├── tests/
│   └── smoke_test.py              # lightweight local verification
├── projA1.ipynb                   # original EDA notebook
├── projA2.ipynb                   # original modeling/fairness notebook
├── projA1.pdf                     # original exported report
├── projA2.pdf                     # original exported report
├── requirements.txt
└── README.md
```

## Run Locally

Create an environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run the smoke test:

```powershell
python tests/smoke_test.py
```

Run the modeling script when the Cook County CSV files are available:

```powershell
python scripts/run_model.py --train data/cook_county_train.csv
```

Generate predictions for a contest/test file:

```powershell
python scripts/run_model.py `
  --train data/cook_county_train.csv `
  --test data/cook_county_contest_test.csv `
  --output predictions.csv
```

## Data Note

The original Cook County data files are not committed to this repository because they are large course-provided artifacts. The notebooks and CLI expect files with the same schema used in the UC Berkeley Data 100 Cook County housing project.

## Interview Talking Points

- I would describe this as an applied fairness project, not just a prediction project.
- The core tradeoff is that aggregate accuracy can hide asymmetric harm.
- RMSE is useful for model selection, but residual parity and proportional error are more directly tied to tax fairness.
- A production version would monitor error by price tier, geography, and demographic proxies where legally and ethically appropriate.
