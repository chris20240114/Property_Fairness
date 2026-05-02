# Original Coding Process and Analysis

This document preserves the reasoning path from the original notebooks while presenting it in a recruiter-friendly format. The original academic artifacts are still included as `projA1.ipynb`, `projA2.ipynb`, `projA1.pdf`, and `projA2.pdf`.

## 1. Framing the Problem

The project starts from a public-impact question: how can a county assess residential property values accurately without creating an unfair tax burden?

Each row in the dataset represents a Cook County residential property record with sale information, structural attributes, location-related fields, and prior assessment estimates. Because assessments influence tax bills, the modeling task is not only to predict sale price, but also to understand whether errors are distributed fairly.

My original framing focused on three ideas:

- Property assessments should be market-based and explainable.
- Average accuracy is not enough if errors systematically burden lower-priced homes.
- A useful model evaluation needs both prediction metrics and fairness diagnostics.

## 2. Exploratory Data Analysis

The first notebook explores the Cook County housing data before modeling. I looked for data quality issues, skewed variables, and relationships that could become useful model features.

Important EDA observations:

- The raw sale price distribution contains implausible low values, including very small placeholder-like prices.
- To avoid those records dominating the model, I filtered training records to properties with `Sale Price >= 500`.
- Sale price is heavily right-skewed, so I modeled `Log Sale Price` instead of raw dollars.
- `Building Square Feet` has a positive relationship with price, and the log transform makes the relationship more linear.
- Land and building estimates also have skew and outliers, so I used transformation and outlier checks before modeling.

Core preprocessing idea:

```python
training_data = initial_data[initial_data["Sale Price"] >= 500]
training_data["Log Sale Price"] = np.log(training_data["Sale Price"])
training_data["Log Building Square Feet"] = np.log(training_data["Building Square Feet"])
```

## 3. Feature Engineering

The notebooks tested features that were predictive, interpretable, and available at prediction time.

Features explored:

- `Log Building Square Feet`: captures the size-price relationship while reducing skew.
- `Bathrooms`: extracted from bathroom fields or property descriptions.
- `Neighborhood Code`: explored as a location proxy, including higher-priced neighborhood groupings.
- `Estimate (Building)` and `Estimate (Land)`: prior assessment estimates that carry signal about property value.
- Categorical fields such as wall material: explored through substitution and one-hot encoding.

The final portfolio model intentionally uses a small, explainable feature set:

```text
Log Estimate (Building)
Bathrooms
Log Building Square Feet
```

This choice keeps the demo understandable. In an interview, it is easy to explain why each variable matters and how the transformations reduce skew.

## 4. Modeling Approach

The modeling notebook fits ordinary least squares linear regression on `Log Sale Price`.

The original course model used `sklearn.linear_model.LinearRegression(fit_intercept=True)`. The refactored repo keeps that path in `scripts/run_model.py` and includes a NumPy least-squares fallback so the demo can still run in lightweight environments.

The final feature pipeline:

- Copies the input dataframe to avoid mutating source data.
- Creates `Log Sale Price` only for training data.
- Creates log-transformed estimate and building-size features.
- Adds bathroom count.
- Fills missing feature values with medians.
- Removes sale-price outliers only during training, never from test data.

This train/test distinction matters because a real assessor cannot refuse to value a property simply because it is unusual.

## 5. Validation Results

In the original notebook run, the final model produced:

```text
Train RMSE:     0.6002
Holdout RMSE:   0.5937
4-fold CV RMSE: 0.6002 +/- 0.0049
```

The close train, holdout, and cross-validation results suggest the model was not simply memorizing the training data. Because the target is log sale price, an RMSE near `0.60` means typical errors are multiplicative rather than fixed-dollar errors.

## 6. Fairness Analysis

The project then asks whether an apparently decent model is fair for tax assessment.

I split predictions into cheaper and more expensive homes and compared both raw-dollar RMSE and overestimation rates.

Original notebook findings:

```text
RMSE for lower-priced homes:  about $76,899
RMSE for higher-priced homes: about $245,632

Lower-priced homes overestimated: 57.78%
Higher-priced homes overestimated: 27.69%
```

The higher dollar RMSE for expensive homes is expected because expensive homes have larger dollar values. But the overestimation pattern is more important for tax fairness: lower-priced homes were overestimated more often, while higher-priced homes were less often overestimated.

That pattern suggests a regressive assessment risk. If predicted value is used for taxation, overestimating lower-priced homes can shift more burden onto homeowners who may be less able to absorb it.

## 7. Metric Reflection

The notebooks also compare RMSE with relative-error thinking.

RMSE is useful because it penalizes large mistakes, but it is dominated by high-priced homes when predictions are evaluated in dollars. That can hide proportional harm to lower-priced homeowners.

I explored MAPE as a fairness-relevant alternative:

```python
def mape(theta, X, y):
    y_pred = X @ theta
    percentage_error = np.abs((y - y_pred) / y)
    return np.mean(percentage_error)
```

The point was not that MAPE is perfect. The point was that metric choice encodes values. For public assessment systems, a responsible evaluation should include:

- aggregate accuracy, such as RMSE;
- proportional error, such as MAPE;
- residual parity across price tiers and communities;
- overestimation and underestimation rates by group.

## 8. My Takeaway

The strongest lesson from the project is that a model can be technically accurate and socially incomplete.

A fair property assessment model should have competitive prediction accuracy, but it should also have residuals centered near zero across meaningful groups, similar error dispersion by group, and no consistent pattern of over-assessing lower-priced homes while under-assessing higher-priced homes.

For a production version, I would add:

- richer neighborhood and geospatial features;
- validation across time, not just random holdout splits;
- fairness dashboards by price tier and geography;
- documentation explaining where the model performs poorly;
- an appeals-aware workflow so model uncertainty is visible to decision makers.
