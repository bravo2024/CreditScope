# CreditScope

> Credit limit assignment and risk segmentation with multi-model comparison.

Trains four classifiers on synthetic credit applicant data to predict default risk and inform credit limit decisions. Provides segment-level portfolio analysis, score distribution, and approval rate optimisation.

## Quickstart

```bash
pip install -r requirements.txt
python train.py
pytest -q
streamlit run app.py
```

## Model Performance

Best model (Logistic Regression) holdout results:

| Metric | Value |
|---|---|
| ROC AUC | 0.657 |
| Gini | 0.314 |
| KS Statistic | 0.331 |
| F1 Score | 0.415 |
| Accuracy | 0.616 |

5-fold CV AUC: 0.658 ± 0.032. Four models compared.

## Features

| Tab | What it does |
|---|---|
| **Explorer** | Applicant data overview, default distribution, feature statistics |
| **Model Lab** | Multi-model comparison table, ROC curves, calibration plots, CV results |
| **Scoring** | Risk score distribution, scorecard bands, approval rate by segment |
| **Limits** | Credit limit assignment rules, utilisation projection, loss-given-default estimates |

## Repo Structure

```
CreditScope/
  src/         data, model, evaluate, persist modules
  train.py     training pipeline (multi-model + CV)
  app.py       Gradio dashboard
  tests/       pytest smoke test
  models/      saved model + metrics (gitignored)
```

## Data

Synthetic credit applicant data: income, loan amount, employment status, credit history length, DTI ratio, and behavioural attributes.

## License

MIT
