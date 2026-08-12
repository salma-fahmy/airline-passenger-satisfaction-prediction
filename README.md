# Data Computation & Model Deployment Project

An end-to-end project: data analysis in a notebook, a trained model pipeline, and a simple app (`app.py`) to serve predictions.

## Contents
- `data_computation.ipynb` – data analysis & model training
- `model_pipeline.pkl` – trained model pipeline
- `app.py` – app to load the model and serve predictions
- `data.csv` – dataset

## How to run
```bash
python app.py
```

Note: consider using Git LFS for `model_pipeline.pkl` and `data.csv` since they're large binary/data files.
