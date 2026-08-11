# Rossmann Sales Forecast — Task 3 Web App

Flask dashboard that serves the Task 2 (Random Forest) model: enter a
Store ID + parameters, or upload a CSV of future dates, and get
predicted **Sales** and **Customers** with a 95% interval, a chart, and
a downloadable CSV — matching the Task 3 spec in the project brief.

## Files
| File | Purpose |
|---|---|
| `features.py` | Feature engineering, ported 1:1 from `Task_2.ipynb` (shared by training + serving so they never drift). |
| `train_model.py` | Trains & serializes the Sales model **and** a Customers model (added — the brief asks the dashboard to show both, which `Task_2.ipynb` didn't originally cover) with the `name-DD-MM-YYYY-HH-MM-SS-00.pkl` timestamp convention. |
| `predict_service.py` | Loads the latest serialized model pair; falls back to a small synthetic-data demo model if none exist yet, so the app is testable before you've run training. |
| `app.py` | Flask routes: dashboard, `/predict`, `/download/<id>`, `/api/predict`. |
| `templates/`, `static/` | Dashboard UI (form + Chart.js chart + results table). |

## Run locally
```bash
pip install -r requirements.txt

# 1. Put the real Kaggle Rossmann files here:
#    data/train.csv  data/test.csv  data/store.csv

# 2. Train and serialize the models (writes to models/)
python train_model.py

# 3. Start the app
python app.py
# -> http://localhost:5000
```

Until step 2 has been run with real data, the app still works — it
shows a clearly labeled **demo mode** banner and uses a small
synthetic-data model so you can test the UI/flow immediately.

## Deploy to Heroku
```bash
heroku create your-app-name
git init && git add . && git commit -m "Task 3 app"
heroku git:remote -a your-app-name
git push heroku main
```
`Procfile` and `requirements.txt` are already set up for this
(`gunicorn app:app`). Note: on Heroku's ephemeral filesystem you'll
typically want to train the model as part of a release step, or bake a
pre-trained `models/*.pkl` + `models/meta-*.json` into the repo before
deploying, since the filesystem resets on dyno restart.

## What's still open from the brief (for the final submission)
- **DVC**: not wired up here — add `dvc init`, track `data/` and
  `models/`, and take the "multiple data/model versions" screenshots
  the brief asks for.
- **MLflow dashboard screenshots**: `Task_2.ipynb` already logs runs
  via `mlflow.start_run(...)`; run `mlflow ui` locally and screenshot
  the run comparison.
- **LSTM**: present in `Task_2.ipynb` with a `try/except` around the
  TensorFlow import — confirm it actually executed (not just skipped)
  in an environment with TensorFlow installed, since the final
  submission rewards deep-learning depth.
