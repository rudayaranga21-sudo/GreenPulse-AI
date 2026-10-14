# GreenPulse AI — Student Prototype

A runnable Streamlit dashboard for:
- water-use anomaly screening with Isolation Forest
- text-based waste classification with TF-IDF + Multinomial Naive Bayes
- charts, batch predictions, and CSV exports

## Requirements
Python 3.10 or newer is recommended.

## Run locally
1. Extract this folder.
2. Open a terminal in the `GreenPulse_AI` folder.
3. Create and activate a virtual environment (recommended).
4. Install packages:
   ```bash
   python -m pip install -r requirements.txt
   ```
5. Start the app:
   ```bash
   streamlit run app.py
   ```
6. Open the local URL printed in the terminal (usually `http://localhost:8501`).

## Files
- `app.py` — Streamlit application
- `data/water_usage_sample.csv` — synthetic daily water records
- `data/waste_training_sample.csv` — sample phrases for training
- `data/waste_test_examples.csv` — labelled examples for a demonstration test
- `PROJECT_REPORT_TEMPLATE.md` — report template
- `TESTING_GUIDE.md` — suggested validation procedure

## Important limitations
The included data is synthetic demonstration data and is not measured data from Bengaluru. A flagged water value does not prove a leak. The waste model is a small educational baseline, not a reliable disposal authority. Validate with separate, labelled local data before making claims about performance.
