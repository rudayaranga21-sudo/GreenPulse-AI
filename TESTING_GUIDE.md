# Testing Guide

## Before testing
- Install dependencies and run `streamlit run app.py`.
- Record Python and package versions.
- Keep a copy of the untouched sample data.
- Separate model-training examples from final evaluation cases.

## Waste model tests
1. Use `data/waste_test_examples.csv` as a practice batch; note that its labels are illustrative and some phrases overlap the training vocabulary.
2. For a credible result, create a new test set with at least 20 varied descriptions labelled independently.
3. Avoid copying test descriptions into the training file.
4. Record the confusion matrix or each correct/incorrect prediction.
5. Calculate accuracy: correct predictions / total labelled cases × 100.
6. For a multi-class classifier, also consider per-category precision, recall, and F1.

## Water model tests
1. Run the sample dataset and note flagged rows.
2. Try uploading a CSV with `date` and `litres` columns.
3. Test a stable baseline, a gradual increase, and a deliberately unusual value.
4. Track expected anomalies, detected anomalies, false positives, and missed anomalies.
5. Explain that the model only flags unusual values; it does not diagnose leaks.

## Demo checklist
- Show the chart and anomaly flags.
- Classify `banana peel`, `plastic bottle`, and `newspaper`.
- Upload a labelled test CSV and inspect its predictions.
- Download the CSV reports.
- State clearly that sample data is synthetic.
