from pathlib import Path
from datetime import date, timedelta
import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from PIL import Image
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).parent
DATA = ROOT / "data"
st.set_page_config(page_title="GreenPulse AI", page_icon="🌱", layout="wide")

st.markdown("""
<style>
.main {background: #f7faf8;}
[data-testid="stMetric"] {background: white; border: 1px solid #dce9e1; padding: 14px; border-radius: 12px;}
div.block-container {padding-top: 1.4rem;}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_water():
    path = DATA / "water_usage_sample.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    return df

@st.cache_data
def load_waste():
    return pd.read_csv(DATA / "waste_training_sample.csv")

@st.cache_resource
def build_waste_model():
    df = load_waste()
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), lowercase=True)),
        ("classifier", MultinomialNB())
    ])
    model.fit(df["description"].astype(str), df["category"].astype(str))
    return model

def classify_water(df, contamination=0.08):
    result = df.copy()
    values = pd.to_numeric(result["litres"], errors="coerce")
    valid = values.dropna()
    if len(valid) < 5 or valid.nunique() < 2:
        result["anomaly"] = False
        result["anomaly_score"] = np.nan
        return result, "Add at least five varied water-use records for anomaly detection."
    model = IsolationForest(n_estimators=150, contamination=contamination, random_state=42)
    model.fit(valid.to_frame())
    result["anomaly"] = False
    result["anomaly_score"] = np.nan
    ok = values.notna()
    result.loc[ok, "anomaly"] = model.predict(values[ok].to_frame()) == -1
    result.loc[ok, "anomaly_score"] = model.decision_function(values[ok].to_frame())
    return result, None

st.title("🌱 GreenPulse AI")
st.caption("A student prototype for water-use anomaly screening and waste segregation support.")
st.info("Sample records are synthetic demonstration data, not measurements from Bengaluru. An anomaly is a prompt to investigate—not proof of a leak. Waste predictions may be wrong.")

water_tab, waste_tab, scan_tab, data_tab, about_tab = st.tabs(["💧 Water monitor", "♻️ Waste classifier", "📷 Camera & code scan", "📊 Data & exports", "ℹ️ About / testing"])

with water_tab:
    st.subheader("Water consumption monitor")
    water = load_water()
    upload = st.file_uploader("Optional: upload a water CSV (columns: date, litres)", type=["csv"], key="water_upload")
    if upload is not None:
        try:
            water = pd.read_csv(upload, parse_dates=["date"])
            if not {"date", "litres"}.issubset(water.columns):
                st.error("CSV must contain date and litres columns.")
                water = load_water()
            else:
                water["litres"] = pd.to_numeric(water["litres"], errors="coerce")
                water = water.dropna(subset=["date", "litres"]).sort_values("date")
        except Exception as exc:
            st.error(f"Could not read the CSV: {exc}")
            water = load_water()
    water["litres"] = pd.to_numeric(water["litres"], errors="coerce")
    water = water.dropna(subset=["date", "litres"]).sort_values("date")
    contamination = st.slider("Expected proportion of unusual records", 0.01, 0.20, 0.08, 0.01,
                              help="This is a model setting, not a measured real-world leak rate.")
    scored, warning = classify_water(water, contamination)
    c1, c2, c3 = st.columns(3)
    c1.metric("Records", f"{len(scored):,}")
    c2.metric("Average use", f"{scored['litres'].mean():,.0f} L/day" if len(scored) else "—")
    c3.metric("Flagged records", int(scored["anomaly"].sum()))
    if warning:
        st.warning(warning)
    fig = px.line(scored, x="date", y="litres", markers=True, title="Daily water use",
                  labels={"date":"Date", "litres":"Litres per day"})
    flagged = scored[scored["anomaly"]]
    if not flagged.empty:
        fig.add_scatter(x=flagged["date"], y=flagged["litres"], mode="markers",
                        marker={"size": 12, "symbol":"x"}, name="Flagged for review")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Isolation Forest flags values that look unusual compared with this dataset. It does not establish the cause.")
    st.dataframe(scored, use_container_width=True, hide_index=True)
    st.download_button("Download scored water records (CSV)", scored.to_csv(index=False).encode("utf-8"),
                       "greenpulse_water_scored.csv", "text/csv")

with waste_tab:
    st.subheader("Waste description classifier")
    model = build_waste_model()
    example = st.selectbox("Try an example", ["banana peel", "plastic bottle", "newspaper", "used tea leaves",
                                             "glass jar", "cardboard box", "vegetable scraps", "metal can"])
    description = st.text_input("Describe the waste item", value=example, max_chars=250)
    if st.button("Classify waste", type="primary"):
        if not description.strip():
            st.warning("Enter a short description first.")
        else:
            pred = model.predict([description.strip()])[0]
            probs = model.predict_proba([description.strip()])[0]
            confidence = float(np.max(probs))
            st.success(f"Predicted category: **{pred}**")
            st.metric("Model confidence (not guaranteed accuracy)", f"{confidence:.0%}")
            st.caption("This is a small demonstration model trained on sample phrases. Check local municipal guidance before disposing of an item.")
            prob_df = pd.DataFrame({"Category": model.classes_, "Model score": probs}).sort_values("Model score", ascending=False)
            st.dataframe(prob_df, use_container_width=True, hide_index=True)
    st.markdown("**Batch testing:** upload a CSV containing a `description` column. If it also has a `true_category` column, the app reports test accuracy.")
    batch_file = st.file_uploader("Upload waste test CSV", type=["csv"], key="waste_batch")
    if batch_file is not None:
        try:
            batch = pd.read_csv(batch_file)
            if "description" not in batch.columns:
                st.error("CSV must contain a description column.")
            else:
                batch["predicted_category"] = model.predict(batch["description"].fillna("").astype(str))
                if "true_category" in batch.columns:
                    batch["correct"] = batch["predicted_category"].astype(str).str.lower() == batch["true_category"].astype(str).str.lower()
                    accuracy = batch["correct"].mean() if len(batch) else float("nan")
                    st.metric("Accuracy on this uploaded labelled set", f"{accuracy:.1%}" if pd.notna(accuracy) else "—")
                    st.caption("Only use this result if the labels are reliable and the test cases were not used to train the model.")
                st.dataframe(batch, use_container_width=True, hide_index=True)
                st.download_button("Download batch predictions", batch.to_csv(index=False).encode("utf-8"),
                                   "greenpulse_waste_predictions.csv", "text/csv")
        except Exception as exc:
            st.error(f"Could not process test CSV: {exc}")


with scan_tab:
    st.subheader("Camera waste scan and barcode / QR reader")
    st.write("Take a photo or upload an image of a waste item. You can also decode a QR code or barcode when one is visible.")
    st.warning("Image-based category predictions use an optional general-purpose vision model. They are suggestions, not guaranteed identification. Product codes may not contain disposal information.")

    scan_mode = st.radio("Choose scan type", ["Identify waste from an image", "Read a QR code / barcode"], horizontal=True)
    camera_image = st.camera_input("Take a photo") if scan_mode == "Identify waste from an image" else None
    uploaded_image = st.file_uploader("Or upload an image", type=["png", "jpg", "jpeg", "webp"], key="scan_image_upload")
    image_file = camera_image if camera_image is not None else uploaded_image

    if image_file is not None:
        try:
            image = Image.open(image_file).convert("RGB")
            st.image(image, caption="Image to scan", use_container_width=True)

            if scan_mode == "Identify waste from an image":
                st.markdown("**Image classification**")
                st.caption("The first run may download a vision model and can take several minutes. Install the optional packages listed in `requirements-vision.txt` first.")
                if st.button("Identify waste from image", type="primary"):
                    try:
                        from transformers import pipeline
                        @st.cache_resource
                        def get_vision_model():
                            return pipeline(
                                "zero-shot-image-classification",
                                model="openai/clip-vit-base-patch32"
                            )
                        classifier = get_vision_model()
                        labels = [
                            "organic food waste, such as a banana peel or vegetable scraps",
                            "plastic waste, such as a plastic bottle or wrapper",
                            "paper or cardboard waste, such as newspaper or a cardboard box",
                            "glass waste, such as a glass bottle or jar",
                            "metal waste, such as an aluminium can",
                            "electronic waste, such as a phone, cable, or circuit board",
                            "hazardous waste, such as a battery or chemical container",
                            "textile waste, such as old clothes or fabric"
                        ]
                        with st.spinner("Analysing image... first use may take a while"):
                            predictions = classifier(image, candidate_labels=labels)
                        top = predictions[0]
                        category_map = {
                            "organic": "Organic",
                            "plastic": "Plastic",
                            "paper": "Paper / cardboard",
                            "glass": "Glass",
                            "metal": "Metal",
                            "electronic": "Electronic waste (e-waste)",
                            "hazardous": "Hazardous",
                            "textile": "Textile"
                        }
                        label_text = top["label"].lower()
                        category = next((v for k, v in category_map.items() if k in label_text), "Needs manual review")
                        st.success(f"Likely category: **{category}**")
                        st.metric("Vision model score", f"{float(top['score']):.1%}")
                        st.caption("The score is a model similarity score, not a validated probability of correctness. Verify the result before disposal.")
                        st.dataframe(
                            pd.DataFrame([{"Suggested label": p["label"], "Model score": p["score"]} for p in predictions]),
                            use_container_width=True, hide_index=True
                        )
                    except ImportError:
                        st.error("Vision packages are not installed. In the project folder, run: python -m pip install -r requirements-vision.txt")
                    except Exception as exc:
                        st.error(f"Image classification failed: {exc}")
            else:
                st.markdown("**QR / barcode reader**")
                if st.button("Scan code in image", type="primary"):
                    try:
                        import zxingcpp
                        results = zxingcpp.read_barcodes(image)
                        if results:
                            for result in results:
                                st.success(f"Code type: {result.format}")
                                st.code(result.text)
                                st.caption("This is the decoded code content. It may be a product identifier or URL; it does not automatically prove what material the item is made from.")
                                text_value = result.text.strip()
                                if text_value.startswith(("http://", "https://")):
                                    st.markdown(f"Decoded link: {text_value}")
                        else:
                            st.warning("No readable QR code or barcode was detected. Try brighter lighting, a sharper image, and a closer view of the code.")
                    except ImportError:
                        st.error("Barcode reader package is not installed. Run: python -m pip install zxing-cpp")
                    except Exception as exc:
                        st.error(f"Code scanning failed: {exc}")
        except Exception as exc:
            st.error(f"Could not open image: {exc}")

    st.markdown("---")
    st.markdown("**What this scanner can and cannot do**")
    st.markdown("""
    - Image mode gives a best-effort visual category suggestion.
    - QR/barcode mode reads the code content; it does not automatically look up product composition unless connected to a product database.
    - Use local municipal guidance for disposal decisions. Keep batteries, chemicals, medical waste, and e-waste separate and follow local collection rules.
    """)

with data_tab:
    st.subheader("Data overview and exports")
    water = load_water()
    waste = load_waste()
    a, b = st.columns(2)
    with a:
        st.markdown("**Water sample data**")
        st.dataframe(water, use_container_width=True, hide_index=True)
        st.download_button("Download sample water CSV", water.to_csv(index=False).encode("utf-8"),
                           "water_usage_sample.csv", "text/csv")
    with b:
        st.markdown("**Waste training examples**")
        st.dataframe(waste, use_container_width=True, hide_index=True)
        st.download_button("Download waste training CSV", waste.to_csv(index=False).encode("utf-8"),
                           "waste_training_sample.csv", "text/csv")

with about_tab:
    st.subheader("How it works")
    st.markdown("""
    - **Water:** an Isolation Forest model scores daily litre values and flags a configurable share of unusual records.
    - **Waste:** TF-IDF converts text into numeric features; Multinomial Naive Bayes predicts one of the sample categories.
    - **Camera scan:** optional CLIP vision model suggests a waste category from an image.
    - **QR/barcode scan:** reads code content with ZXing; a decoded code is not itself a waste-material identification.
    - **Reporting:** records and predictions can be exported as CSV files.

    **Limitations**
    - Included records and training phrases are synthetic teaching examples.
    - One-dimensional anomaly detection does not account for occupancy, weather, meter errors, or day-of-week patterns.
    - Text classification can misclassify unfamiliar descriptions; confidence is not the same as calibrated probability.
    - Do not use this prototype for emergency or operational decisions without validation.

    **Suggested validation**
    1. Gather consented, non-identifying local measurements and record their source.
    2. Keep training and test examples separate.
    3. Test at least 20 labelled waste descriptions and report correct/incorrect predictions.
    4. Test normal and deliberately unusual water values; record false alarms and missed anomalies.
    5. Never describe synthetic demonstration data as real Bengaluru data.
    """)
