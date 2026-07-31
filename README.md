# FabSense — AI Co-Pilot for Semiconductor Yield Loss

**A decision-support tool that predicts wafer defect risk, explains *which specific sensor* is causing it, and estimates the financial impact in USD and INR.**

---

## The Business Problem

A semiconductor fab running 10,000 wafers/month at a 16.8% defect rate is scrapping **$840,000 (₹6.97 Crore) every month.**

Most defect-prediction models stop at "this wafer will probably fail." That is not actionable. An engineer standing at the tool needs to know *which knob to turn* and *whether it's worth stopping the line.*

FabSense closes that gap. Every prediction ships with:
- A **risk score** with an interactive gauge
- The **specific sensor** driving that risk, with how far out of spec it is
- A **plain-English recommendation**
- The **cost impact** in USD and INR

---

## Headline Result

| Metric | Value |
|---|---|
| Baseline monthly scrap loss | **$840,000** (₹6,97,20,000) |
| Defects caught early by FabSense | 957 / month (57% recall) |
| False alarms generated | 1,170 / month |
| Gross savings | $287,280 (₹2,38,44,240) |
| Less: inspection cost of false alarms | –$58,520 (₹48,57,160) |
| **Net monthly savings** | **$228,760 (₹1,89,87,080)** |
| **Net annual savings** | **$2,745,120 (₹22,78,44,960)** |
| Share of scrap cost recovered | **27.2%** |

---

## The Most Important Decision in This Project

The first model scored **84.2% accuracy** — and was useless.

It caught only **16% of actual defects** (recall = 0.16). It hit 84% by predicting "good" for nearly everything, which works when 83% of wafers *are* good. High accuracy, zero value.

The fix was not a better algorithm. It was **changing the decision threshold from 50% to 30%.**

| Model | Threshold | Accuracy | Defect Precision | **Defect Recall** |
|---|---|---|---|---|
| v1 — baseline | 50% | 84.2% | 0.61 | **0.16** |
| v2 — class-balanced | 50% | 81.8% | 0.40 | **0.18** |
| **v2 — tuned threshold** | **30%** | 81.2% | 0.45 | **0.57** |

Accuracy went **down** by 3 points. Recall went **up** by 3.5x.

That is the correct trade for this business. Missing a defect costs **$500** (scrapped wafer after full processing), while a false alarm costs **$50** (one manual inspection). When the error costs are that asymmetric, you optimise for recall and accept the noise. **The metric you optimise has to come from the cost structure, not the textbook.**

---

## App Features

- **Interactive Risk Gauge** — a Plotly speedometer visualising defect probability against the user-defined threshold
- **Sensor Status Table** — colour-coded table showing every sensor's reading, safe range, and SHAP impact
- **Adjustable Threshold Slider** — lets users see the precision/recall tradeoff in real time
- **Dual Currency (USD & INR)** — financial impact calculated in both currencies
- **Plain-English Explanations** — translates SHAP values into actionable recommendations

---

## How It Works

**Input** → 6 sensor readings + process step
**Model** → Random Forest (100 trees, depth 15, class-weight balanced)
**Explain** → SHAP TreeExplainer isolates each sensor's contribution to *this specific* prediction
**Translate** → Compare top SHAP feature against the safe range for that process step
**Cost** → Multiply risk by scrap cost, output in USD + INR

### Worked Example

```
Process Step: Deposition
Temperature: 476.7 C   Pressure: 2.37   Gas Flow: 205.2

Defect Probability: 85.4%

[ALERT] HIGH RISK: Temperature is 476.7C, which is 1.7C (0.4%)
        above the safe upper limit of 475C.
[OK] Pressure is 2.37 (Normal).
[OK] Gas flow is 205.2 (Normal).

RECOMMENDATION: Halt processing. Check the heating element and thermocouple.
Cost if missed: $500 (₹41,500) | Saved if caught now: $300 (₹24,900)
```

SHAP assigned temperature a contribution of **+0.41** — by far the largest driver — while pressure sat at **–0.06**. The model independently recovered the physical failure rule (Deposition temp > 475°C) from data alone. Nobody coded that threshold into the model.

---

## Tech Stack

| Tool | Role |
|---|---|
| Python, pandas, NumPy | Data handling |
| scikit-learn | Random Forest classifier |
| SHAP | Per-prediction explainability |
| Plotly | Interactive gauge chart |
| matplotlib | SHAP bar charts |
| Streamlit | Web app / product layer |
| Google Colab | Development environment |

---

## Run It Locally

```bash
git clone https://github.com/YOUR-USERNAME/FabSense.git   # <<< EDIT username
cd FabSense
pip install -r requirements.txt
streamlit run app.py
```

The app trains the model on startup — no model file to download.

---

## About the Data

**The dataset is synthetic — 2,000 wafers generated programmatically.** This was deliberate.

The only well-known public semiconductor dataset (UCI SECOM) has 1,567 **anonymised** features named `Feature_1` through `Feature_1567`. You cannot build an explainability tool on that, because "Feature_847 is elevated" is not a sentence an engineer can act on. The entire point of this project is actionable explanations.

So the data was generated with **named, physically meaningful sensors** and known failure thresholds per process step. This has a real advantage: because the ground-truth rules are known, it's possible to verify that SHAP recovered the *correct* causal driver rather than just a correlated one — which is exactly what the Deposition example above demonstrates.

**Honest limitation:** synthetic data means the model has not been validated against real fab noise, sensor drift, tool-to-tool variation, or multi-step interaction effects. The pipeline is real; the data is a simulation. Swapping in real labelled fab data would require only changes to the loading step.

---

## What I'd Build Next

- **Threshold optimiser** — sweep 0.05 → 0.95 and pick the threshold that maximises net savings directly, instead of hand-picking 30%
- **Multi-sensor interactions** — currently failures are modelled as single-sensor excursions; real fabs fail on combinations
- **Batch CSV upload** — score a full lot at once and rank wafers by expected cost
- **Time-series drift detection** — flag a tool trending toward spec limits *before* it crosses
- **Validation on SECOM** — accept the loss of interpretability to prove the pipeline holds on real measured data

---

## Project Structure

```
FabSense/
├── app.py              # Streamlit app: model, SHAP, cost logic, UI
├── requirements.txt    # Dependencies
└── README.md
```

---

**Built by <<< Aadhya Joshi* — first hands-on AI project. Built to be used, not just scored.
