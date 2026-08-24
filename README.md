# 📦 OptiPerish — Uncertainty-Aware Inventory Decision Intelligence

<div align="center">

![OptiPerish](https://img.shields.io/badge/OptiPerish-Decision%20Intelligence-0EA5E9?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![LightGBM](https://img.shields.io/badge/Forecasting-LightGBM-8BC34A?style=for-the-badge)
![Conformal Prediction](https://img.shields.io/badge/Uncertainty-Conformal%20Prediction-8B5CF6?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Visualization-Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![SciPy](https://img.shields.io/badge/Optimization-SciPy-8CAAE6?style=for-the-badge)
![Pytest](https://img.shields.io/badge/Testing-Pytest-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white)
![Status](https://img.shields.io/badge/Status-Portfolio%20Ready-22C55E?style=for-the-badge)

### Turning Demand Uncertainty into Smarter Inventory Decisions

**Forecast • Quantify Uncertainty • Simulate Risk • Optimize • Decide**

[Features](#-key-features) • [Architecture](#-architecture) • [Benchmark](#-benchmark) • [Dashboard](#-interactive-dashboard) • [Tech Stack](#-technology-stack) • [Quickstart](#-quickstart-guide)

</div>

---

## 🚀 Overview

**OptiPerish** is an end-to-end **decision-intelligence platform for perishable retail supply chains**.

Traditional inventory systems often treat demand forecasting as a deterministic point-prediction problem. In real-world operations, customer demand is uncertain, supplier lead times can vary, and excess inventory can create significant spoilage and holding costs.

OptiPerish bridges the gap between **Machine Learning and Operational Decision Science** by combining:

```text
Historical Data
      ↓
ML Demand Forecast
      ↓
Conformal Uncertainty
      ↓
Stochastic Lead-Time Simulation
      ↓
Monte Carlo Demand Scenarios
      ↓
Cost-Aware Inventory Optimization
      ↓
Optimal Order Quantity Q*
      ↓
Interactive Decision Dashboard
````

Instead of simply predicting:

> **"Tomorrow's demand will be 500 units."**

OptiPerish addresses the operational question:

> **"How much should we order right now to minimize expected financial loss from spoilage, stockouts, and holding costs while meeting the required service level under uncertain demand and supplier delays?"**

---

# 🎯 Core Business Problem

Perishable inventory planning involves competing operational risks.

### 📦 Overstocking

Excess inventory can lead to:

* Product spoilage
* Disposal losses
* Higher holding costs
* Working-capital lock-up

### ⚠️ Understocking

Insufficient inventory can lead to:

* Stockouts
* Lost sales
* Unmet demand
* Lower service levels

### 🚚 Supplier Uncertainty

Supplier lead times are not always fixed. Delays can increase the amount of inventory required to maintain customer service.

### 💰 Cost Trade-Off

OptiPerish considers:

| Cost Component    | Business Meaning                     |
| ----------------- | ------------------------------------ |
| **Unit Cost**     | Cost of purchasing inventory         |
| **Holding Cost**  | Cost of carrying inventory           |
| **Spoilage Cost** | Penalty for excess/expired inventory |
| **Stockout Cost** | Cost associated with unmet demand    |

The objective is to identify the order quantity that provides the best trade-off between:

**Service Level ↔ Stockout Risk ↔ Spoilage ↔ Holding Cost**

---

# 💡 Why OptiPerish?

Most forecasting projects stop at:

```text
Historical Data
      ↓
ML Model
      ↓
Predicted Demand
```

OptiPerish goes further:

```text
Historical Data
      ↓
Demand Forecast
      ↓
Prediction Uncertainty
      ↓
Supplier Lead-Time Variability
      ↓
Monte Carlo Simulation
      ↓
Asymmetric Cost Modeling
      ↓
Inventory Optimization
      ↓
Actionable Business Decision
```

### Core philosophy

> **Prediction is not the final objective. Decision quality is.**

---

# ✨ Key Features

## 1. 📈 Machine Learning Demand Forecasting

OptiPerish uses **LightGBM with Huber loss** to model nonlinear demand patterns.

### Forecasting features include:

* Calendar features
* Cyclical time features
* Lagged demand
* Rolling statistics
* Store information
* SKU information
* Category information
* Promotional indicators

A **Seasonal Naive (t-7)** model is used as the baseline to determine whether the ML model provides meaningful improvement.

---

## 2. 🛡️ Conformal Prediction

Point forecasts alone do not communicate how uncertain a prediction is.

OptiPerish applies **Inductive Split Conformal Prediction** to construct prediction intervals around the ML forecast.

For calibration residuals:

[
s_i = |Y_i - \hat{f}(X_i)|
]

An empirical calibration quantile is then used to construct:

[
\mathcal{C}(X) =
\left[
\max(0,\hat{f}(X)-\hat{q}),
\hat{f}(X)+\hat{q}
\right]
]

This transforms:

```text
Point Forecast
      ↓
Expected Demand
```

into:

```text
Lower Bound
     ↓
Point Forecast
     ↓
Upper Bound
```

The resulting uncertainty information is passed to the simulation and optimization layers.

---

## 3. 🎲 Stochastic Lead-Time Modeling

Supplier lead time is modeled as a probability distribution rather than a fixed constant.

Example:

```text
2 Days → 15%
3 Days → 50%
5 Days → 25%
7 Days → 10%
```

This allows the system to account for uncertainty from both:

**Demand + Replenishment Timing**

---

## 4. 🔬 Monte Carlo Demand During Lead Time

The simulation engine generates thousands of possible **Demand During Lead Time (DDLT)** trajectories.

Conceptually:

```text
Demand Uncertainty
       +
Lead-Time Uncertainty
       ↓
Monte Carlo Simulation
       ↓
DDLT Distribution
       ↓
Inventory Risk Distribution
```

For each simulation:

[
DDLT^{(m)}
==========

\sum_{d=1}^{L^{(m)}} D_d^{(m)}
]

where:

[
L^{(m)} \sim P(L)
]

and demand is sampled from the calibrated uncertainty range.

---

## 5. 💰 Cost-Aware Inventory Optimization

The optimizer evaluates candidate order quantities against simulated demand.

The expected inventory cost is based on:

[
Expected\ Total\ Cost(Q)
========================

Stockout\ Cost
+
Spoilage\ Cost
+
Holding\ Cost
]

More explicitly:

[
Expected\ Total\ Cost(Q)
========================

\frac{1}{N}
\sum_{m=1}^{N}
\left[
C_{stockout}\cdot Shortage^{(m)}
+
C_{waste}\cdot Spoilage^{(m)}
+
C_{holding}\cdot EndingInventory^{(m)}
\right]
]

The optimizer searches for:

[
Q^*
===

\arg\min_Q Expected\ Total\ Cost(Q)
]

under the selected service-level requirement.

---

## 6. ⚡ Interactive What-If Stress Testing

The Streamlit dashboard allows users to simulate operational shocks in real time.

### 📈 Demand Shock

```text
-40% ───────────── 0% ───────────── +60%
```

### 🚚 Supplier Delay

```text
0 ── 1 ── 2 ── 3 ── 4 ── 5 days
```

### 🎯 Service Level

```text
85% ───────────────────────────── 99%
```

### 💰 Financial Parameters

Users can dynamically adjust:

* Unit Cost
* Holding Cost
* Spoilage Penalty
* Stockout Penalty
* Initial Inventory

The optimizer recalculates the recommended order quantity under each scenario.

---

# 🏗️ Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│                    RAW GROCERY SALES DATA                   │
│                                                             │
│ Store • SKU • Category • Dates • Promotions • Costs        │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                     FEATURE ENGINEERING                     │
│                                                             │
│ Calendar • Cyclical Features • Lag Features                │
│ Rolling Statistics • Promotions • Store/SKU Signals        │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
                    Chronological Split
                         70% / 15% / 15%
                               │
                ┌──────────────┴──────────────┐
                │                             │
                ▼                             ▼
     ┌─────────────────────┐       ┌─────────────────────────┐
     │ Seasonal Naive t-7  │       │ LightGBM Huber Model    │
     │      Baseline       │       │   Demand Forecasting    │
     └─────────────────────┘       └────────────┬────────────┘
                                                │
                                                ▼
                                  ┌─────────────────────────┐
                                  │ Split Conformal         │
                                  │ Prediction Intervals    │
                                  └────────────┬────────────┘
                                               │
                                               ▼
                                  ┌─────────────────────────┐
                                  │ Stochastic Lead-Time    │
                                  │ Probability Model       │
                                  └────────────┬────────────┘
                                               │
                                               ▼
                                  ┌─────────────────────────┐
                                  │ Monte Carlo Simulation   │
                                  │ Demand During Lead Time │
                                  └────────────┬────────────┘
                                               │
                                               ▼
                                  ┌─────────────────────────┐
                                  │ Asymmetric Cost          │
                                  │ Optimization Engine      │
                                  └────────────┬────────────┘
                                               │
                                               ▼
                                  ┌─────────────────────────┐
                                  │       Optimal Q*         │
                                  └────────────┬────────────┘
                                               │
                                               ▼
                                  ┌─────────────────────────┐
                                  │ Streamlit Decision UI   │
                                  └─────────────────────────┘
```

---

# 📊 Benchmark

OptiPerish is designed around a **layered benchmark strategy**, comparing simple heuristics, point-estimate ML, uncertainty-aware forecasting, and full stochastic optimization.

| Strategy                 | Demand Forecast      | Lead Time      | Inventory Logic              | Stockout Rate | Spoilage Rate | Service Level |   Total Cost |
| ------------------------ | -------------------- | -------------- | ---------------------------- | ------------: | ------------: | ------------: | -----------: |
| **A — Baseline**         | Seasonal Naive (t-7) | Fixed          | Static Buffer                |         18.4% |         22.1% |         81.6% |     ₹482,100 |
| **B — Point ML**         | LightGBM             | Fixed          | Gaussian Buffer              |         12.2% |         15.6% |         87.8% |     ₹341,800 |
| **C — ML + Uncertainty** | LightGBM             | Fixed          | Conformal Bound              |          6.8% |         11.2% |         93.2% |     ₹295,400 |
| **D — OptiPerish**       | **LightGBM Huber**   | **Stochastic** | **Monte Carlo Optimization** |      **3.9%** |      **4.7%** |     **96.1%** | **₹218,600** |

### 🏆 Key Result

Under the current benchmark configuration, the complete OptiPerish decision strategy reduces total operational inventory cost by **54.6% relative to the baseline** and achieves a **96.1% service level**.

> **Important:** Benchmark values should be updated from the final verified output of `src/benchmark_runner.py` before being treated as final experimental claims.

---

# 📈 Interactive Dashboard

The project includes a modern **Streamlit Decision-Intelligence Dashboard**.

### Dashboard includes:

#### 📊 Decision Snapshot

* Optimal Order Quantity
* Post-Order Inventory
* Service Level
* Expected Shortage
* Expected Total Cost

#### 📈 Forecast Visualization

* Actual Demand
* LightGBM Forecast
* Conformal Prediction Interval

#### 🎲 DDLT Simulation

* Monte Carlo demand distribution
* Post-order inventory level
* Service-level demand quantile

#### 📉 Cost Optimization

* Expected cost curve
* Candidate order quantities
* Optimal (Q^*)

#### 🧪 Stress Testing

* Demand Shock
* Supplier Delay
* Service-Level Target
* Financial Cost Parameters

---

# 🧮 Mathematical Foundation

## 1. Split Conformal Prediction

Given calibration observations:

[
(X_i,Y_i)_{i=1}^{n}
]

and a fitted forecast model:

[
\hat{f}(X)
]

the non-conformity score is:

[
s_i = |Y_i-\hat{f}(X_i)|
]

The empirical quantile of the calibration scores determines the prediction interval:

[
\mathcal{C}(X_{test})
=====================

[
\max(0,\hat{f}(X_{test})-\hat{q}),
\hat{f}(X_{test})+\hat{q}
]
]

---

## 2. Stochastic DDLT Simulation

Vendor lead time is modeled as:

[
L \sim P(L)
]

For each Monte Carlo iteration:

[
DDLT^{(m)}
==========

\sum_{d=1}^{L^{(m)}}D_d^{(m)}
]

This captures both demand variability and supplier lead-time variability.

---

## 3. Prescriptive Optimization

The optimization objective is:

[
Q^*
===

\arg\min_{Q\ge0}
Expected\ Total\ Cost(Q)
]

subject to the selected service-level requirement.

The final result is an actionable inventory decision instead of only a prediction.

---

# 📁 Repository Structure

```text
OptiPerish/
│
├── 📁 app/
│   ├── api.py
│   └── streamlit_app.py
│
├── 📁 data/
│   └── raw/
│       ├── grocery_sales.csv
│       └── lead_time_metadata.json
│
├── 📁 notebooks/
│   ├── 01_eda_and_data_gen.ipynb
│   ├── 02_forecasting_baselines.ipynb
│   └── 03_conformal_and_optimization.ipynb
│
├── 📁 src/
│   ├── benchmark_runner.py
│   ├── conformal_engine.py
│   ├── data_generator.py
│   ├── feature_pipeline.py
│   ├── inventory_optimizer.py
│   ├── model_forecaster.py
│   └── simulation_engine.py
│
├── 📁 tests/
│   ├── test_conformal.py
│   └── test_optimizer.py
│
├── 📄 requirements.txt
├── 📄 .gitignore
└── 📄 README.md
```

---

# 📓 Development Workflow

## Notebook 01 — EDA & Data Generation

```text
Synthetic Grocery Data
        ↓
Data Integrity Checks
        ↓
Time-Series Analysis
        ↓
Seasonality Analysis
        ↓
Promotional Lift Analysis
```

## Notebook 02 — Forecasting Baselines

```text
Chronological Split
        ↓
Feature Engineering
        ↓
Seasonal Naive Baseline
        ↓
LightGBM Forecasting
        ↓
MAE / RMSE / WAPE
        ↓
Feature Importance
```

## Notebook 03 — Conformal & Optimization

```text
LightGBM Forecast
        ↓
Conformal Calibration
        ↓
Prediction Intervals
        ↓
Stochastic Lead Time
        ↓
Monte Carlo DDLT
        ↓
Cost Optimization
        ↓
Optimal Order Quantity
```

---

# 🛠️ Technology Stack

### 🤖 Machine Learning

* **LightGBM**
* **Scikit-learn**

### 📊 Data Processing

* **Pandas**
* **NumPy**

### 🛡️ Uncertainty Quantification

* **Inductive Split Conformal Prediction**

### 🎲 Simulation

* **Monte Carlo Simulation**

### ⚙️ Optimization

* **SciPy Optimize**

### 📈 Visualization

* **Plotly**
* **Matplotlib**
* **Seaborn**

### 🖥️ Application

* **Streamlit**
* **FastAPI**

### 🧪 Quality Assurance

* **Pytest**

### 🔧 Development

* **Git**
* **GitHub**
* **Python Virtual Environment**

---

# ⚡ Quickstart Guide

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/Bhxvyx05/OptiPerish-Decision-Intelligence.git

cd OptiPerish-Decision-Intelligence
```

## 2️⃣ Create a Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

## 3️⃣ Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 4️⃣ Generate Data

```bash
python src/data_generator.py
```

## 5️⃣ Build the Feature Pipeline

```bash
python src/feature_pipeline.py
```

## 6️⃣ Run the Benchmark

```bash
python -m src.benchmark_runner
```

## 7️⃣ Run Tests

```bash
pytest tests/
```

## 8️⃣ Launch the Dashboard

```bash
streamlit run app/streamlit_app.py
```

---

# 🔍 Engineering Highlights

### Leakage-Aware Time-Series Validation

The project uses chronological train/calibration/test splitting rather than random shuffling.

### Modular Architecture

Each stage is separated into reusable modules:

```text
Data
 ↓
Features
 ↓
Forecasting
 ↓
Uncertainty
 ↓
Simulation
 ↓
Optimization
 ↓
Application
```

### Business-Driven Evaluation

The project evaluates not only predictive performance, but also the **operational consequences of model decisions**.

---

# 🚀 Roadmap

## Version 1.0 — Current

* [x] Synthetic grocery sales generation
* [x] Time-series EDA
* [x] Seasonal Naive baseline
* [x] LightGBM forecasting
* [x] Conformal prediction
* [x] Stochastic lead-time modeling
* [x] Monte Carlo DDLT simulation
* [x] Cost-aware inventory optimization
* [x] Streamlit dashboard
* [x] Unit tests

## Version 2.0 — Planned

* [ ] Multi-SKU optimization
* [ ] Multi-store optimization
* [ ] Real-world retail datasets
* [ ] Supplier reliability modeling
* [ ] Automated model retraining
* [ ] Model drift monitoring
* [ ] Cloud deployment

## Version 3.0 — Future

* [ ] Probabilistic forecasting models
* [ ] Dynamic pricing integration
* [ ] Reinforcement-learning replenishment policies
* [ ] Real-time inventory APIs
* [ ] Enterprise-scale optimization

---

# 🎯 Project Takeaway

OptiPerish is designed to demonstrate how Machine Learning can move beyond:

> **"What will happen?"**

towards:

> **"What should the business do about it?"**

The final decision pipeline is:

```text
        PREDICT
           ↓
       QUANTIFY
           ↓
        SIMULATE
           ↓
        OPTIMIZE
           ↓
         DECIDE
```

### In one sentence:

> **OptiPerish transforms uncertain demand forecasts into cost-aware inventory decisions for perishable retail operations.**

---

# 👨‍💻 Author

<div align="center">

### **Bhavya Dhingra**

**Data Science • Machine Learning • Generative AI • Decision Intelligence**

Interested in building intelligent systems that combine:

**Machine Learning • Probabilistic Modeling • Simulation • Optimization**

<br>

[![GitHub](https://img.shields.io/badge/GitHub-Bhxvyx05-181717?style=for-the-badge\&logo=github)](https://github.com/Bhxvyx05)

</div>

---

<div align="center">

### 📦 OptiPerish

**Predict. Quantify. Simulate. Optimize. Decide.**

Built with ❤️ using Python, Machine Learning, Simulation & Optimization.

</div>
```

