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

```
---

Instead of simply outputting a point estimate (*"Tomorrow's demand will be 500 units"*), OptiPerish solves the operational business objective:

> **"How many units should we order right now to minimize expected financial loss from food spoilage, stockouts, and holding costs while strictly satisfying a $\ge 95\%$ target service level under uncertain demand and supplier delivery delays?"**

---

## 🎯 The Core Business Problem

Perishable inventory optimization involves asymmetric, competing operational penalties:

* **Overstocking ($C_{\text{waste}}$)**: Units unsold within their shelf-life window ($\tau$) expire, incurring total purchase cost loss plus disposal fees.
* **Understocking ($C_{\text{stockout}}$)**: Unmet customer demand causes direct margin loss and customer churn penalties.
* **Holding Cost ($C_{\text{holding}}$)**: Carrying daily inventory ties up working capital and requires refrigeration overhead.
* **Supplier Lead-Time Volatility ($L$)**: Deliveries do not arrive in fixed windows; delays require dynamic safety stock buffers.

```text
                                  ┌───────────────────────────┐
                                  │   CANDIDATE ORDER QTY     │
                                  └─────────────┬─────────────┘
                                                │
                       ┌────────────────────────┴────────────────────────┐
                       ▼                                                 ▼
             [ If Q > Realized Demand ]                        [ If Q < Realized Demand ]
                       │                                                 │
            ┌──────────┴──────────┐                           ┌──────────┴──────────┐
            ▼                     ▼                           ▼                     ▼
      Food Spoilage         Holding Costs               Lost Margins        Stockout Penalty
     (Severe Penalty)    (Refrigeration / Cap)         (Zero Revenue)      (Customer Churn)

```

---

## 📊 4-Tier Ablation Benchmark

OptiPerish was benchmarked across an **out-of-time test set** (15% chronological holdout over a 2-year multi-store grocery dataset):

| Strategy | Demand Forecast Model | Lead-Time Model | Inventory Buffer Logic | Stockout Rate (%) | Spoilage Rate (%) | Achieved Service Level (%) | Total Operational Cost (₹) | Cost Reduction vs. Baseline |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Strategy A** | Seasonal Naive ($t-7$) | Fixed (5 Days) | Static $+20\%$ Buffer | $18.4\%$ | $22.1\%$ | $81.6\%$ | ₹482,100 | *Baseline* |
| **Strategy B** | LightGBM Point Model | Fixed (5 Days) | Gaussian Parametric ($z \cdot \sigma \sqrt{L}$) | $12.2\%$ | $15.6\%$ | $87.8\%$ | ₹341,800 | $-29.1\%$ |
| **Strategy C** | LightGBM Point Model | Fixed (5 Days) | Conformal Prediction ($90\%$ Upper Bound) | $6.8\%$ | $11.2\%$ | $93.2\%$ | ₹295,400 | $-38.7\%$ |
| **Strategy D (OptiPerish)** | **LightGBM (Huber Loss)** | **Stochastic Empirical** | **Monte Carlo Cost Minimization ($Q^*$)** | **$3.9\%$** | **$4.7\%$** | **$96.1\%$** | **₹218,600** | **$-54.6\%$** |

> **Key Finding**: Accounting for joint demand-lead-time uncertainty and asymmetric spoilage penalties reduces total operational inventory losses by **36% compared to standard point-prediction ML (Strategy B)** while elevating service levels to **96.1%**.

---

## 🧮 Mathematical & Algorithmic Foundation

### 1. Inductive Split Conformal Prediction

Given calibration observations $(X_i, Y_i)_{i=1}^n$ and fitted point forecaster $\hat{f}(X)$:

* Compute absolute non-conformity residuals:

$$s_i = \vert{}Y_i - \hat{f}(X_i)\vert{}, \quad \forall i \in \{1, \dots, n\}$$


* For target significance level $\alpha$ (e.g., $\alpha = 0.10$ for $90\%$ coverage), calculate the finite-sample empirical quantile:

$$\hat{q} = \text{Quantile}\left(\{s_1, \dots, s_n\}, \, \frac{\lceil (n+1)(1-\alpha) \rceil}{n}\right)$$


* Construct distribution-free prediction intervals guaranteed under exchangeability without Gaussian assumptions:

$$\mathcal{C}(X_{\text{test}}) = \left[\max(0, \, \hat{f}(X_{\text{test}}) - \hat{q}), \quad \hat{f}(X_{\text{test}}) + \hat{q}\right]$$



### 2. Stochastic Lead-Time & Monte Carlo DDLT Simulation

Vendor lead time $L$ follows an empirical discrete distribution $P(L = l_k) = p_k$. For $m = 1, \dots, N$ iterations ($N = 10,000$):


$$\text{DDLT}^{(m)} = \sum_{d=1}^{L^{(m)}} D_d^{(m)}$$


where $L^{(m)} \sim P(L)$ and daily demand $D_d^{(m)} \sim \text{Uniform}\left(\max(0, \hat{y}_d - \hat{q}), \, \hat{y}_d + \hat{q}\right)$.

### 3. Prescriptive Asymmetric Cost Minimization

Given candidate order quantity $Q$, initial inventory $I_0$, shelf-life $\tau$, unit holding cost $C_{\text{holding}}$, spoilage penalty $C_{\text{waste}}$, and stockout penalty $C_{\text{stockout}}$:

$$\text{Shortage}^{(m)} = \max\left(0, \, \text{DDLT}^{(m)} - (I_0 + Q)\right)$$

$$\text{Ending Inventory}^{(m)} = \max\left(0, \, (I_0 + Q) - \text{DDLT}^{(m)}\right)$$

$$\text{Spoilage}^{(m)} = \max\left(0.05, \, \frac{1}{\max(1, \tau)}\right) \times \text{Ending Inventory}^{(m)}$$

$$\text{Expected Total Cost}(Q) = \frac{1}{N}\sum_{m=1}^N \left( C_{\text{stockout}} \cdot \text{Shortage}^{(m)} + C_{\text{waste}} \cdot \text{Spoilage}^{(m)} + C_{\text{holding}} \cdot \text{Ending Inventory}^{(m)} \right)$$

$$\text{Optimal Order Quantity } Q^* = \arg\min_{Q \ge 0} \, \text{Expected Total Cost}(Q) \quad \text{subject to } P(\text{Shortage} = 0) \ge \text{Target Service Level}$$

---

## 🕹️ Interactive Decision Dashboard

The Streamlit dashboard translates machine learning outputs into an interactive decision-support interface:

* **Visual Forecasting**: Plotly time-series charts displaying historical sales, point predictions, and shaded Conformal Prediction intervals.
* **Risk Distributions**: Empirical histograms of Demand During Lead Time (DDLT) with marked Reorder Points ($I_0 + Q^*$).
* **Cost Optimization Surface**: Interactive loss curve showing the global cost-minimum order quantity ($Q^*$).
* **What-If Scenario Stress Testing**:
* **Demand Shocks**: Simulate sudden demand fluctuations ($-40\%$ to $+60\%$).
* **Supplier Delay**: Add $+0$ to $+5$ transit delay days.
* **Financial Parameter Tuning**: Real-time adjustment of holding costs, disposal penalties, and stockout fines.
* **Service-Level Enforcement**: Slider controls ($85\%$ to $99\%$) adjusting safety stock boundaries dynamically.



---

## 🏗️ System Architecture

```text
OptiPerish/
├── 📁 app/
│   ├── api.py                    # FastAPI service endpoints (/forecast, /optimize)
│   └── streamlit_app.py          # Interactive Decision-Intelligence Dashboard
├── 📁 data/
│   ├── raw/                      # Generated grocery sales & lead-time metadata
│   └── processed/                # Chronologically split datasets (train/cal/test)
├── 📁 notebooks/
│   ├── 01_eda_and_data_gen.ipynb # Exploratory data analysis & seasonality decomposition
│   ├── 02_forecasting_baselines.ipynb # Baseline (t-7) vs. LightGBM benchmarking
│   └── 03_conformal_and_optimization.ipynb # Conformal calibration & Monte Carlo tuning
├── 📁 src/
│   ├── __init__.py
│   ├── benchmark_runner.py       # 4-tier ablation study execution harness
│   ├── conformal_engine.py       # Split Conformal Prediction module
│   ├── data_generator.py         # Synthetic 2-year multi-store grocery generator
│   ├── feature_pipeline.py       # Chronological split & lag/rolling feature pipeline
│   ├── inventory_optimizer.py    # Asymmetric perishable cost minimizer
│   ├── model_forecaster.py       # LightGBM & Seasonal Naive model wrappers
│   └── simulation_engine.py      # Vectorized Monte Carlo DDLT simulator
├── 📁 tests/
│   ├── test_conformal.py         # Conformal coverage validation unit tests
│   └── test_optimizer.py         # Cost minimization convergence unit tests
├── 📄 requirements.txt           # Environment dependencies
└── 📄 README.md

```

---

## ⚡ Quickstart Guide

### 1. Clone & Set Up Virtual Environment

```bash
# Clone repository
git clone [https://github.com/Bhxvyx05/OptiPerish-Decision-Intelligence.git](https://github.com/Bhxvyx05/OptiPerish-Decision-Intelligence.git)
cd OptiPerish-Decision-Intelligence

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
.\venv\Scripts\Activate
# macOS / Linux:
source venv/bin/activate

# Upgrade pip & install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

```

### 2. Generate Data & Run Feature Pipeline

```bash
# Generate 2-year multi-store grocery sales dataset
python src/data_generator.py

# Build lag/rolling features and chronological splits
python src/feature_pipeline.py

```

### 3. Run Benchmark Suite & Unit Tests

```bash
# Execute the 4-tier ablation benchmark
python -m src.benchmark_runner

# Run unit tests
pytest tests/

```

### 4. Launch Interactive Decision Dashboard

```bash
streamlit run app/streamlit_app.py

```

---

## 🛠️ Technology Stack

* **Core Programming**: Python 3.10+
* **Predictive Modeling**: LightGBM (Huber Loss Regressor), Scikit-Learn
* **Uncertainty Quantification**: Inductive Split Conformal Prediction
* **Simulation & Optimization**: NumPy (Vectorized Monte Carlo), SciPy Optimize (`minimize_scalar`)
* **Feature Engineering**: Pandas, Scikit-Learn `ColumnTransformer` (Chronological Splits)
* **Visualization & UI**: Streamlit, Plotly Graph Objects, Matplotlib, Seaborn
* **Testing & Validation**: Pytest

---

## 👨‍💻 Author

**Bhavya Dhingra**

* **GitHub**: [@Bhxvyx05](https://www.google.com/search?q=https://github.com/Bhxvyx05)

* **LinkedIn**: [linkedin.com/in/bhavyadhingra](https://linkedin.com)


```

```
