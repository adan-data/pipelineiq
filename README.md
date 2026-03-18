# PipelineIQ, CRM Analytics Dashboard

A full-stack analytics project built on 18 months of SaaS customer data (598 customers, Jan 2022 to Jun 2023). The Streamlit dashboard surfaces churn drivers, cohort retention, survival economics, pricing elasticity, and marketing attribution. All findings are reproducible via SQL against `saas_intel.db`.

---

## Live Demos

| | Link |
|---|---|
| **Streamlit App** | [pipelineiq.streamlit.app](https://pipelineiq.streamlit.app/) |
| **Tableau Dashboard** | [PipelineIQ on Tableau Public](https://public.tableau.com/views/PipelineIQSaaSCRMAnalytics/PipelineIQDashboard) |
| **GitHub** | [adan-data/pipelineiq](https://github.com/adan-data/pipelineiq) |

---

## What This Project Demonstrates

This project is an end-to-end analytics workflow built to move from raw dirty data to prioritized, quantified business decisions. It is not a model showcase. The goal at every step was to answer a specific business question, not to demonstrate a technique.

**The central question:** *A SaaS business has three customer tiers, a worsening churn problem, and limited CS capacity. Where does every dollar of retention investment produce the most return?*

The project answers that question through six interconnected analyses, each building on the last.

---

## Business Intelligence, What the Data Actually Says

### 1. The Starter Tier Is a Capital Destruction Machine

The single most financially consequential finding: **61% of customers are Starters, but the average Starter destroys value**. At 10.22%/mo churn, the expected survival to breakeven (month 9.3) is only 39%. That means 61 out of every 100 Starter customers acquired churn before repaying the $196 CAC. Net LTV is roughly +$13, indistinguishable from zero given estimation error.

This is not a rounding error. The Starter unit economics are structurally broken under current churn conditions. Every dollar of acquisition budget spent on Starters without fixing the onboarding funnel has a near-zero expected return.

**Why it matters for decision-making:** This changes the resource allocation question entirely. The correct response is not "get more Starters", it is "fix months 1 to 3 for Starters, or shift acquisition budget toward Pro and Enterprise where unit economics are healthy."

### 2. Churn Is Worsening Structurally, This Is Not Noise

Monthly churn rose from 5.97% (Q1 2022) to 9.37% (Q2 2023), a 57% relative increase over six quarters. This pattern holds across all tiers and cohorts, it is not driven by one bad quarter or one bad channel. It is a structural signal that something in the product or onboarding experience has been degrading.

The cohort heatmap (NB11) makes this visible: early cohorts (2022) retain meaningfully better at every month than later cohorts (2023). The business is not just struggling with churn, it is becoming worse at retaining customers over time.

**Why it matters:** A static churn rate is a manageable operating problem. A *rising* churn rate is an existential one. The first priority action is diagnosing what changed between Q3 2022 and Q1 2023.

### 3. Annual Billing Exhibits Adverse Selection, Not Retention Benefit

The conventional wisdom in SaaS is that annual billing reduces churn because customers are financially committed for 12 months. The data contradicts this: annual billing customers churn at 8.17%/mo vs 6.85% for monthly, and the gap is consistent within every tier (Enterprise annual: 5.06% vs 2.30% monthly).

The causal inference analysis (NB10, Regression Discontinuity) confirms this is not a statistical artifact. The RD design, comparing customers just above and below the annual billing discount threshold, shows the discount attracts price-sensitive customers who still churn, rather than committed customers who were already planning to stay.

**Why it matters:** The discount structure is actively selecting for the wrong customers. Raising the annual discount threshold or adding onboarding gates for annual sign-ups would reduce adverse selection without reducing genuine annual commitment.

### 4. Enterprise Is the Only Reliable Growth Lever

Enterprise represents 45% of active MRR with only 15% of customers. Monthly churn is 3.10%, breakeven is month 10, and 74% of Enterprise customers survive to that point. Gross LTV is ~$6,000, net LTV ~$4,200, 323x the expected net LTV of a Starter customer.

The gap analysis (NB08, benchmarked against OpenView 2024 and KeyBanc 2024) shows Enterprise LTV:CAC of 3.27x is the only metric at benchmark level. Everything else, NRR, logo churn, and Magic Number are all critical or below benchmark.

**Why it matters:** The growth strategy implied by the data is simple: stop acquiring Starters at scale, invest in Starter onboarding to improve survival to breakeven, and allocate acquisition budget toward Enterprise. One additional Enterprise customer per month generates more expected LTV than 30 additional Starter customers.

### 5. Marketing Attribution: Referral Is Undercounted by 13 Percentage Points

The last-click attribution model credits Organic Search with 42% of signups and Referral with only 18%. The RidgeCV marketing mix model (NB04) corrects this: Organic drops to 28% (it is a demand *capture* channel, not demand *creation*) and Referral rises to 31%, the single largest reallocation.

The model uses TimeSeriesSplit cross-validation to prevent future data leakage, and bootstrap confidence intervals to quantify coefficient uncertainty. A VIF analysis confirmed multicollinearity was present (channels are correlated) and Ridge regression was selected over OLS specifically because it handles correlated predictors without producing negative channel coefficients.

**Why it matters:** Last-click attribution will systematically underfund Referral and overfund Paid Search. The data suggests reallocating 5 to 8% of Paid Search budget toward a structured referral programme would produce a higher marginal return per dollar.

### 6. Pricing: Enterprise Is Underpriced, Pro Is Overpriced

The log-log demand model (NB06) estimates price elasticity per tier from observed conversion data. Enterprise shows elasticity ε ≈ -0.8, inelastic demand, meaning a 10% price increase reduces conversion by only 8%, producing net positive revenue. Enterprise customers are buying on functionality and integration cost, not price sensitivity.

Pro shows ε ≈ -2.1, highly elastic. The model estimates the revenue-maximising Pro price is $40 to 50, well below the current $74. The wide confidence interval (±35%) means the exact number is uncertain, but the *direction* is robust across the full CI range: Pro is overpriced and is generating lower conversion revenue than a lower price would.

**Why it matters:** Raising Enterprise price to $400 is the lowest-risk revenue lever in this dataset. Lowering Pro price requires A/B validation first (designed in NB09) because the conversion uplift assumption needs empirical confirmation before committing.

---

## What This Project Shows Analytically

Beyond the business findings, the project was designed to demonstrate specific analytical skills that are easy to claim but hard to evidence:

- **Honest uncertainty quantification:** every model includes confidence intervals, explicit assumptions, and a required limitations section. The pricing notebook literally has a section titled "What This Is and What It Is Not."
- **Methodological justification:** Ridge over OLS because LASSO eliminates channels we need in the attribution model; customer-months churn rate instead of `churned.mean()` because the latter is a lifetime proportion, not a rate.
- **Causal thinking:** the adverse selection finding in annual billing was validated with a Regression Discontinuity design, not just a group comparison. The onboarding finding was validated with Difference-in-Differences.
- **Data integrity as a first-class concern:** NB01 and NB02 treat data cleaning as an audit trail, documenting every anomaly (negative MRR, 9999 outliers, mixed-case tier names, future churn dates) and explaining why each one would have distorted downstream models if uncorrected.
- **SQL as a production artefact:** NB11 contains 10 production-ready SQL queries against `saas_intel.db` with documented business purpose, not just exploratory queries.

---

## Quick Start

```bash
git clone https://github.com/adan-data/pipelineiq.git
cd pipelineiq
pip install -r requirements.txt
streamlit run app.py
```

The app runs without the database, it falls back to `clean_saas_customers.csv` automatically. To build the database:

```bash
python src/11_survival_analysis.py
```

---

## Project Structure

```
pipelineiq/
├── app.py                          # Streamlit dashboard (7 pages)
├── saas_intel.db                   # SQLite database (built by src scripts)
├── requirements.txt
│
├── data/
│   └── processed/
│       ├── clean_saas_customers.csv
│       └── cohort_retention_grid.csv
│
├── src/
│   └── 11_survival_analysis.py     # Builds km_survival + cac_breakeven tables
│
├── notebooks/
│   ├── 01_eda_narrative.ipynb           # Data quality audit trail
│   ├── 02_cleaning_decisions.ipynb      # Every cleaning decision documented
│   ├── 03_churn_model_development.ipynb # GBM + SHAP feature importance
│   ├── 04_mmm_development.ipynb         # RidgeCV marketing mix model
│   ├── 05_survival_analysis.ipynb       # KM curves + CAC breakeven
│   ├── 06_pricing_optimisation.ipynb    # Log-log demand + elasticity
│   ├── 07_rfm_segmentation.ipynb        # SaaS-adapted RFM scoring
│   ├── 08_risk_gap_analysis.ipynb       # Monte Carlo + benchmark gaps
│   ├── 09_experiment_design.ipynb       # Power analysis for 5 experiments
│   ├── 10_causal_inference.ipynb        # RD + DiD causal validation
│   └── 11_cohort_retention_sql.ipynb    # Cohort grid + 10 SQL queries
│
└── excel/
    └── PipelineIQ_Analytics.xlsx   # 9-sheet stakeholder workbook
```

---

## Dashboard Pages

| Page | What it shows |
|---|---|
| Executive Overview | 4 KPI metrics, priority action items, active MRR by tier and quarter |
| Customer Churn Analysis | Churn by tier, channel, billing cycle, and cohort quarter |
| Cohort Retention | 18-cohort heatmap + average retention curve with P25–P75 band |
| Survival & CAC Breakeven | One panel per tier, survival curve, breakeven month, % surviving to BE |
| Pricing Elasticity | Log-log demand model, tier-level elasticity, optimal price ranges |
| Marketing Attribution (MMM) | Last-click vs RidgeCV attribution, channel churn enrichment |
| Customer Segmentation (RFM) | 5 RFM segments, MRR at risk, segment action matrix |

---

## Key Metrics (from actual data)

| Metric | Value | Benchmark | Status |
|---|---|---|---|
| Overall monthly churn | 7.18%/mo | 2%/mo | Critical |
| Starter monthly churn | 10.22%/mo | 2%/mo | Critical |
| Enterprise monthly churn | 3.10%/mo | 2%/mo | Below benchmark |
| Starter survival to breakeven | 39% | >50% | Below threshold |
| Enterprise net LTV | +$4,195 |, | Healthy |
| Churn trend | +57% over 6 quarters | Flat | Worsening |
| NRR (estimated) | 85% | 106% | Critical |
| Annual vs monthly churn | Annual > Monthly | Annual < Monthly | Adverse selection |

---

## Notebooks

| # | Notebook | Business Question Answered |
|---|---|---|
| 01 | EDA | What data quality failures exist and what would they have distorted? |
| 02 | Cleaning | What cleaning decisions were made and why? |
| 03 | Churn Model | What features predict churn and how predictive are they? |
| 04 | MMM | What is each marketing channel actually contributing to signups? |
| 05 | Survival | At what month does a customer from each tier become profitable? |
| 06 | Pricing | Is each tier priced above or below its revenue-maximising point? |
| 07 | RFM | Which active customers need CS attention most urgently? |
| 08 | Risk/Gap | How bad do outcomes get in the bear case, and where is the business vs benchmarks? |
| 09 | Experiments | What experiments would validate the pricing and onboarding recommendations? |
| 10 | Causal | Does annual billing causally reduce churn, or is the correlation spurious? |
| 11 | Cohort + SQL | How does retention vary by cohort, and what SQL powers the dashboard? |

---

## Excel Deliverable (`excel/PipelineIQ_Analytics.xlsx`)

9 sheets for non-technical stakeholders, all numbers from actual data:

| Sheet | Contents |
|---|---|
| 1_Executive KPIs | 13 headline metrics with units and context notes |
| 2_Tier Breakdown | Unit economics, hazard, avg MRR, CAC, breakeven month, LTV, net LTV |
| 3_Channel Breakdown | Churn rate and avg MRR per acquisition channel |
| 4_Quarterly Trend | Monthly churn per cohort quarter with QoQ change |
| 5_Cohort Retention | Retention % grid, 18 cohorts × Mo 0 to 12 |
| 6_Survival Breakeven | Survival % at Mo 1, 3, 6, 9, 12, 18, 24 + breakeven summary per tier |
| 7_RFM Segments | 5 segments with MRR at risk, priority, and recommended action |
| 8_Pricing Elasticity | Elasticity estimates, optimal ranges, expected revenue impact |
| 9_Gap Analysis | 9 metrics benchmarked against OpenView/Bessemer/KeyBanc 2024 |

---

## Data Dictionary

`clean_saas_customers.csv`, 598 rows, one per customer:

| Column | Type | Description |
|---|---|---|
| customer_id | str | Unique identifier |
| signup_month | date | First month of subscription |
| tier | str | Starter / Pro / Enterprise |
| billing_cycle | str | Annual / Monthly |
| channel | str | Acquisition channel |
| mrr | float | Monthly recurring revenue, cleaned to $5 to $500 |
| churned | int | 1 = churned, 0 = active as of Jan 2024 |
| churn_date | date | Month of churn (null if active) |

---

## Tech Stack

- **Dashboard:** Streamlit, Plotly
- **Analysis:** pandas, numpy, scipy, scikit-learn, statsmodels
- **Database:** SQLite via sqlite3
- **Notebooks:** Jupyter
- **Export:** openpyxl

---

## Important Notes on Methodology

**Churn rate definition:** All churn rates use the customer-months method: `churned ÷ total customer-months observed`. This produces a monthly rate. The figure `churned.mean() = 68.2%` is the lifetime churn proportion, not a monthly rate, so it should not be compared to the 2%/mo industry benchmark.

**Survival curves:** The dashboard uses the exponential approximation `exp(-h×t)` for smooth rendering. The `src/11_survival_analysis.py` script stores actual Kaplan-Meier step curves in `saas_intel.db`. The KM median (8.1 months for Starter) differs from the exponential approximation (ln(2)/h = 6.78 months) because Starter hazard is not perfectly constant over time.

**Pricing and MMM estimates:** Observational estimates carry ±35% uncertainty. They indicate direction and order of magnitude, not precise optimal values. All pricing recommendations require A/B validation (experimental designs provided in NB09) before full rollout.

**Causal claims:** Only findings validated in NB10 (RD and DiD designs) are stated as causal. All other findings are associations with stated assumptions.
