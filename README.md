# RFM Customer Segmentation Analysis

A Python-based customer segmentation project using RFM (Recency, Frequency, Monetary) modeling on real retail transactional data. The goal was to identify distinct customer groups and translate them into actionable marketing strategies.

---

## Project Overview

RFM analysis is a proven marketing technique that scores customers based on three dimensions:

- **Recency** — How recently they made a purchase
- **Frequency** — How often they buy
- **Monetary** — How much they spend in total

By combining these scores, customers are grouped into meaningful segments — each with a tailored retention or growth strategy.

---

## Key Numbers

| Metric | Value |
|---|---|
| Transactions Analyzed | 800,000+ |
| Customers Segmented | 5,860 |
| Segments Discovered | 7 |
| Top 5 Customer Revenue | $1.9M+ |
| High-Value Customers at Risk | 615 |

---

## Customer Segments

| Segment | Customers | Priority | Strategy |
|---|---|---|---|
| Champions | 720 | 🟢 High | VIP rewards, early access, premium offers |
| Loyal Customers | 506 | 🟡 Medium | Tiered loyalty programs, exclusive discounts |
| Potential Loyalists | 1,430 | 🟡 Medium | Personalized recommendations, gentle nudges |
| New Customers | 123 | 🟡 Medium | Welcome series, onboarding journey |
| At Risk Customers | 1,470 | 🔴 High | Win-back campaigns, 15–20% discounts |
| Can't Lose Them | 615 | 🔴 Critical | Personal outreach, strong retention offers |
| Lost Customers | 842 | ⚫ Low | Surveys, last-chance comeback deals |

---

## Visualizations

**Page 1 — Customer Overview**
- Customer segment distribution (horizontal bar chart)
- Average RFM scores across all customers
- Key metrics summary
- Top 5 customers by total revenue

**Page 2 — RFM Deep Dive**
- RFM heatmap: average spending by Recency × Frequency score
- Recency distribution histogram
- Frequency distribution histogram
- Monetary distribution histogram

---

## Technical Implementation

**RFM Score Calculation**
```python
snapshot_date = df_clean['InvoiceDate'].max() + pd.Timedelta(days=1)

rfm = df_clean.groupby('Customer ID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,  # Recency
    'Invoice': 'nunique',                                      # Frequency
    'TotalSales': 'sum'                                        # Monetary
})
rfm.columns = ['Recency', 'Frequency', 'Monetary']
```

**Scoring Logic**
- Customers scored 1–4 on each dimension using quartile-based binning
- Scores combined into a 3-digit RFM code (e.g. 444 = Champion)
- Segments assigned using rule-based logic on R, F, M score combinations

---

## Tools & Libraries

| Tool | Purpose |
|---|---|
| Python 3.8+ | Core language |
| pandas | Data cleaning and transformation |
| numpy | Numerical operations |
| matplotlib | Chart rendering |
| seaborn | Statistical visualizations |

---

## Dataset

This project uses the **Online Retail Dataset** from the UCI Machine Learning Repository.

- Download it here: [UCI Online Retail Dataset](https://archive.ics.uci.edu/ml/datasets/online+retail)
- Save as `Online_Retail.csv` in the project root before running

---

## How to Run

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/rfm-segmentation.git
cd rfm-segmentation

# 2. Install dependencies
pip install pandas numpy matplotlib seaborn

# 3. Add the dataset
# Place Online_Retail.csv in the project root

# 4. Run the analysis
python RFM_Analysis.py
```

---

## Output Files

```
├── RFM_Analysis.py                    # Main analysis script
├── Online_Retail.csv                  # Dataset (not included)
├── rfm_analysis_results.csv          # Full RFM scores per customer
├── RFM Analysis Visuals (Page 1).png # Overview dashboard
├── RFM Analysis Visuals (Page 2).png # Distribution dashboard
└── README.md                         # Project documentation
```

---

## Business Impact

This segmentation enables marketing teams to:

- Retain high-value customers before they churn
- Re-engage at-risk segments with targeted campaigns
- Allocate marketing budget more precisely
- Manage the full customer lifecycle with data-backed decisions

---

