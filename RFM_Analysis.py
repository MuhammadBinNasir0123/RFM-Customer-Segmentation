# =============================================================================
# RFM Customer Segmentation Analysis
#
# Goal: Understand customer purchasing behavior by breaking it down into:
# - Recency (R): How recently a customer made a purchase
# - Frequency (F): How often they buy
# - Monetary (M): How much money they spend
#
# By combining these metrics, we can group customers into segments and give
# each group specific marketing strategies to maximize retention and revenue.
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Setting up a clean style for plots (so visuals are easy to read)
sns.set_style("whitegrid")
print(" All libraries imported successfully!")

# =============================================================================
# STEP 1: LOAD AND EXPLORE THE DATA
# =============================================================================
print("\n STEP 1: Loading and exploring the dataset...")

# Load the dataset (should be in the working directory)
file_path = 'Online_Retail.csv'

try:
    df = pd.read_csv(file_path, encoding='iso-8859-1')
    print(" Dataset loaded successfully!")
    print(f"   - Shape: {df.shape[0]} rows and {df.shape[1]} columns")
except FileNotFoundError:
    print(" File not found! Please check the file path.")
    print("   Current file path:", file_path)
    exit()

# Quick preview of the data
print("\nFirst few rows of the dataset:")
print(df.head())

# =============================================================================
# STEP 2: CLEAN AND PREPARE THE DATA
# =============================================================================
print("\n\n STEP 2: Cleaning the data...")
print("   - Removing records without Customer ID")
print("   - Dropping returns/cancellations")
print("   - Adding a total sales column")

# Work on a copy to keep the raw data safe
df_clean = df.copy()

# Remove customers with missing IDs
initial_count = df_clean.shape[0]
df_clean = df_clean.dropna(subset=['Customer ID'])
print(f"   - Removed {initial_count - df_clean.shape[0]} records without Customer ID")

# Filter out returns (negative qty) and invalid prices
df_clean = df_clean[df_clean['Quantity'] > 0]
df_clean = df_clean[df_clean['Price'] > 0]
print(f"   - Removed returns/invalid rows → {df_clean.shape[0]} records left")

# Calculate sales per line item
df_clean['TotalSales'] = df_clean['Quantity'] * df_clean['Price']

# Convert invoice dates
print("   - Converting invoice dates...")
df_clean['InvoiceDate'] = pd.to_datetime(df_clean['InvoiceDate'], format='%d/%m/%Y %H:%M', errors='coerce')

# Drop rows where date conversion failed
failed_dates = df_clean['InvoiceDate'].isnull().sum()
if failed_dates > 0:
    print(f"   ⚠  {failed_dates} invalid dates dropped")
    df_clean = df_clean.dropna(subset=['InvoiceDate'])

print(" Data cleaned!")
print(f"   - Final size: {df_clean.shape[0]} rows")
print(f"   - Time span: {df_clean['InvoiceDate'].min().date()} → {df_clean['InvoiceDate'].max().date()}")

# =============================================================================
# STEP 3: CALCULATE RFM METRICS
# =============================================================================
print("\n\n STEP 3: Calculating RFM metrics...")

# Snapshot date = day after last transaction in dataset
snapshot_date = df_clean['InvoiceDate'].max() + pd.Timedelta(days=1)
print(f"   - Snapshot date: {snapshot_date.date()}")

# Compute metrics per customer
rfm = df_clean.groupby('Customer ID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,  # Recency
    'Invoice': 'nunique',  # Frequency (unique purchases)
    'TotalSales': 'sum'  # Monetary (total spend)
})

rfm.columns = ['Recency', 'Frequency', 'Monetary']
print(f" RFM metrics calculated for {rfm.shape[0]} customers!")

# Overview of the metrics
print("\nRFM Summary Stats:")
print(rfm.describe())

# =============================================================================
# STEP 4: ASSIGN RFM SCORES
# =============================================================================
print("\n\n STEP 4: Scoring customers (1-4 scale)...")
print("   - Recency: lower = better (score 4 = most recent)")
print("   - Frequency/Monetary: higher = better (score 4 = top buyers)")

# Recency: most recent → score 4
rfm['R_Score'] = pd.qcut(rfm['Recency'], q=4, labels=[4, 3, 2, 1])

# Frequency & Monetary: higher values = higher scores
freq_bins = [rfm['Frequency'].min() - 1, 1, 2, 3, rfm['Frequency'].max()]
monetary_bins = [rfm['Monetary'].min() - 1,
                 rfm['Monetary'].quantile(0.25),
                 rfm['Monetary'].quantile(0.5),
                 rfm['Monetary'].quantile(0.75),
                 rfm['Monetary'].max()]

rfm['F_Score'] = pd.cut(rfm['Frequency'], bins=freq_bins, labels=[1, 2, 3, 4], include_lowest=True)
rfm['M_Score'] = pd.cut(rfm['Monetary'], bins=monetary_bins, labels=[1, 2, 3, 4], include_lowest=True)

# Fill missing values (edge cases)
for column in ['R_Score', 'F_Score', 'M_Score']:
    if rfm[column].isnull().any():
        null_count = rfm[column].isnull().sum()
        median_val = rfm[column].median()
        rfm[column].fillna(median_val, inplace=True)

# Combine into single RFM code (e.g., 444 = top tier)
rfm['RFM_Segment'] = (
    rfm['R_Score'].astype(int).astype(str) +
    rfm['F_Score'].astype(int).astype(str) +
    rfm['M_Score'].astype(int).astype(str)
)

print(" Scoring done!")
print("\nScore Distributions:")
print("Recency:", rfm['R_Score'].value_counts().sort_index().to_dict())
print("Frequency:", rfm['F_Score'].value_counts().sort_index().to_dict())
print("Monetary:", rfm['M_Score'].value_counts().sort_index().to_dict())

# =============================================================================
# STEP 5: CREATE CUSTOMER SEGMENTS
# =============================================================================
print("\n\n STEP 5: Putting customers into meaningful groups...")

def get_segment(rfm_row):
    """
    Maps RFM scores into easy-to-understand groups
    """
    r_score = rfm_row['R_Score']
    f_score = rfm_row['F_Score']
    m_score = rfm_row['M_Score']

    if r_score == 4 and f_score == 4 and m_score == 4:
        return 'Champions'
    elif r_score == 4 and f_score >= 3:
        return 'Loyal Customers'
    elif r_score == 4 and f_score == 1:
        return 'New Customers'
    elif r_score == 3:
        return 'Potential Loyalists'
    elif r_score == 2:
        return 'At Risk Customers'
    elif r_score == 1 and f_score >= 2:
        return 'Cant Lose Them'
    elif r_score == 1:
        return 'Lost Customers'
    else:
        return 'Others'

rfm['Customer_Group'] = rfm.apply(get_segment, axis=1)
segment_counts = rfm['Customer_Group'].value_counts()

print(" Customers segmented!")
print("\nSegment Breakdown:")
for segment, count in segment_counts.items():
    print(f"   - {segment}: {count} customers ({(count / len(rfm) * 100):.1f}%)")

# =============================================================================
# STEP 6: MARKETING RECOMMENDATIONS
# =============================================================================
print("\n\n STEP 6: What should we do with each group?")

suggestions = {
    'Champions': "Reward them: VIP perks, early access, premium offers",
    'Loyal Customers': "Keep them engaged: tiered loyalty programs, exclusive discounts",
    'New Customers': "Welcome properly: onboarding series, first-purchase discount",
    'Potential Loyalists': "Nurture them: recommendations, educational content, gentle offers",
    'At Risk Customers': "Bring them back: 'We miss you' emails, 15–20% discounts",
    'Cant Lose Them': "High stakes: strong offers, personal outreach, feedback calls",
    'Lost Customers': "Last try: surveys to learn why they left, comeback deals",
    'Others': "Broad marketing: newsletters, brand awareness campaigns"
}

print("=" * 60)
for segment, count in segment_counts.items():
    print(f"\n {segment} ({count} customers):")
    print(f"   {suggestions.get(segment, 'General marketing approach')}")
print("=" * 60)

# =============================================================================
# STEP 7: CREATE VISUALIZATIONS - PAGE 1
# =============================================================================
print("\n\n STEP 7: Building the first dashboard...")
print(" - Page 1: High-level customer overview & quick insights")

# Convert categorical RFM scores to numbers so we can plot averages
rfm['R_Score_Num'] = rfm['R_Score'].astype(int)
rfm['F_Score_Num'] = rfm['F_Score'].astype(int)
rfm['M_Score_Num'] = rfm['M_Score'].astype(int)

# Create a 2x2 dashboard layout
fig1, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Customer Segment Distribution
ax1 = axes[0, 0]
segment_counts_sorted = segment_counts.sort_values(ascending=True)
colors = plt.cm.Set3(np.linspace(0, 1, len(segment_counts_sorted)))
bars = ax1.barh(range(len(segment_counts_sorted)), segment_counts_sorted.values, color=colors, alpha=0.85)
ax1.set_title('CUSTOMER SEGMENT DISTRIBUTION', fontweight='bold', pad=10)
ax1.set_yticks(range(len(segment_counts_sorted)))
ax1.set_yticklabels(segment_counts_sorted.index, fontsize=9)
ax1.set_xlabel('Number of Customers', fontsize=10)
ax1.grid(True, alpha=0.3, axis='x')

# Add labels on bars so it’s easier to read the counts directly
for i, (bar, count) in enumerate(zip(bars, segment_counts_sorted.values)):
    ax1.text(bar.get_width() + max(segment_counts_sorted.values) * 0.01,
             bar.get_y() + bar.get_height() / 2,
             f'{count}', ha='left', va='center', fontweight='bold', fontsize=9)

# 2. Average RFM Scores
ax2 = axes[0, 1]
score_means = [rfm['R_Score_Num'].mean(),
               rfm['F_Score_Num'].mean(),
               rfm['M_Score_Num'].mean()]
score_labels = ['Recency', 'Frequency', 'Monetary']
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
bars = ax2.bar(score_labels, score_means, color=colors, alpha=0.8)
ax2.set_title('AVERAGE RFM SCORES (1-4 Scale)', fontweight='bold', pad=10)
ax2.set_ylabel('Average Score', fontsize=10)
ax2.set_ylim(0, 4.5)
ax2.grid(True, alpha=0.3, axis='y')

# Add score values above each bar
for bar, value in zip(bars, score_means):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
             f'{value:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

# 3. Quick Metrics Summary (plain text box)
ax3 = axes[1, 0]
ax3.axis('off')
summary_text = (
    f"KEY METRICS SUMMARY\n\n"
    f"Total Customers: {len(rfm):,}\n"
    f"Most Common Segment: {segment_counts.idxmax()}\n"
    f"Largest Segment Size: {segment_counts.max():,}\n"
    f"Top Spending Segment: {rfm.groupby('Customer_Group')['Monetary'].mean().idxmax()}\n"
    f"Avg Recency: {rfm['Recency'].mean():.0f} days\n"
    f"Avg Frequency: {rfm['Frequency'].mean():.1f} purchases\n"
    f"Avg Spending: ${rfm['Monetary'].mean():.0f}\n"
    f"Total Revenue: ${rfm['Monetary'].sum():,.0f}"
)
ax3.text(0.5, 0.5, summary_text, transform=ax3.transAxes, fontsize=11,
         ha='center', va='center', fontweight='bold',
         bbox=dict(boxstyle="round,pad=1", facecolor="lightblue", alpha=0.8))

# 4. Top 5 Customers by Spending
ax4 = axes[1, 1]
top_customers = rfm.nlargest(5, 'Monetary')
bars = ax4.barh(range(len(top_customers)), top_customers['Monetary'], color='orange', alpha=0.7)
ax4.set_title('TOP 5 CUSTOMERS BY SPENDING', fontweight='bold', pad=10)
ax4.set_xlabel('Total Spending ($)', fontsize=10)
ax4.set_yticks(range(len(top_customers)))
ax4.set_yticklabels([f'Customer {int(idx)}' for idx in top_customers.index], fontsize=9)
ax4.grid(True, alpha=0.3, axis='x')

# Add spending labels next to each bar
for i, (bar, value) in enumerate(zip(bars, top_customers['Monetary'])):
    ax4.text(bar.get_width() + max(top_customers['Monetary']) * 0.01,
             bar.get_y() + bar.get_height() / 2,
             f'${value:,.0f}', ha='left', va='center', fontweight='bold', fontsize=9)

plt.tight_layout()
plt.subplots_adjust(top=0.95, hspace=0.4, wspace=0.3)
plt.savefig('rfm_page1_overview.png', dpi=300, bbox_inches='tight', facecolor='white')
print(" Page 1 dashboard saved as 'rfm_page1_overview.png'")
plt.show()

# Drop temporary numeric columns
rfm = rfm.drop(['R_Score_Num', 'F_Score_Num', 'M_Score_Num'], axis=1)


# =============================================================================
# STEP 8: CREATE VISUALIZATIONS - PAGE 2
# =============================================================================
print("\n\n STEP 8: Creating the second dashboard...")
print(" - Page 2: RFM heatmap and value distributions")

# Re-add numeric versions for plotting
rfm['R_Score_Num'] = rfm['R_Score'].astype(int)
rfm['F_Score_Num'] = rfm['F_Score'].astype(int)
rfm['M_Score_Num'] = rfm['M_Score'].astype(int)

# Make another 2x2 dashboard
fig2, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Heatmap of Recency vs Frequency with Average Spending
ax1 = axes[0, 0]
heatmap_data = rfm.groupby(['R_Score_Num', 'F_Score_Num'])['Monetary'].mean().unstack().fillna(0)
heatmap_data = heatmap_data.sort_index(ascending=False)  
im = ax1.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
ax1.set_title('RFM HEATMAP: AVERAGE SPENDING', fontweight='bold', pad=10)
ax1.set_xlabel('Frequency Score', fontsize=10)
ax1.set_ylabel('Recency Score', fontsize=10)
ax1.set_xticks(range(4))
ax1.set_yticks(range(4))
ax1.set_xticklabels([1, 2, 3, 4], fontsize=9)
ax1.set_yticklabels([4, 3, 2, 1], fontsize=9)

# Add dollar values inside each heatmap cell
for i in range(4):
    for j in range(4):
        if j < heatmap_data.shape[1] and i < heatmap_data.shape[0]:
            value = heatmap_data.iloc[i, j]
            ax1.text(j, i, f'${value:.0f}', ha='center', va='center',
                     color='black', fontweight='bold', fontsize=8)

plt.colorbar(im, ax=ax1, shrink=0.8, label='Average Spending ($)')

# 2. Recency Histogram
ax2 = axes[0, 1]
recency_data = rfm[rfm['Recency'] <= 365]  # Focus on customers active in the last year
sns.histplot(data=recency_data, x='Recency', bins=25, ax=ax2, color='skyblue', alpha=0.7)
ax2.set_title('RECENCY DISTRIBUTION', fontweight='bold', pad=10)
ax2.set_xlabel('Days Since Last Purchase', fontsize=10)
ax2.set_ylabel('Number of Customers', fontsize=10)
ax2.grid(True, alpha=0.3)

# 3. Frequency Histogram
ax3 = axes[1, 0]
frequency_data = rfm[rfm['Frequency'] <= 25]  # Cut off at 25 for clarity
sns.histplot(data=frequency_data, x='Frequency', bins=15, ax=ax3, color='lightgreen', alpha=0.7)
ax3.set_title('FREQUENCY DISTRIBUTION', fontweight='bold', pad=10)
ax3.set_xlabel('Number of Purchases', fontsize=10)
ax3.set_ylabel('Number of Customers', fontsize=10)
ax3.grid(True, alpha=0.3)

# 4. Monetary Histogram
ax4 = axes[1, 1]
monetary_data = rfm[rfm['Monetary'] <= 5000]  # Focus on typical spenders under 5k
sns.histplot(data=monetary_data, x='Monetary', bins=20, ax=ax4, color='salmon', alpha=0.7)
ax4.set_title('SPENDING DISTRIBUTION', fontweight='bold', pad=10)
ax4.set_xlabel('Spending per Customer ($)', fontsize=10)
ax4.set_ylabel('Number of Customers', fontsize=10)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.subplots_adjust(top=0.95, hspace=0.4, wspace=0.3)
plt.savefig('rfm_page2_distributions.png', dpi=300, bbox_inches='tight', facecolor='white')
print(" Page 2 dashboard saved as 'rfm_page2_distributions.png'")
plt.show()

# Clean up numeric helper columns
rfm = rfm.drop(['R_Score_Num', 'F_Score_Num', 'M_Score_Num'], axis=1)


# =============================================================================
# STEP 9: SAVE RESULTS
# =============================================================================
print("\n\n STEP 9: Exporting results...")
rfm.to_csv('rfm_analysis_results.csv')
print(" RFM results saved as 'rfm_analysis_results.csv'")

print("\n ANALYSIS COMPLETE!")
print(" Final outputs generated:")
print(" - rfm_page1_overview.png (Customer overview dashboard)")
print(" - rfm_page2_distributions.png (RFM heatmap + distributions)")
print(" - rfm_analysis_results.csv (Full RFM scores & segments)")
print("\nNext step: Plug these insights into targeted marketing campaigns ")

