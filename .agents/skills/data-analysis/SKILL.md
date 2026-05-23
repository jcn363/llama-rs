---
name: data-analysis
description: "Use when exploring a new dataset, generating data-driven reports, validating data quality, performing statistical analysis, creating visualizations, or making data-driven recommendations - provides full analysis pipeline from loading and cleaning through statistical analysis, visualization, and structured reporting in Python and SQL."
---

# Data Analysis

## Overview

Full analysis pipeline covering dataset exploration, cleaning, statistical analysis, and visualization in Python (pandas) and SQL. Produces structured markdown reports with key findings and actionable recommendations.

## When to Use

- **Data exploration**: Understand a new dataset
- **Report generation**: Derive data-driven insights
- **Quality validation**: Check data consistency and integrity
- **Decision support**: Make data-driven recommendations

## Core Workflow

### Step 1: Load and Explore Data

**Python (pandas):**

```python
import pandas as pd
import numpy as np

# Load CSV
df = pd.read_csv('data.csv')

# Basic info
print(df.info())
print(df.describe())
print(df.head(10))

# Check missing values
print(df.isnull().sum())

# Data types
print(df.dtypes)
```

**SQL:**

```sql
-- Inspect table schema
DESCRIBE table_name;

-- Sample data
SELECT * FROM table_name LIMIT 10;

-- Basic stats
SELECT
    COUNT(*) as total_rows,
    COUNT(DISTINCT column_name) as unique_values,
    MIN(numeric_column) as min_val,
    MAX(numeric_column) as max_val,
    AVG(numeric_column) as avg_val
FROM table_name;
```

### Step 2: Data Cleaning

```python
# Handle missing values
df['column'].fillna(df['column'].mean(), inplace=True)
df.dropna(subset=['required_column'], inplace=True)

# Remove duplicates
df.drop_duplicates(inplace=True)

# Type conversions
df['date'] = pd.to_datetime(df['date'])
df['category'] = df['category'].astype('category')

# Remove outliers (IQR method)
Q1 = df['value'].quantile(0.25)
Q3 = df['value'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['value'] >= Q1 - 1.5 * IQR) & (df['value'] <= Q3 + 1.5 * IQR)]
```

### Step 3: Statistical Analysis

```python
# Descriptive statistics
print(df['numeric_column'].describe())

# Grouped analysis
grouped = df.groupby('category').agg({
    'value': ['mean', 'sum', 'count'],
    'other': 'nunique'
})
print(grouped)

# Correlation
correlation = df[['col1', 'col2', 'col3']].corr()
print(correlation)

# Pivot table
pivot = pd.pivot_table(df,
    values='sales',
    index='region',
    columns='month',
    aggfunc='sum'
)
```

### Step 4: Visualization

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Histogram
plt.figure(figsize=(10, 6))
df['value'].hist(bins=30)
plt.title('Distribution of Values')
plt.savefig('histogram.png')

# Boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(x='category', y='value', data=df)
plt.title('Value by Category')
plt.savefig('boxplot.png')

# Heatmap (correlation)
plt.figure(figsize=(10, 8))
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.savefig('heatmap.png')

# Time series
plt.figure(figsize=(12, 6))
df.groupby('date')['value'].sum().plot()
plt.title('Time Series of Values')
plt.savefig('timeseries.png')
```

### Step 5: Derive Insights

```python
# Top/bottom analysis
top_10 = df.nlargest(10, 'value')
bottom_10 = df.nsmallest(10, 'value')

# Trend analysis
df['month'] = df['date'].dt.to_period('M')
monthly_trend = df.groupby('month')['value'].sum()
growth = monthly_trend.pct_change() * 100

# Segment analysis
segments = df.groupby('segment').agg({
    'revenue': 'sum',
    'customers': 'nunique',
    'orders': 'count'
})
segments['avg_order_value'] = segments['revenue'] / segments['orders']
```

## Output Format

```markdown
# Data Analysis Report

## 1. Dataset Overview
- Dataset: [name]
- Records: X,XXX
- Columns: XX
- Date range: YYYY-MM-DD ~ YYYY-MM-DD

## 2. Key Findings
- Insight 1
- Insight 2
- Insight 3

## 3. Statistical Summary
| Metric | Value |
|--------|-------|
| Mean   | X.XX  |
| Median | X.XX  |
| Std dev | X.XX |

## 4. Recommendations
1. [Recommendation 1]
2. [Recommendation 2]
```

## Constraints

### Required (MUST)
- **Preserve raw data**: Always work on a copy, never modify the original
- **Document the process**: Code, assumptions, and decisions must be recorded
- **Validate results**: Cross-check findings against raw data

### Prohibited (MUST NOT)
- Do not expose sensitive personal data in reports or visualizations
- Do not draw unsupported conclusions — correlation ≠ causation

## Best Practices

- **Understand the data first**: Learn column meanings, units, and collection methods before analysis
- **Incremental analysis**: Move from simple summaries to complex models
- **Use multiple visualization types**: Histograms, boxplots, scatter plots, and time series reveal different patterns
- **Validate assumptions**: Always verify normality, linearity, independence assumptions
- **Reproducibility**: Document every step so results can be recreated from raw data

## Common Mistakes

| Mistake | Correction |
|---------|-----------|
| Analyzing without understanding column semantics | Read data dictionary or consult domain experts first |
| Ignoring missing values | Always check and document null counts before analysis |
| Confusing correlation with causation | State clearly when findings are correlational only |
| Data leakage when cleaning | Drop outliers/duplicates before any grouped analysis |
| Not preserving raw data | Work on `df_clean = df.copy()` |
| Overplotting in visualizations | Use alpha transparency, sampling, or hexbin for dense data |

## References

- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Matplotlib Gallery](https://matplotlib.org/stable/gallery/)
- [Seaborn Tutorial](https://seaborn.pydata.org/tutorial.html)
