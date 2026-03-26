# Descriptive Statistics

## What is it?
Simple tools to **summarize and understand data** — the first step in any data analysis.

Three categories:
- **Central Tendency** — where does the data center?
- **Variability** — how spread out is the data?
- **Frequency Distribution** — how is the data distributed?

---

## 1. Measures of Central Tendency

### Mean
Average of all values:
```
x̄ = Σx / n
```
```python
import numpy as np
arr = [5, 6, 11]
print(np.mean(arr))  # 7.33
```

### Median
Middle value of a sorted dataset.
- Odd count → center value
- Even count → average of two middle values

```python
arr = [1, 2, 3, 4]
print(np.median(arr))  # 2.5
```
> Better than mean for **skewed data**

### Mode
Most frequently occurring value — useful for categorical data.
```python
from scipy import stats
arr = [1, 2, 2, 3]
print(stats.mode(arr))  # mode=2, count=2
```

| Measure | Best Used When |
|---------|---------------|
| Mean | Symmetric, no outliers |
| Median | Skewed data or outliers present |
| Mode | Categorical data, most common value |

---

## 2. Measures of Variability

### Range
```
Range = Max - Min
```
> Simple but sensitive to outliers

```python
arr = [1, 2, 3, 4, 5]
print(max(arr) - min(arr))  # 4
```

### Variance
Average squared deviation from the mean:
```
σ² = Σ(x - μ)² / N
```
```python
import statistics
arr = [1, 2, 3, 4, 5]
print(statistics.variance(arr))  # 2.5
```

### Standard Deviation
Square root of variance — same units as data:
```
σ = √[Σ(x - μ)² / N]
```
```python
print(statistics.stdev(arr))  # 1.58
```

| Measure | Formula | Sensitive to Outliers? |
|---------|---------|----------------------|
| Range | Max - Min | ✅ Very |
| Variance | Σ(x-μ)²/N | Moderate |
| Std Dev | √Variance | Moderate |

---

## 3. Measures of Frequency Distribution

Summarizes how data is distributed across categories or intervals.

Includes:
- Data intervals or categories
- Frequency counts
- Relative frequencies (percentages)
- Cumulative frequencies

> First step before creating histograms, pie charts, or applying advanced methods

---

## ML Relevance
- **Mean & Std Dev** — used in feature normalization / standardization
- **Variance** — used in PCA to measure how much information each feature carries
- **Standard deviation** — used to detect outliers (`> 3σ` rule)
- **Frequency distribution** — basis for understanding data before modelling
