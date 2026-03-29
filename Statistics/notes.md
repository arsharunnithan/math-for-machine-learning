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
---
# Inferential Statistics

## What is it?
Making **predictions and conclusions about a population** based on sample data.

| | Descriptive Statistics | Inferential Statistics |
|---|---|---|
| Purpose | Summarize data | Draw conclusions beyond the data |
| Scope | The sample itself | The broader population |
| Output | Mean, std dev, charts | Predictions, confidence intervals, p-values |

---

## Key Techniques

### 1. Confidence Intervals
A range of values that likely contains the true population parameter.

```
CI = x̄ ± Z(α/2) × (σ / √n)
```

| Symbol | Meaning |
|--------|---------|
| x̄ | Sample mean |
| Z(α/2) | Z-value (1.96 for 95% CI) |
| σ | Population standard deviation |
| n | Sample size |

> 95% CI means: if we repeated the experiment 100 times, 95 of those intervals would contain the true mean.

---

### 2. Hypothesis Testing

A formal procedure to test claims about data.

| Term | Meaning |
|------|---------|
| **H₀ (Null)** | Default assumption — "no difference" |
| **H₁ (Alternative)** | Claim to prove — "there is a difference" |
| **α (significance level)** | Threshold — usually 0.05 |
| **p-value** | Probability of observing results if H₀ is true |

**Test statistic (Z-test):**
```
Z = (x̄ - μ₀) / (σ / √n)
```

**Decision rule:**
```
p-value < α (0.05)  →  Reject H₀
p-value ≥ α         →  Fail to reject H₀
```

---

### 3. Central Limit Theorem (CLT)
As sample size increases, the **distribution of sample means approaches normal** — regardless of the original distribution.

```
X̄ ~ N(μ, σ/√n)
```

> This is why we can apply normal-based tests even on skewed data (income, purchases, etc.) — as long as n is large enough (usually n ≥ 30)

---

## Errors in Hypothesis Testing

| Error | Description | Probability |
|-------|-------------|-------------|
| **Type I (False Positive)** | Reject H₀ when it's actually true | α |
| **Type II (False Negative)** | Fail to reject H₀ when it's false | β |
| **Power of test** | Probability of correctly rejecting false H₀ | 1 - β |

---

## Parametric vs Non-Parametric Tests

| | Parametric | Non-Parametric |
|---|---|---|
| **Assumes distribution?** | ✅ Yes (usually normal) | ❌ No |
| **Data type** | Continuous | Categorical, ranked, or non-normal |
| **Examples** | Z-test, T-test, ANOVA | Chi-square, Mann-Whitney, Kruskal-Wallis |
| **When to use** | Large samples, normal data | Small samples, skewed or categorical data |

---

## Worked Example — Delivery Algorithm Test

**Setup:** 100 orders split — 50 with new algorithm, 50 with current system

1. **H₀:** New algorithm does not reduce delivery time
2. **H₁:** New algorithm reduces delivery time
3. **α = 0.05**
4. Run t-test on delivery times
5. If p-value < 0.05 → reject H₀ → new algorithm is better
6. CI of (-5, -2) minutes → deliveries are 2–5 minutes faster

---

## ML Relevance
- **A/B testing** — comparing model versions uses hypothesis testing
- **CLT** — justifies using normal-based tests on large datasets
- **Confidence intervals** — quantify uncertainty in model performance metrics
- **p-values** — used to evaluate feature significance in linear models
