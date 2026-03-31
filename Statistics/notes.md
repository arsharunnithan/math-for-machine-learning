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
---
# Covariance and Correlation

## Covariance
Measures how two variables **change together** — direction of relationship.

### Formulas

**Sample Covariance** (use n-1, Bessel's correction):
```
Cov(X,Y) = Σ(Xᵢ - X̄)(Yᵢ - Ȳ) / (n-1)
```

**Population Covariance** (use n):
```
Cov(X,Y) = Σ(Xᵢ - μX)(Yᵢ - μY) / n
```

### Types

| Type | Meaning |
|------|---------|
| **Positive** | Both variables increase together |
| **Negative** | One increases, other decreases |
| **Zero** | No linear relationship |

> Range: **-∞ to +∞** — hard to interpret magnitude

---

## Correlation
Standardized version of covariance — measures **direction AND strength** of relationship.

```
Corr(X,Y) = Σ(xᵢ - x̄)(yᵢ - ȳ) / √[Σ(xᵢ-x̄)² · Σ(yᵢ-ȳ)²]
```

> Equivalently: `Corr(X,Y) = Cov(X,Y) / (σX · σY)`

### Interpretation

| Value | Meaning |
|-------|---------|
| Close to **+1** | Strong positive relationship |
| Close to **-1** | Strong negative relationship |
| **0** | No linear relationship |

> Range: **-1 to +1** — always interpretable

---

## Covariance vs Correlation

| Aspect | Covariance | Correlation |
|--------|-----------|-------------|
| Range | -∞ to +∞ | -1 to +1 |
| Tells you | Direction only | Direction + Strength |
| Scale dependent? | ✅ Yes | ❌ No (dimensionless) |
| Comparable across datasets? | ❌ No | ✅ Yes |

---

## Applications

| Domain | Covariance | Correlation |
|--------|-----------|-------------|
| Finance | Portfolio risk (how stocks move together) | — |
| ML | PCA (covariance matrix of features) | Feature selection |
| Medical | — | Blood pressure vs cholesterol |
| Weather | — | Temperature vs humidity |

---

## ML Relevance
- **PCA** is built on the **covariance matrix** of features
- **Feature selection** — drop highly correlated features to reduce redundancy
- **Multicollinearity** in regression — high correlation between predictors causes instability
- Correlation ≠ causation — a key principle when interpreting model features
---
# Confidence Intervals

## What is it?
A range of values that likely contains the **true population parameter**.

> Instead of "average height is 165 cm" → "we are 95% confident the average height is between 160–170 cm"

---

## Confidence Levels

| Level | Meaning |
|-------|---------|
| 90% | 90 out of 100 intervals contain the true value |
| **95%** | 95 out of 100 — most commonly used |
| 99% | 99 out of 100 — more conservative, wider interval |

```
Confidence Level = 1 - α
```
> α = significance level (0.05 for 95% CI)

---

## Formula

```
CI = Point Estimate ± Margin of Error
   = x̄ ± (Critical Value × Standard Error)
```

**Standard Error:**
```
SE = Standard Deviation / √n
```

---

## When to Use t vs Z Distribution

| | t-distribution | Z-distribution |
|---|---|---|
| Sample size | Small (n < 30) | Large (n > 30) |
| Population std dev | Unknown | Known |
| Critical value (95%) | From t-table (varies with df) | 1.96 |

### Common Z-values

| Confidence Level | Z-value |
|-----------------|---------|
| 90% | 1.645 |
| 95% | 1.960 |
| 99% | 2.576 |

---

## Worked Example — t-distribution

n=10, mean=240 kg, std=25 kg, 95% CI

```
df = n - 1 = 9
α = 0.025
t-value (df=9, α=0.025) = 2.262

CI = 240 ± 2.262 × (25/√10)
   = (222.12, 257.88)
```

```python
import scipy.stats as stats
import math

mean, std, n = 240, 25, 10
t = stats.t.ppf(0.975, df=n-1)
moe = t * (std / math.sqrt(n))
print(f"CI: ({mean-moe:.2f}, {mean+moe:.2f})")
# CI: (222.12, 257.88)
```

## Worked Example — Z-distribution

n=50, mean=4.63, std=0.54, 95% CI

```
SE = 0.54 / √50 = 0.0764
MOE = 1.96 × 0.0764 = 0.1497

CI = (4.480, 4.780)
```

```python
from scipy import stats
import numpy as np

mean, std_dev, n = 4.63, 0.54, 50
se = std_dev / np.sqrt(n)
moe = 1.960 * se
print(f"CI: ({mean-moe:.3f}, {mean+moe:.3f})")
# CI: (4.480, 4.780)
```

---

## Types of Confidence Intervals

| Type | When to Use |
|------|------------|
| **Mean (normal data)** | t-dist (n<30) or Z-dist (n≥30) |
| **Proportions** | % of population with a trait |
| **Non-normal data** | Bootstrap resampling method |

---

## ML Relevance
- **A/B testing** — CI tells you if a model improvement is statistically meaningful
- **Model evaluation** — report accuracy with a CI, not just a single number
- **Survey/data collection** — determine sample size needed for a given CI width
- Wider CI = more uncertainty; narrower CI = more precision (needs larger n)
---
# Hypothesis Testing

## What is it?
Comparing two opposing claims about a population using sample data to decide which is more likely true.

---

## Key Terms

| Term | Meaning |
|------|---------|
| **H₀ (Null Hypothesis)** | Default assumption — "no effect / no difference" |
| **H₁ (Alternative Hypothesis)** | The claim to prove — "there is an effect" |
| **α (Significance Level)** | Threshold for rejecting H₀ — usually 0.05 |
| **p-value** | Probability of observing the data if H₀ is true |
| **Test Statistic** | Number measuring how far data deviates from H₀ |
| **Critical Value** | Cutoff to compare test statistic against |
| **Degrees of Freedom** | Depends on sample size — used to find critical value |

---

## Types of Tests

| Type | When to Use | Example |
|------|-------------|---------|
| **One-tailed (left)** | Expect decrease only | H₁: μ < 50 |
| **One-tailed (right)** | Expect increase only | H₁: μ > 50 |
| **Two-tailed** | Expect change in either direction | H₁: μ ≠ 50 |

---

## Type I & Type II Errors

| | H₀ is True | H₀ is False |
|---|---|---|
| **Fail to Reject H₀** | ✅ Correct | ❌ Type II Error (β) — False Negative |
| **Reject H₀** | ❌ Type I Error (α) — False Positive | ✅ Correct |

> **Power of test** = 1 - β (probability of correctly rejecting a false H₀)

---

## Steps in Hypothesis Testing

1. **Define hypotheses** — H₀ and H₁
2. **Choose significance level** — usually α = 0.05
3. **Collect and analyze data**
4. **Calculate test statistic**
5. **Make decision** — compare p-value to α or test statistic to critical value
6. **Interpret results**

**Decision rules:**
```
p-value ≤ α   →  Reject H₀
p-value > α   →  Fail to reject H₀
```

---

## Choosing the Right Test

| Test | When to Use |
|------|-------------|
| **Z-test** | Large sample (n≥30), population variance known |
| **T-test** | Small sample or unknown population variance |
| **Chi-square** | Categorical data — observed vs expected counts |
| **ANOVA / F-test** | Comparing means across 3+ groups |

---

## Worked Example — Drug Trial (Paired T-test)

**H₀:** Drug has no effect on blood pressure
**H₁:** Drug has an effect

```python
import numpy as np
from scipy import stats

before = np.array([120, 122, 118, 130, 125, 128, 115, 121, 123, 119])
after  = np.array([115, 120, 112, 128, 122, 125, 110, 117, 119, 114])

t_stat, p_val = stats.ttest_rel(after, before)
print(f"T: {t_stat:.2f}")       # T: -9.0
print(f"P: {p_val:.8f}")        # P: 0.00000854
```

> T = -9.0, p ≈ 0.0000085 → p < 0.05 → **Reject H₀** → drug significantly lowers blood pressure

**T-statistic formula:**
```
t = m / (s / √n)     where m = mean of differences, s = std, n = sample size
```

---

## Limitations

- p < 0.05 doesn't mean the effect is large — just statistically significant
- Sensitive to data quality
- Focuses on one specific claim — can miss broader patterns

## ML Relevance
- **Feature selection** — test if a feature is significantly correlated with target
- **A/B testing** — compare model versions statistically
- **Model evaluation** — check if accuracy improvement is significant or by chance
- **Linear regression** — p-values of coefficients are hypothesis tests
---
# P-Value

## What is it?
The probability of observing results **as extreme as the ones obtained**, assuming the null hypothesis is true.

> "If nothing unusual is happening, how surprising are these results?"

- **Small p-value** → results unlikely by chance → strong evidence against H₀
- **Large p-value** → results consistent with random variation → insufficient evidence to reject H₀

---

## Decision Rule

```
p-value ≤ α  →  Reject H₀       (result is statistically significant)
p-value > α  →  Fail to reject H₀
```

> α is usually **0.05** (5% significance level)

---

## How to Calculate P-Value (Steps)

1. State H₀ and H₁
2. Choose the appropriate statistical test
3. Calculate the test statistic
4. Determine the sampling distribution (t, Z, chi-square, F)
5. Find area in the tail(s) of the distribution = p-value
6. Compare p-value to α and decide

---

## Statistical Tests

| Test | When to Use |
|------|-------------|
| **Z-test** | Large sample, known population variance |
| **T-test** | Small sample or unknown variance |
| **Chi-square** | Categorical data — observed vs expected |
| **F-test** | Comparing variances or 3+ group means |
| **Correlation test** | Testing linear relationship between variables |

---

## Worked Example — Two-Sample T-test

Males: n=30, mean=175, std=5
Females: n=35, mean=168, std=6

**H₀:** No difference in mean height
**H₁:** Significant difference exists

```
t = (175 - 168) / √(5²/30 + 6²/35) = 7 / √1.8619 ≈ 5.13
df = (30 + 35) - 2 = 63
```

```python
import scipy.stats as stats

t_statistic = 5.13
df = 63

p_value = 2 * (1 - stats.t.cdf(abs(t_statistic), df))
print("P value:", p_value)  # 2.99e-06
```

> p ≈ 0.000003 << 0.05 → **Reject H₀** → significant height difference exists

---

## One-Sample T-test (Python)

```python
import numpy as np
import scipy.stats as stats

sample_data = [78, 82, 88, 95, 79, 92, 85, 88, 75, 80]
population_mean = 85

t_stat, p_value = stats.ttest_1samp(sample_data, population_mean)
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.4f}")
# p > 0.05 → fail to reject H₀
```

---

## What Influences the P-Value?

| Factor | Effect |
|--------|--------|
| **Larger sample size** | Smaller p-value (easier to detect significance) |
| **Larger effect size** | Smaller p-value |
| **Higher data variability** | Larger p-value |
| **Lower α** | Higher bar for significance |
| **Choice of test** | Different tests → different p-values |

---

## Common Misunderstandings

| ❌ Wrong | ✅ Correct |
|---------|----------|
| p < 0.05 means H₀ is false | p < 0.05 means data is unlikely under H₀ |
| p-value = probability H₀ is true | p-value = probability of data given H₀ is true |
| Small p = large effect | Small p ≠ practically significant |

---

## ML Relevance
- **Feature selection** — features with p < 0.05 are statistically significant predictors
- **Linear regression** — each coefficient has a p-value showing its significance
- **A/B testing** — p-value determines if one model/version is truly better
- **Model comparison** — check if performance improvement is statistically significant

---
