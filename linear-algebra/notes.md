# Vectors for ML

## What is a Vector?
An ordered list of numbers where each number represents a feature of a data point.

```
v = (170, 65)  # height (cm), weight (kg)
```

In n-dimensions: `v = (x1, x2, x3, ..., xn)`

```python
import numpy as np
vector = np.array([170, 65])
print("Vector:", vector)
```

---

## Scalars, Vectors and Matrices

| Structure | Description |
|-----------|-------------|
| **Scalar** | A single value (integer or float) |
| **Vector** | 1D array — a linear sequence of numbers |
| **Matrix** | 2D array — multiple vectors in rows and columns |

```python
import numpy as np
mat = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])
print(mat)
```

---

## Vectors in ML Models

- **Input** — Data (images, text, sensor readings) converted to numerical vectors
- **Model** — Operations like matrix multiplication update model parameters
- **Output** — Output vectors used for similarity, clustering, or further predictions

---

## Types of Vectors

| Type | Description | Use Case |
|------|-------------|----------|
| **Row Vector** | `(x1, x2, x3)` — horizontal | Standard input representation |
| **Column Vector** | Vertical arrangement of elements | Linear algebra operations |
| **Zero Vector** | All elements = 0, e.g. `(0,0,0)` | Optimization problems, origin in vector space |
| **Unit Vector** | Magnitude = 1, `u = v / ‖v‖` | Representing direction |
| **Sparse Vector** | Mostly zeros | Text analysis, recommendation systems |
| **Dense Vector** | Mostly non-zero | Image processing, deep learning |

---

## Why Vectors Matter in ML

- **Feature Representation** — Data points expressed as numbers; e.g. words → vectors via Word2Vec, TF-IDF
- **Distance & Similarity** — Euclidean distance (straight-line), Cosine similarity (angle between vectors)
- **Transformations** — Rotation, scaling, projection; used in PCA for dimensionality reduction

---

## Vector Operations

### Addition & Subtraction
```python
a = np.array([2, 3])
b = np.array([1, 4])
print(a + b)  # [3 7]
print(a - b)  # [1 -1]
```

### Scalar Multiplication
```python
a = np.array([1, 2, 3])
print(3 * a)  # [3 6 9]
```

### Dot Product
```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print(np.dot(a, b))  # 32
```
> `1×4 + 2×5 + 3×6 = 32` — measures similarity between vectors

### Cross Product
```python
c = np.array([1, 2, 3])
d = np.array([4, 5, 6])
print(np.cross(c, d))  # [-3  6 -3]
```
> Result is a vector perpendicular to both inputs

---

## Applications in ML Algorithms

| Algorithm | How Vectors are Used |
|-----------|----------------------|
| **Linear Regression** | `Y = Xw + b` — X is feature vector, w is weights vector |
| **SVM** | Vector math to find the best separating hyperplane |
| **Neural Networks** | Weights, activations, gradients all stored as vectors |
| **K-Means Clustering** | Points assigned to clusters based on vector distances |
| **Word2Vec / GloVe** | Words mapped to vectors capturing semantic meaning |
| **BERT / Doc2Vec** | Sentences and documents encoded as vectors |
