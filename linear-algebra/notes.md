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
---
# Matrices and Matrix Arithmetic for ML

## What is a Matrix?
A 2D array of numbers arranged in rows and columns.

- Element notation: `a[i][j]` — i = row, j = column
- Order: `m x n` — m rows, n columns

```python
import numpy as np
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])  # Order: 3x3
```

---

## Matrix Operations

### 1. Addition
- Same order required
- Corresponding elements are added

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
C = A + B  # [[6, 8], [10, 12]]
```

### 2. Subtraction
- Same order required
- Corresponding elements are subtracted

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[0, 1], [2, 1]])
C = A - B  # [[1, 1], [1, 3]]
```

### 3. Division
- Same order required
- Element-wise division (`/` for float, `//` for integer)

```python
A = np.array([[4, 2], [6, 8]])
B = np.array([[2, 2], [3, 4]])
C = A // B  # [[2, 1], [2, 2]]
```

### 4. Matrix Multiplication (Dot Product)
- Columns of A must equal rows of B
- ⚠️ **Not commutative** — `AB ≠ BA`
- Each element = dot product of row from A × column from B

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
C = A.dot(B)  # [[19, 22], [43, 50]]
```

> `C[0][0] = 1×5 + 2×7 = 19`

### 5. Vector Multiplication
- Columns of matrix must equal rows of vector
- Matrix × column vector → column vector

```python
A = np.array([[1, 2], [1, 1]])
V = np.array([[1], [1]])
C = A.dot(V)  # [[3], [2]]
```

### 6. Scalar Multiplication
- Scalar multiplied with every element
- Order remains the same

```python
A = np.array([[1, 2], [3, 4]])
C = A * 2  # [[2, 4], [6, 8]]
```

---

## Quick Reference

| Operation | Requirement | Commutative? |
|-----------|-------------|--------------|
| Addition | Same order | ✅ Yes |
| Subtraction | Same order | ❌ No |
| Division | Same order | ❌ No |
| Matrix Multiplication | Cols of A = Rows of B | ❌ No |
| Scalar Multiplication | None | ✅ Yes |
