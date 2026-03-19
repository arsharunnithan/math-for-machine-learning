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
---
# Scalar Product of Vectors (Dot Product)

## Definition
The scalar product of two vectors A and B:

```
A · B = |A| |B| cos θ
```

- Result is a **scalar** (not a vector)
- θ = angle between the two vectors

---

## Formula (Component Form)

```
A · B = Ax·Bx + Ay·By + Az·Bz
```

Where:
- A = Ax·i + Ay·j + Az·k
- B = Bx·i + By·j + Bz·k

---

## Matrix Representation

Treat A as a **row matrix** (transpose) and B as a **column matrix**:

```
[Ax  Ay  Az] · [Bx]  =  Ax·Bx + Ay·By + Az·Bz
               [By]
               [Bz]
```

---

## Sign Based on Angle

| Angle (θ) | cos θ | Dot Product |
|-----------|-------|-------------|
| 0° < θ < 90° | Positive | Positive |
| θ = 90° | 0 | **Zero** |
| 90° < θ < 180° | Negative | Negative |

---

## Special Cases

| Case | Angle | Result |
|------|-------|--------|
| Parallel vectors | 0° | `A · B = \|A\| \|B\|` |
| Antiparallel vectors | 180° | `A · B = -\|A\| \|B\|` |
| Orthogonal vectors | 90° | `A · B = 0` |

---

## Properties

| Property | Formula |
|----------|---------|
| Commutative | `a · b = b · a` |
| Distributive | `a · (b + c) = a·b + a·c` |
| Associative (scalar) | `c(da) = (cd)a` |
| Scalar consistency | `(ua) · (vb) = uv(a · b)` |
| Orthogonality | `u · v = 0` → vectors are perpendicular |

---

## Applications

- **Projection of a onto b** → `(a · b) / |b|`
- **Angle between two vectors** → `cos θ = (a · b) / (|a| |b|)`
- **Scalar Triple Product** → `a · (b × c) = b · (c × a) = c · (a × b)`

---

## ML Relevance
- Dot product measures **similarity** between vectors
- Used in **cosine similarity**, **neural network activations**, **attention mechanisms**
---
# Euclidean Distance & Manhattan Distance

## Euclidean Distance

The shortest straight-line distance between two points — like a bird flying directly from one spot to another. Based on the **Pythagorean theorem**.

### Formulas

**2D:**
```
d = √[(x2 - x1)² + (y2 - y1)²]
```

**3D:**
```
d = √[(x2 - x1)² + (y2 - y1)² + (z2 - z1)²]
```

**n-Dimensions:**
```
d = √[Σ(x2i - x1i)²]   for i = 1 to n
```

### Derivation (2D)
- Draw a right-angled triangle with AB as hypotenuse
- Apply Pythagorean theorem: `d² = (x2 - x1)² + (y2 - y1)²`
- Take square root → Euclidean distance formula

---

## Euclidean vs Manhattan Distance

| Aspect | Euclidean Distance | Manhattan Distance |
|--------|-------------------|-------------------|
| **Definition** | Shortest straight-line distance | Distance along axes at right angles |
| **Formula (2D)** | `√[(x2-x1)² + (y2-y1)²]` | `\|x2-x1\| + \|y2-y1\|` |
| **Path** | Direct straight line | City blocks / grid pattern |
| **Metric Name** | L2 norm | L1 norm |
| **Use Cases** | Physics, direct distance problems | Urban planning, optimization algorithms |
| **Sensitivity to Scaling** | Less sensitive | More sensitive (adds absolute differences) |

---

## ML Relevance
- **K-Means & KNN** — use Euclidean distance to find nearest points/clusters
- **Recommendation systems** — Manhattan distance used in certain optimization setups
- Choice of distance metric directly affects model performance
