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
Shortest straight-line distance between two points — like a bird flying directly. Based on the **Pythagorean theorem**.

### Formulas

| Dimensions | Formula |
|------------|---------|
| 2D | `d = √[(x2-x1)² + (y2-y1)²]` |
| 3D | `d = √[(x2-x1)² + (y2-y1)² + (z2-z1)²]` |
| nD | `d = √[Σ(x2i - x1i)²]` |

### Derivation (2D)
Draw a right-angled triangle with AB as hypotenuse:
```
d² = (x2 - x1)² + (y2 - y1)²
d  = √[(x2 - x1)² + (y2 - y1)²]
```

---

## Manhattan Distance
Also known as **L1** or **taxicab distance**. Measures distance along grid-like paths — like a taxi navigating city streets.

### Formulas

| Dimensions | Formula |
|------------|---------|
| 2D | `d = \|x1-x2\| + \|y1-y2\|` |
| 3D | `d = \|x1-x2\| + \|y1-y2\| + \|z1-z2\|` |
| nD | `d = Σ\|xi - yi\|` |

### Worked Examples

| Points | Dimensions | Calculation | Result |
|--------|------------|-------------|--------|
| P=(1,2), Q=(4,0) | 2D | \|1-4\| + \|2-0\| | **5** |
| P=(1,2,3), Q=(4,0,1) | 3D | 3+2+2 | **7** |
| P=(1,2,3,4), Q=(4,0,1,2) | 4D | 3+2+2+2 | **9** |

### Python

```python
# NumPy
import numpy as np
a = np.array([2, 3, 5])
b = np.array([7, 1, 9])
d = np.sum(np.abs(a - b))  # 11

# SciPy
from scipy.spatial.distance import cityblock
d = cityblock((2,3,5), (7,1,9))  # 11
```

---

## Euclidean vs Manhattan — Comparison

| Aspect | Euclidean | Manhattan |
|--------|-----------|-----------|
| **Definition** | Shortest straight-line distance | Distance along axes at right angles |
| **Formula (2D)** | `√[(x2-x1)² + (y2-y1)²]` | `\|x2-x1\| + \|y2-y1\|` |
| **Path** | Direct straight line | City blocks / grid pattern |
| **Metric Name** | L2 norm | L1 norm |
| **Sensitivity to Scaling** | Less sensitive | More sensitive |
| **Use Cases** | Physics, open-space distances | Urban planning, optimization |

> ⚠️ **Euclidean ≤ Manhattan always** — straight line is always shorter than grid path.
> Example: P=(1,2), Q=(4,0) → Manhattan = **5**, Euclidean ≈ **3.61**

---

## When to Use Which?

| Use Manhattan when... | Use Euclidean when... |
|----------------------|----------------------|
| Movement is grid-restricted | Open space / physical distances |
| Diagonal movement not allowed | Diagonal movement allowed |
| High-dimensional ML data | Continuous data |
| Discrete or ordinal data | Straight-line reflects reality |

---

## Applications

| Algorithm/Domain | Distance Used | Why |
|-----------------|--------------|-----|
| KNN, K-Means | Euclidean | Finds nearest points in space |
| A* Pathfinding | Manhattan | Grid-based routing heuristic |
| Text / Document Clustering | Manhattan | Works well with sparse/high-dim data |
| Fraud / Anomaly Detection | Manhattan | Less sensitive to extreme values |
| GIS / Urban Planning | Manhattan | Models street grid movement |
| Computer Vision | Both | Comparing pixel/feature vectors |
---
# Eigenvalues & Eigenvectors

## Core Equation
```
Av = λv
```
- **A** — square matrix
- **v** — eigenvector (direction unchanged under transformation)
- **λ** — eigenvalue (how much the vector is stretched/compressed)

> If λ is negative, the direction is **reversed**. If λ = 0, the vector is squished to zero.

---

## How to Find Eigenvalues & Eigenvectors

**Step 1** — Solve for eigenvalues using:
```
det(A - λI) = 0
```

**Step 2** — For each eigenvalue λ, solve for eigenvector v using:
```
(A - λI)v = 0
```

---

## Worked Example — 2×2 Matrix

Matrix A = `[[1, 2], [5, 4]]`

**Finding eigenvalues:**
```
det(A - λI) = 0
(1-λ)(4-λ) - (2×5) = 0
λ² - 5λ - 6 = 0
(λ-6)(λ+1) = 0
→ λ = 6 and λ = -1
```

**Eigenvector for λ = 6:**
```
(A - 6I)v = 0  →  5a = 2b  →  v = [2, 5]
```

**Eigenvector for λ = -1:**
```
(A + I)v = 0  →  a = -b  →  v = [1, -1]
```

---

## Worked Example — 3×3 Matrix

Matrix A = all 2s (3×3)

**Eigenvalues:** λ = 6, λ = 0, λ = 0

**Eigenvectors:**

| λ | Eigenvector |
|---|-------------|
| 0 | `[-1, 1, 0]` and `[-1, 0, 1]` |
| 6 | `[1, 1, 1]` |

---

## Types of Eigenvectors

| Type | Equation | Shape |
|------|----------|-------|
| Right eigenvector | `AVR = λVR` | Column vector (n×1) |
| Left eigenvector | `VLA = VLλ` | Row vector (1×n) |

---

## Eigenspace
The **set of all eigenvectors** of a matrix — all linearly independent of each other.

For the 3×3 example above, eigenspace = `{[-1,1,0], [-1,0,1], [1,1,1]}`

---

## Matrix Diagonalization
A matrix can be written as:
```
A = X D X⁻¹
```
- **X** — matrix of eigenvectors (columns)
- **D** — diagonal matrix with eigenvalues on the diagonal
- **X⁻¹** — inverse of X

For the 3×3 example:
```
D = [[6,0,0], [0,0,0], [0,0,0]]
X = [[1,1,1], [1,-1,0], [1,0,-1]] (columns = eigenvectors)
```

---

## Applications

| Domain | How Eigenvalues/Vectors are Used |
|--------|----------------------------------|
| **PCA** | Eigenvectors = principal components; eigenvalues = variance explained |
| **Google PageRank** | Eigenvector for λ=1 gives page importance scores |
| **Eigenfaces (CV)** | Eigenvectors of face data used for recognition |
| **NLP (LSA)** | Eigen-decomposition finds word-document relationships |
| **Markov Processes** | Eigenvector for λ=1 gives long-term stable probabilities |
| **Graph/Network Analysis** | Eigenvalues of graph Laplacian detect communities |
| **Signal Processing** | Optimizes channels, noise filtering |
| **Robotics/Control Systems** | Eigenvalues determine system stability |

---

## ML Relevance
- **PCA is built entirely on eigenvectors** — must know this for interviews
- Eigenvalues tell you how much variance each principal component captures
- Dimensionality reduction, noise removal, image compression all rely on this
