# Differential Calculus

## What is it?
Studies how functions change — the **rate of change** at any given point.

---

## Key Concepts

### Limits
Describes behaviour of a function as input approaches a value:
```
lim f(x) = L  as x → c
```

### Continuity
f(x) is continuous at x = a if:
- f(a) exists
- lim f(x) as x → a exists (left and right limits are equal)
- lim f(x) = f(a)

### Derivative (First Principle)
```
f'(x) = lim [f(x+h) - f(x)] / h   as h → 0
```
> Instantaneous rate of change = slope of the tangent line at a point

---

## Notation

| Notation | Form |
|----------|------|
| Leibniz | `dy/dx` |
| Lagrange | `f'(x)` |
| Newton | `ẏ` |
| Euler | `Df(x)` |

---

## Differentiation Rules

| Rule | Formula |
|------|---------|
| **Power** | `d/dx (xⁿ) = nxⁿ⁻¹` |
| **Sum** | `(u + v)' = u' + v'` |
| **Constant Multiple** | `(cf)' = c·f'` |
| **Product** | `(uv)' = u·v' + v·u'` |
| **Quotient** | `(h/g)' = [g·h' - h·g'] / g²` |
| **Chain** | `f(g(x))' = f'(g(x))·g'(x)` |

---

## Common Derivatives

| Function | Derivative |
|----------|------------|
| constant c | 0 |
| xⁿ | nxⁿ⁻¹ |
| eˣ | eˣ |
| aˣ | aˣ ln a |
| ln x | 1/x |
| log_a(x) | 1 / (x ln a) |
| sin x | cos x |
| cos x | -sin x |
| tan x | sec²x |
| sin⁻¹x | 1/√(1-x²) |
| cos⁻¹x | -1/√(1-x²) |
| tan⁻¹x | 1/(1+x²) |

---

## Differentiation Techniques

### Implicit Differentiation
Used when y is not explicitly solved — differentiate both sides w.r.t x, then solve for dy/dx.

**Example:** x² + y² = 1
```
2x + 2y(dy/dx) = 0
dy/dx = -x/y
```

### Logarithmic Differentiation
Useful for functions like y = xˣ — take ln of both sides first.

**Example:** y = xˣ
```
ln(y) = x·ln(x)
(1/y)·dy/dx = ln(x) + 1
dy/dx = xˣ(ln(x) + 1)
```

### Parametric Differentiation
When x = x(t) and y = y(t):
```
dy/dx = (dy/dt) / (dx/dt)
```
**Example:** x = cos(t), y = sin(t)
```
dx/dt = -sin(t),  dy/dt = cos(t)
dy/dx = cos(t) / -sin(t) = -cot(t)
```

---

## Polynomial Differentiation
Apply power rule term by term:
```
P(x) = a₀ + a₁x + a₂x² + ... + aₙxⁿ
P'(x) = a₁ + 2a₂x + ... + naₙxⁿ⁻¹
```

---

## ML Relevance
- **Gradient Descent** — uses derivatives to find the direction of steepest descent
- **Backpropagation** — chain rule applied repeatedly through neural network layers
- **Loss function optimization** — finding minimum via derivatives
- **Activation functions** — sigmoid, ReLU derivatives needed during training
---
# Gradient

## What is it?
Extends the derivative to **multiple dimensions** — a vector of all partial derivatives.

```
∇f = (∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ)
```

Each component = rate of change of f with respect to that variable.

---

## Formulas

| Dimensions | Gradient |
|------------|---------|
| 2D f(x,y) | `∇f = (∂f/∂x, ∂f/∂y)` |
| 3D f(x,y,z) | `∇f = (∂f/∂x, ∂f/∂y, ∂f/∂z)` |
| nD | `∇f = (∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ)` |

---

## Geometric Interpretation

- **Direction** → points toward steepest **ascent**
- **Magnitude ‖∇f‖** → how steep the function is at that point
- **Perpendicular to level curves** — gradient is always at 90° to contour lines f(x,y) = c

> In gradient descent we go **opposite** to the gradient (steepest descent)

---

## Worked Example

f(x, y) = x² + 3y²

**Step 1 — Compute partial derivatives:**
```
∂f/∂x = 2x
∂f/∂y = 6y
∇f = (2x, 6y)
```

**Step 2 — Evaluate at (1, 2):**
```
∇f(1,2) = (2×1, 6×2) = (2, 12)
```
→ Function increases most rapidly in direction (2, 12)

---

## Python

```python
import sympy as sp

x, y = sp.symbols('x y')
f = x**2 + 3*y**2

grad_f = [sp.diff(f, var) for var in (x, y)]
print("Gradient:", grad_f)  # [2*x, 6*y]

# Evaluate at (1, 2)
grad_f_func = [sp.lambdify((x, y), expr) for expr in grad_f]
grad_value = [func(1, 2) for func in grad_f_func]
print("Gradient at (1,2):", grad_value)  # [2, 12]
```

---

## Applications

| Domain | Use |
|--------|-----|
| **Gradient Descent (ML)** | `θ ← θ - α∇f(θ)` — update model parameters |
| **Backpropagation** | Gradient of loss w.r.t. each weight |
| **Edge Detection (CV)** | Sobel filters compute image gradients to find edges |
| **Physics** | E = -∇V (electric field from potential) |
| **Robotics** | Gradient-based path planning |

---
# Higher Order Derivatives

## What are they?
Derivatives obtained by repeatedly differentiating a function.

| Derivative | Notation | Measures |
|------------|----------|---------|
| 1st | `f'(x)` or `dy/dx` | Rate of change / slope |
| 2nd | `f''(x)` or `d²y/dx²` | Curvature / concavity |
| 3rd | `f'''(x)` or `d³y/dx³` | Rate of change of curvature |
| nth | `fⁿ(x)` or `dⁿy/dxⁿ` | nth rate of change |

---

## Second Order Derivative

Differentiate f'(x) again:
```
y = f(x)
dy/dx = f'(x)
d²y/dx² = d/dx[f'(x)] = f''(x)
```

**Example:** y = x / (x² + 1), find y'' at x = 1

```
Step 1 — First derivative (quotient rule):
y' = (1 - x²) / (x² + 1)²

Step 2 — Second derivative at x = 1:
y''(1) = 0
```

---

## Third Order Derivative

Differentiate f''(x) again:
```
d³y/dx³ = d/dx[f''(x)] = f'''(x)
```

**Example:** y = 3x³ + 12x + 4, find y''' at x = 1

```
y'(x)   = 9x² + 12
y''(x)  = 18x
y'''(x) = 18

y'''(1) = 18
```

---

## Higher-Order Derivatives in Parametric Form

When x = x(t) and y = y(t):
```
dy/dx = (dy/dt) / (dx/dt)

d²y/dx² = d/dt[y'(t)/x'(t)] × (1/x'(t))
```

---

## Applications

| Use | How |
|-----|-----|
| **Maxima & Minima** | f'(x) = 0 and f''(x) < 0 → maximum; f''(x) > 0 → minimum |
| **Concavity** | f''(x) > 0 → concave up; f''(x) < 0 → concave down |
| **Acceleration** | 2nd derivative of position = acceleration |
| **Graph Shape** | Inflection points where f''(x) = 0 |

---

## ML Relevance
- **Loss landscape analysis** — 2nd derivative tells if a critical point is a minimum or saddle point
- **Newton's Method** — uses 2nd derivative for faster optimization than gradient descent
- **Hessian Matrix** — matrix of 2nd order partial derivatives, used in advanced optimizers

## ML Relevance
- The gradient of the **loss function** tells us which direction increases the error
- We move **against** the gradient to reduce loss — this is gradient descent
- Every weight update in neural networks uses the gradient
---
# Multivariable Calculus

## What is it?
Extends single-variable calculus to functions with **multiple input variables** — essential for ML since models have thousands of parameters.

```
f(x₁, x₂, ..., xₙ) = y
```

---

## Key Concepts

### Partial Derivatives
Rate of change of f with respect to **one variable**, keeping all others constant.

```
∂f/∂x  →  how f changes as x changes (y fixed)
∂f/∂y  →  how f changes as y changes (x fixed)
```

### Gradient Vector
Vector of all partial derivatives — points in direction of **steepest ascent**:
```
∇f = (∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ)
```
> In ML we follow the **negative gradient** to descend toward the minimum

---

## Gradient Vector Field — Key Observations
For f(x,y) = x² + y²:
- Gradient vectors point **radially outward** from origin
- Vector magnitude **increases** away from origin (steeper further out)
- Level curves (contours) are always **perpendicular** to gradient vectors

---

## Optimization with Constraints

General form:
```
minimize  f₀(x)
subject to  fᵢ(x) ≤ 0    (inequality constraints)
            hⱼ(x) = 0    (equality constraints)
```

**Lagrange Multipliers** — technique to incorporate constraints into optimization by finding stationary points of the Lagrangian.

```python
from scipy.optimize import minimize
import numpy as np

def objective(x):
    return (x[0] - 2)**2 + (x[1] - 3)**2

constraints = [
    {'type': 'ineq', 'fun': lambda x: x[0] - 1},   # x1 >= 1
    {'type': 'ineq', 'fun': lambda x: x[1] - 2},   # x2 >= 2
    {'type': 'eq',   'fun': lambda x: x[0] + x[1] - 4}  # x1 + x2 = 4
]

result = minimize(objective, [0, 0], method='SLSQP',
                  bounds=((1, None), (2, None)),
                  constraints=constraints)

print("Optimal x:", result.x)       # [1.5, 2.5]
print("Min value:", result.fun)     # 0.5
```

---

## Applications in ML

| Application | How Multivariable Calculus is Used |
|-------------|-----------------------------------|
| **Gradient Descent** | Partial derivatives of loss w.r.t. each parameter |
| **Backpropagation** | Chain rule applied across layers to compute gradients |
| **Neural Network Training** | Update weights using ∇Loss at each step |
| **Hessian / 2nd Order Optimization** | Matrix of 2nd partial derivatives for Newton's method |
| **Constrained Optimization** | Lagrange multipliers for regularization (L1, L2) |
| **MLE / Bayesian Inference** | Gradient of likelihood function w.r.t. model parameters |

---

## Optimization Variants

| Method | Gradient Used | Speed | Stability |
|--------|--------------|-------|-----------|
| **Batch Gradient Descent** | Full dataset | Slow | Stable |
| **Stochastic GD (SGD)** | Single sample | Fast | Noisy |
| **Mini-Batch GD** | Subset of data | Balanced | Balanced |
| **Newton's Method** | Gradient + Hessian | Faster convergence | Expensive |
| **Quasi-Newton (BFGS)** | Approximate Hessian | Efficient | Good |

---

## ML Relevance
- Every weight update in a neural network is a multivariable calculus operation
- The loss function is a **scalar function of thousands of variables** (weights)
- Gradient descent only works because of partial derivatives
- Backpropagation = chain rule applied to a multivariable composite function
---
# Chain Rule

## What is it?
Computes the derivative of a **composite function** — a function inside another function.

```
y = f(g(x))
dy/dx = (df/dg) · (dg/dx)
```

> In plain terms: multiply the derivatives of each layer together

---

## Why it Matters in ML
Neural networks are composed of many layers — each a function of the previous.
The chain rule lets us compute **how much each weight contributed to the final loss**, layer by layer → this is **backpropagation**.

---

## Neural Network Example (Step by Step)

**Setup:**
- Input: x = [x1, x2]
- Hidden layer: a1 = σ(W1·x + b1)
- Output: z = σ(W2·a1 + b2)
- Loss: L = ½(z - y)²

### Step 1 — Forward Pass
```
a1 = σ(W1·x + b1)     ← hidden layer activation
z  = σ(W2·a1 + b2)    ← final output
```

### Step 2 — Loss
```
L = ½(z - y)²
```

### Step 3 — Chain Rule (Backpropagation)

**Output layer gradient:**
```
∂L/∂z  = z - y
∂z/∂W2 = z(1-z) · a1ᵀ       ← sigmoid derivative
∂L/∂W2 = ∂L/∂z · ∂z/∂W2    ← chain rule
∂L/∂b2 = ∂L/∂z · z(1-z)
```

### Step 4 — Parameter Update
```
W1 = W1 - α · ∂L/∂W1
b1 = b1 - α · ∂L/∂b1
W2 = W2 - α · ∂L/∂W2
b2 = b2 - α · ∂L/∂b2
```

---

## PyTorch Implementation

```python
import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(2, 2)
        self.output = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        a1 = self.sigmoid(self.hidden(x))
        return self.sigmoid(self.output(a1))

net = SimpleNet()
x = torch.tensor([[0.5, 1.5]], dtype=torch.float32)
target = torch.tensor([[1.0]], dtype=torch.float32)

output = net(x)                          # forward pass
loss = nn.MSELoss()(output, target)      # compute loss
loss.backward()                          # chain rule applied automatically

print(net.hidden.weight.grad)            # gradients via backprop
print(net.output.weight.grad)
```

---

## Applications

| Application | How Chain Rule is Used |
|-------------|----------------------|
| **Backpropagation** | Propagates loss gradient backward through each layer |
| **Gradient Descent** | Computes ∂L/∂w for every weight |
| **RNNs** | Chain rule applied through time steps |
| **CNNs** | Gradients through convolutional layers |
| **AutoDiff (PyTorch/TensorFlow)** | Chain rule automated via computation graph |

---

## Advantages & Limitations

| ✅ Advantages | ⚠️ Limitations |
|--------------|--------------|
| Enables backpropagation at scale | Vanishing / exploding gradients in deep networks |
| Integrated into all major frameworks | Requires differentiable functions |
| Works across all architectures | Computationally expensive for very deep networks |

---

## ML Relevance
- **Backpropagation = chain rule applied repeatedly** from output to input
- Every framework (PyTorch, TensorFlow, JAX) automates this via **autograd**
- Understanding chain rule = understanding how neural networks actually learn
---
# Jacobian & Hessian Matrices

---

## Jacobian Matrix

Matrix of **all first-order partial derivatives** of a vector-valued function.

For f: Rⁿ → Rᵐ (n inputs, m outputs):

```
J(x) = | ∂f₁/∂x₁  ∂f₁/∂x₂  ...  ∂f₁/∂xₙ |
       | ∂f₂/∂x₁  ∂f₂/∂x₂  ...  ∂f₂/∂xₙ |
       |    ⋮          ⋮      ⋱      ⋮    |
       | ∂fₘ/∂x₁  ∂fₘ/∂x₂  ...  ∂fₘ/∂xₙ |
```

- Size: **m × n**
- Generalizes the gradient to vector-valued functions

### Worked Example

f(x,y) = (x⁴ + 3y²x,  5y² - 2xy + 1) at point (1, 2)

**Partial derivatives:**
```
∂f₁/∂x = 4x³ + 3y²    ∂f₁/∂y = 6yx
∂f₂/∂x = -2y           ∂f₂/∂y = 10y - 2x
```

**Jacobian:**
```
Jf(x,y) = | 4x³+3y²   6yx    |
           | -2y        10y-2x |
```

**At (1, 2):**
```
Jf(1,2) = | 16   12 |
           | -4   18 |
```

---

## Hessian Matrix

Matrix of **all second-order partial derivatives** of a scalar function — captures curvature.

For f(x₁, x₂, ..., xₙ):

```
H(f) = | ∂²f/∂x₁²      ∂²f/∂x₁∂x₂  ...  ∂²f/∂x₁∂xₙ |
       | ∂²f/∂x₂∂x₁    ∂²f/∂x₂²    ...  ∂²f/∂x₂∂xₙ |
       |      ⋮               ⋮       ⋱       ⋮       |
       | ∂²f/∂xₙ∂x₁    ∂²f/∂xₙ∂x₂  ...  ∂²f/∂xₙ²   |
```

- Always a **square n × n** matrix
- Symmetric: ∂²f/∂xᵢ∂xⱼ = ∂²f/∂xⱼ∂xᵢ

### Worked Example

f(x,y) = y⁴ + x³ + 3x² + 4y² - 4xy - 5y + 8 at point (1, 0)

**First derivatives:**
```
∂f/∂x = 3x² + 6x - 4y
∂f/∂y = 4y³ + 8y - 4x - 5
```

**Second derivatives:**
```
∂²f/∂x²   = 6x + 6
∂²f/∂y²   = 12y² + 8
∂²f/∂x∂y  = -4
```

**Hessian:**
```
Hf(x,y) = | 6x+6    -4    |
           | -4    12y²+8  |
```

**At (1, 0):**
```
Hf(1,0) = | 12  -4 |
           | -4   8 |
```

---

## Jacobian vs Hessian

| | Jacobian | Hessian |
|---|---|---|
| **Derivative order** | 1st order | 2nd order |
| **Input function** | Vector-valued f: Rⁿ → Rᵐ | Scalar f: Rⁿ → R |
| **Output shape** | m × n matrix | n × n square matrix |
| **Captures** | How outputs change w.r.t. inputs | Curvature of the function |
| **Always symmetric?** | Not necessarily | ✅ Yes |

---

## Applications

### Jacobian
| Use | Description |
|-----|-------------|
| **Backpropagation** | Gradients of vector-valued layer outputs |
| **Feature sensitivity** | How sensitive model output is to each input |
| **Change of variables** | Used in integral transformations |

### Hessian
| Use | Description |
|-----|-------------|
| **Newton's Method** | Uses H to find minima faster than gradient descent |
| **Curvature Analysis** | Positive definite H → local minimum; negative definite → maximum |
| **L-BFGS optimizer** | Approximates Hessian for efficient second-order optimization |
| **Saddle point detection** | Mixed eigenvalues of H → saddle point |

---

## ML Relevance
- **Jacobian** is used in backprop when layer outputs are vectors
- **Hessian** used in second-order optimizers (Newton, L-BFGS) — faster convergence but expensive to compute for large models
- Hessian eigenvalues reveal the loss landscape: flat regions, sharp minima, saddle points
---
# Gradient Descent

## What is it?
An iterative optimization algorithm that minimizes a cost function by moving parameters in the **direction opposite to the gradient**.

> Analogy: Standing on a hill, feeling the slope, and taking steps downhill until you reach the valley.

---

## Core Update Rule

```
w = w - γ · ∂J/∂w
```

- **w** — model parameter (weight)
- **γ (alpha)** — learning rate
- **∂J/∂w** — gradient of the loss w.r.t. w

| Gradient sign | Effect |
|---------------|--------|
| Positive | Subtract → w decreases |
| Negative | Subtract negative → w increases |

---

## Mathematics (Linear Regression Example)

**Loss function (MSE):**
```
J(w, b) = (1/n) Σ(yₚ - y)²    where yₚ = x·w + b
```

**Gradient w.r.t. w:**
```
∂J/∂w = (2/n) Σ(yₚ - y)·x
```

**Parameter update:**
```
w = w - γ · ∂J/∂w
b = b - γ · ∂J/∂b
```

---

## Learning Rate

| Learning Rate | Effect |
|--------------|--------|
| Too small | Tiny steps → very slow convergence |
| Too large | Overshoots minimum → oscillates, diverges |
| Just right | Smooth, fast convergence |

**Solutions for bad learning rates:**
- **Gradient clipping** — cap gradients to a predefined range
- **Batch normalization** — normalize layer inputs to stabilize training
- **Weight initialization** — start weights in appropriate range

---

## How it Works (Steps)

1. Initialize parameters randomly
2. Compute gradient of cost function w.r.t. each parameter
3. Update parameters in the opposite direction of the gradient
4. Repeat until convergence

---

## Python Implementation

```python
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler

diabetes = load_diabetes()
X = diabetes.data[:, [2]]
y = diabetes.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

m, c = 0.0, 0.0
learning_rate = 0.05
iterations = 1000

for i in range(iterations):
    y_pred = m * X_scaled.flatten() + c
    error = y_pred - y

    dm = (2 / len(X_scaled)) * np.dot(error, X_scaled.flatten())
    dc = (2 / len(X_scaled)) * np.sum(error)

    m -= learning_rate * dm
    c -= learning_rate * dc
```

---

## Variants of Gradient Descent

| Variant | Data Used | Speed | Stability |
|---------|-----------|-------|-----------|
| **Batch GD** | Entire dataset | Slow | Stable |
| **Stochastic GD (SGD)** | 1 sample | Fast | Noisy |
| **Mini-Batch GD** | Small batch | Balanced | Balanced |
| **Momentum** | Batch + previous gradient | Faster | Smoother |
| **Adagrad** | Adapts lr per parameter | — | Good for sparse data |
| **RMSprop** | Moving avg of squared gradients | — | Better than Adagrad |
| **Adam** | Momentum + RMSprop combined | Fast | Most widely used |

---

## Advantages & Limitations

| ✅ Advantages | ⚠️ Limitations |
|--------------|--------------|
| Works with any differentiable loss | Sensitive to learning rate choice |
| Scalable to large datasets | Can get stuck in local minima |
| Flexible across model types | Sensitive to weight initialization |
| Converges to global min (convex) | Slow on large datasets (Batch GD) |

---

## ML Relevance
- **Every neural network** is trained using some variant of gradient descent
- **Adam** is the default optimizer in most modern deep learning workflows
- Vanishing/exploding gradients are the main failure modes — understand them
- Learning rate is the most important hyperparameter to tune
---
# Stochastic Gradient Descent (SGD)

## What is it?
A variant of gradient descent that updates parameters using **one sample (or small batch)** at a time instead of the entire dataset.

---

## SGD vs Batch Gradient Descent

| | Batch GD | SGD |
|---|---|---|
| Gradient computed on | Entire dataset | 1 sample at a time |
| Speed per iteration | Slow | Fast |
| Memory usage | High | Low |
| Convergence path | Smooth | Noisy / erratic |
| Local minima escape | ❌ Harder | ✅ Easier (noise helps) |
| Best for | Small datasets | Large datasets |

---

## Update Rule

```
θ = θ - η · ∇J(θ; xᵢ, yᵢ)
```

- **θ** — model parameters
- **η** — learning rate
- **∇J(θ; xᵢ, yᵢ)** — gradient computed on single sample (xᵢ, yᵢ)

---

## Python Implementation from Scratch

```python
import numpy as np

# Generate data: y = 4 + 3X + noise
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

def sgd(X, y, learning_rate=0.1, epochs=1000, batch_size=1):
    m = len(X)
    theta = np.random.randn(2, 1)
    X_bias = np.c_[np.ones((m, 1)), X]   # add bias column
    cost_history = []

    for epoch in range(epochs):
        # Shuffle data each epoch
        indices = np.random.permutation(m)
        X_shuffled = X_bias[indices]
        y_shuffled = y[indices]

        for i in range(0, m, batch_size):
            X_batch = X_shuffled[i:i + batch_size]
            y_batch = y_shuffled[i:i + batch_size]

            gradients = (2 / batch_size) * X_batch.T.dot(
                X_batch.dot(theta) - y_batch)
            theta -= learning_rate * gradients

        cost = np.mean((X_bias.dot(theta) - y) ** 2)
        cost_history.append(cost)

    return theta, cost_history

theta_final, cost_history = sgd(X, y, learning_rate=0.1, epochs=1000)
print(f"Final parameters: {theta_final}")
# θ₀ ≈ 4.35 (intercept), θ₁ ≈ 3.45 (slope)
# Fitted model: y = 4.35 + 3.45·X
```

---

## Advantages & Challenges

| ✅ Advantages | ⚠️ Challenges |
|--------------|--------------|
| Fast — fewer computations per step | Noisy updates, erratic convergence |
| Memory efficient — no full dataset needed | Sensitive to learning rate |
| Can escape local minima / saddle points | May take longer overall to converge |
| Supports online learning (incremental) | Needs learning rate scheduling |

---

## SGD Variants

| Variant | Key Idea |
|---------|---------|
| **SGD** | 1 sample per update |
| **Mini-Batch SGD** | Small batch (e.g. 32, 64 samples) |
| **Momentum SGD** | Adds fraction of previous gradient to dampen oscillation |
| **Adam** | Adaptive learning rate = Momentum + RMSprop |

> **Mini-Batch SGD** is what most people mean when they say "SGD" in practice

---

## Applications

| Domain | Use |
|--------|-----|
| **Deep Learning** | Default optimizer for large neural networks |
| **NLP** | Training Word2Vec, transformers |
| **Computer Vision** | Training CNNs for image classification/detection |
| **Reinforcement Learning** | Optimizing DQN and policy gradient models |
| **Online Learning** | Incremental updates as new data arrives |

---

## ML Relevance
- SGD is the **foundation of all modern optimizers** (Adam, RMSprop, Adagrad)
- The "noise" in SGD is actually useful — helps escape bad local minima
- In practice: use **Adam** for most tasks, **SGD + momentum** for CNNs
- Batch size is a key hyperparameter — larger batch = more stable but slower
