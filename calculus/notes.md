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
