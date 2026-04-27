# Basic Probability: Sample Space & Types of Events

## Sample Space
The **set of all possible outcomes** of a random experiment.

### Types of Sample Spaces

| Type | Description | Example |
|------|-------------|---------|
| **Finite** | Countable, limited outcomes | Die roll: S = {1,2,3,4,5,6} |
| **Infinite** | Outcomes go on forever | Coin tosses until first head: S = {1,2,3,...} |
| **Continuous** | Any real value in a range | S = {x ∈ R \| 0 ≤ x ≤ 1} |

### Common Sample Spaces

| Experiment | Sample Space | Size |
|------------|-------------|------|
| 1 coin toss | {H, T} | 2 |
| 2 coin tosses | {HH, HT, TH, TT} | 4 |
| 3 coin tosses | {HHH, HHT, HTH,...} | 8 (2³) |
| 1 die roll | {1,2,3,4,5,6} | 6 |
| 2 dice rolls | {(1,1),(1,2),...,(6,6)} | 36 (6²) |

> **Rule:** n coins → 2ⁿ outcomes, n dice → 6ⁿ outcomes

---

## Types of Events

| Event Type | Definition | Example |
|------------|-----------|---------|
| **Impossible** | P = 0, empty set ∅ | Multiple of 7 on a die |
| **Sure/Certain** | P = 1, entire sample space | Number < 7 on a die |
| **Simple** | Single outcome | E = {1} from die roll |
| **Compound** | Multiple outcomes | E = {3,4,5} from die roll |
| **Independent** | Previous outcome doesn't affect next | Rolling a die repeatedly |
| **Dependent** | Previous outcome affects next | Drawing without replacement |
| **Equally Likely** | All outcomes have same probability | Any face of a fair die (1/6) |
| **Mutually Exclusive** | Cannot occur at same time, A∩B = ∅ | Getting 2 AND 5 on same die roll |
| **Exhaustive** | Together cover all outcomes | {Head, Tail} for a coin toss |

---

## Key Distinctions

### Independent vs Dependent
```
Independent: P(A after B) = P(A)
Dependent:   P(A after B) ≠ P(A)
```

**Dependent example — drawing without replacement:**
```
Bag: 4 black, 3 red balls
P(black on 1st draw) = 4/7 = 0.571
P(black on 2nd draw | black on 1st) = 3/6 = 0.5   ← changed!
```

**Independent example — rolling a die:**
```
P(even) = 0.5 → always 0.5, regardless of previous rolls
```

### Mutually Exclusive vs Exhaustive
```
Mutually Exclusive:  A ∩ B = ∅   (can't both happen)
Exhaustive:          A ∪ B = S   (one must happen)
```
> Head and Tail on a coin toss are **both** mutually exclusive AND exhaustive

---

## Worked Examples

**Two coins, A = at least one head, B = no head:**
```
S = {HH, HT, TH, TT}
A = {HH, HT, TH}
B = {TT}
A ∩ B = ∅  →  Mutually exclusive ✅
```

**Cards: A = King, B = Red card:**
```
A ∩ B = {K♥, K♦}                    → 2 outcomes
A ∪ B = 26 red cards + 2 black kings → 28 outcomes
```

---

## ML Relevance
- **Sample space** = foundation of all probability calculations in ML
- **Independent events** → basis of Naive Bayes (assumes feature independence)
- **Mutually exclusive** → used in multi-class classification (one class at a time)
- **Dependent events** → Markov chains, Hidden Markov Models, RNNs
---
 
