## Problem Setup

We have a diagonal multivariate normal distribution:

$$
p(x \mid \mu, \sigma^2) = \prod_{i=1}^d \frac{1}{\sqrt{2\pi \sigma_i^2}} \exp\left(-\frac{(x_i - \mu_i)^2}{2\sigma_i^2}\right)
$$

where:

- $ \mu \in \mathbb{R}^d $ is the mean vector
- $ \sigma^2 \in \mathbb{R}^d $ is the variance vector
- $ x $ is a sample drawn from the distribution

The goal is to optimize the parameters $ \mu $ and $ \sigma^2 $ to maximize the expected fitness:

$$
J(\mu, \sigma^2) = \mathbb{E}_{x \sim p(x \mid \mu, \sigma^2)}[f(x)]
$$

---

## Gradient Derivation

### 1. Gradient of the Expected Fitness

We use the score function estimator:

$$
\nabla_{\mu, \sigma^2} J(\mu, \sigma^2) = \mathbb{E}_{x \sim p(x \mid \mu, \sigma^2)} \left[ f(x) \nabla_{\mu, \sigma^2} \log p(x \mid \mu, \sigma^2) \right]
$$

### 2. Log Probability Gradient

The log of the diagonal normal distribution is:

$$
\log p(x \mid \mu, \sigma^2) = -\frac{1}{2} \sum_{i=1}^d \left[ \log(2\pi \sigma_i^2) + \frac{(x_i - \mu_i)^2}{\sigma_i^2} \right]
$$

Gradients:

- With respect to $ \mu $:

$$
\frac{\partial \log p(x \mid \mu, \sigma^2)}{\partial \mu_i} = \frac{x_i - \mu_i}{\sigma_i^2}
$$

- With respect to $ \sigma^2 $:

$$
\frac{\partial \log p(x \mid \mu, \sigma^2)}{\partial \sigma_i^2} = -\frac{1}{2\sigma_i^2} + \frac{(x_i - \mu_i)^2}{2\sigma_i^4}
$$

---

### 3. Weighted Gradients

Substituting these into the NES gradient:

- Gradient w.r.t. $ \mu $:

$$
\nabla_{\mu} J(\mu, \sigma^2) = \mathbb{E}_{x \sim p} \left[ f(x) \cdot \frac{x - \mu}{\sigma^2} \right]
$$

- Gradient w.r.t. $ \sigma^2 $:

$$
\nabla_{\sigma^2} J(\mu, \sigma^2) = \mathbb{E}_{x \sim p} \left[ f(x) \cdot \left( -\frac{1}{2\sigma^2} + \frac{(x - \mu)^2}{2\sigma^4} \right) \right]
$$

---

### 4. Monte Carlo Approximation

Using $ n $ samples $ \{x^{(i)}\}_{i=1}^n $:

- For $ \mu $:

$$
\nabla_{\mu} J(\mu, \sigma^2) \approx \frac{1}{n} \sum_{i=1}^n f(x^{(i)}) \cdot \frac{x^{(i)} - \mu}{\sigma^2}
$$

- For $ \sigma^2 $:

$$
\nabla_{\sigma^2} J(\mu, \sigma^2) \approx \frac{1}{n} \sum_{i=1}^n f(x^{(i)}) \cdot \left( -\frac{1}{2\sigma^2} + \frac{(x^{(i)} - \mu)^2}{2\sigma^4} \right)
$$

---

## Final Update Rule

- For $ \mu $:

$$
\mu \leftarrow \mu + \eta \cdot \nabla_{\mu} J(\mu, \sigma^2)
$$

- For $ \sigma^2 $:

$$
\sigma^2 \leftarrow \sigma^2 + \eta \cdot \nabla_{\sigma^2} J(\mu, \sigma^2)
$$

Or with Monte Carlo gradients:

- $ \mu \leftarrow \mu + \eta \cdot \frac{1}{n} \sum f(x^{(i)}) \cdot \frac{x^{(i)} - \mu}{\sigma^2} $
- $ \sigma^2 \leftarrow \sigma^2 + \eta \cdot \frac{1}{n} \sum f(x^{(i)}) \cdot \left( -\frac{1}{2\sigma^2} + \frac{(x^{(i)} - \mu)^2}{2\sigma^4} \right) $

---

## Analytical Fisher Matrix

### Fisher Matrix Derivation

The Fisher matrix quantifies the curvature of the log-likelihood function with respect to the parameters. For the diagonal multivariate normal distribution, the Fisher matrix can be derived analytically.

#### Log Probability Gradient Recap

- Gradient w.r.t. $ \mu $:

$$
\frac{\partial \log p(x \mid \mu, \sigma^2)}{\partial \mu_i} = \frac{x_i - \mu_i}{\sigma_i^2}
$$

- Gradient w.r.t. $ \sigma^2 $:

$$
\frac{\partial \log p(x \mid \mu, \sigma^2)}{\partial \sigma_i^2} = -\frac{1}{2\sigma_i^2} + \frac{(x_i - \mu_i)^2}{2\sigma_i^4}
$$

#### Fisher Matrix Components

The Fisher matrix is computed as the expected value of the outer product of the log probability gradients:

$$
F_{ij} = \mathbb{E}_{x \sim p(x \mid \mu, \sigma^2)} \left[ \frac{\partial \log p(x \mid \mu, \sigma^2)}{\partial \theta_i} \frac{\partial \log p(x \mid \mu, \sigma^2)}{\partial \theta_j} \right]
$$

For the diagonal normal distribution:

1. **Diagonal Terms for $ \mu $**:

$$
F_{\mu_i, \mu_i} = \mathbb{E}_{x \sim p} \left[ \left( \frac{x_i - \mu_i}{\sigma_i^2} \right)^2 \right] = \frac{1}{\sigma_i^2}
$$

2. **Diagonal Terms for $ \sigma^2 $**:

$$
F_{\sigma_i^2, \sigma_i^2} = \mathbb{E}_{x \sim p} \left[ \left( -\frac{1}{2\sigma_i^2} + \frac{(x_i - \mu_i)^2}{2\sigma_i^4} \right)^2 \right] = \frac{1}{2\sigma_i^4}
$$

3. **Cross Terms**:

The cross terms between $ \mu $ and $ \sigma^2 $ vanish due to independence of the gradients.

#### Final Fisher Matrix

The Fisher matrix is block diagonal:

$$
F = \begin{bmatrix}
\text{diag}\left(\frac{1}{\sigma^2}\right) & 0 \\
0 & \text{diag}\left(\frac{1}{2\sigma^4}\right)
\end{bmatrix}
$$

This analytical form can be used for efficient computation of the inverse Fisher matrix.

---

## 🔒 Notes

- **Clamp Variance**: Ensure $ \sigma^2 > 0 $, e.g., $ \sigma^2 = \max(\sigma^2, \epsilon) $
- **Normalize Fitness**: Normalize $ f(x) $ before use
- **Separate Learning Rates**: For $ \mu $ and $ \sigma^2 $, choose different $ \eta $ if needed

---

## Implementation Notes

### Practical Considerations

1. **Clamp Variance**: Ensure $ \sigma^2 > 0 $ to avoid numerical instability. For example:
   $$ \sigma^2 = \max(\sigma^2, \epsilon) $$
   where $ \epsilon $ is a small positive constant.

2. **Normalize Fitness**: Normalize $ f(x) $ to have zero mean and unit variance before using it in the gradient calculations. This helps stabilize the optimization process.

3. **Separate Learning Rates**: Use different learning rates $ \eta_\mu $ and $ \eta_{\sigma^2} $ for $ \mu $ and $ \sigma^2$ updates, as they may require different scales.

4. **Monte Carlo Sampling**: Ensure sufficient samples $ n $ are drawn to approximate the gradients accurately. Larger $ n $ reduces variance but increases computational cost.

5. **Gradient Clipping**: Apply gradient clipping to prevent excessively large updates, which can destabilize training.

6. **Batch Updates**: Perform updates in mini-batches to improve computational efficiency and reduce noise in gradient estimates.

---

# Summary: Issues with NES and Review of Implementations

## 1. Core Issues in Vanilla NES

### 1.1 Premature Variance Collapse

* NES reduces variance based on fitness-weighted gradients.
* If the global optimum lies far outside the initial support, NES may **collapse the variance** before the mean can approach the optimum.
* This leads to convergence on suboptimal regions, especially on **convex or ridge-like functions**.

### 1.2 Premature Variance Decay Explained

* When high-fitness samples appear close to the current mean, NES updates may favor shrinking variance to exploit the local region.
* This causes **exploration capacity to collapse** prematurely.
* Once variance is too small, future samples can no longer reach the global optimum, even if it lies nearby.
* This behavior is especially problematic in **multimodal landscapes** or when initialization is far from the optimum.
* NES reduces variance based on fitness-weighted gradients.
* If the global optimum lies far outside the initial support, NES may **collapse the variance** before the mean can approach the optimum.
* This leads to convergence on suboptimal regions, especially on **convex or ridge-like functions**.

### 1.2 Empirical Fisher Approximation

* Vanilla NES estimates the Fisher Information Matrix (FIM) from sampled data.
* The empirical FIM is **noisy, biased**, and sensitive to sample quality.
* Inversion may be unstable or expensive, especially in high dimensions.

### 1.3 Additive Updates

* Variance/covariance updates are typically additive.
* Additive updates do not preserve positive-definiteness and can lead to instability.

## 2. Analytical Fisher Advantage

* For diagonal Gaussians, NES can compute the **analytical Fisher matrix**:

  * $F_{\mu_i} = 1/\sigma_i^2$
  * $F_{\sigma_i^2} = 1/(2\sigma_i^4)$
* Inversion becomes trivial:

  * $F^{-1}_{\mu_i} = \sigma_i^2$
  * $F^{-1}_{\sigma_i^2} = 2\sigma_i^4$
* Avoids sample noise and improves stability.

## 3. Review of Provided Implementations

### 3.1 `DiagNormal`

* Diagonal Gaussian with analytical Fisher and natural gradient update.
* Efficient and robust, avoids empirical Fisher completely.
* Limitation: Cannot represent covariances or elongated search directions.

### 3.2 `Normal`

* Full-covariance Gaussian using Cholesky parameterization.
* Estimates the **empirical Fisher** from samples.
* Applies **additive** updates to mean and Cholesky factor `A`.
* Advantages:

  * Supports full covariance modeling.
  * Natural gradient step via solving `F @ x = grad` using Cholesky solve.
* Limitations:

  * Not an xNES implementation (no exponential map).
  * Still subject to instability from sample-based Fisher estimation.

## 4. What Makes xNES Different?

### 4.1 Natural Coordinates

* Reparameterizes distribution into $\mu, \sigma, B$ where $B$ is shape matrix.
* Fisher becomes **identity** in these coordinates.
* No need to estimate or invert Fisher.

### 4.2 Exponential Update of Covariance

* Updates $B$ and $\sigma$ via matrix exponential:

  * $B \leftarrow B \cdot \exp(\eta_B \cdot G_B/2)$
  * $\sigma \leftarrow \sigma \cdot \exp(\eta_\sigma \cdot G_\sigma/2)$
* Maintains positive-definiteness and proportional scaling.

### 4.3 Robust Exploration

* Less prone to premature convergence.
* More invariant to transformations of the search space.

## 5. Recommendations

* `DiagNormal` is efficient for separable problems and small dimensions.
* `Normal` can be enhanced by switching to **exponential updates** and **analytical Fisher** if possible.
* Consider full transition to **xNES** for stability, robustness, and simplicity of natural gradients.

---

Feel free to request an implementation template or test suite for validating your future xNES variant.
