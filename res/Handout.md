# NES - Natural Evolution Strategies

Natural Evolution Strategies are a class of evolutionary optimization strategies (ES) that optimize the parameters of distributions to find optima in a (high dimensionsional) potential landscape. These strategies combine the use of stochastic gradients with further distribution specific properties like fisher information matrix to produce more reliable results. Since the choice of distribution and its parametrization have an impact on the performance of the specific algorithm a comparioson becomes exceedingly interesting.

## Theoretical Background

The general idea of NES is to choose parameters ($\theta_i$) for a more or less problem-specific distribution into a parameter vector

$$
\boldsymbol{\theta} = (\theta_1, \theta_2, \dots, \theta_n)^T
$$

that is able to resolve the potential space. Commonly, this is done by using a distribution centered around $\boldsymbol{0}$ in combination with a mean shift of appropriate dimension $d$:

$$
\boldsymbol{\mu} = (\mu_1, \mu_2, \dots, \mu_d)^T
$$

For a separable Normal distribution in 3 dimensions, we define:

$$
\pi(\mathbf{z} \mid \boldsymbol{\theta}) = \mathcal{N}(\mathbf{z}; \boldsymbol{\mu}, \operatorname{diag}(\boldsymbol{\sigma}^2)) = \prod_{i=1}^3 \frac{1}{\sqrt{2\pi\sigma_i^2}} \exp\left(-\frac{(z_i - \mu_i)^2}{2\sigma_i^2}\right)
$$

This parametrization therefore results in the following free parameters:

$$
\boldsymbol{\theta} = (\mu_1, \mu_2, \mu_3, \sigma_1, \sigma_2, \sigma_3)^T
$$

Using this parameterization, it becomes viable to compute the stochastic gradient in the parameter space based on samples evaluated on the potential surface. Specifically, one simply takes the gradient of the probability density function (PDF) with respect to the parameters:

$$
\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}) \approx \frac{1}{\lambda} \sum_{k=1}^{\lambda} f(\mathbf{z}_k)\, \nabla_{\boldsymbol{\theta}} \log \pi(\mathbf{z}_k \mid \boldsymbol{\theta})
$$

where $\pi(\mathbf{z}_k \mid \boldsymbol{\theta})$ is the PDF at sample $\mathbf{z}_k$, and $f(\mathbf{z}_k)$ is its evaluated fitness (i.e., potential). This is the log-likelihood trick, which enables Monte Carlo–based estimation of the expected gradient.

Specializing this expression for the 3D separable Gaussian case, we obtain the following analytic gradient of the log-likelihood:

$$
\nabla_{\mu_i} \log \pi(\mathbf{z}_k \mid \boldsymbol{\theta}) = \frac{z_{k,i} - \mu_i}{\sigma_i^2}, \quad \nabla_{\sigma_i} \log \pi(\mathbf{z}_k \mid \boldsymbol{\theta}) = \frac{(z_{k,i} - \mu_i)^2}{\sigma_i^3} - \frac{1}{\sigma_i}
$$

Therefore, the full gradient vector for a single sample $\mathbf{z}_k$ is:

$$
\nabla_{\boldsymbol{\theta}} \log \pi(\mathbf{z}_k \mid \boldsymbol{\theta}) =
\begin{bmatrix}
\frac{z_{k,1}-\mu_1}{\sigma_1^2} \\
\frac{z_{k,2}-\mu_2}{\sigma_2^2} \\
\frac{z_{k,3}-\mu_3}{\sigma_3^2} \\
\frac{(z_{k,1}-\mu_1)^2}{\sigma_1^3} - \frac{1}{\sigma_1} \\
\frac{(z_{k,2}-\mu_2)^2}{\sigma_2^3} - \frac{1}{\sigma_2} \\
\frac{(z_{k,3}-\mu_3)^2}{\sigma_3^3} - \frac{1}{\sigma_3}
\end{bmatrix}
$$

This expression can be plugged into the Monte Carlo estimator to compute the stochastic gradient update in NES.

The well understood issue with wimply appying this gradient in  a gradient descent manner is the inverse relation of the mean update and the covariance:

$$
\nabla_{\mu_i} J(\boldsymbol{\theta}) \propto \frac{1}{\sigma_i^2}
$$

The result is that the closer the distribution zeros in on the minimum, reducing variance in the process, the more volatile the mean becomes preventing high accuracy.

To address this fundamental limitation, Natural Evolution Strategies (NES) introduces a principled approach by leveraging the structure of the underlying distribution. Specifically NES introduces the Fisher information matrix as a correction (essentially preconditioner) to the update

$$
\widetilde{\nabla}_{\boldsymbol{\theta}} J(\boldsymbol{\theta}) := \mathbf{F}^{-1}(\boldsymbol{\theta}) \, \nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta})
$$

This ensures invariance under the parametrization and allows for scale invariance between different parameters.

The General Fisher matrix for a distribution $\pi(\mathbf{z} \mid \boldsymbol{\theta})$ is given by:

$$
\mathbf{F}(\boldsymbol{\theta}) = \mathbb{E}_{\mathbf{z} \sim \pi(\cdot \mid \boldsymbol{\theta})} \left[ \nabla_{\boldsymbol{\theta}} \log \pi(\mathbf{z} \mid \boldsymbol{\theta}) \, \nabla_{\boldsymbol{\theta}} \log \pi(\mathbf{z} \mid \boldsymbol{\theta})^T \right]
$$

Generally this is done stchastically using a similar approach as for the stochastic gradient.

For specific cases (multivariate normals) however there exists an acceptable closed form solution provided by the equation:

$$
\mathcal{I}_{m,n} = \frac{\partial \boldsymbol{\mu}^{\mathrm{T}}}{\partial \theta_m}
\boldsymbol{\Sigma}^{-1} \frac{\partial \boldsymbol{\mu}}{\partial \theta_n} + \frac{1}{2} \, \mathrm{tr} \left( \boldsymbol{\Sigma}^{-1} \frac{\partial \boldsymbol{\Sigma}}{\partial \theta_m} \boldsymbol{\Sigma}^{-1} \frac{\partial \boldsymbol{\Sigma}}{\partial \theta_n} \right)
$$

In the case of our separable Gaussian distribution we can attain the Fisher matrix analytically:

$$
\mathbf{F}_{\mu_i, \mu_i} = \frac{1}{\sigma_i^2} \hspace{20px} \mathbf{F}_{\sigma_i, \sigma_i} =
$$

Therefore, the natural gradient corrects the problematic inverse scaling observed in the standard gradient, yielding updates like:

$$
\widetilde{\nabla}_{\mu_i} J(\boldsymbol{\theta}) \propto \sigma_i^2 \cdot \nabla_{\mu_i} J(\boldsymbol{\theta}) \propto f(\mathbf{z}_k) (z_{k,i} - \mu_i)
$$

which are no longer destabilized by small $\sigma_i$.

This allows for the usage of a wide variety of functions to better fit the circumstance.

## Subclasses of NES

Normal Distributions are a common choice for Stochastic gradient methods making up a large part of the implementations. Especially exponential representations are common due to the reduction of complexity in the calculation. Exponential (xNES) and non exponential parametrizations are implemented for this project allowing the comparison between the two strategies based on stability and performance. This group is also split further in separable (SNES) and full representations since advantages, especially from a computational point of view, can vary quite significantly. Heavy Tailed distributions represented by the Students-t class of distributions are also represented as they can allow for avoidance of local minima.

To appropriately assess the performance depending on the following distributions and parametrizations were implemented.

### Separable Multivariate Normal

The separable multivariate normal distribution is characterized by its diagonal covariance matrix, simplifying computations and allowing for closed-form solutions for gradients and Fisher matrices.

#### Parametrization

The distribution is parameterized by:

$$
\boldsymbol{\theta} = (\mu_1, \mu_2, \dots, \mu_d, \sigma_1^2, \sigma_2^2, \dots, \sigma_d^2)^T
$$

where $\mu_i$ represents the mean for dimension $i$, and $\sigma_i^2$ represents the variance for dimension $i$.

#### Gradients

The gradients of the log-likelihood with respect to the parameters are:

$$
\nabla_{\mu_i} \log \pi(\mathbf{z}_k \mid \boldsymbol{\theta}) = \frac{z_{k,i} - \mu_i}{\sigma_i^2}, \quad \nabla_{\sigma_i^2} \log \pi(\mathbf{z}_k \mid \boldsymbol{\theta}) = \frac{(z_{k,i} - \mu_i)^2}{2\sigma_i^4} - \frac{1}{2\sigma_i^2}
$$

#### Fisher Matrix

The Fisher matrix for the separable multivariate normal distribution is given analytically as:

$$
\mathbf{F}_{\mu_i, \mu_i} = \frac{1}{\sigma_i^2}, \quad \mathbf{F}_{\sigma_i^2, \sigma_i^2} = \frac{1}{2\sigma_i^4}
$$

The separable multivariate normal exponential representation uses logarithmic parameters for the covariance, simplifying updates and ensuring positive variances.

#### Parametrization

The distribution is parameterized by:

$$
\boldsymbol{\theta} = (\mu_1, \mu_2, \dots, \mu_d, \ln(\sigma_1), \ln(\sigma_2), \dots, \ln(\sigma_d))^T
$$

where $\mu_i$ represents the mean for dimension $i$, and $\ln(\sigma_i)$ represents the logarithm of the standard deviation for dimension $i$.

#### Gradients

The gradients of the log-likelihood with respect to the parameters are:

$$
\nabla_{\mu_i} \log \pi(\mathbf{z}_k \mid \boldsymbol{\theta}) = \frac{z_{k,i} - \mu_i}{\sigma_i^2}, \quad \nabla_{\ln(\sigma_i)} \log \pi(\mathbf{z}_k \mid \boldsymbol{\theta}) = \frac{(z_{k,i} - \mu_i)^2}{\sigma_i^2} - 1
$$

#### Fisher Matrix

The Fisher matrix for the separable multivariate normal exponential representation is given analytically as:

$$
\mathbf{F}_{\mu_i, \mu_i} = \sigma_i^2, \quad \mathbf{F}_{\ln(\sigma_i), \ln(\sigma_i)} = 2
$$

### Students-T Exponential Representation (SNES)

The Students-T distribution is a heavy-tailed distribution that allows for sampling points further from the mean, making it suitable for escaping local minima.

#### Parametrization

The distribution is parameterized by:

$$
\boldsymbol{\theta} = (\mu_1, \mu_2, \dots, \mu_d, \ln(\sigma_1), \ln(\sigma_2), \dots, \ln(\sigma_d), \ln(\nu))^T
$$

where $\mu_i$ represents the mean for dimension $i$, $\ln(\sigma_i)$ represents the logarithm of the standard deviation for dimension $i$, and $\ln(\nu)$ represents the logarithm of the degrees of freedom.

#### Gradients

The gradients of the log-likelihood with respect to the parameters are:

$$
\nabla_{\mu_i} \log \pi(\mathbf{z}_k \mid \boldsymbol{\theta}) = \frac{(\nu + d) z_i}{(\nu + \|z\|^2) \sigma_i}, \quad \nabla_{\ln(\sigma_i)} \log \pi(\mathbf{z}_k \mid \boldsymbol{\theta}) = \frac{(\nu + d) z_i^2}{\nu + \|z\|^2} - 1
$$

$$
\nabla_{\ln(\nu)} \log \pi(\mathbf{z}_k \mid \boldsymbol{\theta}) = \nu \left( \frac{1}{2} \psi\left(\frac{\nu + d}{2}\right) - \frac{1}{2} \psi\left(\frac{\nu}{2}\right) - \frac{1}{2} \ln\left(1 + \frac{\|z\|^2}{\nu}\right) - \frac{d}{2\nu} + \frac{(\nu + d) \|z\|^2}{2(\nu^2 + \nu \|z\|^2)} \right)
$$

where $z_i = \frac{z_{k,i} - \mu_i}{\sigma_i}$ and $\|z\|^2 = \sum_{i=1}^d z_i^2$.

#### Fisher Matrix

The Fisher matrix for the Students-T distribution is computed numerically due to the complexity of the closed-form solution. However, the inverse Fisher matrix diagonal can be approximated as:

$$
\mathbf{F}_{\mu_i, \mu_i}^{-1} \approx \sigma_i^2, \quad \mathbf{F}_{\ln(\sigma_i), \ln(\sigma_i)}^{-1} \approx 2, \quad \mathbf{F}_{\ln(\nu), \ln(\nu)}^{-1} \approx \frac{1}{\text{Var}(\nabla_{\ln(\nu)} \log \pi)}
$$

### Multivariate Normal

The multivariate normal distribution uses a full covariance matrix, allowing for more flexible modeling of dependencies between dimensions.

#### Parametrization

The distribution is parameterized by:

$$
\boldsymbol{\theta} = (\boldsymbol{\mu}, \mathbf{A})^T
$$

where $\boldsymbol{\mu}$ represents the mean vector, and $\mathbf{A}$ is the Cholesky decomposition of the covariance matrix $\mathbf{\Sigma}$ such that $\mathbf{\Sigma} = \mathbf{A}^T \mathbf{A}$.

#### Gradients

The gradients of the log-likelihood with respect to the parameters are:

$$
\nabla_{\boldsymbol{\mu}} \log \pi(\mathbf{z}_k \mid \boldsymbol{\theta}) = \mathbf{\Sigma}^{-1} (\mathbf{z}_k - \boldsymbol{\mu}),
$$

$$
\nabla_{\mathbf{A}} \log \pi(\mathbf{z}_k \mid \boldsymbol{\theta}) = \mathbf{A} \left(-\frac{1}{2} \mathbf{\Sigma}^{-1} + \frac{1}{2} \mathbf{\Sigma}^{-1} (\mathbf{z}_k - \boldsymbol{\mu})(\mathbf{z}_k - \boldsymbol{\mu})^T \mathbf{\Sigma}^{-1} \right).
$$

#### Fisher Matrix

The Fisher matrix for the multivariate normal distribution is computed numerically due to the complexity of the closed-form solution. However, it can be approximated using sample-based methods.

### Multivariate Normal Exponential Representations (xNES)

The multivariate normal exponential representation uses matrix exponential updates for the covariance, ensuring positive-definiteness and simplifying parameter updates.

#### Parametrization

The distribution is parameterized by:

$$
\boldsymbol{\theta} = (\boldsymbol{\mu}, \mathbf{A})^T
$$

where $\boldsymbol{\mu}$ represents the mean vector, and $\mathbf{A}$ is the Cholesky decomposition of the covariance matrix $\mathbf{\Sigma}$ such that $\mathbf{\Sigma} = \mathbf{A}^T \mathbf{A}$.

#### Gradients

The gradients of the log-likelihood with respect to the parameters are:

$$
\nabla_{\boldsymbol{\mu}} \log \pi(\mathbf{z}_k \mid \boldsymbol{\theta}) = \mathbf{A} (\mathbf{z}_k - \boldsymbol{\mu}),
$$

$$
\nabla_{\mathbf{A}} \log \pi(\mathbf{z}_k \mid \boldsymbol{\theta}) = \frac{1}{2} \mathbf{A} \left[ \mathbf{A}^{-1} (\mathbf{z}_k - \boldsymbol{\mu})(\mathbf{z}_k - \boldsymbol{\mu})^T \mathbf{A}^{-T} - \mathbf{I} \right].
$$

#### Fisher Matrix

The Fisher matrix for the multivariate normal exponential representation is computed numerically due to the complexity of the closed-form solution. However, it can be approximated using sample-based methods.

## Learning Rates

Despite seeming similar to many other optimization strategies we cannot use common optimtazion strategies around learning rates. Since there exists essentially a feedback loop between the ditsribution and sample fitnesses it becomes quite fragile to adaptively change step sizes:

In the case this introduces addition step size consistently we get recursive instability. In the case of essentially overdamping we get premature convergence.

Therefore we must choose the learning rates very carefully. Generally we cannot touch the learning rate of the means at all making parameter specific learning rates necessary. This already makes usual approaches for avoiding local minima impossible: Momentum based approaches and adaptive increases/decreases based on performance become impossible for the means.

Commonly used are however approaches scaling or even resetting the variances at certain stages of the process. This essentially equates to heat based approaches that repetetively attempt to avoid local minima by temporarily increasing the variance potentially allowing an escape from the local minium.

An entirely different approach is provided by the use of Heavy Tailed functions such as Cauchy (or more generally Students-T) distributions that by default sample more points at further out points. If this successfully lands outside the local minimum it allows a so called hop to a new location by being the largest contributor the to gradient by magnitudes.

The learning rates for the parameters can be adjusted based on different strategies that were found to work:

1. **Constant Learning Rates**: Heuristically chosen rates, such as:
   - $\mu$ rates: $1$
   - $\mathbf{A}$ rates: $\frac{3 + \log(d)}{5 \sqrt{d}}$ (Schaul et al. 2013 "No More Pesky Learning Rates")


2. **Adaptive Learning Rates**: KL-based rates with EMA stabilization:
   - Covariance rates are adjusted dynamically using a KL budget and exponential moving average (EMA) of gradient norms.

Generally both of these perform quite similarily for low ($\approx 30$) dimensions while for higher dimensional problems the KL-based approach dominates by far.
