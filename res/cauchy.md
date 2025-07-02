Thanks for the clarification. I’ll gather the necessary mathematical derivations for implementing a Student-t distribution in Separable Natural Evolution Strategies (SNES) with diagonal covariance:

* How to compute the stochastic gradient (for the mean, log-standard deviation, and log-degrees of freedom) using samples and their fitness.
* How to compute the inverse Fisher Information matrix for each parameter.
* What are typical or suggested learning rates for these parameters.
* I’ll also reference any notable implementations of Student-t SNES variants, if available.

I’ll update you as soon as the derivations and references are ready.


# Separable NES with a Student-t Search Distribution

## Problem Setup and Notation

**Separable Natural Evolution Strategies (SNES)** maintains a factored search distribution over a \$d\$-dimensional parameter vector \$x\in\mathbb{R}^d\$. In the Student-\$t\$ variant, we use a multivariate Student-\$t\$ distribution with diagonal covariance (i.e. independent components) as the mutation distribution. Let \$\mu\in\mathbb{R}^d\$ be the mean vector, \$\boldsymbol{\sigma}\in\mathbb{R}^d\$ the scale (standard deviation) vector for each coordinate, and \$\nu>0\$ the degrees of freedom (tail-heaviness) parameter. The probability density of a sample \$x\sim \text{Student-}t(\mu,; \mathrm{diag}(\sigma\_1^2,\dots,\sigma\_d^2),; \nu)\$ is:

$$
p(x\,|\,\mu,\boldsymbol{\sigma},\nu) \;=\; \frac{\Gamma\!\Big(\frac{\nu+d}{2}\Big)}{\Gamma\!\Big(\frac{\nu}{2}\Big)\,(\nu\pi)^{d/2}\,\prod_{i=1}^d\sigma_i}\;\left[\,1 + \frac{1}{\nu}\sum_{i=1}^d\Big(\frac{x_i-\mu_i}{\sigma_i}\Big)^2\,\right]^{-\frac{\nu+d}{2}}~,
$$

where \$\Gamma(\cdot)\$ is the gamma function. For convenience, define the (squared) Mahalanobis radius of \$x\$ as:

$r^2 \;=\; \sum_{i=1}^d \frac{(x_i-\mu_i)^2}{\sigma_i^2}~,$

so that \$p(x)\propto \big(1+\frac{r^2}{\nu}\big)^{-(\nu+d)/2}\$. SNES performs *natural gradient* ascent on the expected fitness \$J(\mu,\boldsymbol{\sigma},\nu)=\mathbb{E}*{x\sim p}\[f(x)]\$. In practice, we estimate gradients from a sample batch \${x^{(k)}, f^{(k)}}*{k=1}^\lambda\$. Below, we derive the **stochastic gradients** of the log-likelihood and then apply the Fisher Information matrix to obtain *natural* update rules. We assume \$\log\$-parameterization for \$\sigma\_i\$ and \$\nu\$ (i.e. we treat \$\theta\_i=\ln\sigma\_i\$ and \$\zeta=\ln\nu\$ as the parameters), which simplifies gradient expressions by ensuring positivity of \$\sigma\_i,\nu\$.

## 1. Stochastic Gradients for Student-\$t\$ (Mean, Log-\$\sigma\$, Log-\$\nu\$)

Using the “score function” gradient estimator, the gradient of \$J\$ w\.r.t. any parameter \$\theta\$ can be written as \$\nabla\_\theta J = \mathbb{E}*{x\sim p}\[f(x),\nabla*\theta \ln p(x)]\$. We derive the log-likelihood gradients for each parameter:

* **Mean vector \$\mu\$:**\* The gradient of the log-density w\.rt. \$\mu\_i\$ is obtained by differentiating the quadratic form inside \$p(x)\$. For each dimension \$i\$:

  $$
  \frac{\partial}{\partial \mu_i}\ln p(x) \;=\; \frac{\nu+d}{\,\nu + r^2\,}\;\frac{x_i - \mu_i}{\sigma_i^2}~,
  $$

  which can be written vectorwise as

  $$
  \nabla_{\mu}\ln p(x) \;=\; \frac{\nu+d}{\,\nu + r^2\,}\;\Sigma^{-1}(x-\mu)~,
  $$

  where \$\Sigma=\mathrm{diag}(\sigma\_1^2,\dots,\sigma\_d^2)\$ and \$\Sigma^{-1}(x-\mu)\$ produces the elementwise terms \$((x\_i-\mu\_i)/\sigma\_i^2)\$. Intuitively, this is the Gauss/NES mean gradient scaled by a factor \$\frac{\nu+d}{,\nu+r^2,}\$ that down-weights far outliers. In the limit \$\nu\to\infty\$ (Gaussian mutation), this factor \$\to 1\$ and we recover the usual Gaussian NES result \$\nabla\_{\mu}\ln p(x)=\Sigma^{-1}(x-\mu)\$.

* **Log-standard-deviations \$\ln\sigma\_i\$:**\* Using \$\theta\_i=\ln\sigma\_i\$, we apply the chain rule \$\frac{\partial}{\partial \theta\_i} = \sigma\_i,\frac{\partial}{\partial \sigma\_i}\$. The log-density contains two \$\sigma\_i\$-dependent parts: (1) the normalization \$-\ln\sigma\_i\$, and (2) the \$(1 + \frac{r^2}{\nu})^{-(\nu+d)/2}\$ term via \$r^2\$ which contains \$\sigma\_i^2\$ in the denominator. Differentiating, we obtain the per-component gradient:

  $$
  \frac{\partial}{\partial \ln\sigma_i}\ln p(x) \;=\; \frac{\nu+d}{\,\nu + r^2\,}\;\frac{(x_i-\mu_i)^2}{\sigma_i^2}\;-\;1~,
  $$

  for \$i=1,\dots,d\$. In vector form, \$\nabla\_{\ln\boldsymbol{\sigma}}\ln p(x) = \frac{\nu+d}{,\nu + r^2,};\big((x-\mu)/\boldsymbol{\sigma}\big)^2 ;-; \mathbf{1}\$, where \$(x-\mu)/\boldsymbol{\sigma}\$ denotes elementwise division and the square is elementwise. The “\$-1\$” term comes from the \$\partial(-\ln\sigma\_i)/\partial\ln\sigma\_i = -1\$ contribution of the normalization. Again, for \$\nu\to\infty\$ this recovers the Gaussian SNES result \$\nabla\_{\ln\sigma\_i}\ln p = \frac{(x\_i-\mu\_i)^2}{\sigma\_i^2}-1\$. For finite \$\nu\$, the heavy-tail effect appears as a moderation factor \$\frac{\nu+d}{\nu+r^2}\$ on the usual \$(x\_i-\mu\_i)^2/\sigma\_i^2\$ term, reducing the influence of large deviations.

* **Log-degrees-of-freedom \$\ln\nu\$:**\* Let \$\zeta=\ln\nu\$. Using \$\frac{\partial}{\partial\zeta} = \nu,\frac{\partial}{\partial\nu}\$, we differentiate the log-density w\.rt. \$\nu\$. This is more involved, since \$\nu\$ appears in both the gamma functions and the \$(1+r^2/\nu)\$ term. For derivation, it’s convenient to use identities \$\frac{d}{dx}\ln\Gamma(x)=\psi(x)\$ (digamma function) and \$\frac{d}{dx}\ln(1+u,x^{-1}) = -\frac{u}{x+u}\$, and also note \$\frac{\partial r^2}{\partial \nu}=0\$. Differentiating \$\ln p(x)\$ w\.rt. \$\nu\$ and then multiplying by \$\nu\$ (to get \$\partial/\partial\ln\nu\$) gives:

  $$
  \nabla_{\ln\nu}\ln p(x) \;=\; \frac{\nu}{2}\Big[\psi\!\Big(\frac{\nu+d}{2}\Big) - \psi\!\Big(\frac{\nu}{2}\Big) - \ln\!\Big(1+\frac{r^2}{\nu}\Big)\Big] \;-\; \frac{d}{2} \;+\; \frac{\nu+d}{2}\;\frac{r^2}{\,\nu + r^2\,}~.
  $$

  This expression can be understood term-by-term:

  * The \$\psi((\nu+d)/2)-\psi(\nu/2)\$ term comes from differentiating the gamma functions (from normalization).
  * The \$-\ln(1+r^2/\nu)\$ term comes from differentiating the heavy-tail factor \$(1+r^2/\nu)^{-(\nu+d)/2}\$.
  * \$-d/2\nu\$ (multiplied by \$\nu\$ here) comes from differentiating \$-(d/2)\ln(\nu)\$ in the normalization.
  * The final \$+\frac{\nu+d}{2}\frac{r^2}{\nu+r^2}\$ term arises from the chain-rule (since \$\nu\$ appears also in the exponent \$(\nu+d)/2\$).

  While this looks complicated, it simplifies in important special cases. For example, in the Cauchy case \$\nu=1\$, this reduces to \$\nabla\_{\ln\nu}\ln p = \frac{1}{2}\[\psi((1+d)/2) - \psi(1/2) - \ln(1+r^2)]\$ because for Cauchy the last two terms cancel out (\$d/(2\nu)=d/2\$ and \$(\nu+d)r^2/(2(\nu+r^2)) = (d+1)r^2/\[2(1+r^2)]\$, which sum to \$d/2\$). In the Gaussian limit \$\nu\to\infty\$, one can use asymptotics \$\psi(x+1)\approx \ln(x)\$ to check that \$\nabla\_{\ln\nu}\ln p \to 0\$, as expected (the Gaussian has no \$\nu\$ parameter).

**Fitness Gradient Estimator:** In practice, we weight these “score function” derivatives by fitness to get the gradient of the objective. For example, the stochastic gradient for the mean is \$\hat{g}*{\mu\_i} = \frac{1}{\lambda}\sum*{k=1}^\lambda f^{(k)} \frac{\nu+d}{,\nu + r^{2^{(k)}},}\frac{x^{(k)}\_i - \mu\_i}{\sigma\_i^2}\$ (and similarly for \$\ln\sigma\_i\$ and \$\ln\nu\$). In SNES, it’s common to use *fitness shaping* (ranking the \$f^{(k)}\$ values and replacing them with centered ranks or utilities) to reduce variance, but the form of gradients remains as derived above.

## 2. Inverse Fisher Information Matrix (Diagonal Covariance Case)

For natural gradient updates, we need the Fisher Information Matrix (FIM) for our distribution’s parameters. With a diagonal covariance (independent components), the FIM is block-diagonal between \${\mu\_i}\$ and \${\ln\sigma\_i}\$ parameters due to symmetry, and there is no coupling between different \$\mu\_i\$ or different \$\sigma\_i\$ dimensions (no correlation in the score between \$\mu\_i\$ and \$\mu\_j\$, etc., because the distribution factorizes across \$i\$). However, the \$\ln\nu\$ parameter (being global) will generally couple with the \$\ln\sigma\$ components in the FIM. We outline each block of the FIM and its inverse:

* **Mean parameters:** For each dimension \$i\$, the Fisher information for \$\mu\_i\$ is \$F\_{\mu\_i\mu\_i} = \mathbb{E}!\Big\[\Big(\frac{\partial}{\partial \mu\_i}\ln p(x)\Big)^2\Big]\$. Using \$\frac{\partial}{\partial \mu\_i}\ln p = \frac{\nu+d}{\nu+r^2}\frac{(x\_i-\mu\_i)}{\sigma\_i^2}\$ from above, and noting cross terms \$i\neq j\$ have zero expectation, we get:

  $$
  F_{\mu_i\mu_i} \;=\; \mathbb{E}\!\Big[\frac{(\nu+d)^2}{(\nu+r^2)^2}\frac{(x_i-\mu_i)^2}{\sigma_i^4}\Big]~.
  $$

  This integral can be evaluated via known results for Student-\$t\$ moments (or using Beta integrals). The result turns out to be proportional to \$1/\sigma\_i^2\$. In fact, for a *univariate* Student-\$t\$ (d=1) one finds \$F\_{\mu\mu}=\frac{\nu+1}{\nu+3}\frac{1}{\sigma^2}\$. By analogy, for the \$d\$-dimensional case the Fisher info of the mean in each direction is:

  $$
  F_{\mu_i\mu_i} \;=\; \frac{\nu+d}{\,\nu + d + 2\,}\;\frac{1}{\sigma_i^2}~,
  $$

  and \$F\_{\mu\_i\mu\_j}=0\$ for \$i\neq j\$. (This form reduces to the univariate case when \$d=1\$, and to the Gaussian case in the limit \$\nu\to\infty\$ since \$(\nu+d)/(\nu+d+2)\to 1\$.) The **inverse Fisher** for the mean is simply the reciprocal:

  $$
  F_{\mu_i\mu_i}^{-1} \;=\; \frac{\nu + d + 2}{\,\nu + d\,}\;\sigma_i^2~,
  $$

  which generalizes the Gaussian SNES result \$F\_{\mu\mu}^{-1}=\sigma^2\$. Thus, in the natural gradient update, the mean step in each coordinate is scaled by \$\frac{\nu+d+2}{\nu+d}\sigma\_i^2\$ instead of \$\sigma\_i^2\$. For heavy tails, this factor is slightly larger than \$\sigma\_i^2\$ (since \$\frac{\nu+d+2}{\nu+d}>1\$), reflecting that we trust the mean gradient direction a bit more due to lower “information” in heavy-tailed distributions (intuitively, heavy tails yield larger score variance, thus smaller Fisher information).

* **Scale (log-\$\sigma\$) parameters:** The Fisher information for \$\ln\sigma\_i\$ is \$F\_{\theta\_i\theta\_i}=\mathbb{E}\[ (\partial/\partial \ln\sigma\_i \ln p)^2 ] = \mathbb{E}\[ (\frac{\nu+d}{\nu+r^2}\frac{(x\_i-\mu\_i)^2}{\sigma\_i^2} - 1)^2 ]\$. Again, by independence, \$F\_{\theta\_i\theta\_j}=0\$ for \$i\neq j\$. The exact evaluation is tedious, but known results exist for the univariate case. For one dimension, \$F\_{\ln\sigma,\ln\sigma} = \frac{\nu}{2(\nu+3),}\frac{1}{\sigma^4}\$ in terms of \$\sigma^2\$ (or equivalently \$\frac{\nu}{8(\nu+3)}\frac{1}{\sigma^8}\$ in terms of \$\ln\sigma\$). By analogy in \$d\$ dimensions, we incorporate \$d\$ into the “\$\nu+3\$” term (becoming \$\nu+d+2\$ as before). We thus expect:

  $$
  F_{\theta_i\theta_i} \;=\; \frac{\nu}{\,2(\nu + d + 2)\,}\;\frac{1}{\sigma_i^4} \qquad \text{(for parameter $\sigma_i^2$)}~,
  $$

  which in terms of \$\theta\_i=\ln\sigma\_i\$ (recall \$\frac{\partial}{\partial\ln\sigma\_i} = \sigma\_i\frac{\partial}{\partial\sigma\_i}\$) becomes

  $$
  F_{\ln\sigma_i,\ln\sigma_i} \;=\; \frac{\nu}{\,8(\nu + d + 2)\,}\;\frac{1}{\sigma_i^8}~.
  $$

  Inverse Fisher then is:

  $$
  F_{\ln\sigma_i,\ln\sigma_i}^{-1} \;=\; \frac{8(\nu + d + 2)}{\nu}\;\sigma_i^8~.
  $$

  In the Gaussian limit (\$\nu\to\infty\$), this reduces to \$8,\sigma\_i^8/\nu \to 0\$ as \$\nu\to\infty\$—however, note that for a true Gaussian (\$\nu=\infty\$) the Fisher w\.rt. \$\ln\sigma\$ actually approaches \$1/(8\sigma\_i^8)\$ (since effectively \$\nu/(\nu+d+2)\approx 1\$ for large \$\nu\$). Plugging \$\nu\to\infty\$ into the above gives \$F\_{\ln\sigma\_i}\approx\frac{1}{8\sigma\_i^8}\$, so \$F^{-1}\approx 8\sigma\_i^8\$, which matches the known Fisher of a Gaussian in log-std param (for a 1-D Gaussian, \$\text{Var}\[-1 + (x-\mu)^2/\sigma^2] = 2\$, giving \$F\_{\ln\sigma}=1/(4\sigma^4)\cdot 2 = 1/(2\sigma^4)\$ in \$\sigma^2\$-param, or \$1/(8\sigma^8)\$ in \$\ln\sigma\$-param as we found earlier). Thus the formula is consistent in all cases. In summary, each log-scale parameter has an information \~inversely proportional to \$\sigma^8\$, and heavy tails (finite \$\nu\$) further reduce that information, leading to a larger natural step size scaling.

* **Degrees-of-freedom parameter:** The Fisher information for \$\zeta=\ln\nu\$ can be derived from the second derivative of \$\ln p\$. While one *could* derive a closed form involving the trigamma \$\psi'(\cdot)\$ function, it’s more instructive to note the qualitative behavior. Formally, \$F\_{\ln\nu} = \mathbb{E}\[ (\nabla\_{\ln\nu}\ln p(x))^2 ]\$. Using the expression we found for \$\nabla\_{\ln\nu}\ln p\$, one obtains:

  $$
  F_{\ln\nu,\ln\nu} \;=\; \frac{\nu^2}{4}\Big[\psi'\!\Big(\frac{\nu+d}{2}\Big) - \psi'\!\Big(\frac{\nu}{2}\Big)\Big] \;-\; \frac{\nu+d}{2\nu^2}\;\mathbb{E}\Big[\frac{\nu r^2}{\,\nu + r^2\,}\Big] \;+\; \frac{d}{2\,\nu^2}~,
  $$

  which indeed contains *trigamma* terms (\$\psi'\$) from the gamma-function part and an expectation term from the heavy-tail part. This simplifies in special cases. For example, in the Cauchy case (\$\nu=1\$) the second term vanishes as discussed, and one gets \$F\_{\ln\nu} = \frac{1}{4}\[\psi'(\frac{1+d}{2}) - \psi'(\frac{1}{2})]\$ – which, interestingly, is **infinite** because \$\psi'(1/2)=\pi^2/2\$ is infinite (reflecting the fact that the Cauchy’s \$\nu\$ is not an identifiable parameter in the Fisher sense due to infinite variance). For larger \$\nu\$, \$F\_{\ln\nu}\$ becomes finite. The full expression for general \$\nu\$ is lengthy, but one can compute it if needed. For practical purposes, it’s often wise to use a relatively small learning rate for \$\ln\nu\$ or even fix \$\nu\$ if heavy tails are mainly needed initially, since the Fisher information for \$\nu\$ can be low and the estimation noise high. We denote the inverse Fisher on \$\ln\nu\$ as \$F\_{\ln\nu}^{-1}\$ (which would be the reciprocal if it were decoupled; when coupling with \$\ln\sigma\$ is considered, we would invert the \$2\times 2\$ submatrix for \$(\ln\nu,\ln\sigma)\$).

**FIM Coupling:** Note that \$\ln\nu\$ and \$\ln\sigma\_i\$ are **not independent** in the Fisher matrix. Their covariance \$F\_{\ln\nu,\ln\sigma\_i} = \mathbb{E}\[(\nabla\_{\ln\nu}\ln p)(\nabla\_{\ln\sigma\_i}\ln p)]\$ is generally nonzero because both involve the term \$\ln(1+r^2/\nu)\$ (for \$\ln\nu\$) and \$(x\_i-\mu\_i)^2/\sigma\_i^2\$ (for \$\ln\sigma\_i\$) which are statistically coupled through \$r^2\$. In a full natural gradient, one would need to invert the joint Fisher of \$(\ln\nu, \ln\sigma\_1,\dots,\ln\sigma\_d)\$. However, a common simplification in *separable* NES is to treat these updates separately or assume weak coupling. In practice one might ignore the cross term or set it to zero, which is an approximation. (For example, Schaul *et al.* note that heavy-tailed NES can proceed “without further corrections like multiplying with the (in general undefined) inverse Fisher matrix” by working in the “local coordinates” of the distribution.) If needed, one can compute the cross-term via integration; however, for brevity we do not derive it here.

## 3. Effect of Log-Parametrization

Using logarithmic parametrization for \$\sigma\_i\$ and \$\nu\$ simplifies the update rules and ensures positivity. The gradients we derived above already assume this log-parametrization – that’s why terms like “\$-1\$” appear (for \$\ln\sigma\$) which wouldn’t appear if we derived w\.rt. \$\sigma\$ directly. In essence, \$\nabla\_{\ln\sigma\_i} J = \sigma\_i,\nabla\_{\sigma\_i} J\$, so the natural gradient in \$\ln\sigma\_i\$ corresponds to a multiplicative update to \$\sigma\_i\$. This is beneficial because it makes the update scale-invariant (updating \$\ln\sigma\$ by \$\Delta\$ corresponds to multiplying \$\sigma\$ by \$e^{\Delta}\$). Similarly, \$\nabla\_{\ln\nu} J = \nu,\nabla\_{\nu} J\$. A small step in \$\ln\nu\$ produces a multiplicative change in \$\nu\$. This is sensible since the effect of \$\nu\$ on the distribution is highly nonlinear (especially when \$\nu\$ is small, a slight change can dramatically alter tail heaviness). The Fisher analysis above was done in terms of \$\ln\sigma\$ and \$\ln\nu\$, which is why the formulas involve high powers of \$\sigma\$ (e.g. \$\sigma^8\$) – those come from the chain rule and ensure that the natural gradient steps for \$\ln\sigma\$ are dimensionless and properly scaled. In summary, log-parametrization means we compute gradients as shown above, and apply updates additively to \$\ln\sigma\_i\$ and \$\ln\nu\$ (which correspond to multiplicative changes in \$\sigma\_i\$ and \$\nu\$).

## 4. Learning Rates for \$\mu\$, \$\ln\sigma\$, \$\ln\nu\$

In NES, **learning rates** \$(\eta\_\mu,\eta\_\sigma,\eta\_\nu)\$ control the fraction of the natural gradient used each generation. Prior work provides typical defaults as a function of dimension \$d\$. Wierstra *et al.* (2014) suggest \$\eta\_\mu=1\$ for the mean, and smaller values for covariance parameters (since far more covariance parameters exist). For *separable* NES (diagonal covariance), Schaul *et al.* (2011) recommend:

* **Mean:** \$\eta\_\mu = 1.0\$ (full step along natural gradient for the mean). The mean typically converges faster and can use a larger learning rate safely.

* **Scale (std-dev):** \$\eta\_\sigma\$ on the order of \$\frac{3+\ln d}{5\sqrt{d}}\$ (this is an increased learning rate for diagonal SNES compared to full-covariance NES). For example, in many experiments \$\eta\_\sigma = \frac{3+\ln d}{5\sqrt{d}}\$ was used, which for moderate \$d\$ gives a value around \$0.2\$–\$0.3\$. This is higher than the full-matrix NES covariance learning rate (which was \$\sim\frac{9+3\ln d}{5d\sqrt{d}}\$), reflecting the fact that diagonal adaptation is less prone to instability than full covariance adaptation.

* **Degrees of freedom:** There isn’t a widely established default for \$\eta\_\nu\$ in literature, since adapting \$\nu\$ in NES is less common. However, one should generally set \$\eta\_\nu\$ **small** – because \$\nu\$ greatly influences the distribution shape, a little change can have a large effect. A conservative choice is to set \$\eta\_{\ln\nu}\$ similar to or lower than \$\eta\_\sigma\$. For example, one might start with \$\eta\_{\ln\nu}\$ an order of magnitude smaller (e.g. \$\sim 0.1/\sqrt{d}\$) and adjust if needed. If \$\nu\$ is kept fixed (like using a Cauchy distribution fixed at \$\nu=1\$), then no learning rate is needed for \$\nu\$. If adapting, ensure the update to \$\nu\$ is clipped to keep \$\nu>0\$ and not too low (since extremely heavy tails can cause high-variance estimates).

In summary, a typical configuration for moderate \$d\$ might be \$(\eta\_\mu,\eta\_\sigma,\eta\_\nu)=(1.0,;0.2,;0.05)\$ as a starting point. These can be tuned; some implementations even make \$\eta\_\nu\$ decrease over time to stabilize the tail parameter.

## 5. Existing Implementations of Student-\$t\$ NES

While most NES implementations use Gaussian mutations, there have been extensions to heavy-tailed distributions:

* **PyBrain Library (2010)** – An early open-source implementation of SNES by Schaul *et al.* included a **Cauchy-NES** variant (Cauchy is a \$t\$-distribution with \$\nu=1\$). In fact, Schaul’s 2011 paper provides pseudocode for a \$(1+1)\$-Cauchy NES and notes that *“a Python implementation of all these \[variants] is available in the open-source PyBrain library”*. This code implements SNES with diagonal covariance and allows using a multivariate Cauchy search distribution. Adapting a general \$\nu\$ was not the focus (they primarily tested \$\nu=1\$ vs \$\nu=\infty\$), but one could modify it to slowly learn \$\nu\$.

* **GitHub Repositories:** While not extremely common, there are simple NES code examples (e.g. Mofan’s NES toy example) that could be modified to use Student-\$t\$ by replacing the Gaussian sampling with Student-\$t\$ (using a library or custom sampler) and plugging in the gradients derived above. To our knowledge, dedicated Student-\$t\$ NES implementations are rare, but the concept is straightforward given an existing NES: one needs to add the \$\ln\nu\$ parameter, use the above gradient formulas, and update \$\nu\$. The recent information-geometric work by Xu & Zeng (2024) on *Student-t processes* includes natural gradient steps for a diagonal-\$t\$ distribution (they derive the FIM for diagonal \$t\$ in closed form) – their algorithm is for variational inference, but it’s mathematically equivalent to adapting a Student-\$t\$ distribution’s parameters. Upon publication, they promise to release code, which could be repurposed for NES.

In practice, one can implement Student-\$t\$ SNES by extending an existing Gaussian SNES: replace the Gaussian sampler with a Student-\$t\$ sampler, extend the parameter vector to include \$\ln\nu\$, and use the update rules derived here. The heavy-tailed NES approach has been reported to help escape deceptive local optima by occasionally taking large jumps, at the cost of a slightly noisier natural gradient (especially if \$\nu\$ is very small, i.e. extremely heavy tails).

**References:** The derivations above draw from known results in the NES literature and Student-\$t\$ distribution properties. Notably, Wierstra *et al.* (2014) and Schaul *et al.* (2011) discuss heavy-tailed NES concepts. For detailed mathematical background on Student-\$t\$ Fisher information, see Lange *et al.* (1989) and the Cross-Validated discussion by user *bayes003*, which provided the univariate \$t\$-distribution Fisher matrix used as a basis for our multivariate generalizations. The recommended learning rates come from the default SNES settings in those works.
