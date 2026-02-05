# Changelog

All notable changes to STAD80 Assignment 2 Part 2 (Simulating ODEs and SDEs) will be documented in this file.

## [2.0.0] - 2026-02-05 00:31:50 EST

### Major Update - Comprehensive Mathematical Answers
All conceptual and mathematical questions updated with rigorous, research-backed derivations based on web searches of current literature.

### Cell 16 - Brownian Motion Trajectory Behavior
- Added mathematical analysis: solution X_t = sigma * W_t, variance Var(X_t) = sigma^2 * t
- Explained path properties: continuous but nowhere differentiable, quadratic variation [X]_t = sigma^2 * t
- Added Levy's modulus of continuity: Holder continuous with exponent alpha < 1/2

### Cell 22 - Variance Derivation for Brownian Motion
- Full Ito isometry derivation with step-by-step proof
- Distribution result: X_t ~ N(0, sigma^2 * t)
- Physical interpretation of sigma as noise intensity

### Cell 24 - OU Process Theta Parameter Analysis
- Explicit solution: X_t = X_0 * e^(-theta*t) + sigma * integral of e^(-theta*(t-s)) dW_s
- Relaxation time tau = 1/theta with physical interpretation
- Autocorrelation function: C(tau) = (sigma^2/2theta) * e^(-theta|tau|)

### Cell 28 - OU Process Convergence Analysis
- Time-dependent variance formula: Var(X_t) = (sigma^2/2theta)(1 - e^(-2*theta*t))
- Convergence rate: exponential with rate 2*theta
- Stationary distribution: N(0, sigma^2/(2*theta))

### Cell 31 - Ratio D = sigma^2/(2*theta) Analysis
- D as sufficient statistic for long-term behavior
- Stationary density formula with explicit derivation
- Fluctuation-dissipation relation analogy to statistical mechanics

### Cell 43 - Langevin Dynamics Parameter Effects
- Discretization error analysis: weak error O(h), total error O(sqrt(T/h))
- Mixing time scaling for log-concave targets
- Fokker-Planck equation proof that p(x) is invariant measure

### Cell 48 - Score Function Derivation
- General Gaussian score formula: nabla log p(x) = -Sigma^(-1)(x - mu)
- Step-by-step derivation from density to score
- Interpretation as restoring force toward mean

### Cell 49 - OU-Langevin Equivalence Proof
- Complete substitution showing Langevin reduces to OU
- Fluctuation-dissipation theorem connection
- Temperature analogy: D = k_B * T

## [1.1.0] - 2026-02-04 23:47:21 EST

### Updated
- Cell 22: Added rigorous mathematical derivation for variance formula Var(X_t) = sigma^2 * t
- Derivation uses Ito isometry: E[(sigma * W_t)^2] = sigma^2 * E[W_t^2] = sigma^2 * t
- Explained that standard Wiener process has Var(W_t) = t by definition

## [1.0.1] - 2026-02-04 19:45:28 EST

### Verified
- Re-executed entire notebook (50 cells) with no errors
- All 21 code cells executed successfully
- All 6 visualization cells generated proper outputs (Brownian motion, OU process, scaled trajectories, density plots, Langevin dynamics)
- Confirmed all implementations and conceptual answers are complete

## [1.0.0] - 2026-02-04 19:28:11 EST

### Added
- Implemented EulerSimulator class with step method for ODE integration
- Implemented EulerMaruyamaSimulator class with step method for SDE integration
- Implemented BrownianMotion SDE class with drift and diffusion coefficients
- Implemented OUProcess (Ornstein-Uhlenbeck) SDE class with mean-reverting drift
- Implemented LangevinSDE class for score-based sampling dynamics

### Conceptual Answers
- Question 2.1 (Cell 16): Large sigma produces erratic trajectories; small sigma produces smooth paths near origin
- Question 2.1 (Cell 22): Varying sigma scales amplitude of fluctuations; variance scales as sigma^2 * t
- Question 2.2 (Cell 24): Small theta gives weak mean-reversion (Brownian-like); large theta gives strong pull to zero
- Question 2.2 (Cell 28): Increasing sigma widens stationary distribution; increasing theta narrows it and speeds convergence
- Question 2.2 (Cell 31): Ratio D = sigma^2/(2*theta) determines stationary variance N(0, D)
- Question 3.1 (Cell 43): Larger sigma speeds mixing but adds noise; more timesteps improve convergence; source affects transient only

### Mathematical Derivations
- Question 3.2 (Cell 48): Derived score of Gaussian N(0, sigma^2/(2*theta)) as -2*theta*x/sigma^2
- Question 3.2 (Cell 49): Proved OU process is Langevin dynamics for Gaussian target by substituting score into drift

### Technical Details
- Euler method: X_{t+h} = X_t + h * u_t(X_t)
- Euler-Maruyama method: X_{t+h} = X_t + h * u_t(X_t) + sqrt(h) * sigma_t * z_t
- Brownian motion: dX_t = sigma * dW_t (zero drift)
- OU process: dX_t = -theta * X_t * dt + sigma * dW_t
- Langevin dynamics: dX_t = 0.5 * sigma^2 * score(X_t) * dt + sigma * dW_t

### Verified
- All implementations tested and verified working with PyTorch on CPU
- Notebook kernel updated from 'mtds' to 'python3' for compatibility
- Optional animation cells (45-46) require 'celluloid' library (not installed)
