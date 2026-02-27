# Changelog

## [2026-02-26 20:55 EST] - Full audit and final touches

### Added
- **Problem 4.3 (Cell 79)**: Written observations on bridging arbitrary distributions with linear probability paths
- MPS device support for Apple Silicon acceleration

### Changed
- Updated device selection in Cell 3 to use MPS when available (CUDA > MPS > CPU)
- Verified all 80 cells: all implementations correct, all visualization cells have outputs

## [2026-02-26 19:05 EST] - Complete all assignment cells

### Added
- **Problem 2.1 (Cell 18)**: Implemented `LinearAlpha.__call__` returning `t` and `SquareRootBeta.__call__` returning `sqrt(1-t)`
- **Problem 2.2 (Cell 20)**: Implemented `GaussianConditionalProbabilityPath.sample_conditional_path` using reparameterization trick: `x = alpha_t * z + beta_t * eps`
- **Problem 2.3 (Cell 20)**: Implemented `GaussianConditionalProbabilityPath.conditional_vector_field` computing `u_t(x|z) = (alpha_dot - beta_dot/beta * alpha) * z + (beta_dot/beta) * x`
- **Problem 2.4 (Cell 20)**: Implemented `GaussianConditionalProbabilityPath.conditional_score` computing `nabla_x log p_t(x|z) = (alpha_t * z - x) / beta_t^2`
- **Problem 3.1 (Cell 44)**: Implemented `ConditionalFlowMatchingTrainer.get_train_loss` with Monte Carlo estimate of the CFM objective
- **Problem 3.2 (Cell 53)**: Implemented `ConditionalScoreMatchingTrainer.get_train_loss` with Monte Carlo estimate of the CSM objective
- **Problem 3.3 (Cell 61)**: Implemented `ScoreFromVectorField.forward` deriving score from learned vector field via `(alpha_t * u_theta - alpha_dot * x) / (beta_t^2 * alpha_dot - alpha_t * beta_dot * beta_t)`
- **Problem 4.1 (Cell 68)**: Implemented `LinearConditionalProbabilityPath.sample_conditional_path` as `X_t = (1-t)*X_0 + t*z` and `conditional_vector_field` as `u_t(x|z) = (z-x)/(1-t)`
