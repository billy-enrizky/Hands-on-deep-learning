# Changelog

All notable changes to this project will be documented in this file.

## [2026-03-10 14:10:42]

### Added
- Q2.2: Implemented `CFGTrainer.get_train_loss` with classifier-free guidance training objective
- Q2.3: Implemented `MLPConditionalVectorField.forward` with class embedding concatenation
- Q3.1: Implemented `FourierEncoder` with random Fourier features for time embedding
- Q3.2: Implemented `Patchifier` with Conv2d and einops rearrange
- Q3.3: Implemented `MHA` (multi-headed self-attention), `DiffusionTransformerLayer` (adaLN-Zero conditioning), and `DiffusionTransformer` (learned positional encodings + layer stack)
- Q3.4: Implemented `Depatchifier` with LayerNorm, MLP, rearrange, and final Conv2d
- Q3.5: Implemented `MNISTDiffusionTransformer` combining Fourier time embedding, class embedding, patchifier, DiT, and depatchifier

### Changed
- Added MPS (Apple Silicon) device fallback to all device selection lines
