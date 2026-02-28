# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- 2026-02-27 23:52:35 - Completed `CharDataset.__getitem__`: one-hot encoding of input sequences and class index target
- 2026-02-27 23:52:35 - Completed `CharLSTM` model: LSTM layer with linear output head
- 2026-02-27 23:52:35 - Completed `sample_from_logits`: temperature-scaled softmax sampling via `torch.multinomial`
- 2026-02-27 23:52:35 - Completed `generate_text`: autoregressive character-level text generation
- 2026-02-27 23:52:35 - Added MPS (Apple Silicon) device fallback: CUDA -> MPS -> CPU

### Verified
- 2026-02-28 00:26:02 - Full notebook audit: all 27 cells verified (code + outputs)
- 2026-02-28 00:26:02 - Training confirmed: early stopping at val_loss=1.5848, 103097 trainable parameters
- 2026-02-28 00:26:02 - All outputs present: loss curves plot rendered, text generation working at both diversity levels

### Changed

### Fixed

### Removed
