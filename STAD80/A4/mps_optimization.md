# MPS Optimization Guide for Diffusion Transformer Training

## Hardware Context

- Apple M1 Pro, 16-core GPU, 16GB unified memory
- PyTorch 2.9.1 with MPS backend
- Primary bottleneck: **memory bandwidth** (shared between CPU and GPU on unified memory)

## Optimizations Applied

### 1. Fused Scaled Dot-Product Attention (`F.scaled_dot_product_attention`)

**What**: Replaced manual attention (separate Q@K, softmax, attn@V operations) with PyTorch's fused SDPA kernel.

**Why it helps**: The naive implementation performs 3 separate GPU kernel launches (two `bmm` + one `softmax`), each requiring a full read/write of intermediate tensors to memory. SDPA fuses these into a single kernel, eliminating intermediate memory round-trips. For transformers, attention is the dominant cost -- this is the single largest speedup.

**References**:
- PyTorch SDPA docs: https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
- PyTorch SDPA tutorial: https://docs.pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html

### 2. Mixed Precision via `torch.autocast('mps', dtype=torch.float16)`

**What**: Wraps the forward pass in float16 autocast so eligible operations (matmuls, convolutions) run in half precision.

**Why it helps**: float16 uses half the memory bandwidth of float32. On Apple Silicon, the GPU and CPU share the same memory bus, making bandwidth the primary bottleneck. Halving the data moved per operation directly improves throughput. Operations that need numerical stability (reductions, norms) automatically stay in float32.

**References**:
- PyTorch AMP docs: https://docs.pytorch.org/docs/stable/amp.html
- PyTorch performance tuning guide: https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html

### 3. `torch.compile(model)`

**What**: JIT-compiles the model graph using TorchInductor, fusing pointwise operations (SiLU, LayerNorm, element-wise adds/multiplies) into single kernels.

**Why it helps**: Without compilation, each operation (e.g., `x * (1 + gamma) + beta`) launches a separate GPU kernel, and each kernel reads/writes the full tensor from memory. `torch.compile` fuses chains of these into one kernel, drastically reducing kernel launch overhead and memory traffic. The first step is slow (~30-60s) due to tracing and compilation, but all subsequent steps benefit.

**References**:
- torch.compile docs: https://docs.pytorch.org/docs/stable/generated/torch.compile.html
- torch.compile tutorial: https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html
- Performance tuning guide (fusion section): https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html

## Results

| Metric | Before optimization | After optimization |
|---|---|---|
| Training speed | ~2.1 it/s | ~5-8 it/s (estimated) |
| Estimated total time (20k steps) | ~2h 35min | ~45-75min |
| Memory per batch | Higher (float32 throughout) | Lower (float16 for matmuls) |

## Additional Tips (Not Applied)

- **Reduce `num_steps`**: MNIST often converges well before 20k steps. Monitor the loss curve and stop early if plateaued.
- **Reduce `batch_size`**: If memory pressure is high, smaller batches can avoid swapping and actually improve throughput.
- **`torch.set_float32_matmul_precision('medium')`**: Allows TF32-like reduced precision for remaining float32 matmuls. May help on future Apple Silicon with wider ALUs.
- **Channels-last memory format** (`model.to(memory_format=torch.channels_last)`): Can improve conv2d performance by matching Metal's preferred memory layout. Not applied here since the patchifier conv is a small part of total compute.
