# LoRA-Safe TorchCompile Node for ComfyUI

`TorchCompileModel_LoRASafe` is a drop-in replacement for ComfyUI's stock
`TorchCompileModel` node, designed for workflows that rely on model patching
(e.g., LoRA stacks, TEA-Cache, Sage-Attention, and related runtime patches).

The main difference from the stock node is **when** compilation is applied:
this node compiles after patching so your patches stay active instead of being
silently bypassed.

---

## Why this exists

In many optimized ComfyUI workflows, the diffusion model is patched before
sampling starts. If compile wrapping happens at the wrong stage, you can end up
compiling a pre-patch object and losing expected patch behavior.

This node is built to avoid that failure mode:

- Patch first
- Compile lazily on first real execution
- Keep patched behavior intact while still benefiting from `torch.compile`

---

## Feature summary

- **LoRA-safe compile flow** for patched diffusion models
- **Lazy compilation** (first forward pass triggers compile)
- **Whole-model compile** by default
- **Transformer-only compile mode** for architectures where compiling only block
  lists is more stable or faster to warm up
- **ComfyUI-aligned stability behavior**: attempts `model.clone(disable_dynamic=True)`
  and (when available) uses ComfyUI's `guard_filter_fn` to avoid unnecessary
  guards on dynamic `transformer_options` dictionaries
- Backend support:
  - `inductor`
  - `cudagraphs` (CUDA required; generally best on newer GPUs)
  - `nvfuser` (CUDA required)

---

## FLUX support (including FLUX.2 KLEIN 9B)

This node includes Flux-specific compile target discovery intended to improve
compatibility with **FLUX.2 KLEIN 9B** and other Flux-style workflows.

For Flux-style models in ComfyUI, block forward passes are highly dynamic
(`transformer_options` mutation, dynamic patch replacement, and patch hooks
around attention). Because of that, this node **does not** compile whole
`double_blocks.{i}` / `single_blocks.{i}` forwards.

Instead, when `compile_transformer_only = true`, Flux targets leaf tensor-heavy
submodules. It first tries preferred names and then falls back to inferred leaf
modules from each block when naming differs across ComfyUI builds.

Preferred Flux names attempted first:

- `double_blocks.{i}.img_attn.qkv`
- `double_blocks.{i}.img_attn.proj`
- `double_blocks.{i}.txt_attn.qkv`
- `double_blocks.{i}.txt_attn.proj`
- `double_blocks.{i}.img_mlp`
- `double_blocks.{i}.txt_mlp`
- `single_blocks.{i}.linear1`
- `single_blocks.{i}.linear2`

Note: MLP containers (`img_mlp`/`txt_mlp`) are often not leaf modules in Flux
builds, so the node may compile inferred leaf submodules inside those MLP paths
instead of compiling the container module directly.

Fallback inference (when preferred names are missing, or when preferred names
resolve to non-leaf containers) looks for leaf modules matching attention
`qkv/proj`, `img_mlp/txt_mlp`, and `linear1/linear2` patterns inside Flux
blocks.

For non-Flux transformer-style models, it compiles known block containers such
as:

- `layers`
- `transformer_blocks`
- `blocks`
- `visual_transformer_blocks`
- `text_transformer_blocks`

If no known transformer compile targets are found, the node safely falls back
to compiling `diffusion_model` as a whole.

---

## Installation

1. Clone or download this repository.
2. Copy the `lora_safe_compile` folder into your ComfyUI `custom_nodes/` directory.
3. Restart ComfyUI.
4. Search for **TorchCompileModel_LoRASafe** in:
   **model / optimisation 🛠️**

---

## Node inputs (detailed)

- `model` (`MODEL`)
  - The model object to patch and compile.

- `backend` (`inductor`, `cudagraphs`, `nvfuser`)
  - `inductor`: default recommendation for most users.
  - `cudagraphs`: CUDA-only; can reduce overhead in stable-shape loops.
  - `nvfuser`: CUDA-only; may vary by PyTorch/CUDA environment.

- `mode` (`default`, `reduce-overhead`, `max-autotune`)
  - Passed through to `torch.compile` mode behavior.

- `fullgraph` (`BOOLEAN`)
  - Requests full-graph capture. Can improve performance in some setups, but
    may reduce tolerance for graph breaks.

- `dynamic` (`BOOLEAN`)
  - Enables dynamic-shape aware compilation behavior.
  - Note: the node still attempts `model.clone(disable_dynamic=True)` first to
    match ComfyUI stock compile behavior when supported by the model object.

- `disable_cudagraphs` (`BOOLEAN`, default `true`)
  - For `backend = inductor`, adds
    `torch.compile(..., options={'triton.cudagraphs': False})`.
  - Not applied to the `cudagraphs` backend.
  - Recommended if you hit `cudaMallocAsync` / cudagraph instability in
    Inductor-based runs.

- `compile_transformer_only` (`BOOLEAN`)
  - `false` (default): compile the entire `diffusion_model`.
  - `true`: compile discovered transformer targets only.
  - Flux behavior: compiles leaf heavy submodules (qkv/proj/mlp/linear) instead
    of full block forwards to avoid compiling across dynamic hook-heavy paths.
  - Fallback behavior: if no recognized transformer compile targets are
    detected, compile
    `diffusion_model` anyway.

---

## Recommended starting points

If you're unsure where to begin:

- Backend: `inductor`
- `disable_cudagraphs`: `true` (recommended default for stability)
- Mode: `default`
- `fullgraph`: `false`
- `dynamic`: `false`
- `compile_transformer_only`: `true` for FLUX-style models (including
  FLUX.2 KLEIN 9B) to target leaf heavy modules; otherwise start with `false`
  and compare.

Then iterate one setting at a time based on stability and speed.

---

## Performance and stability notes

- First generation after graph changes can be slower (compile warm-up).
- Recompiles can occur when shapes/settings change substantially.
- Best backend can differ across GPUs, drivers, and PyTorch versions.
- If a backend fails, switch to `inductor` first.
- If you encounter `cudaMallocAsync`/cudagraph issues, keep
  `disable_cudagraphs = true`.

- Guard filtering: when ComfyUI's `skip_torch_compile_dict` is available, this
  node passes it as `options['guard_filter_fn']` to reduce guard churn on
  dynamic `transformer_options` dictionaries.
- If full-model compile is unstable on your graph, enable
  `compile_transformer_only`.

---

## Troubleshooting quick guide

- **Issue:** Backend error on startup or first sample.
  - Try `backend = inductor`.
  - Keep `disable_cudagraphs = true`.

- **Issue:** Very long first run.
  - Expected in compile warm-up; test subsequent runs before evaluating.

- **Issue:** Unstable behavior with full model compile.
  - Set `compile_transformer_only = true`.

- **Issue:** CUDA backend unavailable.
  - Use `inductor`, or verify CUDA-compatible PyTorch/GPU environment.
