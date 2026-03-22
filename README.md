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

Instead, when `compile_transformer_only = true`, Flux targets strict leaf
submodules by default.

Strict Flux names targeted by default:

- `double_blocks.{i}.img_attn.qkv`
- `double_blocks.{i}.img_attn.proj`
- `double_blocks.{i}.txt_attn.qkv`
- `double_blocks.{i}.txt_attn.proj`
- `single_blocks.{i}.linear1`
- `single_blocks.{i}.linear2`

For cudagraph-capable Flux runs (`backend = cudagraphs`, or `backend = inductor`
with `disable_cudagraphs = false`), the behavior splits into two safe modes:

- `compile_transformer_only = true`
  - target set is narrowed to the cudagraph-safe qkv subset only:
    - `double_blocks.{i}.img_attn.qkv`
    - `double_blocks.{i}.txt_attn.qkv`
- `compile_transformer_only = false`
  - the node does **not** compile the outer `diffusion_model` wrapper path.
  - instead, it widens to a **Flux full-leaf compile** across the model's
    parameterized leaf modules.

This avoids full-model cudagraph capture through Flux / ComfyUI
`WrapperExecutor` paths, which are prone to graph-break / replay failures when
runtime patches such as Sage attention are active.

In cudagraph-capable Flux mode, inferred Flux targets and full-model fallback
are intentionally disabled to avoid widening back into unstable outer-wrapper
paths.

Optional widening controls:

- `allow_flux_inferred_targets` (`false` by default)
  - `false`: use only strict names above.
  - `true`: allow inferred Flux leaf targets (including `img_mlp/txt_mlp`
    pattern matches) for compatibility across variants.
  - Ignored for cudagraph-capable Flux runs, which stay on the strict qkv-only
    set.

- `fallback_to_full_model_if_no_targets` (`false` by default)
  - `false`: if transformer-only target discovery finds nothing, skip compile
    instead of silently widening scope.
  - `true`: if discovery finds nothing, fall back to compiling
    `diffusion_model`.
  - Ignored for cudagraph-capable Flux runs, which skip compile instead of
    widening scope.

For non-Flux transformer-style models, it compiles known block containers such
as:

- `layers`
- `transformer_blocks`
- `blocks`
- `visual_transformer_blocks`
- `text_transformer_blocks`

If no known transformer compile targets are found, behavior is controlled by
`fallback_to_full_model_if_no_targets` (default `false`, meaning skip compile).

---

## Installation

1. Clone or download this repository.
2. Copy the `lora_safe_compile` folder into your ComfyUI `custom_nodes/` directory.
3. Restart ComfyUI.
4. Search for **TorchCompileModel_LoRASafe** in:
   **model / optimisation 🛠️**

or search for this node in Comfyui-Manager and install.

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
  - Set to `false` only when you specifically want Inductor cudagraphs.

- `compile_transformer_only` (`BOOLEAN`)
  - `false` (default): compile the entire `diffusion_model` for non-Flux
    models.
  - `true`: compile discovered transformer targets only.
  - Flux behavior (default strict mode): targets only `qkv/proj` attention
    leaves plus `linear1/linear2` leaves.
  - For cudagraph-capable Flux runs:
    - `true` -> narrowed further to `img_attn.qkv` and `txt_attn.qkv` only.
    - `false` -> widens to a Flux **full-leaf compile** instead of compiling
      the outer `diffusion_model` wrapper path.

- `allow_flux_inferred_targets` (`BOOLEAN`, default `false`)
  - `false` (recommended): strict Flux target set only.
  - `true`: allow inferred Flux leaf targets (`img_mlp/txt_mlp`-style matches)
    when preferred names differ.
  - Ignored for cudagraph-capable Flux runs, which stay on the strict qkv-only
    set.

- `fallback_to_full_model_if_no_targets` (`BOOLEAN`, default `false`)
  - `false` (recommended): if transformer-only discovery finds nothing, skip
    compile rather than widening scope.
  - `true`: if transformer-only discovery finds nothing, compile
    `diffusion_model`.
  - Ignored for cudagraph-capable Flux runs, which skip compile instead of
    widening scope.

---

## Recommended starting points

If you're unsure where to begin:

- Backend: `inductor`
- `disable_cudagraphs`: `true` (recommended default for stability). Set to
  `false` only when you specifically want Inductor cudagraphs.
- Mode: `default`
- `fullgraph`: `false`
- `dynamic`: `false`
- `compile_transformer_only`: `true` for FLUX-style models (including
  FLUX.2 KLEIN 9B); otherwise start with `false` and compare.
- `allow_flux_inferred_targets`: `false` (recommended for debugging and stable
  targeting).
- `fallback_to_full_model_if_no_targets`: `false` (recommended to avoid silent
  whole-model escalation).

Then iterate one setting at a time based on stability and speed.

---

## Performance and stability notes

- First generation after graph changes can be slower (compile warm-up).
- Recompiles can occur when shapes/settings change substantially.
- Best backend can differ across GPUs, drivers, and PyTorch versions.
- If a backend fails, switch to `inductor` first.
- If you encounter `cudaMallocAsync`/cudagraph issues, keep
  `disable_cudagraphs = true`.
- For cudagraph-enabled Flux validation, start with qkv-only targeting and do
  not widen to `proj` unless logs prove the path is stable.

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
