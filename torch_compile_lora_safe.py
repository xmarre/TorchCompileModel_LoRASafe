# Torch 2.3-ready “LoRA-safe” compile node for ComfyUI
# ---------------------------------------------------
# – Fixes accidental `mode=None`
# – Ensures .train() / .eval() are forwarded to the real UNet
# – Handles concurrent first-calls safely with a lock
# – Performs a quick backend sanity-check so users get friendly errors
#
# Drop this file in your ComfyUI/custom_nodes folder and restart.

import torch
import torch.nn as nn
import threading


# --------------------------------------------------------------------- #
#  Helper – transparent wrapper that compiles itself on first forward()
# --------------------------------------------------------------------- #
class _LazyCompiled(nn.Module):
    """
    Wraps any nn.Module (e.g. a UNet or a single transformer block)
    and replaces its forward pass with torch.compile **lazily** on the
    first call.  All other attributes / helpers are proxied so that the
    module behaves like the original one.
    """
    def __init__(self, module: nn.Module, **compile_kw):
        super().__init__()
        self._orig        = module
        self._compiled    = None
        self._compile_kw  = compile_kw
        self._compile_lock = threading.Lock()   # avoid double-compile races

    # ---------- Attribute & module proxies -------------------------------- #
    def __getattr__(self, name):
        if name in {"_orig", "_compiled", "_compile_kw", "_compile_lock"}:
            return super().__getattr__(name)
        return getattr(self._orig, name)        # delegate to real module

    def modules(self):         return self._orig.modules()
    def children(self):        return self._orig.children()
    def named_modules(self, *a, **k): return self._orig.named_modules(*a, **k)
    def state_dict(self,  *a, **k): return self._orig.state_dict(*a, **k)

    # dtype / device queries that ComfyUI calls during sampler setup
    @property
    def dtype(self):  return self._orig.dtype
    @property
    def device(self): return self._orig.device

    # ---------- Training / device helpers --------------------------------- #
    def train(self, mode: bool = True):
        # keep both wrapper *and* wrapped module in sync
        self._orig.train(mode)
        return super().train(mode)

    def eval(self):         # convenience alias
        return self.train(False)

    def to(self, *args, **kwargs):
        self._orig.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    # ---------- first call → compile -------------------------------------- #
    def forward(self, *args, **kwargs):
        if self._compiled is None:
            # Only one thread actually compiles
            with self._compile_lock:
                if self._compiled is None:
                    self._compiled = torch.compile(self._orig, **self._compile_kw)
        return self._compiled(*args, **kwargs)


# --------------------------------------------------------------------- #
#                       ComfyUI node definition
# --------------------------------------------------------------------- #
class TorchCompileModel_LoRASafe:
    """LoRA-safe torch.compile with extra options."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),

                # same four knobs as the stock node
                "backend": (["inductor", "cudagraphs", "nvfuser"],),
                "mode":    (["default", "reduce-overhead", "max-autotune"],),
                "fullgraph": ("BOOLEAN", {"default": False}),
                "dynamic":   ("BOOLEAN", {"default": False}),

                # replicate compile_transformer_block_only
                "compile_transformer_only": (
                    "BOOLEAN",
                    {"default": False,
                     "tooltip":
                     "True → compile each transformer block lazily; "
                     "False → compile whole UNet once"}
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION     = "patch"
    CATEGORY     = "model/optimisation 🛠️"
    EXPERIMENTAL = True

    # ----------------------------------------------------------------- #
    @staticmethod
    def _check_backend(backend: str):
        """Raise a friendly error if the chosen backend cannot run."""
        if backend == "nvfuser" and not torch.cuda.is_available():
            raise ValueError("nvfuser backend requires a CUDA GPU. "
                             "Select 'inductor' instead.")
        if backend == "cudagraphs":
            if not torch.cuda.is_available():
                raise ValueError("cudagraphs backend needs a CUDA GPU.")
            cap = torch.cuda.get_device_capability()
            if cap[0] < 7:
                raise ValueError("cudagraphs works reliably on GPUs with "
                                 "compute capability 7.0 or higher "
                                 f"(yours is {cap[0]}.{cap[1]}).")

    # ----------------------------------------------------------------- #
    def patch(self,
              model, backend, mode,
              fullgraph, dynamic,
              compile_transformer_only):

        # backend sanity-check before we go any further
        self._check_backend(backend)

        m  = model.clone()                              # don’t mutate input
        dm = m.get_model_object("diffusion_model")      # real UNet

        # build compile() kwargs
        compile_kw = dict(
            backend   = backend,
            fullgraph = fullgraph,
            dynamic   = dynamic,
        )
        if mode != "default":                           # fix for mode=None
            compile_kw["mode"] = mode

        # ---------- A) whole-UNet compile (default) ------------------- #
        if not compile_transformer_only:
            m.add_object_patch(
                "diffusion_model",
                _LazyCompiled(dm, **compile_kw)
            )
            return (m,)

        # ---------- B) transformer-only compile ----------------------- #
        if hasattr(dm, "transformer_blocks"):
            for i, blk in enumerate(dm.transformer_blocks):
                m.add_object_patch(
                    f"diffusion_model.transformer_blocks.{i}",
                    _LazyCompiled(blk, **compile_kw)
                )
        else:  # fallback – compile whole UNet
            m.add_object_patch(
                "diffusion_model",
                _LazyCompiled(dm, **compile_kw)
            )
        return (m,)


# --------------------------------------------------------------------- #
#                     ComfyUI registration shim
# --------------------------------------------------------------------- #
NODE_CLASS_MAPPINGS = {
    "TorchCompileModel_LoRASafe": TorchCompileModel_LoRASafe,
}
