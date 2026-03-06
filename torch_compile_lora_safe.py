import logging
import torch
import torch.nn as nn
import threading

_LOG = logging.getLogger(__name__)

try:
    from comfy_api.torch_helpers import set_torch_compile_wrapper as _set_torch_compile_wrapper
except Exception:
    _set_torch_compile_wrapper = None


class _LazyCompiled(nn.Module):
    """
    Wraps any nn.Module (e.g. a UNet or a single transformer block)
    and replaces its forward pass with torch.compile **lazily** on the
    first call.  All other attributes / helpers are proxied so that the
    module behaves like the original one.
    """

    def __init__(self, module: nn.Module, **compile_kw):
        super().__init__()
        self._orig = module
        self._compiled = None
        self._compile_kw = compile_kw
        self._compile_lock = threading.Lock()

    def __getattr__(self, name):
        if name in {"_orig", "_compiled", "_compile_kw", "_compile_lock"}:
            return super().__getattr__(name)
        return getattr(self._orig, name)

    def modules(self):
        return self._orig.modules()

    def children(self):
        return self._orig.children()

    def named_modules(self, *a, **k):
        return self._orig.named_modules(*a, **k)

    def state_dict(self, *a, **k):
        return self._orig.state_dict(*a, **k)

    @property
    def dtype(self):
        return self._orig.dtype

    @property
    def device(self):
        return self._orig.device

    def train(self, mode: bool = True):
        self._orig.train(mode)
        return super().train(mode)

    def eval(self):
        return self.train(False)

    def to(self, *args, **kwargs):
        self._orig.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def forward(self, *args, **kwargs):
        if self._compiled is None:
            with self._compile_lock:
                if self._compiled is None:
                    self._compiled = torch.compile(self._orig, **self._compile_kw)
        return self._compiled(*args, **kwargs)


class TorchCompileModel_LoRASafe:
    """LoRA-safe torch.compile that also works with Flux block layouts."""

    _BLOCK_LAYER_TYPES = (
        "double_blocks",
        "single_blocks",
        "layers",
        "transformer_blocks",
        "blocks",
        "visual_transformer_blocks",
        "text_transformer_blocks",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "backend": (["inductor", "cudagraphs", "nvfuser"],),
                "mode": (["default", "reduce-overhead", "max-autotune"],),
                "fullgraph": ("BOOLEAN", {"default": False}),
                "dynamic": ("BOOLEAN", {"default": False}),
                "compile_transformer_only": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "True -> compile known transformer block lists only. "
                        "Flux models use double_blocks/single_blocks; "
                        "SD-style models often use transformer_blocks/blocks.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "model/optimisation 🛠️"
    EXPERIMENTAL = True

    @staticmethod
    def _check_backend(backend: str):
        if backend == "nvfuser" and not torch.cuda.is_available():
            raise ValueError("nvfuser backend requires a CUDA GPU. " "Select 'inductor' instead.")
        if backend == "cudagraphs":
            if not torch.cuda.is_available():
                raise ValueError("cudagraphs backend needs a CUDA GPU.")
            cap = torch.cuda.get_device_capability()
            if cap[0] < 7:
                raise ValueError(
                    "cudagraphs works reliably on GPUs with "
                    "compute capability 7.0 or higher "
                    f"(yours is {cap[0]}.{cap[1]})."
                )

    @classmethod
    def _discover_compile_keys(cls, diffusion_model):
        keys = []
        for layer_name in cls._BLOCK_LAYER_TYPES:
            if hasattr(diffusion_model, layer_name):
                blocks = getattr(diffusion_model, layer_name)
                if not isinstance(blocks, (nn.ModuleList, list, tuple)):
                    continue
                for i in range(len(blocks)):
                    keys.append(f"diffusion_model.{layer_name}.{i}")
        return list(dict.fromkeys(keys))

    @staticmethod
    def _build_compile_kwargs(backend, mode, fullgraph, dynamic):
        compile_kw = dict(
            backend=backend,
            fullgraph=fullgraph,
            dynamic=dynamic,
        )
        if mode != "default":
            compile_kw["mode"] = mode
        return compile_kw

    def patch(self, model, backend, mode, fullgraph, dynamic, compile_transformer_only):
        self._check_backend(backend)

        m = model.clone()
        dm = m.get_model_object("diffusion_model")

        compile_kw = self._build_compile_kwargs(backend, mode, fullgraph, dynamic)

        if compile_transformer_only:
            compile_keys = self._discover_compile_keys(dm)
            if not compile_keys:
                _LOG.warning(
                    "TorchCompileModel_LoRASafe: no known transformer block lists found; "
                    "falling back to whole diffusion model compile."
                )
                compile_keys = ["diffusion_model"]
        else:
            compile_keys = ["diffusion_model"]

        if _set_torch_compile_wrapper is not None:
            try:
                _set_torch_compile_wrapper(model=m, keys=compile_keys, **compile_kw)
                return (m,)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to apply torch.compile wrapper via comfy_api.torch_helpers: {e}"
                ) from e

        for key in compile_keys:
            if key == "diffusion_model":
                target = dm
            else:
                obj = dm
                for part in key.split(".")[1:]:
                    obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
                target = obj
            m.add_object_patch(key, _LazyCompiled(target, **compile_kw))

        return (m,)


NODE_CLASS_MAPPINGS = {
    "TorchCompileModel_LoRASafe": TorchCompileModel_LoRASafe,
}
