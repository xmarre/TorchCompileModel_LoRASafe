import logging
import os
import threading

import torch
import torch.nn as nn

_LOG = logging.getLogger(__name__)

_DEFAULT_FORCE_PARAMETER_STATIC_SHAPES = getattr(
    torch._dynamo.config, "force_parameter_static_shapes", None
)
_DEFAULT_GRAPH_PARTITION = getattr(torch._inductor.config, "graph_partition", None)

try:
    from comfy_api.torch_helpers import set_torch_compile_wrapper as _set_torch_compile_wrapper
except Exception:
    _set_torch_compile_wrapper = None

try:
    from comfy_extras.nodes_torch_compile import skip_torch_compile_dict as _skip_torch_compile_dict
except Exception:
    _skip_torch_compile_dict = None


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

    _GENERIC_BLOCK_LAYER_TYPES = (
        "layers",
        "transformer_blocks",
        "blocks",
        "visual_transformer_blocks",
        "text_transformer_blocks",
    )

    _FLUX_DOUBLE_PREFERRED_SUFFIXES = (
        "img_attn.qkv",
        "img_attn.proj",
        "txt_attn.qkv",
        "txt_attn.proj",
    )
    _FLUX_DOUBLE_CUDAGRAPH_SAFE_SUFFIXES = (
        "img_attn.qkv",
        "txt_attn.qkv",
    )

    _FLUX_SINGLE_PREFERRED_SUFFIXES = ("linear1", "linear2")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "backend": (["inductor", "cudagraphs", "nvfuser"],),
                "mode": (["default", "reduce-overhead", "max-autotune"],),
                "fullgraph": ("BOOLEAN", {"default": False}),
                "dynamic": ("BOOLEAN", {"default": False}),
                "disable_cudagraphs": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "True -> pass torch.compile options {'triton.cudagraphs': False} "
                        "for inductor backend. Useful to avoid cudaMallocAsync/cudagraph issues "
                        "in some environments.",
                    },
                ),
                "compile_transformer_only": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "True -> compile discovered transformer targets only. "
                        "Flux models compile leaf submodules inside double_blocks/single_blocks; "
                        "SD-style models often compile transformer_blocks/blocks.",
                    },
                ),
                "debug_torch_logs": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Enable PyTorch compile/cudagraph debug logs via "
                        "torch._logging.set_logs (ignored if TORCH_LOGS env var is already set).",
                    },
                ),
                "skip_dynamic_cudagraphs": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "For cudagraph-capable runs, skip cudagraphing functions with "
                        "dynamic-shape inputs. Recommended for Flux.",
                    },
                ),
                "allow_flux_inferred_targets": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "False (recommended) -> Flux compile targets are strict and limited "
                        "to qkv/proj/linear1/linear2 leaves only. "
                        "True -> also allow inferred Flux target names for compatibility.",
                    },
                ),
                "fallback_to_full_model_if_no_targets": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "If transformer-only target discovery finds nothing, "
                        "fall back to compiling diffusion_model. "
                        "False keeps execution uncompiled instead of widening target scope.",
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

    @staticmethod
    def _is_leaf_compile_module(module: nn.Module) -> bool:
        if not isinstance(module, nn.Module):
            return False
        if any(True for _ in module.children()):
            return False
        return any(True for _ in module.parameters(recurse=False))

    @classmethod
    def _discover_flux_all_leaf_keys(cls, diffusion_model: nn.Module):
        keys = []
        for name, module in diffusion_model.named_modules():
            if not name or not cls._is_leaf_compile_module(module):
                continue
            keys.append(f"diffusion_model.{name}")
        return list(dict.fromkeys(keys))

    @staticmethod
    def _is_flux_model(diffusion_model: nn.Module) -> bool:
        return hasattr(diffusion_model, "double_blocks") and hasattr(
            diffusion_model, "single_blocks"
        )

    @classmethod
    def _try_preferred_suffixes(cls, block: nn.Module, suffixes):
        targets = []
        for suffix in suffixes:
            obj = block
            ok = True
            for part in suffix.split("."):
                if not hasattr(obj, part):
                    ok = False
                    break
                obj = getattr(obj, part)
            if ok and cls._is_leaf_compile_module(obj):
                targets.append(suffix)
        return targets

    @classmethod
    def _discover_flux_double_block_suffixes(
        cls,
        block: nn.Module,
        allow_inferred_targets: bool = False,
        cudagraph_capable: bool = False,
    ):
        preferred_suffixes = (
            cls._FLUX_DOUBLE_CUDAGRAPH_SAFE_SUFFIXES
            if cudagraph_capable
            else cls._FLUX_DOUBLE_PREFERRED_SUFFIXES
        )
        preferred = cls._try_preferred_suffixes(block, preferred_suffixes)
        if preferred or not allow_inferred_targets:
            return preferred

        inferred = []
        for name, module in block.named_modules():
            if not name or not cls._is_leaf_compile_module(module):
                continue
            lname = name.lower()
            parts = lname.split(".")
            tail = parts[-1]
            if (
                (("img_attn" in lname or "txt_attn" in lname) and tail in {"qkv", "proj"})
                or ("img_mlp" in lname)
                or ("txt_mlp" in lname)
            ):
                inferred.append(name)
        return list(dict.fromkeys(inferred))

    @classmethod
    def _discover_flux_single_block_suffixes(
        cls,
        block: nn.Module,
        allow_inferred_targets: bool = False,
        cudagraph_capable: bool = False,
    ):
        if cudagraph_capable:
            return []

        preferred = cls._try_preferred_suffixes(block, cls._FLUX_SINGLE_PREFERRED_SUFFIXES)
        if preferred or not allow_inferred_targets:
            return preferred

        inferred = []
        for name, module in block.named_modules():
            if not name or not cls._is_leaf_compile_module(module):
                continue
            lname = name.lower()
            if lname.endswith("linear1") or lname.endswith("linear2"):
                inferred.append(name)
        return list(dict.fromkeys(inferred))

    @classmethod
    def _discover_compile_keys(
        cls,
        diffusion_model,
        allow_flux_inferred_targets: bool = False,
        cudagraph_capable: bool = False,
    ):
        keys = []

        # Flux special-case: compile only leaf tensor-heavy modules, not full block forward.
        if cls._is_flux_model(diffusion_model):
            double_blocks = getattr(diffusion_model, "double_blocks")
            single_blocks = getattr(diffusion_model, "single_blocks")

            if isinstance(double_blocks, (nn.ModuleList, list, tuple)):
                for i, block in enumerate(double_blocks):
                    for suffix in cls._discover_flux_double_block_suffixes(
                        block,
                        allow_inferred_targets=allow_flux_inferred_targets,
                        cudagraph_capable=cudagraph_capable,
                    ):
                        keys.append(f"diffusion_model.double_blocks.{i}.{suffix}")

            if isinstance(single_blocks, (nn.ModuleList, list, tuple)):
                for i, block in enumerate(single_blocks):
                    for suffix in cls._discover_flux_single_block_suffixes(
                        block,
                        allow_inferred_targets=allow_flux_inferred_targets,
                        cudagraph_capable=cudagraph_capable,
                    ):
                        keys.append(f"diffusion_model.single_blocks.{i}.{suffix}")

            if keys:
                return list(dict.fromkeys(keys))

        for layer_name in cls._GENERIC_BLOCK_LAYER_TYPES:
            if hasattr(diffusion_model, layer_name):
                blocks = getattr(diffusion_model, layer_name)
                if not isinstance(blocks, (nn.ModuleList, list, tuple)):
                    continue
                for i in range(len(blocks)):
                    keys.append(f"diffusion_model.{layer_name}.{i}")
        return list(dict.fromkeys(keys))

    @staticmethod
    def _build_compile_kwargs(backend, mode, fullgraph, dynamic, disable_cudagraphs):
        compile_kw = dict(
            backend=backend,
            fullgraph=fullgraph,
            dynamic=dynamic,
        )
        if mode != "default":
            compile_kw["mode"] = mode

        options = {}
        if _skip_torch_compile_dict is not None:
            options["guard_filter_fn"] = _skip_torch_compile_dict
        if disable_cudagraphs and backend == "inductor":
            options["triton.cudagraphs"] = False
        if options:
            compile_kw["options"] = options

        return compile_kw

    @staticmethod
    def _configure_torch_logs(debug_torch_logs: bool):
        if not debug_torch_logs:
            return
        # TORCH_LOGS takes precedence over torch._logging.set_logs(...).
        if os.environ.get("TORCH_LOGS"):
            _LOG.warning(
                "TORCH_LOGS is already set; using environment logging instead of "
                "torch._logging.set_logs."
            )
            return
        try:
            torch._logging.set_logs(
                dynamo=logging.INFO,
                inductor=logging.INFO,
                cudagraphs=True,
                cudagraph_static_inputs=True,
                perf_hints=True,
                graph_breaks=True,
                guards=True,
                recompiles=True,
                recompiles_verbose=True,
            )
        except Exception as e:
            _LOG.warning(f"Failed to enable torch logging: {e}")

    def patch(
        self,
        model,
        backend,
        mode,
        fullgraph,
        dynamic,
        disable_cudagraphs,
        compile_transformer_only,
        debug_torch_logs=False,
        skip_dynamic_cudagraphs=True,
        allow_flux_inferred_targets=False,
        fallback_to_full_model_if_no_targets=False,
    ):
        self._check_backend(backend)

        self._configure_torch_logs(debug_torch_logs)

        cudagraph_capable = backend == "cudagraphs" or (
            backend == "inductor" and not disable_cudagraphs
        )

        if _DEFAULT_FORCE_PARAMETER_STATIC_SHAPES is not None:
            torch._dynamo.config.force_parameter_static_shapes = (
                False if cudagraph_capable else _DEFAULT_FORCE_PARAMETER_STATIC_SHAPES
            )
        if _DEFAULT_GRAPH_PARTITION is not None:
            torch._inductor.config.graph_partition = (
                True if cudagraph_capable else _DEFAULT_GRAPH_PARTITION
            )

        # This flag is process-global, so set it on every call to match the
        # current node input instead of leaking state from earlier runs.
        if backend in ("inductor", "cudagraphs"):
            torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = bool(
                skip_dynamic_cudagraphs
            )

        try:
            m = model.clone(disable_dynamic=True)
        except TypeError:
            m = model.clone()
        dm = m.get_model_object("diffusion_model")

        compile_kw = self._build_compile_kwargs(
            backend, mode, fullgraph, dynamic, disable_cudagraphs
        )

        is_flux_model = self._is_flux_model(dm)

        if compile_transformer_only:
            compile_keys = self._discover_compile_keys(
                dm,
                allow_flux_inferred_targets=(
                    allow_flux_inferred_targets and not cudagraph_capable
                ),
                cudagraph_capable=cudagraph_capable,
            )
            if not compile_keys:
                if fallback_to_full_model_if_no_targets and not (
                    cudagraph_capable and is_flux_model
                ):
                    _LOG.warning(
                        "TorchCompileModel_LoRASafe: no known transformer compile targets found; "
                        "falling back to whole diffusion model compile."
                    )
                    compile_keys = ["diffusion_model"]
                else:
                    if cudagraph_capable and is_flux_model:
                        _LOG.warning(
                            "TorchCompileModel_LoRASafe: no cudagraph-safe Flux transformer "
                            "compile targets found; skipping compile instead of widening target scope."
                        )
                    else:
                        _LOG.warning(
                            "TorchCompileModel_LoRASafe: no known transformer compile targets found; "
                            "skipping compile instead of falling back to whole model."
                        )
                    return (m,)
        elif cudagraph_capable and is_flux_model:
            compile_keys = self._discover_flux_all_leaf_keys(dm)
            if not compile_keys:
                _LOG.warning(
                    "TorchCompileModel_LoRASafe: Flux full-leaf cudagraph compile requested, "
                    "but no leaf targets were found; skipping compile instead of compiling the "
                    "whole WrapperExecutor-driven diffusion_model path."
                )
                return (m,)
            _LOG.warning(
                "TorchCompileModel_LoRASafe: Flux whole-model cudagraph compile requested; "
                "using wide leaf-module compile instead to avoid WrapperExecutor / "
                "torch.compiler.disable replay failures."
            )
        else:
            compile_keys = ["diffusion_model"]

        if cudagraph_capable and is_flux_model:
            label = (
                "Flux cudagraph transformer targets"
                if compile_transformer_only
                else "Flux cudagraph wide-leaf targets"
            )
            _LOG.warning("%s: %s", label, compile_keys[:16])
            _LOG.warning("%s count: %d", label, len(compile_keys))

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



