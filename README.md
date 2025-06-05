# LoRA-Safe TorchCompile Node for ComfyUI

Drop-in replacement for the stock `TorchCompileModel` node.  
Compiles after all LoRA / TEA-Cache / Sage-Attention patches, so LoRAs stay
active while you still get the compile speed-up.

## Install
1. Clone / download this repo.
2. Copy **lora_safe_compile** into `ComfyUI/custom_nodes/`.
3. Restart ComfyUI and search for **TorchCompileModel_LoRASafe** under
   *model / optimisation 🛠️*.
