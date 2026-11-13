"""histomatch: match coronal brain slab photographs to the T1 MRI they were cut from."""
__version__ = "0.1.0"
import torch as _torch

# Let the CUDA allocator grow its segments instead of reserving a fresh block for every larger
# request: LoFTR calls of varying size otherwise leave gigabytes of fragmented, unusable cache,
# and on WSL that cache silently spills into system memory and the run crawls. Must be set
# before the first tensor lands on the GPU. (The PYTORCH_ALLOC_CONF environment variable is
# not honoured by every torch build, the setter is.)
try:
    _torch.cuda.memory._set_allocator_settings("expandable_segments:True")
except Exception:       # CPU-only torch, or an old build without the setting
    pass
