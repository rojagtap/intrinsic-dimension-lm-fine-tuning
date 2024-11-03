import evaluate
import torch

from ..wrappers.modeling_embedding import EmbeddingSubspaceWrapper
from ..wrappers.modeling_layernorm import LayerNormSubspaceWrapper
from ..wrappers.modeling_linear import LinearSubspaceWrapper

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

LAYER_MAP = {
    torch.nn.Linear: LinearSubspaceWrapper,
    torch.nn.Embedding: EmbeddingSubspaceWrapper,
    torch.nn.LayerNorm: LayerNormSubspaceWrapper
}


METRIC = evaluate.load("accuracy")
