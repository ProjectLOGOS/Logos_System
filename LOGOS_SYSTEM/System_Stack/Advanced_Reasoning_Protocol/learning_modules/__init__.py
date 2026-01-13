"""Entry points for ARP learning module utilities."""

from importlib import import_module


_pytorch = import_module(".pytorch_ml_adapters", package=__name__)
_ml_components = import_module(".ml_components", package=__name__)
_deep_adapter = import_module(".deep_learning_adapter", package=__name__)

UnifiedTorchAdapter = _pytorch.UnifiedTorchAdapter
FeatureExtractor = _ml_components.FeatureExtractor
DeepLearningAdapter = _deep_adapter.DeepLearningAdapter

__all__ = [
    "UnifiedTorchAdapter",
    "FeatureExtractor",
    "DeepLearningAdapter",
]
