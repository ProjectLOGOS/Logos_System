"""Lightweight machine-learning helpers with optional dependencies."""

import importlib
import logging
from typing import Any, Dict, List, Sequence

import numpy as np

from .translation_engine import TranslationVector, translate

logger = logging.getLogger(__name__)


def _load_optional(name: str):
    try:  # pragma: no cover - optional dependency
        return importlib.import_module(name)
    except ModuleNotFoundError:  # pragma: no cover - fallback path
        return None


_sk_cluster = _load_optional("sklearn.cluster")
_sk_decomposition = _load_optional("sklearn.decomposition")
_sk_ensemble = _load_optional("sklearn.ensemble")
_sk_text = _load_optional("sklearn.feature_extraction.text")

_sk_modules = (_sk_cluster, _sk_decomposition, _sk_ensemble, _sk_text)

if all(module is not None for module in _sk_modules):
    DBSCAN = getattr(_sk_cluster, "DBSCAN")
    TruncatedSVD = getattr(_sk_decomposition, "TruncatedSVD")
    RandomForestRegressor = getattr(_sk_ensemble, "RandomForestRegressor")
    TfidfVectorizer = getattr(_sk_text, "TfidfVectorizer")
    SKLEARN_AVAILABLE = True
else:
    DBSCAN = None  # type: ignore
    TruncatedSVD = None  # type: ignore
    RandomForestRegressor = None  # type: ignore
    TfidfVectorizer = None  # type: ignore
    SKLEARN_AVAILABLE = False

_umap_module = _load_optional("umap")
UMAP_AVAILABLE = _umap_module is not None


def _basic_text_embedding(
    texts: Sequence[str],
    n_components: int,
) -> np.ndarray:
    """Generate deterministic embeddings without external libraries."""

    def features(text: str) -> List[float]:
        stripped = text.strip()
        length = len(stripped)
        vowel_ratio = (
            sum(stripped.lower().count(v) for v in "aeiou") / length
            if length
            else 0.0
        )
        consonant_ratio = (
            sum(ch.isalpha() for ch in stripped) / length
            if length
            else 0.0
        )
        digit_ratio = (
            sum(ch.isdigit() for ch in stripped) / length
            if length
            else 0.0
        )
        return [length, vowel_ratio, consonant_ratio, digit_ratio]

    matrix = np.array([features(text) for text in texts], dtype=float)
    if not matrix.size:
        return np.zeros((0, max(1, n_components)), dtype=float)

    if matrix.shape[1] >= n_components:
        return matrix[:, :n_components]

    pad_width = n_components - matrix.shape[1]
    padding = np.zeros((matrix.shape[0], pad_width), dtype=float)
    return np.hstack((matrix, padding))


class FeatureExtractor:
    """Combine ontological axes with text embeddings."""

    def __init__(self, n_components: int = 50):
        self.n_components = n_components
        self._fitted = False

        if SKLEARN_AVAILABLE:
            self.vectorizer = TfidfVectorizer(max_features=1000)
            self.svd = TruncatedSVD(n_components=n_components)
        else:
            self.vectorizer = None
            self.svd = None
            logger.warning(
                "scikit-learn not available; using basic embeddings"
            )

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        if SKLEARN_AVAILABLE and self.vectorizer and self.svd:
            tfidf = self.vectorizer.fit_transform(texts)
            emb = self.svd.fit_transform(tfidf)
            self._fitted = True
            return emb

        self._fitted = True
        return _basic_text_embedding(texts, self.n_components)

    def transform(self, texts: List[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("FeatureExtractor: call fit_transform() first")

        if SKLEARN_AVAILABLE and self.vectorizer and self.svd:
            tfidf = self.vectorizer.transform(texts)
            return self.svd.transform(tfidf)

        return _basic_text_embedding(texts, self.n_components)

    def extract(self, payloads: List[Any]) -> np.ndarray:
        texts = [
            item if isinstance(item, str) else item.get("text", "")
            for item in payloads
        ]

        if not self._fitted:
            text_embs = self.fit_transform(texts)
        else:
            text_embs = self.transform(texts)

        axes = []
        for item in payloads:
            text = item if isinstance(item, str) else item.get("text", "")
            translation: TranslationVector = translate(text)
            axes.append([
                translation.existence,
                translation.goodness,
                translation.truth,
            ])

        axes_arr = (
            np.array(axes, dtype=float)
            if axes
            else np.zeros((0, 3), dtype=float)
        )
        return np.hstack([axes_arr, text_embs]) if axes_arr.size else text_embs


class ClusterAnalyzer:
    """Dimensionality reduction and clustering helper."""

    def __init__(
        self,
        eps: float = 0.5,
        min_samples: int = 5,
        n_neighbors: int = 15,
    ):
        if UMAP_AVAILABLE and _umap_module is not None:
            self.reducer = _umap_module.UMAP(
                n_neighbors=n_neighbors,
                min_dist=0.1,
            )
        elif SKLEARN_AVAILABLE and TruncatedSVD is not None:
            self.reducer = TruncatedSVD(n_components=2)
        else:
            self.reducer = None

        if SKLEARN_AVAILABLE and DBSCAN is not None:
            self.clusterer = DBSCAN(eps=eps, min_samples=min_samples)
        else:
            self.clusterer = None

    def fit(self, features: np.ndarray) -> Dict[str, np.ndarray]:
        if self.reducer is None:
            emb2d = features[:, :2] if features.shape[1] >= 2 else features
        else:
            emb2d = self.reducer.fit_transform(features)

        if self.clusterer is None:
            labels = np.zeros(features.shape[0], dtype=int)
        else:
            labels = self.clusterer.fit_predict(emb2d)

        return {"embedding_2d": emb2d, "labels": labels}

    def find_gaps(self, labels: np.ndarray) -> np.ndarray:
        if self.clusterer is None:
            return np.array([], dtype=int)
        return np.where(labels == -1)[0]


class NextNodePredictor:
    """Predict the next coordinate in trinity space."""

    def __init__(self):
        if SKLEARN_AVAILABLE and RandomForestRegressor is not None:
            self.model = RandomForestRegressor(n_estimators=50)
        else:
            self.model = None
            self._centroid = np.zeros(3)

    def train(self, X: np.ndarray, y: np.ndarray):
        if self.model is None:
            self._centroid = y.mean(axis=0) if len(y) else np.zeros(3)
            return

        self.model.fit(X, y)

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.model is None:
            return np.repeat(
                self._centroid.reshape(1, -1),
                features.shape[0],
                axis=0,
            )

        return self.model.predict(features)
