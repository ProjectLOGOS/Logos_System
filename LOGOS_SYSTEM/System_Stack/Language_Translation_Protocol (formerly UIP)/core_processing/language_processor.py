"""
core_processing/language_processor.py

Consolidated language processing module for UIP protocol.
Integrates linguistic analysis, advanced NLP, and translation bridge components.
Merges robust features from linguistic_analysis.py, nlp_processor.py, and translation_bridge.py.

Key Features:
- Intent classification and entity extraction with theological/philosophical focus.
- Semantic parsing and triple generation.
- Bidirectional translation between natural language and Lambda Logos ontological representations.
- Trinity vector extraction and handling.
- Fallback mechanisms for unavailable libraries (e.g., spaCy, NLTK).
"""

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

# Optional dependencies; fall back to light-weight mocks when missing.
try:  # pragma: no cover - runtime optional dependency
    import spacy

    SPACY_AVAILABLE = True
except ImportError:  # pragma: no cover - runtime optional dependency
    spacy = None
    SPACY_AVAILABLE = False

try:  # pragma: no cover - runtime optional dependency
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize

    NLTK_AVAILABLE = True
except ImportError:  # pragma: no cover - runtime optional dependency
    stopwords = None
    WordNetLemmatizer = None
    word_tokenize = None
    NLTK_AVAILABLE = False


LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lightweight fallbacks so the module runs even without heavy NLP stacks.
# ---------------------------------------------------------------------------
if not SPACY_AVAILABLE:
    class _DummySpacyDoc:
        """Minimal stand-in for a spaCy Doc."""

        def __init__(self, text: str) -> None:
            self.text = text
            self.ents: List[Any] = []

    class _DummySpacy:
        def __call__(self, text: str) -> "_DummySpacyDoc":
            return _DummySpacyDoc(text)

    def _spacy_load(model: str) -> "_DummySpacy":  # type: ignore[override]
        return _DummySpacy()
else:
    def _spacy_load(model: str):  # pragma: no cover - heavy dependency
        return spacy.load(model)


if not NLTK_AVAILABLE:
    def _word_tokenize(text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())

    class _DummyLemmatizer:
        def lemmatize(self, word: str) -> str:
            return word

    _stop_words = {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "in",
        "on",
        "at",
        "to",
        "of",
        "for",
        "with",
        "by",
    }
else:
    _word_tokenize = word_tokenize  # pragma: no cover - heavy dependency
    _DummyLemmatizer = WordNetLemmatizer  # pragma: no cover - heavy dependency
    _stop_words = set(stopwords.words("english"))  # pragma: no cover - heavy dependency


# ---------------------------------------------------------------------------
# Minimal domain stubs to avoid leaking heavy dependencies into core logic.
# ---------------------------------------------------------------------------
@dataclass
class TrinityVector:
    existence: float = 0.5
    goodness: float = 0.5
    truth: float = 0.5
    confidence: float = 0.5

    def to_dict(self) -> Dict[str, float]:
        return {
            "existence": self.existence,
            "goodness": self.goodness,
            "truth": self.truth,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> "TrinityVector":
        return cls(
            existence=data.get("existence", 0.5),
            goodness=data.get("goodness", 0.5),
            truth=data.get("truth", 0.5),
            confidence=data.get("confidence", 0.5),
        )


class LogosExpr:
    """Placeholder for Lambda Logos expressions."""


class LambdaEngine:
    """Minimal lambda-engine façade."""

    def expr_to_dict(self, expr: LogosExpr) -> Dict[str, Any]:
        return {"expression": repr(expr)}


class OntologicalType(Enum):
    EXISTENCE = "𝔼"
    GOODNESS = "𝔾"
    TRUTH = "𝕋"


# ---------------------------------------------------------------------------
# Enumerations and dataclasses aggregated from the legacy modules.
# ---------------------------------------------------------------------------
class IntentCategory(Enum):
    THEOLOGICAL_INQUIRY = "theological"
    PHILOSOPHICAL_QUESTION = "philosophical"
    LOGICAL_ANALYSIS = "logical"
    MODAL_REASONING = "modal"
    TRINITY_ANALYSIS = "trinity"
    ONTOLOGICAL_QUERY = "ontological"
    COMPUTATIONAL_REQUEST = "computational"
    DEFINITION_REQUEST = "definition"
    COMPARISON_REQUEST = "comparison"
    VALIDATION_REQUEST = "validation"
    GENERAL_CONVERSATION = "general"


class EntityType(Enum):
    THEOLOGICAL_CONCEPT = "theological_concept"
    PHILOSOPHICAL_TERM = "philosophical_term"
    MODAL_OPERATOR = "modal_operator"
    LOGICAL_CONNECTIVE = "logical_connective"
    TRINITY_DIMENSION = "trinity_dimension"
    ONTOLOGICAL_CATEGORY = "ontological_category"
    DIVINE_ATTRIBUTE = "divine_attribute"
    PERSON_NAME = "person_name"
    NUMERICAL_VALUE = "numerical"
    TEMPORAL_REFERENCE = "temporal"


class SemanticRelation(Enum):
    IMPLIES = "implies"
    CONTRADICTS = "contradicts"
    ENTAILS = "entails"
    PRESUPPOSES = "presupposes"
    EXEMPLIFIES = "exemplifies"
    ANALOGOUS_TO = "analogous_to"
    PART_OF = "part_of"
    INSTANCE_OF = "instance_of"
    CAUSE_OF = "cause_of"
    DEFINED_BY = "defined_by"


@dataclass
class NamedEntity:
    text: str
    entity_type: EntityType
    start_pos: int
    end_pos: int
    confidence: float
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SemanticTriple:
    subject: str
    relation: SemanticRelation
    obj: str
    confidence: float


@dataclass
class IntentClassification:
    primary_intent: IntentCategory
    confidence: float
    secondary_intents: List[Tuple[IntentCategory, float]] = field(default_factory=list)
    reasoning: str = ""


@dataclass
class SemanticParse:
    logical_form: str
    confidence: float
    modal_operators: List[str] = field(default_factory=list)
    quantifiers: List[str] = field(default_factory=list)
    theological_concepts: List[str] = field(default_factory=list)


@dataclass
class NLPProcessingResult:
    original_text: str
    intent_classification: IntentClassification
    named_entities: List[NamedEntity]
    semantic_triples: List[SemanticTriple]
    semantic_parse: Optional[SemanticParse]
    sentiment_analysis: Dict[str, float]
    complexity_metrics: Dict[str, float]
    trinity_vector_mapping: TrinityVector
    theological_analysis: Dict[str, Any]
    processing_metadata: Dict[str, Any]


@dataclass
class TranslationResult:
    query: str
    trinity_vector: TrinityVector
    layers: Dict[str, Any] = field(default_factory=lambda: {"SIGN": [], "MIND": {}, "BRIDGE": {}})

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "trinity_vector": self.trinity_vector.to_dict(),
            "layers": self.layers,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TranslationResult":
        return cls(
            query=data.get("query", ""),
            trinity_vector=TrinityVector.from_dict(data.get("trinity_vector", {})),
            layers=data.get("layers", {}),
        )


# ---------------------------------------------------------------------------
# Consolidated processing pipeline.
# ---------------------------------------------------------------------------
class LanguageProcessor:
    """Merged linguistic, NLP, and translation logic for the UIP."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.logger = LOGGER.getChild(self.__class__.__name__)
        self.config = config or {}

        model_name = self.config.get("spacy_model", "en_core_web_sm")
        self.nlp = _spacy_load(model_name)
        self.lemmatizer = _DummyLemmatizer()
        self.stop_words = set(_stop_words)

        self.theological_terms = self._initialize_theological_terms()
        self.divine_attributes = self._initialize_divine_attributes()
        self.philosophical_concepts = self._initialize_philosophical_concepts()
        self.trinity_mappings = self._initialize_trinity_mappings()
        self.ontological_keywords = self._load_ontological_keywords()
        self.semantic_categories = self._initialize_semantic_categories()
        self.common_terms = self._initialize_common_terms()

        self.lambda_engine = LambdaEngine()
        self.logger.info("LanguageProcessor initialised with fallback-friendly components")

    # ------------------------------------------------------------------
    # Initialisation helpers.
    # ------------------------------------------------------------------
    def _initialize_theological_terms(self) -> Dict[str, float]:
        return {
            "god": 1.0,
            "divine": 1.0,
            "trinity": 1.0,
            "incarnation": 0.6,
            "atonement": 0.6,
            "salvation": 0.6,
            "grace": 0.6,
            "resurrection": 0.6,
        }

    def _initialize_divine_attributes(self) -> Dict[str, float]:
        return {
            "omnipotent": 0.9,
            "omniscient": 0.9,
            "omnibenevolent": 0.9,
            "eternal": 0.7,
            "immutable": 0.7,
            "holy": 0.7,
        }

    def _initialize_philosophical_concepts(self) -> Dict[str, float]:
        return {
            "ontology": 0.7,
            "epistemology": 0.7,
            "metaphysics": 0.7,
            "ethics": 0.5,
            "logic": 0.5,
            "teleology": 0.5,
        }

    def _initialize_trinity_mappings(self) -> Dict[str, List[str]]:
        return {
            "existence": ["father", "creator", "source"],
            "goodness": ["son", "redeemer", "savior"],
            "truth": ["holy", "spirit", "guide", "revealer"],
        }

    def _load_ontological_keywords(self) -> Dict[str, List[str]]:
        return {
            "existence": ["exist", "being", "real", "substance"],
            "goodness": ["good", "just", "moral", "virtue"],
            "truth": ["true", "truth", "fact", "knowledge"],
        }

    def _initialize_semantic_categories(self) -> Dict[str, List[str]]:
        return {
            "ontological": ["exist", "being", "essence"],
            "moral": ["good", "evil", "virtue", "vice"],
            "epistemic": ["know", "truth", "belief"],
            "causal": ["cause", "effect", "created"],
            "modal": ["necessary", "possible", "contingent"],
            "logical": ["entails", "contradicts", "implies"],
        }

    def _initialize_common_terms(self) -> Dict[str, LogosExpr]:
        return {
            "existence": LogosExpr(),
            "goodness": LogosExpr(),
            "truth": LogosExpr(),
            "implication": LogosExpr(),
        }

    # ------------------------------------------------------------------
    # Public API.
    # ------------------------------------------------------------------
    async def process(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> NLPProcessingResult:
        self.logger.debug("Processing text: %s", text)
        tokens = _word_tokenize(text)
        lemmas = [self.lemmatizer.lemmatize(tok) for tok in tokens if tok not in self.stop_words]

        entities = self._extract_entities(text, lemmas)
        intent = self._classify_intent(text, lemmas)
        triples = self._generate_semantic_triples(entities)
        semantic_parse = self._generate_semantic_parse(text, intent, entities)

        result = NLPProcessingResult(
            original_text=text,
            intent_classification=intent,
            named_entities=entities,
            semantic_triples=triples,
            semantic_parse=semantic_parse,
            sentiment_analysis=self._analyze_sentiment(text),
            complexity_metrics=self._calculate_complexity(text),
            trinity_vector_mapping=self._extract_trinity_vector(text),
            theological_analysis=self._perform_theological_analysis(text, entities),
            processing_metadata={"timestamp": datetime.utcnow().isoformat(), "metadata": metadata or {}},
        )
        return result

    def natural_to_lambda(self, query: str, metadata: Optional[Dict[str, Any]] = None) -> Tuple[LogosExpr, TranslationResult]:
        sign_layer = self._natural_to_sign(query)
        mind_layer = self._sign_to_mind(sign_layer)
        bridge_layer = self._mind_to_bridge(mind_layer)

        expr = self.common_terms.get("implication", LogosExpr())
        trinity_vector = TrinityVector(
            existence=bridge_layer["ontological_dimensions"].get("existence", 0.5),
            goodness=bridge_layer["ontological_dimensions"].get("goodness", 0.5),
            truth=bridge_layer["ontological_dimensions"].get("truth", 0.5),
            confidence=bridge_layer["confidence"],
        )

        translation = TranslationResult(query=query, trinity_vector=trinity_vector, layers={
            "SIGN": sign_layer,
            "MIND": mind_layer,
            "BRIDGE": bridge_layer,
            "METADATA": metadata or {},
        })
        return expr, translation

    def lambda_to_natural(self, expr: LogosExpr, translation: Optional[TranslationResult] = None) -> str:
        if translation:
            return json.dumps({
                "translation": translation.to_dict(),
                "expression": self.lambda_engine.expr_to_dict(expr),
            })
        return self.lambda_engine.expr_to_dict(expr)["expression"]

    # ------------------------------------------------------------------
    # Internal helpers.
    # ------------------------------------------------------------------
    def _extract_entities(self, text: str, lemmas: List[str]) -> List[NamedEntity]:
        entities: List[NamedEntity] = []
        doc = self.nlp(text)
        for ent in getattr(doc, "ents", []):
            entity_type = EntityType.PERSON_NAME if ent.label_ == "PERSON" else EntityType.TEMPORAL_REFERENCE
            entities.append(NamedEntity(ent.text, entity_type, ent.start_char, ent.end_char, 0.75))

        text_lower = text.lower()
        for lemma in lemmas:
            if lemma in self.theological_terms:
                pos = text_lower.find(lemma)
                if pos >= 0:
                    entities.append(NamedEntity(lemma, EntityType.THEOLOGICAL_CONCEPT, pos, pos + len(lemma), 0.9))
            if lemma in self.divine_attributes:
                pos = text_lower.find(lemma)
                if pos >= 0:
                    entities.append(NamedEntity(lemma, EntityType.DIVINE_ATTRIBUTE, pos, pos + len(lemma), 0.85))
            if lemma in self.philosophical_concepts:
                pos = text_lower.find(lemma)
                if pos >= 0:
                    entities.append(NamedEntity(lemma, EntityType.PHILOSOPHICAL_TERM, pos, pos + len(lemma), 0.8))

        return entities

    def _classify_intent(self, text: str, lemmas: List[str]) -> IntentClassification:
        text_lower = text.lower()
        if any(term in text_lower for term in self.theological_terms):
            primary = IntentCategory.THEOLOGICAL_INQUIRY
            confidence = 0.9
        elif any(term in text_lower for term in self.philosophical_concepts):
            primary = IntentCategory.PHILOSOPHICAL_QUESTION
            confidence = 0.8
        elif "necess" in text_lower or "modal" in text_lower:
            primary = IntentCategory.MODAL_REASONING
            confidence = 0.7
        else:
            primary = IntentCategory.GENERAL_CONVERSATION
            confidence = 0.6

        return IntentClassification(primary_intent=primary, confidence=confidence)

    def _generate_semantic_triples(self, entities: List[NamedEntity]) -> List[SemanticTriple]:
        triples: List[SemanticTriple] = []
        if len(entities) >= 2:
            triples.append(SemanticTriple(entities[0].text, SemanticRelation.INSTANCE_OF, entities[1].text, 0.65))
        return triples

    def _generate_semantic_parse(
        self,
        text: str,
        intent: IntentClassification,
        entities: List[NamedEntity],
    ) -> SemanticParse:
        modal_operators = [tok for tok in _word_tokenize(text) if tok in {"necessarily", "possibly", "must", "may"}]
        logical_form = "".join([
            "□" if "necessarily" in modal_operators else "",
            "P(x) → Q(x)",
        ])
        concepts = [entity.text for entity in entities if entity.entity_type == EntityType.THEOLOGICAL_CONCEPT]
        return SemanticParse(
            logical_form=logical_form or "P(x)",
            confidence=intent.confidence * 0.7,
            modal_operators=modal_operators,
            quantifiers=[quant for quant in ("∀", "∃") if quant in text],
            theological_concepts=concepts,
        )

    def _analyze_sentiment(self, text: str) -> Dict[str, float]:
        positive = sum(1 for token in _word_tokenize(text.lower()) if token in {"good", "true", "just"})
        negative = sum(1 for token in _word_tokenize(text.lower()) if token in {"bad", "false", "evil"})
        total = len(text.split()) or 1
        polarity = (positive - negative) / total
        return {"polarity": polarity, "subjectivity": 0.5}

    def _calculate_complexity(self, text: str) -> Dict[str, float]:
        words = text.split()
        diversity = len(set(words)) / len(words) if words else 0.0
        return {"lexical_diversity": diversity, "sentence_length": len(words)}

    def _perform_theological_analysis(self, text: str, entities: List[NamedEntity]) -> Dict[str, Any]:
        concepts = [entity.text for entity in entities if entity.entity_type == EntityType.THEOLOGICAL_CONCEPT]
        attributes = [entity.text for entity in entities if entity.entity_type == EntityType.DIVINE_ATTRIBUTE]
        return {"concepts": concepts, "attributes": attributes}

    def _extract_trinity_vector(self, text: str) -> TrinityVector:
        text_lower = text.lower()
        scores = {dim: sum(1 for kw in kws if kw in text_lower) for dim, kws in self.ontological_keywords.items()}
        total = sum(scores.values()) or 1
        return TrinityVector(
            existence=scores["existence"] / total,
            goodness=scores["goodness"] / total,
            truth=scores["truth"] / total,
            confidence=min(total / 10, 1.0),
        )

    def _natural_to_sign(self, query: str) -> List[str]:
        return _word_tokenize(query)

    def _sign_to_mind(self, sign_layer: List[str]) -> Dict[str, float]:
        mind: Dict[str, float] = defaultdict(float)
        for token in sign_layer:
            for category, keywords in self.semantic_categories.items():
                if token in keywords:
                    mind[category] += 1.0
        total = sum(mind.values()) or 1
        return {category: value / total for category, value in mind.items()}

    def _mind_to_bridge(self, mind_layer: Dict[str, float]) -> Dict[str, Any]:
        ontological_dimensions = {
            "existence": 0.5 + 0.5 * mind_layer.get("ontological", 0.0),
            "goodness": 0.5 + 0.5 * mind_layer.get("moral", 0.0),
            "truth": 0.5 + 0.5 * mind_layer.get("epistemic", 0.0),
        }
        confidence = min(sum(mind_layer.values()), 1.0)
        return {
            "ontological_dimensions": ontological_dimensions,
            "confidence": confidence,
        }


language_processor = LanguageProcessor()

__all__ = [
    "LanguageProcessor",
    "language_processor",
    "NLPProcessingResult",
    "TranslationResult",
    "TrinityVector",
    "IntentCategory",
    "EntityType",
    "SemanticRelation",
]
