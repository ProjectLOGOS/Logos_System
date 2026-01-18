# HEADER_TYPE: LEGACY_REWRITE_CANDIDATE
# EXECUTION: FORBIDDEN
# IMPORT: FORBIDDEN
# AUTHORITY: NONE
# DESTINATION: Logos_System_Rebuild
# ARCHIVE_AFTER_REWRITE: REQUIRED

"""
Bridge Principle implementation for I2.
Deterministic translation layer: external meaning → Structured Meaning Packets (SMPs).

Core contract:
- NO inference
- NO belief formation
- NO will-based decisions
- Pure ontology-anchored semantic mapping
- Privation preservation without remediation

Author: Divine Necessity Framework
Version: 1.0.0
Schema: SMP v0.1 (canonical)
"""

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS & TYPE DEFINITIONS (Aligned with SMP v0.1 Schema)
# ============================================================================

class Origin(Enum):
    """Source origin types"""
    USER = "user"
    LLM = "llm"
    AGENT = "agent"
    SYSTEM = "system"
    TOOL = "tool"


class Channel(Enum):
    """Communication channels"""
    USER = "user"
    SYSTEM = "system"
    AGENT = "agent"
    TOOL = "tool"


class IntentClass(Enum):
    """Intent classification"""
    QUESTION = "question"
    CLAIM = "claim"
    REQUEST = "request"
    INSTRUCTION = "instruction"
    CONVERSATION = "conversation"
    UNKNOWN = "unknown"


class EntityType(Enum):
    """Entity type taxonomy"""
    PERSON = "person"
    OBJECT = "object"
    CONCEPT = "concept"
    PLACE = "place"
    EVENT = "event"
    UNKNOWN = "unknown"


class Polarity(Enum):
    """Propositional polarity"""
    AFFIRM = "affirm"
    NEGATE = "negate"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class Modality(Enum):
    """Modal operators (PXL-aligned)"""
    ASSERT = "assert"
    POSSIBLE = "possible"
    NECESSARY = "necessary"
    OBLIGATORY = "obligatory"
    FORBIDDEN = "forbidden"
    UNKNOWN = "unknown"


class RelationType(Enum):
    """Typed relations"""
    CAUSAL = "causal"
    TEMPORAL = "temporal"
    IDENTITY = "identity"
    EQUIVALENCE = "equivalence"
    PART_OF = "part_of"
    NORMATIVE = "normative"
    COHERENCE = "coherence"      # PXL: x ⧟ y
    EXCLUSIVITY = "exclusivity"  # PXL: x ⇎ y
    INTERCHANGE = "interchange"  # PXL: x ⇌ y
    GROUNDED_IN = "grounded_in"  # PXL: x ⟹ y
    UNKNOWN = "unknown"


class DomainHint(Enum):
    """MESH domain hints"""
    EPISTEMIC = "epistemic"
    ONTOLOGICAL = "ontological"
    AXIOLOGICAL = "axiological"
    TELEOLOGICAL = "teleological"
    LINGUISTIC = "linguistic"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class PrivationDomain(Enum):
    """Privation domain classification"""
    ONTOLOGICAL = "ontological"
    AXIOLOGICAL = "axiological"
    EPISTEMIC = "epistemic"
    TELEOLOGICAL = "teleological"
    LINGUISTIC = "linguistic"
    AGENTIC = "agentic"
    UNKNOWN = "unknown"


class PrivationAction(Enum):
    """Privation handling action"""
    ALLOW = "allow"
    TRANSFORM = "transform"
    QUARANTINE = "quarantine"
    ESCALATE = "escalate"


class RiskLevel(Enum):
    """Risk severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TraceStage(Enum):
    """Processing stages for provenance trace"""
    I2_INTAKE = "i2_intake"
    UIP_PARSE = "uip_parse"
    PRIVATION_GATE = "privation_gate"
    PRIVATION_CLASSIFY = "privation_classify"
    PRIVATION_ANALYZE = "privation_analyze"
    PRIVATION_TRANSFORM = "privation_transform"
    BRIDGE_EMIT = "bridge_emit"


class TraceActor(Enum):
    """Actors in provenance chain"""
    I2 = "I2"
    UIP = "UIP"
    SOP = "SOP"
    LOGOS = "LOGOS"


# ============================================================================
# SMP v0.1 DATA STRUCTURES
# ============================================================================

@dataclass
class Source:
    """SMP source metadata"""
    origin: Origin
    origin_id: Optional[str] = None
    i2_session_id: str = ""
    parent_smp_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "origin": self.origin.value,
            "origin_id": self.origin_id,
            "i2_session_id": self.i2_session_id,
            "parent_smp_id": self.parent_smp_id,
        }


@dataclass
class Entity:
    """Ontology-anchored entity"""
    eid: str
    label: str
    type: EntityType
    ontology_ref: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "eid": self.eid,
            "label": self.label,
            "type": self.type.value,
            "ontology_ref": self.ontology_ref,
        }


@dataclass
class Proposition:
    """Propositional content"""
    pid: str
    surface: str
    polarity: Polarity
    modality: Modality

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pid": self.pid,
            "surface": self.surface,
            "polarity": self.polarity.value,
            "modality": self.modality.value,
        }


@dataclass
class Relation:
    """Typed relation between entities/propositions"""
    rid: str
    type: RelationType
    from_ref: str  # eid or pid
    to_ref: str    # eid or pid
    label: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rid": self.rid,
            "type": self.type.value,
            "from": self.from_ref,
            "to": self.to_ref,
            "label": self.label,
        }


@dataclass
class Ontology:
    """Ontological metadata"""
    ontoprop_refs: List[str] = field(default_factory=list)
    domain_hint: DomainHint = DomainHint.UNKNOWN

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ontoprop_refs": self.ontoprop_refs,
            "domain_hint": self.domain_hint.value,
        }


@dataclass
class Payload:
    """Core meaning payload (lean at ingress)"""
    text_canonical: str
    language: str = "en"
    entities: List[Entity] = field(default_factory=list)
    propositions: List[Proposition] = field(default_factory=list)
    relations: List[Relation] = field(default_factory=list)
    ontology: Ontology = field(default_factory=Ontology)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text_canonical": self.text_canonical,
            "language": self.language,
            "entities": [e.to_dict() for e in self.entities],
            "propositions": [p.to_dict() for p in self.propositions],
            "relations": [r.to_dict() for r in self.relations],
            "ontology": self.ontology.to_dict(),
        }


@dataclass
class PrivationMetadata:
    """Privation detection and handling"""
    is_privative: bool = False
    tags: List[str] = field(default_factory=list)
    domain: PrivationDomain = PrivationDomain.UNKNOWN
    severity: float = 0.0  # 0.0 to 1.0
    action: PrivationAction = PrivationAction.ALLOW
    override_applied: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_privative": self.is_privative,
            "tags": self.tags,
            "domain": self.domain.value,
            "severity": self.severity,
            "action": self.action.value,
            "override_applied": self.override_applied,
        }


@dataclass
class Risk:
    """Risk assessment metadata"""
    level: RiskLevel = RiskLevel.LOW
    flags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level.value,
            "flags": self.flags,
        }


@dataclass
class Security:
    """Security gate results and risk levels"""
    privation: PrivationMetadata = field(default_factory=PrivationMetadata)
    risk: Risk = field(default_factory=Risk)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "privation": self.privation.to_dict(),
            "risk": self.risk.to_dict(),
        }


@dataclass
class TraceEntry:
    """Single provenance trace entry"""
    stage: TraceStage
    actor: TraceActor
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage.value,
            "actor": self.actor.value,
            "note": self.note,
        }


@dataclass
class UIPArtifacts:
    """UIP processing artifacts"""
    embedding_id: Optional[str] = None
    parser_version: Optional[str] = None
    semantic_model: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "embedding_id": self.embedding_id,
            "parser_version": self.parser_version,
            "semantic_model": self.semantic_model,
        }


@dataclass
class Provenance:
    """Traceability and confidence information"""
    confidence: float = 1.0
    uip_artifacts: UIPArtifacts = field(default_factory=UIPArtifacts)
    trace: List[TraceEntry] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "confidence": self.confidence,
            "uip_artifacts": self.uip_artifacts.to_dict(),
            "trace": [t.to_dict() for t in self.trace],
        }


@dataclass
class Enrichment:
    """Optional enrichment (ARP/SCP may add later)"""
    bayesian: Dict[str, Any] = field(default_factory=dict)
    mvs_bdn: Dict[str, Any] = field(default_factory=dict)
    iel: Dict[str, Any] = field(default_factory=dict)
    proof_refs: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bayesian": self.bayesian,
            "mvs_bdn": self.mvs_bdn,
            "iel": self.iel,
            "proof_refs": self.proof_refs,
        }


@dataclass
class Routing:
    """Optional routing (Logos/SOP may add)"""
    recommended_targets: List[str] = field(default_factory=list)
    final_target: str = "LOGOS"
    requires_will: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "recommended_targets": self.recommended_targets,
            "final_target": self.final_target,
            "requires_will": self.requires_will,
        }


@dataclass
class StructuredMeaningPacket:
    """
    SMP v0.1 - Canonical schema for LOGOS stack.

    Required fields (lean at ingress from I2):
    - smp_version, smp_id, timestamp_utc
    - source, channel, intent_class
    - payload, provenance, security

    Optional fields (for data explosion by ARP/SCP):
    - enrichment, routing
    """

    smp_version: str = "0.1"
    smp_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    source: Source = field(default_factory=lambda: Source(origin=Origin.USER))
    channel: Channel = Channel.USER
    intent_class: IntentClass = IntentClass.UNKNOWN
    payload: Payload = field(default_factory=Payload)
    provenance: Provenance = field(default_factory=Provenance)
    security: Security = field(default_factory=Security)
    enrichment: Optional[Enrichment] = None
    routing: Optional[Routing] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "smp_version": self.smp_version,
            "smp_id": self.smp_id,
            "timestamp_utc": self.timestamp_utc,
            "source": self.source.to_dict(),
            "channel": self.channel.value,
            "intent_class": self.intent_class.value,
            "payload": self.payload.to_dict(),
            "provenance": self.provenance.to_dict(),
            "security": self.security.to_dict(),
        }

        if self.enrichment:
            result["enrichment"] = self.enrichment.to_dict()
        if self.routing:
            result["routing"] = self.routing.to_dict()

        return result

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


# ============================================================================
# ONTOLOGY MAPPING
# ============================================================================

class OntologyMapper:
    """
    Maps external references to OntoGrid.
    NO inference - pure lookup and alignment.
    """

    def __init__(self, ontogrid_schema: Dict[str, Any]):
        self.ontogrid = ontogrid_schema
        self.entities = ontogrid_schema.get("entities", {})
        self.relations = ontogrid_schema.get("relations", {})
        self.domains = ontogrid_schema.get("domains", {})
        logger.info("OntologyMapper initialized")

    def anchor_entity(self, label: str, entity_type: EntityType) -> Entity:
        eid = f"e_{uuid.uuid4().hex[:8]}"
        ontology_ref = self._exact_lookup(label)
        if not ontology_ref:
            ontology_ref = self._fuzzy_lookup(label)
        entity = Entity(eid=eid, label=label, type=entity_type, ontology_ref=ontology_ref)
        if ontology_ref:
            logger.debug(f"Entity anchored: {label} -> {ontology_ref}")
        else:
            logger.debug(f"Entity unanchored: {label} (flagged for Logos review)")
        return entity

    def _exact_lookup(self, ref: str) -> Optional[str]:
        return self.entities.get(ref)

    def _fuzzy_lookup(self, ref: str) -> Optional[str]:
        ref_lower = ref.lower()
        for key, value in self.entities.items():
            if key.lower() == ref_lower:
                return value
        return None

    def get_relation_type(self, predicate: str) -> RelationType:
        predicate_lower = predicate.lower()
        pxl_map = {
            "coherence": RelationType.COHERENCE,
            "⧟": RelationType.COHERENCE,
            "exclusivity": RelationType.EXCLUSIVITY,
            "⇎": RelationType.EXCLUSIVITY,
            "interchange": RelationType.INTERCHANGE,
            "⇌": RelationType.INTERCHANGE,
            "grounded_in": RelationType.GROUNDED_IN,
            "⟹": RelationType.GROUNDED_IN,
        }
        if predicate in pxl_map or predicate_lower in pxl_map:
            return pxl_map.get(predicate, pxl_map.get(predicate_lower, RelationType.UNKNOWN))
        standard_map = {
            "causes": RelationType.CAUSAL,
            "requires": RelationType.CAUSAL,
            "before": RelationType.TEMPORAL,
            "after": RelationType.TEMPORAL,
            "is": RelationType.IDENTITY,
            "equals": RelationType.EQUIVALENCE,
            "part_of": RelationType.PART_OF,
            "ought": RelationType.NORMATIVE,
            "should": RelationType.NORMATIVE,
        }
        return standard_map.get(predicate_lower, RelationType.UNKNOWN)


# ============================================================================
# INTENT CLASSIFICATION
# ============================================================================

class IntentClassifier:
    """
    Classifies user intent from text patterns.
    NO inference - pattern matching only.
    """

    QUESTION_MARKERS = ["?", "what", "why", "how", "when", "where", "who", "which"]
    CLAIM_MARKERS = ["is", "are", "necessarily", "must", "always"]
    REQUEST_MARKERS = ["please", "can you", "could you", "would you"]
    INSTRUCTION_MARKERS = ["do", "make", "create", "build", "write", "generate"]

    def classify(self, text: str) -> IntentClass:
        text_lower = text.lower()
        if "?" in text or any(marker in text_lower for marker in self.QUESTION_MARKERS):
            return IntentClass.QUESTION
        if any(marker in text_lower for marker in self.REQUEST_MARKERS):
            return IntentClass.REQUEST
        text_start = text_lower[:20]
        if any(text_start.startswith(marker) for marker in self.INSTRUCTION_MARKERS):
            return IntentClass.INSTRUCTION
        if any(marker in text_lower for marker in self.CLAIM_MARKERS):
            return IntentClass.CLAIM
        return IntentClass.CONVERSATION


# ============================================================================
# MODALITY & POLARITY EXTRACTION
# ============================================================================

class ModalityExtractor:
    """
    Detects modal operators and polarity.
    NO inference - pattern matching only.
    """

    NECESSARY_MARKERS = ["necessary", "must", "always", "□", "necessarily", "cannot"]
    POSSIBLE_MARKERS = ["possible", "might", "could", "◇", "possibly", "may"]
    OBLIGATORY_MARKERS = ["ought", "should", "obligatory", "duty", "required"]
    FORBIDDEN_MARKERS = ["forbidden", "must not", "cannot", "prohibited"]
    NEGATION_MARKERS = ["not", "no", "never", "¬", "∼", "neither"]

    def extract_modality(self, text: str) -> Modality:
        text_lower = text.lower()
        if any(marker in text_lower for marker in self.NECESSARY_MARKERS):
            return Modality.NECESSARY
        if any(marker in text_lower for marker in self.OBLIGATORY_MARKERS):
            return Modality.OBLIGATORY
        if any(marker in text_lower for marker in self.FORBIDDEN_MARKERS):
            return Modality.FORBIDDEN
        if any(marker in text_lower for marker in self.POSSIBLE_MARKERS):
            return Modality.POSSIBLE
        return Modality.ASSERT

    def extract_polarity(self, text: str) -> Polarity:
        text_lower = text.lower()
        negation_count = sum(1 for marker in self.NEGATION_MARKERS if marker in text_lower)
        if negation_count == 0:
            return Polarity.AFFIRM
        if negation_count == 1:
            return Polarity.NEGATE
        return Polarity.MIXED


# ============================================================================
# PRIVATION DETECTION
# ============================================================================

class PrivationDetector:
    """
    Detects privation (coherence failures) without remediation.
    Tags for SOP/Logos handling.
    """

    PRIVATION_TAGS = {
        "negation_present": ["not", "no", "never", "¬", "∼"],
        "ontological_nullity": ["nothing", "nonexistent", "void", "null"],
        "contradiction": ["both and not", "simultaneously opposite", "⧟.*⇎"],
        "identity_violation": ["not equal to itself", "∼(.*⧟.*)"],
        "grounding_failure": ["no sufficient reason", "∼grounded", "brute fact"],
        "excluded_middle_failure": ["neither nor", "∼(.*⫴.*)"],
        "coherence_failure": ["incoherent", "inconsistent", "∼coherence"],
        "axiological_privation": ["evil", "bad", "wrong", "immoral"],
        "epistemic_privation": ["unknown", "unknowable", "uncertain", "ignorance"],
        "teleological_privation": ["purposeless", "meaningless", "aimless"],
    }

    def detect(self, text: str, semantic_data: Dict[str, Any]) -> PrivationMetadata:
        import re

        privation = PrivationMetadata()
        if semantic_data.get("privation_detected"):
            privation.is_privative = True
            privation.tags = semantic_data.get("privation_tags", [])
            privation.domain = self._map_domain(semantic_data.get("privation_domain", ""))
            privation.severity = semantic_data.get("privation_severity", 0.5)
            privation.action = self._determine_action(privation.severity)
            logger.info(
                f"Privation detected (UIP): {privation.domain.value}, severity {privation.severity:.2f}"
            )
            return privation

        text_lower = text.lower()
        detected_tags = []
        for tag, patterns in self.PRIVATION_TAGS.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    detected_tags.append(tag)
                    break

        if detected_tags:
            privation.is_privative = True
            privation.tags = detected_tags
            privation.domain = self._infer_domain(detected_tags)
            privation.severity = self._calculate_severity(detected_tags)
            privation.action = self._determine_action(privation.severity)
            logger.info(
                f"Privation detected (pattern): {privation.domain.value}, tags: {detected_tags}"
            )

        return privation

    def _map_domain(self, domain_str: str) -> PrivationDomain:
        domain_map = {
            "ontological": PrivationDomain.ONTOLOGICAL,
            "axiological": PrivationDomain.AXIOLOGICAL,
            "epistemic": PrivationDomain.EPISTEMIC,
            "teleological": PrivationDomain.TELEOLOGICAL,
            "linguistic": PrivationDomain.LINGUISTIC,
            "agentic": PrivationDomain.AGENTIC,
        }
        return domain_map.get(domain_str.lower(), PrivationDomain.UNKNOWN)

    def _infer_domain(self, tags: List[str]) -> PrivationDomain:
        if any("ontological" in tag for tag in tags):
            return PrivationDomain.ONTOLOGICAL
        if any("axiological" in tag for tag in tags):
            return PrivationDomain.AXIOLOGICAL
        if any("epistemic" in tag for tag in tags):
            return PrivationDomain.EPISTEMIC
        if any("teleological" in tag for tag in tags):
            return PrivationDomain.TELEOLOGICAL
        return PrivationDomain.LINGUISTIC

    def _calculate_severity(self, tags: List[str]) -> float:
        severity_weights = {
            "contradiction": 0.9,
            "ontological_nullity": 0.8,
            "grounding_failure": 0.8,
            "coherence_failure": 0.7,
            "axiological_privation": 0.6,
            "identity_violation": 0.7,
            "excluded_middle_failure": 0.7,
            "epistemic_privation": 0.4,
            "teleological_privation": 0.5,
            "negation_present": 0.3,
        }
        if not tags:
            return 0.0
        total = sum(severity_weights.get(tag, 0.5) for tag in tags)
        return min(total / len(tags), 1.0)

    def _determine_action(self, severity: float) -> PrivationAction:
        if severity >= 0.8:
            return PrivationAction.ESCALATE
        if severity >= 0.6:
            return PrivationAction.TRANSFORM
        if severity >= 0.3:
            return PrivationAction.ALLOW
        return PrivationAction.ALLOW


# ============================================================================
# DOMAIN CLASSIFIER
# ============================================================================

class DomainClassifier:
    """Classify domain hints for ontological routing"""

    DOMAIN_MARKERS = {
        DomainHint.EPISTEMIC: ["know", "believe", "uncertain", "evidence", "truth"],
        DomainHint.ONTOLOGICAL: ["exist", "being", "reality", "substance", "essence"],
        DomainHint.AXIOLOGICAL: ["good", "bad", "ought", "value", "moral", "evil"],
        DomainHint.TELEOLOGICAL: ["purpose", "goal", "end", "telos", "design"],
        DomainHint.LINGUISTIC: ["meaning", "semantics", "syntax", "language", "proposition"],
    }

    def classify(self, text: str) -> DomainHint:
        text_lower = text.lower()
        domain_scores: Dict[DomainHint, int] = {}
        for domain, markers in self.DOMAIN_MARKERS.items():
            score = sum(1 for marker in markers if marker in text_lower)
            if score > 0:
                domain_scores[domain] = score
        if not domain_scores:
            return DomainHint.UNKNOWN
        if len(domain_scores) > 1:
            return DomainHint.MIXED
        return max(domain_scores, key=domain_scores.get)


# ============================================================================
# SMP BUILDER
# ============================================================================

class SMPBuilder:
    """
    Constructs Structured Meaning Packets (SMP v0.1).
    Orchestrates all extraction and mapping components.
    """

    def __init__(self, ontogrid_schema: Dict[str, Any], session_id: str):
        self.ontology_mapper = OntologyMapper(ontogrid_schema)
        self.intent_classifier = IntentClassifier()
        self.modality_extractor = ModalityExtractor()
        self.privation_detector = PrivationDetector()
        self.domain_classifier = DomainClassifier()
        self.session_id = session_id
        logger.info("SMPBuilder initialized")

    def build_smp(
        self,
        raw_input: str,
        semantic_data: Dict[str, Any],
        channel: Channel = Channel.USER,
        origin: Origin = Origin.USER,
        parent_smp_id: Optional[str] = None,
    ) -> StructuredMeaningPacket:
        logger.info("Building SMP v0.1...")
        trace = [
            TraceEntry(TraceStage.I2_INTAKE, TraceActor.I2, "Raw input received"),
            TraceEntry(TraceStage.UIP_PARSE, TraceActor.UIP, "Semantic extraction complete"),
        ]
        source = Source(
            origin=origin,
            origin_id=semantic_data.get("origin_id"),
            i2_session_id=self.session_id,
            parent_smp_id=parent_smp_id,
        )
        intent_class = self.intent_classifier.classify(raw_input)
        payload = self._build_payload(raw_input, semantic_data)
        trace.append(TraceEntry(TraceStage.BRIDGE_EMIT, TraceActor.I2, "Payload constructed"))
        privation = self.privation_detector.detect(raw_input, semantic_data)
        if privation.is_privative:
            trace.append(
                TraceEntry(
                    TraceStage.PRIVATION_CLASSIFY,
                    TraceActor.I2,
                    f"Privation detected: {privation.domain.value}",
                )
            )
        risk = self._assess_risk(privation, payload)
        security = Security(privation=privation, risk=risk)
        provenance = Provenance(
            confidence=semantic_data.get("confidence", 0.95),
            uip_artifacts=UIPArtifacts(
                embedding_id=semantic_data.get("embedding_id"),
                parser_version=semantic_data.get("parser_version", "uip-v1.0"),
                semantic_model=semantic_data.get("semantic_model", "default"),
            ),
            trace=trace,
        )
        smp = StructuredMeaningPacket(
            source=source,
            channel=channel,
            intent_class=intent_class,
            payload=payload,
            provenance=provenance,
            security=security,
        )
        logger.info(
            f"SMP built successfully (ID: {smp.smp_id}, confidence: {provenance.confidence:.2f})"
        )
        return smp

    def _build_payload(self, text: str, semantic_data: Dict[str, Any]) -> Payload:
        entities: List[Entity] = []
        for entity_data in semantic_data.get("entities", []):
            label = entity_data if isinstance(entity_data, str) else entity_data.get("label", "")
            type_hint = (
                entity_data.get("type") if isinstance(entity_data, dict) else "unknown"
            )
            entity_type = self._map_entity_type(type_hint)
            entities.append(self.ontology_mapper.anchor_entity(label, entity_type))

        propositions: List[Proposition] = []
        for prop_data in semantic_data.get("propositions", []):
            surface = prop_data if isinstance(prop_data, str) else prop_data.get("surface", "")
            pid = f"p_{uuid.uuid4().hex[:8]}"
            modality = self.modality_extractor.extract_modality(surface)
            polarity = self.modality_extractor.extract_polarity(surface)
            propositions.append(
                Proposition(pid=pid, surface=surface, polarity=polarity, modality=modality)
            )

        relations: List[Relation] = []
        for rel_data in semantic_data.get("relations", []):
            if not isinstance(rel_data, dict):
                continue
            rid = f"r_{uuid.uuid4().hex[:8]}"
            predicate = rel_data.get("predicate", "")
            relation_type = self.ontology_mapper.get_relation_type(predicate)
            from_ref = self._find_ref(rel_data.get("subject", ""), entities, propositions)
            to_ref = self._find_ref(rel_data.get("object", ""), entities, propositions)
            if from_ref and to_ref:
                relations.append(
                    Relation(
                        rid=rid,
                        type=relation_type,
                        from_ref=from_ref,
                        to_ref=to_ref,
                        label=predicate,
                    )
                )

        domain_hint = self.domain_classifier.classify(text)
        ontology = Ontology(
            ontoprop_refs=semantic_data.get("ontoprop_refs", []),
            domain_hint=domain_hint,
        )
        return Payload(
            text_canonical=text.strip(),
            language=semantic_data.get("language", "en"),
            entities=entities,
            propositions=propositions,
            relations=relations,
            ontology=ontology,
        )

    def _map_entity_type(self, type_str: str) -> EntityType:
        type_map = {
            "person": EntityType.PERSON,
            "object": EntityType.OBJECT,
            "concept": EntityType.CONCEPT,
            "place": EntityType.PLACE,
            "event": EntityType.EVENT,
        }
        return type_map.get(type_str.lower(), EntityType.UNKNOWN)

    def _find_ref(self, label: str, entities: List[Entity], propositions: List[Proposition]) -> Optional[str]:
        for entity in entities:
            if entity.label.lower() == label.lower():
                return entity.eid
        for prop in propositions:
            if label.lower() in prop.surface.lower():
                return prop.pid
        return None

    def _assess_risk(self, privation: PrivationMetadata, payload: Payload) -> Risk:
        flags: List[str] = []
        if privation.is_privative:
            if privation.severity >= 0.7:
                flags.append("high_privation_severity")
            flags.append("privation_present")
        if any(not e.ontology_ref for e in payload.entities):
            flags.append("ambiguous_entities")
        if len(payload.relations) > 5:
            flags.append("complex_relational_structure")
        if privation.severity >= 0.8:
            level = RiskLevel.CRITICAL
        elif privation.severity >= 0.6 or "high_privation_severity" in flags:
            level = RiskLevel.HIGH
        elif privation.is_privative or flags:
            level = RiskLevel.MEDIUM
        else:
            level = RiskLevel.LOW
        if level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            flags.append("requires_logos_review")
        return Risk(level=level, flags=flags)
