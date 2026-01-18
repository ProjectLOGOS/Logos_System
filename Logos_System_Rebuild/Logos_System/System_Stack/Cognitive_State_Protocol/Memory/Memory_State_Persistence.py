# HEADER_TYPE: CANONICAL_REBUILD_MODULE
# EXECUTION: CONTROLLED
# AUTHORITY: GOVERNED
# ORIGIN: SYSTEMATIC_REWRITE

"""
CROSS-SESSION REPLAY & REHYDRATION INFRASTRUCTURE
Enables deterministic replay, forensic analysis, and full system state reconstruction
"""

from typing import Dict, List, Optional, Set, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum, auto
import numpy as np
import json
import pickle
import zlib
import msgpack
import hashlib
from datetime import datetime, timezone, timedelta
from pathlib import Path
import sqlite3
import lmdb
import gzip
import bz2
from collections import defaultdict, deque, OrderedDict
import asyncio
from concurrent.futures import ThreadPoolExecutor
import struct
import csv
import yaml
import tomli

# ==================== REPLAY EVENT TYPES ====================

class ReplayEventType(Enum):
    """Types of events that can be replayed"""
    # System Events
    SYSTEM_INIT = auto()
    SYSTEM_SHUTDOWN = auto()
    CHECKPOINT_CREATED = auto()
    CHECKPOINT_RESTORED = auto()
    
    # UWM Events (from Phase 5)
    UWM_EVENT_APPROVED = auto()
    UWM_ATOM_CREATED = auto()
    UWM_ATOM_UPDATED = auto()
    UWM_ATOM_SUPERSEDED = auto()
    
    # SMP Events
    SMP_GENERATED = auto()
    SMP_ENRICHED = auto()
    SMP_EVALUATED = auto()
    SMP_APPROVED = auto()
    SMP_REJECTED = auto()
    SMP_ROUTED = auto()
    
    # Memory Events
    MEMORY_CREATED = auto()
    MEMORY_ACCESSED = auto()
    MEMORY_CONSOLIDATED = auto()
    MEMORY_FORGOTTEN = auto()
    
    # Agent Events
    AGENT_CREATED = auto()
    AGENT_ACTION = auto()
    AGENT_HYPOTHESIS = auto()
    AGENT_LEARNED = auto()
    
    # Evaluation Events
    EVALUATION_STARTED = auto()
    EVALUATION_COMPLETED = auto()
    TRIUNE_COMMUTATION = auto()
    CONVERGENCE_ANALYSIS = auto()

class ReplayGranularity(Enum):
    """Level of detail for replay"""
    SUMMARY = auto()       # Just major events
    OPERATIONAL = auto()   # All system operations
    DEBUG = auto()        # Every detail including internal state
    FORENSIC = auto()     # Everything + timing and context

# ==================== REPLAY EVENT STRUCTURE ====================

@dataclass
class ReplayEvent:
    """
    Immutable event record for deterministic replay
    Contains everything needed to reconstruct system state
    """
    
    # Core identification
    event_id: str
    event_type: ReplayEventType
    sequence_number: int
    timestamp: datetime
    
    # Content
    payload: Dict[str, Any]
    
    # Context
    session_id: str
    process_id: str
    thread_id: Optional[str] = None
    
    # Dependencies
    parent_event_ids: List[str] = field(default_factory=list)
    causes_event_ids: List[str] = field(default_factory=list)
    
    # Verification
    checksum: str = ""
    signature: Optional[str] = None
    
    # Metadata
    source_component: str = ""
    criticality: int = 1  # 1-5, higher = more critical for replay
    
    def __post_init__(self):
        # Generate deterministic event ID if not provided
        if not self.event_id:
            self.event_id = self._generate_event_id()
        
        # Calculate checksum
        if not self.checksum:
            self.checksum = self._calculate_checksum()
    
    def _generate_event_id(self) -> str:
        """Generate deterministic event ID"""
        components = [
            self.session_id,
            str(self.sequence_number),
            self.event_type.name,
            self.timestamp.isoformat()
        ]
        content = "|".join(components)
        return f"event_{hashlib.sha256(content.encode()).hexdigest()[:16]}"
    
    def _calculate_checksum(self) -> str:
        """Calculate checksum of event content"""
        # Serialize deterministic components
        check_data = {
            'type': self.event_type.name,
            'sequence': self.sequence_number,
            'timestamp': self.timestamp.isoformat(),
            'payload': self.payload,
            'session': self.session_id,
            'parents': sorted(self.parent_event_ids)
        }
        
        # Deterministic JSON serialization
        check_json = json.dumps(check_data, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(check_json.encode()).hexdigest()
    
    def to_serializable(self) -> Dict:
        """Convert to serializable format for storage"""
        return {
            'event_id': self.event_id,
            'event_type': self.event_type.name,
            'sequence_number': self.sequence_number,
            'timestamp': self.timestamp.isoformat(),
            'payload': self.payload,
            'session_id': self.session_id,
            'process_id': self.process_id,
            'thread_id': self.thread_id,
            'parent_event_ids': self.parent_event_ids,
            'causes_event_ids': self.causes_event_ids,
            'checksum': self.checksum,
            'signature': self.signature,
            'source_component': self.source_component,
            'criticality': self.criticality
        }
    
    @classmethod
    def from_serializable(cls, data: Dict) -> 'ReplayEvent':
        """Reconstruct from serialized data"""
        return cls(
            event_id=data['event_id'],
            event_type=ReplayEventType[data['event_type']],
            sequence_number=data['sequence_number'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            payload=data['payload'],
            session_id=data['session_id'],
            process_id=data['process_id'],
            thread_id=data.get('thread_id'),
            parent_event_ids=data.get('parent_event_ids', []),
            causes_event_ids=data.get('causes_event_ids', []),
            checksum=data['checksum'],
            signature=data.get('signature'),
            source_component=data.get('source_component', ''),
            criticality=data.get('criticality', 1)
        )
    
    def verify(self) -> bool:
        """Verify event integrity"""
        calculated = self._calculate_checksum()
        return calculated == self.checksum

# ==================== REPLAY LOGGER ====================

class DeterministicReplayLogger:
    """
    Logs all system events for deterministic replay
    Ensures complete audit trail
    """
    
    def __init__(self, log_path: Path, session_id: str):
        self.log_path = log_path
        self.session_id = session_id
        self.sequence_counter = 0
        
        # Create log directory
        self.log_path.mkdir(parents=True, exist_ok=True)
        
        # Open log files
        self.event_log = self._open_event_log()
        self.checkpoint_log = self._open_checkpoint_log()
        self.state_log = self._open_state_log()
        
        # In-memory event buffer (for fast access during session)
        self.event_buffer: List[ReplayEvent] = []
        self.event_index: Dict[str, ReplayEvent] = {}
        
        # Thread safety
        self.lock = asyncio.Lock()
        
        # Performance optimization
        self.batch_size = 100
        self.batch_buffer: List[ReplayEvent] = []
        
        # Metrics
        self.metrics = {
            'events_logged': 0,
            'bytes_written': 0,
            'batch_flushes': 0,
            'verification_errors': 0
        }
    
    def log_event(self,
                 event_type: ReplayEventType,
                 payload: Dict,
                 parent_events: Optional[List[ReplayEvent]] = None,
                 criticality: int = 1,
                 source_component: str = "") -> ReplayEvent:
        """
        Log an event for replay
        Returns the created event
        """
        
        self.sequence_counter += 1
        
        # Create event
        event = ReplayEvent(
            event_id="",  # Will be auto-generated
            event_type=event_type,
            sequence_number=self.sequence_counter,
            timestamp=datetime.now(timezone.utc),
            payload=payload,
            session_id=self.session_id,
            process_id=str(hashlib.sha256(self.session_id.encode()).hexdigest()[:8]),
            parent_event_ids=[e.event_id for e in (parent_events or [])],
            source_component=source_component,
            criticality=criticality
        )
        
        # Store in memory
        self.event_buffer.append(event)
        self.event_index[event.event_id] = event
        
        # Add to batch buffer
        self.batch_buffer.append(event)
        
        # Flush if batch is full
        if len(self.batch_buffer) >= self.batch_size:
            self._flush_batch()
        
        self.metrics['events_logged'] += 1
        
        return event
    
    async def log_event_async(self,
                            event_type: ReplayEventType,
                            payload: Dict,
                            **kwargs) -> ReplayEvent:
        """Async version of log_event"""
        async with self.lock:
            return self.log_event(event_type, payload, **kwargs)
    
    def create_checkpoint(self,
                         system_state: Dict,
                         description: str = "") -> ReplayEvent:
        """
        Create a system checkpoint for faster replay
        """
        
        # Serialize and compress state
        state_serialized = self._serialize_state(system_state)
        state_checksum = hashlib.sha256(state_serialized).hexdigest()
        
        # Store checkpoint
        checkpoint_id = f"checkpoint_{self.session_id}_{self.sequence_counter + 1}"
        checkpoint_path = self.log_path / "checkpoints" / f"{checkpoint_id}.cpt"
        checkpoint_path.parent.mkdir(exist_ok=True)
        
        with open(checkpoint_path, 'wb') as f:
            # Write metadata
            metadata = {
                'checkpoint_id': checkpoint_id,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'sequence_number': self.sequence_counter + 1,
                'state_checksum': state_checksum,
                'description': description,
                'event_count': len(self.event_buffer)
            }
            f.write(json.dumps(metadata).encode() + b'\n')
            
            # Write compressed state
            compressed = zlib.compress(state_serialized, level=9)
            f.write(compressed)
        
        # Log checkpoint event
        checkpoint_event = self.log_event(
            event_type=ReplayEventType.CHECKPOINT_CREATED,
            payload={
                'checkpoint_id': checkpoint_id,
                'path': str(checkpoint_path),
                'state_checksum': state_checksum,
                'event_count_at_checkpoint': len(self.event_buffer),
                'description': description
            },
            criticality=5,  # Highest criticality
            source_component='replay_logger'
        )
        
        return checkpoint_event
    
    def _flush_batch(self):
        """Flush batch buffer to disk"""
        if not self.batch_buffer:
            return
        
        try:
            # Write events to WAL (Write-Ahead Log)
            wal_path = self.log_path / f"wal_{datetime.now().strftime('%Y%m%d_%H%M%S')}.bin"
            
            with open(wal_path, 'ab') as f:
                for event in self.batch_buffer:
                    # Serialize event
                    event_data = msgpack.packb(event.to_serializable(), use_bin_type=True)
                    
                    # Write with length prefix
                    length = len(event_data)
                    f.write(struct.pack('>I', length))
                    f.write(event_data)
                    
                    self.metrics['bytes_written'] += length + 4
            
            # Also append to main event log
            self._append_to_event_log(self.batch_buffer)
            
            # Clear batch buffer
            self.batch_buffer.clear()
            self.metrics['batch_flushes'] += 1
            
        except Exception as e:
            print(f"Error flushing replay batch: {e}")
            # Keep events in buffer for retry
    
    def _append_to_event_log(self, events: List[ReplayEvent]):
        """Append events to main event log"""
        try:
            # Use LMDB for efficient append
            with self.event_log.begin(write=True) as txn:
                for event in events:
                    txn.put(
                        f"event_{event.sequence_number:010d}".encode(),
                        msgpack.packb(event.to_serializable(), use_bin_type=True)
                    )
                    
                    # Also store by event_id for quick lookup
                    txn.put(
                        f"id_{event.event_id}".encode(),
                        msgpack.packb(event.to_serializable(), use_bin_type=True)
                    )
        except Exception as e:
            print(f"Error appending to event log: {e}")

# ==================== REPLAY ENGINE ====================

class DeterministicReplayEngine:
    """
    Engine for deterministic replay of system sessions
    Can rebuild full system state from logs
    """
    
    def __init__(self, log_base_path: Path):
        self.log_base_path = log_base_path
        
        # Replay state
        self.current_session: Optional[str] = None
        self.current_sequence: int = 0
        self.replay_speed: float = 1.0  # 1.0 = realtime, 0.0 = instant
        
        # Component rebuilders
        self.uwm_rebuilder = UWMRebuilder()
        self.smp_rebuilder = SMPRebuilder()
        self.memory_rebuilder = MemoryRebuilder()
        self.evaluation_rebuilder = EvaluationRebuilder()
        
        # Replay cache
        self.event_cache: Dict[str, ReplayEvent] = {}
        self.checkpoint_cache: Dict[str, Dict] = {}
        
        # Performance optimization
        self.preload_window = 1000  # Preload this many events ahead
        
        # Verification state
        self.verification_passed = True
        self.checksum_errors = []
        
    def find_sessions(self) -> List[Dict]:
        """Find all replayable sessions"""
        sessions = []
        
        for session_dir in self.log_base_path.iterdir():
            if session_dir.is_dir():
                # Check if it has replay logs
                event_log = session_dir / "events.lmdb"
                if event_log.exists():
                    session_info = self._read_session_info(session_dir)
                    sessions.append(session_info)
        
        # Sort by start time
        sessions.sort(key=lambda s: s.get('start_time', ''), reverse=True)
        return sessions
    
    def replay_session(self,
                      session_id: str,
                      start_sequence: int = 0,
                      end_sequence: Optional[int] = None,
                      granularity: ReplayGranularity = ReplayGranularity.OPERATIONAL,
                      callback: Optional[callable] = None) -> Dict:
        """
        Replay a session deterministically
        Returns replay report
        """
        
        session_path = self.log_base_path / session_id
        if not session_path.exists():
            raise ValueError(f"Session {session_id} not found")
        
        # Load session metadata
        session_info = self._read_session_info(session_path)
        
        # Find best starting checkpoint
        checkpoint = self._find_best_checkpoint(session_path, start_sequence)
        
        # Initialize system state from checkpoint
        system_state = {}
        if checkpoint:
            system_state = self._restore_from_checkpoint(checkpoint)
            start_sequence = checkpoint['sequence_number']
            print(f"Starting replay from checkpoint at sequence {start_sequence}")
        else:
            print(f"Starting replay from beginning (no suitable checkpoint found)")
        
        # Load events
        events = self._load_events(
            session_path, start_sequence, end_sequence, granularity
        )
        
        # Prepare replay report
        replay_report = {
            'session_id': session_id,
            'start_sequence': start_sequence,
            'end_sequence': end_sequence or events[-1].sequence_number if events else start_sequence,
            'total_events': len(events),
            'start_time': datetime.now(timezone.utc).isoformat(),
            'checkpoint_used': checkpoint['checkpoint_id'] if checkpoint else None,
            'granularity': granularity.name,
            'verification_errors': []
        }
        
        # Execute replay
        replayed_events = 0
        for event in events:
            try:
                # Verify event integrity
                if not event.verify():
                    self.checksum_errors.append({
                        'event_id': event.event_id,
                        'sequence': event.sequence_number,
                        'expected': event.checksum,
                        'calculated': event._calculate_checksum()
                    })
                    replay_report['verification_errors'].append(event.event_id)
                    continue
                
                # Apply event to system state
                system_state = self._apply_event(event, system_state, granularity)
                
                # Update sequence
                self.current_sequence = event.sequence_number
                replayed_events += 1
                
                # Call callback if provided
                if callback:
                    should_continue = callback(event, system_state)
                    if not should_continue:
                        break
                
                # Sleep for realistic timing if needed
                if self.replay_speed > 0 and len(events) > 1:
                    self._sleep_for_event_timing(event, events)
                    
            except Exception as e:
                print(f"Error replaying event {event.event_id}: {e}")
                replay_report.setdefault('replay_errors', []).append({
                    'event_id': event.event_id,
                    'error': str(e),
                    'sequence': event.sequence_number
                })
        
        # Finalize replay
        replay_report.update({
            'end_time': datetime.now(timezone.utc).isoformat(),
            'events_replayed': replayed_events,
            'final_system_state': self._summarize_state(system_state),
            'checksum_errors': len(self.checksum_errors),
            'replay_errors': len(replay_report.get('replay_errors', [])),
            'success_rate': replayed_events / len(events) if events else 0.0
        })
        
        return replay_report
    
    def rebuild_full_state(self,
                          session_id: str,
                          target_sequence: Optional[int] = None) -> Dict:
        """
        Rebuild complete system state at specific point
        Returns fully rehydrated system
        """
        
        session_path = self.log_base_path / session_id
        
        # Find checkpoint closest to target
        if target_sequence:
            checkpoint = self._find_best_checkpoint(session_path, target_sequence)
        else:
            # Find latest checkpoint
            checkpoint = self._find_latest_checkpoint(session_path)
        
        if checkpoint:
            # Restore from checkpoint
            system_state = self._restore_from_checkpoint(checkpoint)
            start_sequence = checkpoint['sequence_number']
        else:
            # Start from scratch
            system_state = {}
            start_sequence = 0
        
        # Replay events from checkpoint to target
        if target_sequence and target_sequence > start_sequence:
            events = self._load_events(
                session_path, start_sequence, target_sequence, ReplayGranularity.OPERATIONAL
            )
            
            for event in events:
                system_state = self._apply_event(event, system_state, ReplayGranularity.OPERATIONAL)
        
        # Rehydrate all components
        rehydrated = {
            'uwm': self.uwm_rebuilder.rehydrate(system_state.get('uwm', {})),
            'smps': self.smp_rebuilder.rehydrate(system_state.get('smps', {})),
            'memory': self.memory_rebuilder.rehydrate(system_state.get('memory', {})),
            'evaluations': self.evaluation_rebuilder.rehydrate(system_state.get('evaluations', {})),
            'agents': system_state.get('agents', {}),
            'metadata': {
                'session_id': session_id,
                'sequence_number': target_sequence or self.current_sequence,
                'rebuilt_at': datetime.now(timezone.utc).isoformat(),
                'source_checkpoint': checkpoint['checkpoint_id'] if checkpoint else None
            }
        }
        
        return rehydrated
    
    def _apply_event(self,
                    event: ReplayEvent,
                    system_state: Dict,
                    granularity: ReplayGranularity) -> Dict:
        """
        Apply event to system state
        """
        
        # Skip low-criticality events if granularity is SUMMARY
        if (granularity == ReplayGranularity.SUMMARY and 
            event.criticality < 3):
            return system_state
        
        event_type = event.event_type
        payload = event.payload
        
        if event_type == ReplayEventType.UWM_EVENT_APPROVED:
            # Apply UWM event
            uwm_state = system_state.get('uwm', {})
            uwm_state = self.uwm_rebuilder.apply_event(payload, uwm_state)
            system_state['uwm'] = uwm_state
            
        elif event_type == ReplayEventType.SMP_GENERATED:
            # Add SMP
            smps = system_state.get('smps', {})
            smp_id = payload.get('smp_id')
            if smp_id:
                smps[smp_id] = payload
                system_state['smps'] = smps
                
        elif event_type == ReplayEventType.SMP_EVALUATED:
            # Update SMP with evaluation
            smps = system_state.get('smps', {})
            smp_id = payload.get('smp_id')
            if smp_id in smps:
                smps[smp_id]['evaluation'] = payload.get('evaluation', {})
                system_state['smps'] = smps
                
        elif event_type == ReplayEventType.MEMORY_CREATED:
            # Add memory
            memories = system_state.get('memory', {})
            memory_id = payload.get('memory_id')
            if memory_id:
                memories[memory_id] = payload
                system_state['memory'] = memories
                
        elif event_type == ReplayEventType.AGENT_ACTION:
            # Record agent action
            agents = system_state.get('agents', {})
            agent_id = payload.get('agent_id')
            if agent_id:
                if agent_id not in agents:
                    agents[agent_id] = {'actions': []}
                agents[agent_id]['actions'].append(payload)
                system_state['agents'] = agents
        
        # Store event in history
        history = system_state.get('event_history', [])
        history.append({
            'sequence': event.sequence_number,
            'type': event.event_type.name,
            'timestamp': event.timestamp.isoformat(),
            'event_id': event.event_id
        })
        system_state['event_history'] = history
        
        return system_state

# ==================== REHYDRATION SYSTEMS ====================

class UWMRebuilder:
    """Rebuilds UWM state from events"""
    
    def rehydrate(self, uwm_state: Dict) -> Dict:
        """Rehydrate UWM from serialized state"""
        
        rehydrated = {
            'atoms': {},
            'events': [],
            'constraints': [],
            'temporal_layers': {}
        }
        
        # Rebuild atoms
        for atom_id, atom_data in uwm_state.get('atoms', {}).items():
            rehydrated['atoms'][atom_id] = self._rehydrate_atom(atom_data)
        
        # Rebuild temporal layers
        for layer_name, layer_data in uwm_state.get('temporal_layers', {}).items():
            rehydrated['temporal_layers'][layer_name] = self._rehydrate_temporal_layer(layer_data)
        
        # Rebuild constraints
        rehydrated['constraints'] = uwm_state.get('constraints', [])
        
        # Rebuild evidence chains
        rehydrated['evidence_chains'] = self._rebuild_evidence_chains(uwm_state)
        
        return rehydrated
    
    def apply_event(self, event_payload: Dict, current_state: Dict) -> Dict:
        """Apply UWM event to current state"""
        
        event_type = event_payload.get('event_type')
        
        if event_type == 'CREATE':
            atom_id = event_payload.get('atom_id')
            if atom_id:
                current_state.setdefault('atoms', {})[atom_id] = {
                    'data': event_payload.get('atom_data', {}),
                    'created_at': event_payload.get('timestamp'),
                    'evidence': event_payload.get('evidence', []),
                    'valid_from': event_payload.get('timestamp'),
                    'valid_to': None
                }
                
        elif event_type == 'UPDATE':
            atom_id = event_payload.get('atom_id')
            if atom_id in current_state.get('atoms', {}):
                # Close old atom
                current_state['atoms'][atom_id]['valid_to'] = event_payload.get('timestamp')
                
                # Create new version
                new_atom_id = f"{atom_id}_v{event_payload.get('version', 2)}"
                current_state['atoms'][new_atom_id] = {
                    'data': event_payload.get('new_data', {}),
                    'created_at': event_payload.get('timestamp'),
                    'evidence': event_payload.get('evidence', []),
                    'valid_from': event_payload.get('timestamp'),
                    'valid_to': None,
                    'supersedes': atom_id
                }
        
        return current_state

class SMPRebuilder:
    """Rebuilds SMP state from events"""
    
    def rehydrate(self, smp_state: Dict) -> Dict:
        """Rehydrate SMPs from serialized state"""
        
        rehydrated = {
            'smps': {},
            'generation_stats': {},
            'evaluation_stats': {},
            'approval_chain': {}
        }
        
        # Rebuild individual SMPs
        for smp_id, smp_data in smp_state.get('smps', {}).items():
            rehydrated['smps'][smp_id] = self._rehydrate_smp(smp_data)
        
        # Rebuild statistics
        rehydrated['generation_stats'] = self._rebuild_generation_stats(smp_state)
        rehydrated['evaluation_stats'] = self._rebuild_evaluation_stats(smp_state)
        rehydrated['approval_chain'] = self._rebuild_approval_chain(smp_state)
        
        return rehydrated

class MemoryRebuilder:
    """Rebuilds memory state from events"""
    
    def rehydrate(self, memory_state: Dict) -> Dict:
        """Rehydrate memory system"""
        
        rehydrated = {
            'traces': {},
            'concepts': {},
            'associations': {},
            'access_patterns': {},
            'consolidation_state': {}
        }
        
        # Rebuild memory traces
        for trace_id, trace_data in memory_state.get('traces', {}).items():
            rehydrated['traces'][trace_id] = self._rehydrate_trace(trace_data)
        
        # Rebuild concepts
        for concept_id, concept_data in memory_state.get('concepts', {}).items():
            rehydrated['concepts'][concept_id] = self._rehydrate_concept(concept_data)
        
        # Rebuild associations graph
        rehydrated['associations'] = self._rebuild_association_graph(memory_state)
        
        # Rebuild access patterns
        rehydrated['access_patterns'] = memory_state.get('access_patterns', {})
        
        return rehydrated

# ==================== FORENSIC ANALYSIS TOOLS ====================

class ForensicAnalyzer:
    """
    Tools for forensic analysis of replay logs
    """
    
    def __init__(self, replay_engine: DeterministicReplayEngine):
        self.engine = replay_engine
        self.analysis_cache = {}
    
    def analyze_causal_chain(self,
                            session_id: str,
                            target_event_id: str,
                            max_depth: int = 10) -> Dict:
        """
        Analyze causal chain leading to an event
        """
        
        # Load session events
        session_path = self.engine.log_base_path / session_id
        events = self._load_all_events(session_path)
        
        # Find target event
        target_event = None
        for event in events:
            if event.event_id == target_event_id:
                target_event = event
                break
        
        if not target_event:
            return {'error': f'Event {target_event_id} not found'}
        
        # Build causal chain
        causal_chain = []
        visited = set()
        
        def trace_causes(event: ReplayEvent, depth: int):
            if depth >= max_depth or event.event_id in visited:
                return
            
            visited.add(event.event_id)
            causal_chain.append({
                'depth': depth,
                'event': event,
                'relationship': 'cause' if depth > 0 else 'target'
            })
            
            # Trace parent events
            for parent_id in event.parent_event_ids:
                parent_event = self._find_event_by_id(events, parent_id)
                if parent_event:
                    trace_causes(parent_event, depth + 1)
        
        trace_causes(target_event, 0)
        
        # Sort by depth (closest causes first)
        causal_chain.sort(key=lambda x: x['depth'])
        
        # Analyze patterns
        patterns = self._analyze_causal_patterns(causal_chain)
        
        return {
            'target_event': target_event.to_serializable(),
            'causal_chain': [item['event'].to_serializable() for item in causal_chain],
            'chain_length': len(causal_chain),
            'max_depth_reached': max(item['depth'] for item in causal_chain),
            'patterns': patterns,
            'critical_events': self._identify_critical_events(causal_chain)
        }
    
    def timeline_analysis(self,
                         session_id: str,
                         time_window: Optional[Tuple[datetime, datetime]] = None) -> Dict:
        """
        Analyze events in a timeline
        """
        
        session_path = self.engine.log_base_path / session_id
        events = self._load_all_events(session_path)
        
        # Filter by time window
        if time_window:
            events = [e for e in events if time_window[0] <= e.timestamp <= time_window[1]]
        
        # Group by event type
        by_type = defaultdict(list)
        for event in events:
            by_type[event.event_type.name].append(event)
        
        # Calculate statistics
        stats = {
            'total_events': len(events),
            'event_types': len(by_type),
            'events_by_type': {k: len(v) for k, v in by_type.items()},
            'time_span': {
                'start': min(e.timestamp for e in events).isoformat() if events else None,
                'end': max(e.timestamp for e in events).isoformat() if events else None,
                'duration_seconds': (
                    (max(e.timestamp for e in events) - min(e.timestamp for e in events)).total_seconds()
                    if len(events) > 1 else 0
                )
            },
            'event_rate': len(events) / max(1, stats['time_span']['duration_seconds']),
            'criticality_distribution': self._calculate_criticality_distribution(events)
        }
        
        # Detect anomalies
        anomalies = self._detect_timeline_anomalies(events)
        
        return {
            'statistics': stats,
            'anomalies': anomalies,
            'event_density': self._calculate_event_density(events),
            'component_activity': self._analyze_component_activity(events)
        }
    
    def compare_sessions(self,
                        session_ids: List[str],
                        comparison_type: str = 'structural') -> Dict:
        """
        Compare multiple sessions
        """
        
        sessions_data = []
        for session_id in session_ids:
            session_path = self.engine.log_base_path / session_id
            if session_path.exists():
                # Rebuild state at end of session
                state = self.engine.rebuild_full_state(session_id)
                sessions_data.append({
                    'session_id': session_id,
                    'state': state,
                    'metadata': self._read_session_info(session_path)
                })
        
        if len(sessions_data) < 2:
            return {'error': 'Need at least 2 sessions to compare'}
        
        # Perform comparison
        comparisons = {}
        
        if comparison_type == 'structural':
            comparisons = self._compare_structural(sessions_data)
        elif comparison_type == 'behavioral':
            comparisons = self._compare_behavioral(sessions_data)
        elif comparison_type == 'outcome':
            comparisons = self._compare_outcomes(sessions_data)
        
        return {
            'sessions_compared': [d['session_id'] for d in sessions_data],
            'comparison_type': comparison_type,
            'similarity_scores': self._calculate_similarity_scores(sessions_data),
            'differences': comparisons.get('differences', {}),
            'common_patterns': comparisons.get('common_patterns', []),
            'divergence_points': self._find_divergence_points(sessions_data)
        }

# ==================== REPLAY VALIDATION SYSTEM ====================

class ReplayValidator:
    """
    Validates replay correctness and determinism
    """
    
    def __init__(self):
        self.validation_rules = self._load_validation_rules()
        self.reference_replays = {}  # Known-good replays for comparison
    
    def validate_replay(self,
                       replay_report: Dict,
                       original_session: Optional[Dict] = None) -> Dict:
        """
        Validate a replay for correctness
        """
        
        validation_results = {
            'passed': True,
            'checks': [],
            'errors': [],
            'warnings': []
        }
        
        # Check 1: Event sequence integrity
        seq_check = self._check_sequence_integrity(replay_report)
        validation_results['checks'].append(seq_check)
        if not seq_check['passed']:
            validation_results['passed'] = False
            validation_results['errors'].append('Sequence integrity check failed')
        
        # Check 2: Checksum validation
        checksum_check = self._check_checksums(replay_report)
        validation_results['checks'].append(checksum_check)
        if not checksum_check['passed']:
            validation_results['passed'] = False
            validation_results['errors'].append('Checksum validation failed')
        
        # Check 3: Temporal consistency
        temporal_check = self._check_temporal_consistency(replay_report)
        validation_results['checks'].append(temporal_check)
        if not temporal_check['passed']:
            validation_results['warnings'].append('Temporal consistency issues')
        
        # Check 4: State consistency
        state_check = self._check_state_consistency(replay_report)
        validation_results['checks'].append(state_check)
        if not state_check['passed']:
            validation_results['passed'] = False
            validation_results['errors'].append('State consistency check failed')
        
        # Check 5: Determinism (if we have reference)
        if original_session:
            determinism_check = self._check_determinism(replay_report, original_session)
            validation_results['checks'].append(determinism_check)
            if not determinism_check['passed']:
                validation_results['passed'] = False
                validation_results['errors'].append('Non-deterministic replay detected')
        
        # Calculate validation score
        passed_checks = sum(1 for check in validation_results['checks'] if check['passed'])
        total_checks = len(validation_results['checks'])
        validation_results['score'] = passed_checks / total_checks if total_checks > 0 else 0
        
        return validation_results
    
    def create_reference_replay(self,
                               session_id: str,
                               description: str = "") -> str:
        """
        Create a reference replay for future validation
        Returns reference ID
        """
        
        reference_id = f"ref_{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Store reference replay
        self.reference_replays[reference_id] = {
            'session_id': session_id,
            'created_at': datetime.now(timezone.utc).isoformat(),
            'description': description,
            'expected_checksums': {},  # Will be populated during replay
            'expected_state_hashes': {},
            'validation_rules_used': list(self.validation_rules.keys())
        }
        
        return reference_id

# ==================== COMPLETE REPLAY INFRASTRUCTURE ====================

class CompleteReplayInfrastructure:
    """
    Complete replay infrastructure integrating all components
    """
    
    def __init__(self, base_path: Path = Path("./replay_logs")):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Core components
        self.logger = None
        self.engine = DeterministicReplayEngine(self.base_path)
        self.analyzer = ForensicAnalyzer(self.engine)
        self.validator = ReplayValidator()
        
        # Session management
        self.active_sessions: Dict[str, DeterministicReplayLogger] = {}
        
        # Configuration
        self.config = {
            'auto_checkpoint_interval': 1000,  # events
            'auto_checkpoint_size_mb': 100,     # MB
            'compression_level': 9,
            'encryption_enabled': False,
            'retention_days': 30
        }
        
        # Statistics
        self.stats = {
            'sessions_recorded': 0,
            'events_logged': 0,
            'replays_performed': 0,
            'storage_used_mb': 0
        }
    
    def start_session(self, session_id: str, description: str = "") -> str:
        """
        Start a new replayable session
        Returns the actual session ID (may have timestamp appended)
        """
        
        # Add timestamp for uniqueness
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = f"{session_id}_{timestamp}"
        
        # Create session directory
        session_path = self.base_path / unique_id
        session_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize logger
        logger = DeterministicReplayLogger(session_path, unique_id)
        self.active_sessions[unique_id] = logger
        
        # Store session metadata
        metadata = {
            'session_id': unique_id,
            'original_id': session_id,
            'description': description,
            'start_time': datetime.now(timezone.utc).isoformat(),
            'config': self.config,
            'system_info': self._get_system_info()
        }
        
        metadata_path = session_path / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Log session start event
        logger.log_event(
            event_type=ReplayEventType.SYSTEM_INIT,
            payload={
                'session_metadata': metadata,
                'config': self.config,
                'infrastructure_version': '1.0.0'
            },
            criticality=5,
            source_component='replay_infrastructure'
        )
        
        self.stats['sessions_recorded'] += 1
        
        return unique_id
    
    def get_logger(self, session_id: str) -> Optional[DeterministicReplayLogger]:
        """Get logger for active session"""
        return self.active_sessions.get(session_id)
    
    async def log_event(self,
                       session_id: str,
                       event_type: ReplayEventType,
                       payload: Dict,
                       **kwargs) -> Optional[ReplayEvent]:
        """
        Log an event in the specified session
        """
        
        logger = self.get_logger(session_id)
        if not logger:
            # Try to find session directory
            session_path = self.base_path / session_id
            if session_path.exists():
                # Reopen logger
                logger = DeterministicReplayLogger(session_path, session_id)
                self.active_sessions[session_id] = logger
            else:
                raise ValueError(f"Session {session_id} not found")
        
        # Log the event
        event = await logger.log_event_async(event_type, payload, **kwargs)
        
        # Check if we need to create an auto-checkpoint
        if (logger.sequence_counter % self.config['auto_checkpoint_interval'] == 0 and
            logger.sequence_counter > 0):
            
            # Estimate state size
            state_size = len(pickle.dumps(logger.event_buffer))
            if state_size > self.config['auto_checkpoint_size_mb'] * 1024 * 1024:
                # Create checkpoint
                system_state = self._capture_current_state(session_id)
                logger.create_checkpoint(
                    system_state,
                    description=f"Auto-checkpoint at event {logger.sequence_counter}"
                )
        
        self.stats['events_logged'] += 1
        
        return event
    
    def end_session(self, session_id: str, final_state: Optional[Dict] = None):
        """
        Properly end a session
        """
        
        logger = self.get_logger(session_id)
        if not logger:
            return
        
        # Create final checkpoint if state provided
        if final_state:
            logger.create_checkpoint(
                final_state,
                description="Final session checkpoint"
            )
        
        # Log session end event
        logger.log_event(
            event_type=ReplayEventType.SYSTEM_SHUTDOWN,
            payload={
                'final_sequence': logger.sequence_counter,
                'total_events': len(logger.event_buffer),
                'session_duration': (
                    datetime.now(timezone.utc) - 
                    logger.event_buffer[0].timestamp if logger.event_buffer else 
                    timedelta(0)
                ).total_seconds()
            },
            criticality=5,
            source_component='replay_infrastructure'
        )
        
        # Flush any remaining events
        logger._flush_batch()
        
        # Close log files
        if hasattr(logger, 'event_log'):
            logger.event_log.close()
        
        # Remove from active sessions
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
        
        # Update session metadata
        session_path = self.base_path / session_id
        metadata_path = session_path / "metadata.json"
        
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            metadata.update({
                'end_time': datetime.now(timezone.utc).isoformat(),
                'total_events': logger.sequence_counter,
                'final_checkpoint': f"checkpoint_{session_id}_{logger.sequence_counter}",
                'metrics': logger.metrics
            })
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
    
    def replay_and_analyze(self,
                          session_id: str,
                          analysis_type: str = "full",
                          **replay_kwargs) -> Dict:
        """
        Perform replay with integrated analysis
        """
        
        # First, replay the session
        replay_report = self.engine.replay_session(session_id, **replay_kwargs)
        
        # Then perform analysis
        analysis_results = {}
        
        if analysis_type == "full" or analysis_type == "causal":
            # Analyze causal chains for key events
            key_events = self._identify_key_events(session_id)
            causal_chains = {}
            
            for event_id in key_events[:5]:  # Limit to first 5 key events
                chain = self.analyzer.analyze_causal_chain(session_id, event_id)
                causal_chains[event_id] = chain
        
        if analysis_type == "full" or analysis_type == "timeline":
            # Perform timeline analysis
            timeline = self.analyzer.timeline_analysis(session_id)
            analysis_results['timeline'] = timeline
        
        if analysis_type == "full" or analysis_type == "validation":
            # Validate the replay
            original_metadata = self._read_session_info(self.base_path / session_id)
            validation = self.validator.validate_replay(replay_report, original_metadata)
            analysis_results['validation'] = validation
        
        # Combine results
        combined_report = {
            'replay': replay_report,
            'analysis': analysis_results,
            'session_id': session_id,
            'analyzed_at': datetime.now(timezone.utc).isoformat()
        }
        
        self.stats['replays_performed'] += 1
        
        return combined_report
    
    def batch_replay(self,
                    session_ids: List[str],
                    comparison_type: str = "structural") -> Dict:
        """
        Replay multiple sessions and compare them
        """
        
        all_reports = {}
        
        for session_id in session_ids:
            try:
                report = self.replay_and_analyze(
                    session_id,
                    analysis_type="validation"
                )
                all_reports[session_id] = report
            except Exception as e:
                all_reports[session_id] = {'error': str(e)}
        
        # Compare sessions if we have at least 2 successful replays
        successful_reports = {
            sid: report for sid, report in all_reports.items()
            if 'error' not in report
        }
        
        comparison = {}
        if len(successful_reports) >= 2:
            comparison = self.analyzer.compare_sessions(
                list(successful_reports.keys()),
                comparison_type=comparison_type
            )
        
        return {
            'individual_reports': all_reports,
            'comparison': comparison,
            'summary': {
                'total_sessions': len(session_ids),
                'successful_replays': len(successful_reports),
                'failed_replays': len(session_ids) - len(successful_reports)
            }
        }
    
    def export_session(self,
                      session_id: str,
                      export_format: str = "json",
                      include_data: List[str] = None) -> Path:
        """
        Export a session for external analysis
        """
        
        session_path = self.base_path / session_id
        export_dir = self.base_path / "exports" / session_id
        export_dir.mkdir(parents=True, exist_ok=True)
        
        # What to include
        if include_data is None:
            include_data = ['events', 'checkpoints', 'metadata', 'analysis']
        
        export_files = {}
        
        # Export events
        if 'events' in include_data:
            events = self._load_all_events(session_path)
            
            if export_format == "json":
                events_file = export_dir / "events.json"
                events_data = [e.to_serializable() for e in events]
                with open(events_file, 'w') as f:
                    json.dump(events_data, f, indent=2)
                export_files['events'] = str(events_file)
            
            elif export_format == "csv":
                events_file = export_dir / "events.csv"
                self._export_events_to_csv(events, events_file)
                export_files['events'] = str(events_file)
        
        # Export checkpoints
        if 'checkpoints' in include_data:
            checkpoints_dir = export_dir / "checkpoints"
            checkpoints_dir.mkdir(exist_ok=True)
            
            checkpoint_files = list((session_path / "checkpoints").glob("*.cpt"))
            for cp_file in checkpoint_files:
                export_cp_file = checkpoints_dir / cp_file.name
                export_cp_file.write_bytes(cp_file.read_bytes())
            
            export_files['checkpoints'] = str(checkpoints_dir)
        
        # Export metadata
        if 'metadata' in include_data:
            metadata_file = export_dir / "metadata.json"
            metadata = self._read_session_info(session_path)
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            export_files['metadata'] = str(metadata_file)
        
        # Export analysis
        if 'analysis' in include_data:
            analysis = self.replay_and_analyze(session_id, analysis_type="full")
            analysis_file = export_dir / "analysis.json"
            with open(analysis_file, 'w') as f:
                json.dump(analysis, f, indent=2)
            export_files['analysis'] = str(analysis_file)
        
        # Create manifest
        manifest = {
            'session_id': session_id,
            'exported_at': datetime.now(timezone.utc).isoformat(),
            'format': export_format,
            'files': export_files,
            'infrastructure_version': '1.0.0'
        }
        
        manifest_file = export_dir / "manifest.json"
        with open(manifest_file, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        # Create archive if requested
        if export_format == "archive":
            archive_path = self.base_path / "exports" / f"{session_id}.zip"
            self._create_archive(export_dir, archive_path)
            return archive_path
        
        return export_dir
    
    def cleanup_old_sessions(self, max_age_days: int = None):
        """
        Clean up old session data
        """
        
        if max_age_days is None:
            max_age_days = self.config['retention_days']
        
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=max_age_days)
        
        for session_dir in self.base_path.iterdir():
            if not session_dir.is_dir():
                continue
            
            # Check metadata for session age
            metadata_path = session_dir / "metadata.json"
            if metadata_path.exists():
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    end_time_str = metadata.get('end_time')
                    if end_time_str:
                        end_time = datetime.fromisoformat(end_time_str)
                        if end_time < cutoff_date:
                            # Archive and delete
                            self._archive_session(session_dir.name)
                            self._delete_session(session_dir)
                except Exception as e:
                    print(f"Error cleaning up session {session_dir.name}: {e}")
    
    # ==================== HELPER METHODS ====================
    
    def _capture_current_state(self, session_id: str) -> Dict:
        """Capture current system state from active session"""
        logger = self.get_logger(session_id)
        if not logger:
            return {}
        
        # Rebuild state from logged events
        return self.engine.rebuild_full_state(session_id)
    
    def _identify_key_events(self, session_id: str) -> List[str]:
        """Identify key/critical events in a session"""
        session_path = self.base_path / session_id
        events = self._load_all_events(session_path)
        
        key_events = []
        for event in events:
            if event.criticality >= 4:  # High criticality events
                key_events.append(event.event_id)
            elif event.event_type in [
                ReplayEventType.SYSTEM_INIT,
                ReplayEventType.SYSTEM_SHUTDOWN,
                ReplayEventType.CHECKPOINT_CREATED,
                ReplayEventType.SMP_APPROVED,
                ReplayEventType.UWM_EVENT_APPROVED,
                ReplayEventType.TRIUNE_COMMUTATION
            ]:
                key_events.append(event.event_id)
        
        return key_events[:20]  # Limit to top 20
    
    def _load_all_events(self, session_path: Path) -> List[ReplayEvent]:
        """Load all events from a session"""
        events = []
        
        # Try LMDB first
        event_log_path = session_path / "events.lmdb"
        if event_log_path.exists():
            env = lmdb.open(str(event_log_path), readonly=True, max_dbs=1)
            with env.begin() as txn:
                cursor = txn.cursor()
                for key, value in cursor:
                    if key.startswith(b"event_"):
                        try:
                            event_data = msgpack.unpackb(value, raw=False)
                            event = ReplayEvent.from_serializable(event_data)
                            events.append(event)
                        except Exception as e:
                            print(f"Error loading event: {e}")
        
        # Sort by sequence number
        events.sort(key=lambda x: x.sequence_number)
        return events
    
    def _export_events_to_csv(self, events: List[ReplayEvent], output_path: Path):
        """Export events to CSV format"""
        if not events:
            return
        
        fieldnames = [
            'sequence_number', 'timestamp', 'event_type', 'event_id',
            'criticality', 'source_component', 'session_id',
            'parent_count', 'payload_keys'
        ]
        
        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for event in events:
                writer.writerow({
                    'sequence_number': event.sequence_number,
                    'timestamp': event.timestamp.isoformat(),
                    'event_type': event.event_type.name,
                    'event_id': event.event_id,
                    'criticality': event.criticality,
                    'source_component': event.source_component,
                    'session_id': event.session_id,
                    'parent_count': len(event.parent_event_ids),
                    'payload_keys': ','.join(event.payload.keys())
                })
    
    def _read_session_info(self, session_path: Path) -> Dict:
        """Read session metadata"""
        metadata_path = session_path / "metadata.json"
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    return json.load(f)
            except:
                pass
        
        # Fallback: infer from directory
        return {
            'session_id': session_path.name,
            'path': str(session_path),
            'exists': True
        }
    
    def _get_system_info(self) -> Dict:
        """Get system information for metadata"""
        import platform
        import sys
        
        return {
            'platform': platform.platform(),
            'python_version': sys.version,
            'system_time': datetime.now(timezone.utc).isoformat(),
            'replay_infrastructure_version': '1.0.0'
        }
    
    def _create_archive(self, source_dir: Path, target_path: Path):
        """Create archive of exported data"""
        import shutil
        
        # Create zip archive
        shutil.make_archive(
            str(target_path.with_suffix('')),  # Remove .zip for make_archive
            'zip',
            str(source_dir)
        )
    
    def _archive_session(self, session_id: str):
        """Archive a session before deletion"""
        archive_dir = self.base_path / "archived"
        archive_dir.mkdir(exist_ok=True)
        
        session_dir = self.base_path / session_id
        if session_dir.exists():
            # Create archive
            timestamp = datetime.now().strftime("%Y%m%d")
            archive_path = archive_dir / f"{session_id}_{timestamp}.zip"
            self._create_archive(session_dir, archive_path)
    
    def _delete_session(self, session_dir: Path):
        """Delete session directory"""
        import shutil
        
        try:
            shutil.rmtree(session_dir)
            print(f"Deleted session: {session_dir.name}")
        except Exception as e:
            print(f"Error deleting session {session_dir.name}: {e}")
    
    def get_statistics(self) -> Dict:
        """Get infrastructure statistics"""
        # Calculate storage usage
        storage_bytes = 0
        for session_dir in self.base_path.iterdir():
            if session_dir.is_dir():
                storage_bytes += sum(f.stat().st_size for f in session_dir.rglob('*') if f.is_file())
        
        self.stats['storage_used_mb'] = storage_bytes / (1024 * 1024)
        
        # Session counts
        session_dirs = [d for d in self.base_path.iterdir() if d.is_dir()]
        active_count = len(self.active_sessions)
        archived_count = len(list((self.base_path / "archived").iterdir())) if (self.base_path / "archived").exists() else 0
        
        return {
            **self.stats,
            'total_sessions_stored': len(session_dirs),
            'active_sessions': active_count,
            'archived_sessions': archived_count,
            'config': self.config,
            'base_path': str(self.base_path)
        }

# ==================== UTILITY FUNCTIONS ====================

def create_replay_infrastructure(config: Dict = None) -> CompleteReplayInfrastructure:
    """Factory function to create replay infrastructure"""
    default_config = {
        'base_path': './replay_logs',
        'auto_checkpoint_interval': 1000,
        'auto_checkpoint_size_mb': 100,
        'compression_level': 9,
        'retention_days': 30
    }
    
    if config:
        default_config.update(config)
    
    infrastructure = CompleteReplayInfrastructure(
        base_path=Path(default_config['base_path'])
    )
    
    # Update config
    for key, value in default_config.items():
        if key != 'base_path' and key in infrastructure.config:
            infrastructure.config[key] = value
    
    return infrastructure

def quick_replay_analysis(session_id: str, 
                         replay_speed: float = 10.0,
                         show_progress: bool = True) -> Dict:
    """
    Quick replay and analysis for debugging
    """
    
    infra = create_replay_infrastructure()
    
    def progress_callback(event, state):
        if show_progress:
            print(f"Replaying event {event.sequence_number}: {event.event_type.name}")
            if event.criticality >= 4:
                print(f"  Critical event: {event.event_id}")
        return True
    
    # Perform replay
    report = infra.replay_and_analyze(
        session_id,
        granularity=ReplayGranularity.SUMMARY,
        replay_speed=replay_speed,
        callback=progress_callback if show_progress else None
    )
    
    # Print summary
    if show_progress:
        print("\n" + "="*60)
        print(f"REPLAY COMPLETE: {session_id}")
        print("="*60)
        print(f"Events replayed: {report['replay']['events_replayed']}")
        print(f"Success rate: {report['replay']['success_rate']:.1%}")
        print(f"Duration: {report['replay']['end_time']}")
        
        if 'analysis' in report and 'validation' in report['analysis']:
            validation = report['analysis']['validation']
            print(f"Validation: {'PASSED' if validation['passed'] else 'FAILED'}")
            print(f"Validation score: {validation['score']:.1%}")
    
    return report

# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    # Example usage
    print("Initializing Replay Infrastructure...")
    
    # Create infrastructure
    infra = create_replay_infrastructure()
    
    # Start a new session
    session_id = infra.start_session(
        "example_session",
        description="Example replay session for demonstration"
    )
    
    print(f"Started session: {session_id}")
    
    # Get logger for the session
    logger = infra.get_logger(session_id)
    
    # Log some example events
    logger.log_event(
        ReplayEventType.AGENT_CREATED,
        payload={
            'agent_id': 'agent_001',
            'agent_type': 'reasoning',
            'capabilities': ['inference', 'planning'],
            'initial_state': {'confidence': 0.8}
        },
        source_component='agent_manager',
        criticality=3
    )
    
    logger.log_event(
        ReplayEventType.SMP_GENERATED,
        payload={
            'smp_id': 'smp_001',
            'content': 'Example strategic plan',
            'generator_agent': 'agent_001',
            'complexity': 0.7,
            'novelty': 0.5
        },
        source_component='smp_generator',
        criticality=4
    )
    
    # Create a checkpoint
    current_state = {
        'agents': {'agent_001': {'status': 'active'}},
        'smps': {'smp_001': {'status': 'generated'}},
        'session_info': {'event_count': logger.sequence_counter}
    }
    
    checkpoint_event = logger.create_checkpoint(
        current_state,
        description="Example checkpoint after initial events"
    )
    
    print(f"Created checkpoint: {checkpoint_event.event_id}")
    
    # End session
    infra.end_session(session_id, final_state=current_state)
    
    print(f"Ended session. Total events: {logger.sequence_counter}")
    
    # Replay the session
    print("\nReplaying session...")
    replay_report = infra.replay_and_analyze(session_id, analysis_type="full")
    
    print(f"\nReplay completed. Success rate: {replay_report['replay']['success_rate']:.1%}")
    
    # Export session
    export_path = infra.export_session(session_id, export_format="json")
    print(f"Session exported to: {export_path}")
    
    # Show statistics
    stats = infra.get_statistics()
    print(f"\nInfrastructure Statistics:")
    print(f"  Sessions recorded: {stats['sessions_recorded']}")
    print(f"  Events logged: {stats['events_logged']}")
    print(f"  Storage used: {stats['storage_used_mb']:.2f} MB")
    
    print("\nReplay infrastructure example completed successfully!")