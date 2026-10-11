"""[NEW] Simulator: Event generation with variable clock speed.

This module provides:
1. Clock speeds: 1x (real-time), 10x, 60x for demo acceleration
2. Scenario generation: surge, doctor drops, simultaneous criticals, empty pool
3. Event injection into the replay stream
4. Reproducible scenarios with seed control
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Callable, Iterator
import random
import uuid

import pandas as pd
import numpy as np


class ClockSpeed(str, Enum):
    """Simulation clock speed multipliers."""
    REAL_TIME = "1x"
    FAST = "10x"
    TURBO = "60x"
    
    @property
    def multiplier(self) -> float:
        return {"1x": 1.0, "10x": 10.0, "60x": 60.0}[self.value]


class ScenarioType(str, Enum):
    """Pre-built stress scenarios."""
    NORMAL = "normal"           # Baseline operation
    SURGE = "surge"             # 2-5x demand spike at one hospital
    DOCTOR_DROPS = "doctor_drops"  # A physician goes offline mid-shift
    SIMULTANEOUS_CRITICALS = "simultaneous_criticals"  # Multiple stroke arrivals
    EMPTY_POOL = "empty_pool"   # No eligible physicians for a task type
    
    @property
    def description(self) -> str:
        return {
            "normal": "Normal operations (baseline demand)",
            "surge": "Demand surge: 2-5x arrivals at one hospital for 30-90 minutes",
            "doctor_drops": "A physician goes offline mid-shift unexpectedly",
            "simultaneous_criticals": "3+ stroke consults arrive within 5 minutes",
            "empty_pool": "No physicians available for video consults",
        }[self.value]


# ============================================================================
# SIMULATED EVENTS
# ============================================================================

@dataclass
class SimulatedEvent:
    """A generated event for injection into the stream."""
    event_id: str
    event_time: pd.Timestamp
    event_type: str
    consult_id: str | None = None
    hospital_id: str | None = None
    physician_id: str | None = None
    details: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "event_time": self.event_time,
            "ingested_at": self.event_time,
            "event_type": self.event_type,
            "consult_id": self.consult_id or "",
            "hospital_id": self.hospital_id or "",
            "physician_id": self.physician_id or "",
            **self.details,
        }


# ============================================================================
# SIMULATOR
# ============================================================================

class Simulator:
    """Event simulator with variable clock speed and scenario support.
    
    Usage:
        sim = Simulator(speed=ClockSpeed.FAST, scenario=ScenarioType.SURGE)
        sim.start(now)
        
        while sim.running:
            events = sim.tick(elapsed_seconds=1)
            for event in events:
                # inject into stream
    """
    
    # Hospital IDs for synthetic data
    HOSPITALS = [f"SYN-{i:03d}" for i in range(1, 13)]
    
    # Task types
    TASK_TYPES = ["contact", "imaging_review", "video", "documentation"]
    
    def __init__(
        self,
        speed: ClockSpeed = ClockSpeed.REAL_TIME,
        scenario: ScenarioType = ScenarioType.NORMAL,
        seed: int | None = None,
    ):
        self.speed = speed
        self.scenario = scenario
        self.seed = seed or random.randint(0, 999999)
        self._rng = random.Random(self.seed)
        self._np_rng = np.random.default_rng(self.seed)
        
        self._running = False
        self._start_time: pd.Timestamp | None = None
        self._sim_time: pd.Timestamp | None = None
        self._event_counter = 0
        
        # Scenario state
        self._surge_hospital: str | None = None
        self._surge_start: pd.Timestamp | None = None
        self._surge_end: pd.Timestamp | None = None
        self._dropped_physician: str | None = None
        self._scenario_triggered = False
    
    @property
    def running(self) -> bool:
        return self._running
    
    @property
    def sim_time(self) -> pd.Timestamp | None:
        return self._sim_time
    
    def start(self, now: pd.Timestamp) -> None:
        """Start the simulation at the given time."""
        self._running = True
        self._start_time = now
        self._sim_time = now
        self._scenario_triggered = False
        
        # Initialize scenario
        self._init_scenario()
    
    def stop(self) -> None:
        """Stop the simulation."""
        self._running = False
    
    def tick(self, elapsed_seconds: float = 1.0) -> list[SimulatedEvent]:
        """Advance simulation clock and generate events.
        
        Args:
            elapsed_seconds: Real-world seconds since last tick
        
        Returns:
            List of generated events
        """
        if not self._running or self._sim_time is None:
            return []
        
        # Advance simulation time
        sim_elapsed = elapsed_seconds * self.speed.multiplier
        self._sim_time += pd.Timedelta(seconds=sim_elapsed)
        
        events = []
        
        # Generate scenario-specific events
        events.extend(self._generate_scenario_events())
        
        # Generate baseline arrivals
        events.extend(self._generate_baseline_arrivals(sim_elapsed))
        
        return events
    
    def _init_scenario(self) -> None:
        """Initialize scenario-specific state."""
        if self.scenario == ScenarioType.SURGE:
            # Pick a hospital for surge
            self._surge_hospital = self._rng.choice(self.HOSPITALS)
            # Surge starts 5-15 min from now
            delay_min = self._rng.randint(5, 15)
            self._surge_start = self._sim_time + pd.Timedelta(minutes=delay_min)
            # Lasts 30-90 min
            duration_min = self._rng.randint(30, 90)
            self._surge_end = self._surge_start + pd.Timedelta(minutes=duration_min)
        
        elif self.scenario == ScenarioType.DOCTOR_DROPS:
            # Will drop a physician after 10-20 min
            pass  # Triggered in _generate_scenario_events
        
        elif self.scenario == ScenarioType.EMPTY_POOL:
            # Will mark all physicians busy for video tasks
            pass
    
    def _generate_scenario_events(self) -> list[SimulatedEvent]:
        """Generate scenario-specific events."""
        events = []
        
        if self.scenario == ScenarioType.SURGE:
            events.extend(self._generate_surge_events())
        
        elif self.scenario == ScenarioType.DOCTOR_DROPS:
            events.extend(self._generate_doctor_drop_events())
        
        elif self.scenario == ScenarioType.SIMULTANEOUS_CRITICALS:
            events.extend(self._generate_simultaneous_criticals())
        
        elif self.scenario == ScenarioType.EMPTY_POOL:
            events.extend(self._generate_empty_pool_events())
        
        return events
    
    def _generate_surge_events(self) -> list[SimulatedEvent]:
        """Generate surge scenario events."""
        events = []
        
        if self._surge_start is None or self._sim_time is None:
            return events
        
        # Check if we're in surge window
        if self._surge_start <= self._sim_time < self._surge_end:
            # 3-5x normal arrival rate during surge
            surge_multiplier = self._rng.uniform(2, 5)
            base_rate = 0.5 / 60  # ~0.5 consults per hour per hospital → per second
            
            # Poisson arrival during this tick
            expected = base_rate * surge_multiplier * (self.speed.multiplier)
            n_arrivals = self._np_rng.poisson(expected)
            
            for _ in range(int(n_arrivals)):
                events.append(self._create_consult_arrived(
                    hospital_id=self._surge_hospital,
                    consult_type=self._rng.choice(["Stroke", "General Neuro"]),
                ))
        
        return events
    
    def _generate_doctor_drop_events(self) -> list[SimulatedEvent]:
        """Generate doctor dropout events."""
        events = []
        
        if self._scenario_triggered:
            return events
        
        # Trigger after 10-20 min of simulation
        elapsed = (self._sim_time - self._start_time).total_seconds() / 60
        trigger_time = 10 + self._rng.random() * 10
        
        if elapsed >= trigger_time:
            self._scenario_triggered = True
            # Pick a physician to drop
            self._dropped_physician = f"PHY-{self._rng.randint(1, 14):03d}"
            
            events.append(SimulatedEvent(
                event_id=self._next_event_id(),
                event_time=self._sim_time,
                event_type="agent_heartbeat_lost",
                physician_id=self._dropped_physician,
                details={"reason": "Connection lost (simulated)"},
            ))
        
        return events
    
    def _generate_simultaneous_criticals(self) -> list[SimulatedEvent]:
        """Generate multiple simultaneous stroke arrivals."""
        events = []
        
        if self._scenario_triggered:
            return events
        
        # Trigger after 5 min of simulation
        elapsed = (self._sim_time - self._start_time).total_seconds() / 60
        
        if elapsed >= 5:
            self._scenario_triggered = True
            
            # 3-5 stroke consults at different hospitals
            n_strokes = self._rng.randint(3, 5)
            hospitals = self._rng.sample(self.HOSPITALS, n_strokes)
            
            for hospital_id in hospitals:
                events.append(self._create_consult_arrived(
                    hospital_id=hospital_id,
                    consult_type="Stroke",
                ))
        
        return events
    
    def _generate_empty_pool_events(self) -> list[SimulatedEvent]:
        """Generate events that result in empty physician pool."""
        events = []
        
        if self._scenario_triggered:
            return events
        
        # Trigger after 3 min
        elapsed = (self._sim_time - self._start_time).total_seconds() / 60
        
        if elapsed >= 3:
            self._scenario_triggered = True
            
            # Create a video consult when all doctors are busy
            events.append(self._create_consult_arrived(
                hospital_id=self._rng.choice(self.HOSPITALS),
                consult_type="Stroke",
                details={"task_type": "video", "note": "Arrives when pool is exhausted"},
            ))
        
        return events
    
    def _generate_baseline_arrivals(self, elapsed_seconds: float) -> list[SimulatedEvent]:
        """Generate baseline consult arrivals."""
        events = []
        
        # Skip if in scenario with its own arrival logic
        if self.scenario == ScenarioType.SURGE:
            return events
        if self.scenario == ScenarioType.SIMULTANEOUS_CRITICALS and self._scenario_triggered:
            return events  # Already generated the criticals
        
        # Normal arrival rate: ~3-5 consults per hour across all hospitals
        base_rate = 4 / 3600  # per second
        expected = base_rate * elapsed_seconds
        
        n_arrivals = self._np_rng.poisson(expected)
        
        for _ in range(int(n_arrivals)):
            events.append(self._create_consult_arrived(
                hospital_id=self._rng.choice(self.HOSPITALS),
                consult_type=self._rng.choices(
                    ["Stroke", "General Neuro"],
                    weights=[0.3, 0.7]  # 30% stroke
                )[0],
            ))
        
        return events
    
    def _create_consult_arrived(
        self,
        hospital_id: str,
        consult_type: str,
        details: dict | None = None,
    ) -> SimulatedEvent:
        """Create a consult arrival event."""
        consult_id = f"SIM-CON-{self.seed:06d}-{self._event_counter:04d}"
        
        return SimulatedEvent(
            event_id=self._next_event_id(),
            event_time=self._sim_time,
            event_type="consult_received",
            consult_id=consult_id,
            hospital_id=hospital_id,
            details={
                "consult_type": consult_type,
                **(details or {}),
            },
        )
    
    def _next_event_id(self) -> str:
        """Generate next event ID."""
        self._event_counter += 1
        return f"SIM-EVT-{self.seed:06d}-{self._event_counter:04d}"
    
    def get_status(self) -> dict:
        """Get current simulation status."""
        return {
            "running": self._running,
            "speed": self.speed.value,
            "scenario": self.scenario.value,
            "scenario_description": self.scenario.description,
            "seed": self.seed,
            "sim_time": self._sim_time.isoformat() if self._sim_time else None,
            "elapsed_sim_minutes": (
                (self._sim_time - self._start_time).total_seconds() / 60
                if self._sim_time and self._start_time else 0
            ),
            "scenario_triggered": self._scenario_triggered,
            "events_generated": self._event_counter,
        }


# ============================================================================
# SCENARIO PRESETS
# ============================================================================

def create_demo_scenario(
    scenario_type: str,
    seed: int | None = None,
) -> Simulator:
    """Create a simulator with a preset demo scenario.
    
    Args:
        scenario_type: One of "surge", "doctor_drops", "criticals", "empty_pool", "normal"
        seed: Random seed for reproducibility
    
    Returns:
        Configured Simulator instance
    """
    type_map = {
        "surge": ScenarioType.SURGE,
        "doctor_drops": ScenarioType.DOCTOR_DROPS,
        "criticals": ScenarioType.SIMULTANEOUS_CRITICALS,
        "empty_pool": ScenarioType.EMPTY_POOL,
        "normal": ScenarioType.NORMAL,
    }
    
    scenario = type_map.get(scenario_type.lower(), ScenarioType.NORMAL)
    
    return Simulator(
        speed=ClockSpeed.FAST,  # Default to 10x for demos
        scenario=scenario,
        seed=seed,
    )


def inject_events(
    base_events: pd.DataFrame,
    simulated_events: list[SimulatedEvent],
) -> pd.DataFrame:
    """Inject simulated events into the base event stream.
    
    Args:
        base_events: Original event DataFrame
        simulated_events: Events from simulator
    
    Returns:
        Combined DataFrame sorted by event_time
    """
    if not simulated_events:
        return base_events
    
    sim_df = pd.DataFrame([e.to_dict() for e in simulated_events])
    
    # Ensure column compatibility
    for col in base_events.columns:
        if col not in sim_df.columns:
            sim_df[col] = ""
    
    combined = pd.concat([base_events, sim_df], ignore_index=True)
    return combined.sort_values("event_time").reset_index(drop=True)
