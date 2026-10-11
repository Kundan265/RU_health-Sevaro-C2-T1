"""Tests for engine/simulator.py - event simulation."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.simulator import (
    Simulator, ClockSpeed, ScenarioType, SimulatedEvent,
    create_demo_scenario, inject_events,
)


def test_clock_speeds():
    """Test clock speed multipliers."""
    assert ClockSpeed.REAL_TIME.multiplier == 1.0
    assert ClockSpeed.FAST.multiplier == 10.0
    assert ClockSpeed.TURBO.multiplier == 60.0
    print("✓ Clock speeds have correct multipliers")


def test_simulator_starts_and_stops():
    """Test simulator lifecycle."""
    sim = Simulator()
    now = pd.Timestamp.now()
    
    assert not sim.running
    
    sim.start(now)
    assert sim.running
    assert sim.sim_time == now
    
    sim.stop()
    assert not sim.running
    print("✓ Simulator starts and stops correctly")


def test_tick_advances_time():
    """Test that tick advances simulation time."""
    sim = Simulator(speed=ClockSpeed.FAST)  # 10x
    now = pd.Timestamp.now()
    
    sim.start(now)
    sim.tick(elapsed_seconds=1.0)  # 1 real second = 10 sim seconds
    
    expected = now + pd.Timedelta(seconds=10)
    assert sim.sim_time == expected
    print("✓ Tick advances simulation time correctly")


def test_turbo_speed():
    """Test 60x clock speed."""
    sim = Simulator(speed=ClockSpeed.TURBO)  # 60x
    now = pd.Timestamp.now()
    
    sim.start(now)
    sim.tick(elapsed_seconds=1.0)  # 1 real second = 60 sim seconds
    
    expected = now + pd.Timedelta(seconds=60)
    assert sim.sim_time == expected
    print("✓ Turbo speed (60x) works correctly")


def test_surge_scenario():
    """Test surge scenario generates extra arrivals."""
    sim = Simulator(
        speed=ClockSpeed.TURBO,
        scenario=ScenarioType.SURGE,
        seed=42,
    )
    now = pd.Timestamp.now()
    
    sim.start(now)
    
    # Tick forward into surge window
    all_events = []
    for _ in range(30):  # 30 seconds * 60x = 30 min sim time
        events = sim.tick(elapsed_seconds=1.0)
        all_events.extend(events)
    
    # Should have generated some consult arrivals during surge
    arrivals = [e for e in all_events if e.event_type == "consult_received"]
    assert len(arrivals) > 0, "Surge should generate consult arrivals"
    
    # Check that surge hospital was targeted
    assert sim._surge_hospital is not None
    print(f"✓ Surge scenario generated {len(arrivals)} arrivals at {sim._surge_hospital}")


def test_doctor_drops_scenario():
    """Test doctor dropout scenario."""
    sim = Simulator(
        speed=ClockSpeed.TURBO,
        scenario=ScenarioType.DOCTOR_DROPS,
        seed=42,
    )
    now = pd.Timestamp.now()
    
    sim.start(now)
    
    # Tick until dropout occurs (10-20 min sim time)
    all_events = []
    for _ in range(25):  # 25 seconds * 60 = 25 min
        events = sim.tick(elapsed_seconds=1.0)
        all_events.extend(events)
    
    # Should have a heartbeat_lost event
    dropouts = [e for e in all_events if e.event_type == "agent_heartbeat_lost"]
    assert len(dropouts) >= 1, "Should have a doctor dropout"
    assert dropouts[0].physician_id is not None
    print(f"✓ Doctor drops scenario: {dropouts[0].physician_id} went offline")


def test_simultaneous_criticals_scenario():
    """Test simultaneous stroke arrivals scenario."""
    sim = Simulator(
        speed=ClockSpeed.TURBO,
        scenario=ScenarioType.SIMULTANEOUS_CRITICALS,
        seed=42,
    )
    now = pd.Timestamp.now()
    
    sim.start(now)
    
    # Tick past trigger point (5 min)
    all_events = []
    for _ in range(10):  # 10 seconds * 60 = 10 min
        events = sim.tick(elapsed_seconds=1.0)
        all_events.extend(events)
    
    # Should have 3-5 stroke arrivals
    strokes = [e for e in all_events 
               if e.event_type == "consult_received" 
               and e.details.get("consult_type") == "Stroke"]
    assert len(strokes) >= 3, f"Should have 3+ strokes, got {len(strokes)}"
    
    # Check they're at different hospitals
    hospitals = {e.hospital_id for e in strokes}
    assert len(hospitals) >= 3, "Strokes should be at different hospitals"
    print(f"✓ Simultaneous criticals: {len(strokes)} strokes at {len(hospitals)} hospitals")


def test_reproducible_with_seed():
    """Test that same seed produces same events."""
    seed = 12345
    now = pd.Timestamp.now()
    
    # Run twice with same seed
    events1 = []
    sim1 = Simulator(speed=ClockSpeed.FAST, scenario=ScenarioType.NORMAL, seed=seed)
    sim1.start(now)
    for _ in range(10):
        events1.extend(sim1.tick(elapsed_seconds=1.0))
    
    events2 = []
    sim2 = Simulator(speed=ClockSpeed.FAST, scenario=ScenarioType.NORMAL, seed=seed)
    sim2.start(now)
    for _ in range(10):
        events2.extend(sim2.tick(elapsed_seconds=1.0))
    
    # Should produce same number of events
    assert len(events1) == len(events2)
    print("✓ Same seed produces reproducible events")


def test_create_demo_scenario():
    """Test demo scenario factory."""
    sim = create_demo_scenario("surge", seed=42)
    assert sim.scenario == ScenarioType.SURGE
    assert sim.speed == ClockSpeed.FAST  # Default for demos
    
    sim = create_demo_scenario("doctor_drops")
    assert sim.scenario == ScenarioType.DOCTOR_DROPS
    
    sim = create_demo_scenario("invalid")
    assert sim.scenario == ScenarioType.NORMAL  # Fallback
    print("✓ Demo scenario factory works")


def test_inject_events():
    """Test injecting simulated events into base stream."""
    # Create base events
    base = pd.DataFrame([
        {"event_id": "E1", "event_time": pd.Timestamp("2026-03-17 18:00"), "event_type": "base"},
        {"event_id": "E2", "event_time": pd.Timestamp("2026-03-17 18:30"), "event_type": "base"},
    ])
    
    # Create simulated events
    simulated = [
        SimulatedEvent(
            event_id="S1",
            event_time=pd.Timestamp("2026-03-17 18:15"),
            event_type="consult_received",
        ),
    ]
    
    combined = inject_events(base, simulated)
    
    assert len(combined) == 3
    # Check sorted by event_time
    assert combined.iloc[0]["event_id"] == "E1"
    assert combined.iloc[1]["event_id"] == "S1"
    assert combined.iloc[2]["event_id"] == "E2"
    print("✓ Event injection maintains chronological order")


def test_status_reporting():
    """Test simulator status reporting."""
    sim = Simulator(speed=ClockSpeed.FAST, scenario=ScenarioType.SURGE, seed=42)
    now = pd.Timestamp.now()
    
    sim.start(now)
    sim.tick(elapsed_seconds=5.0)
    
    status = sim.get_status()
    
    assert status["running"] == True
    assert status["speed"] == "10x"
    assert status["scenario"] == "surge"
    assert status["seed"] == 42
    assert status["elapsed_sim_minutes"] > 0
    print("✓ Status reporting works correctly")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Simulator Module")
    print("=" * 60)
    
    test_clock_speeds()
    test_simulator_starts_and_stops()
    test_tick_advances_time()
    test_turbo_speed()
    test_surge_scenario()
    test_doctor_drops_scenario()
    test_simultaneous_criticals_scenario()
    test_reproducible_with_seed()
    test_create_demo_scenario()
    test_inject_events()
    test_status_reporting()
    
    print()
    print("=" * 60)
    print("All simulator tests passed! ✓")
    print("=" * 60)
