---
title: Unit Testing
parent: Architecture
nav_order: 6
---

# A Python‑Native Unit Testing Framework for Deterministic, Gated Builds

PySilicon provides a unified, Python‑native testing framework that supports verification of hardware modules, simulated environments, and the interaction between the two. Because all hardware behavior, interfaces, and simulation semantics are defined in Python, tests can be written using standard Python testing tools without any special DSLs or hardware‑specific languages.

## Goals of the Testing Framework

The testing system is designed to support:

- **Rich, multi‑layer testing** that covers hardware modules, environmental models, and full system behavior.
- **Co‑development of hardware and environment**, enabling designers to evaluate performance and correctness under realistic scenarios (e.g., radar reflections, wireless channels, robotic motion).
- **Deterministic, reproducible tests** that run identically across machines and CI pipelines.
- **Incremental development**, where tests can be written before, during, or after hardware implementation.

## Python‑Native Test Structure

All tests are written using standard Python unit testing frameworks such as `pytest` or `unittest`. A test is simply a Python function or class that:

- constructs a simulation environment,
- instantiates hardware objects (`HwObj`) and environment objects (`SimObj`),
- connects them using interfaces,
- drives transactions into the system,
- and asserts expected behavior.

This allows hardware verification to integrate seamlessly with existing Python testing ecosystems.

Example structure:

```python
def test_fft_basic():
    env = Environment()
    fft = FFT()
    stim = WaveformGenerator()
    sink = DataCollector()

    # Connect modules
    AxiStreamInterface(master=stim.out, slave=fft.in_stream)
    AxiStreamInterface(master=fft.out_stream, slave=sink.inp)

    # Run simulation
    env.run(until=1000)

    # Assertions
    assert sink.data == expected_fft_output
```

## Testing Hardware Modules Directly

Tests can target hardware modules in isolation by driving their **master ports** directly:

- send data into a slave port,
- observe outputs on master ports,
- check internal state (if exposed),
- validate timing behavior using simulated delays.

This enables unit‑level verification similar to RTL testbenches, but entirely in Python.

Example:

```python
def test_fifo_push_pop():
    fifo = FIFO(depth=4)
    fifo.push(1)
    fifo.push(2)
    assert fifo.pop() == 1
```

## Testing Environment Models

Environment objects (`SimObj`) can be tested independently:

- radar waveform generators,
- wireless channel models,
- robotic motion simulators,
- sensor models.

These tests validate the correctness of the physical or software environment before integrating hardware.

Example:

```python
def test_radar_reflection():
    env = Environment()
    radar = RadarTransmitter()
    target = MovingTarget()
    receiver = RadarReceiver()

    # Connect environment objects
    env.add(radar, target, receiver)

    env.run(until=2000)

    assert receiver.detected_range == approx(target.true_range)
```

## Testing Hardware–Environment Interaction

The most powerful capability is **co‑simulation testing**, where hardware modules interact with realistic environment models:

- a hardware FFT processes radar returns,
- a hardware accelerator processes camera frames from a simulated robot,
- a wireless PHY module interacts with a channel model.

These tests validate system‑level correctness and performance.

Example:

```python
def test_fft_in_radar_pipeline():
    env = Environment()
    radar = RadarTransmitter()
    channel = RadarChannel()
    fft = FFT()
    detector = PeakDetector()

    # Connect environment and hardware
    AxiStreamInterface(master=radar.out, slave=channel.inp)
    AxiStreamInterface(master=channel.out, slave=fft.in_stream)
    AxiStreamInterface(master=fft.out_stream, slave=detector.inp)

    env.run(until=5000)

    assert detector.peaks == expected_targets
```

## Benefits of Python‑Native Testing

- **No HDL testbenches** are required.
- **No firmware scaffolding** is needed to drive hardware behavior.
- **Tests run quickly**, enabling rapid iteration.
- **Tests integrate with CI/CD**, enabling continuous verification.
- **Tests are readable and maintainable**, lowering the barrier for new contributors.

The testing framework becomes the backbone of the entire development process, ensuring that hardware, environment, and system‑level behavior evolve together in a controlled, verifiable way.

---
Go to [Guided and Reproducible AI‑Driven Synthesis for FPGA Hardware](./synthesis.md)