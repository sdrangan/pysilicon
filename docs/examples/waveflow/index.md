---
title: WaveFlow
nav_order: 1
parent: Examples
---

# WaveFlow

WaveFlow is a motivating example that illustrates why PySilicon is needed and what kinds of systems cannot be built reliably with current AI‑assisted hardware tools. WaveFlow represents a class of modern wireless architectures that combine heterogeneous accelerators, high‑rate antenna interfaces, and dynamic dataflow reconfiguration. These systems demand architectural exploration, concurrency modeling, and unified firmware generation—capabilities that existing tools do not provide.

## System Overview

WaveFlow is a tile‑based wireless processing chip designed for real‑time communication, sensing, and beamforming workloads. The architecture consists of:

- heterogeneous processing tiles such as FFT engines, systolic arrays, vector units, and filtering blocks  
- high‑rate antenna array interfaces that ingest wideband I/Q samples from multiple RF chains  
- a message‑passing fabric that routes data between tiles based on workload‑dependent flow graphs  
- flow‑dependent control logic that determines how each tile processes incoming messages and forwards results  
- concurrent dataflows that support communication, spectrum sensing, angle‑of‑arrival estimation, beam nulling, and interference mitigation simultaneously  

WaveFlow is not a fixed pipeline. It is a runtime‑programmable dataflow machine whose behavior changes dynamically based on the wireless environment and application demands.