---
title: The Waveflow flow
parent: Overview
nav_order: 3
---

# The Waveflow flow

Waveflow runs **two coupled loops**, joined by a calibrated timing/resource model.

<p align="center">
  <img src="../assets/waveflow_methodology.svg" width="820"
       alt="Two-loop methodology: a violet build pipeline (Python model, SimPy DES, HLS codegen, RTL synth/sim) with a teal feedback system — Agentic DSE and Functional verification below, Calibration and the Timing/resource model above, and AI assisting codegen and exploration.">
</p>

## The build pipeline

A high-level design (architecture + parameters) becomes a **Python model**, which a **SimPy
discrete-event simulation (DES)** runs **fast** and **bit-exact** using a **calibrated,
cycle-approximate** timing/resource model. The same model generates **HLS codegen** (kernel +
testbench), which **RTL synthesis / simulation** turns into cycle- and resource-**exact** ground
truth.

## Inner loop — fast, all-Python (agentic DSE)

The SimPy DES produces performance — accuracy, throughput, resources — that drives **design-space
exploration**. This loop is cheap enough to sweep bit widths, queue sizes, memory organization,
and iteration counts, so an **agent** can explore many designs per toolchain run. It closes by
refining the Python model's architecture and parameters.

## Outer loop — slow, sparse (calibration)

The same Python model generates HLS; C-sim, co-sim, and synthesis give cycle/resource-**exact**
numbers, and **Calibration** fits the **timing/resource model** the inner loop relies on. A few
sparse toolchain runs keep the fast models trustworthy — the model is the **bridge** the two
loops share.

## Functional verification

In parallel, **functional verification** compares the **SimPy golden** against the generated
**RTL**, **bit-for-bit**. A mismatch is a model or codegen bug to fix — this is the correctness
guarantee that makes the fast loop's numbers worth trusting.

## Where AI fits

AI assists at **codegen** (generate the HLS) and as the **agent driving DSE** (search the design
space). Its output is grounded and verified by the bit-exact substrate — Waveflow is the harness
that makes AI useful here, not the other way around.
