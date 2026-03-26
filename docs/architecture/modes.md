---
title: Planning and Synthesis Modes
parent: Architecture
nav_order: 3
---

# Modes for Planning and Synthesis in PySilicon

PySilicon organizes accelerator development into two complementary modes: **Planning Mode** and **Synthesis Mode**. These modes reflect the natural split between exploratory architectural design and deterministic code generation. Understanding this separation helps contributors navigate the workflow and know where each part of the system fits.


## Planning Mode

Planning Mode is the exploratory phase where developers describe their goals, constraints, and intended accelerator behavior in natural language. This mode is designed to help users converge on a clean architecture before any code is generated. It is especially valuable for developers who are not hardware specialists, since early architectural decisions often require experience with partitioning, interface design, and memory organization.
Planning Mode runs inside VS Code and uses Copilot Chat as its conversational engine. The PySilicon extension augments Copilot with domain‑specific instructions and examples so that the agent can provide grounded, consistent guidance during architectural planning. The details of how Planning Mode works are described in the [planning mode document](./planning.md).

## Synthesis Mode
Synthesis Mode is the deterministic, compiler‑like phase of PySilicon. Once the architecture is defined, this mode generates the required implementation artifacts in a reproducible sequence of steps. Each stage of the pipeline transforms the design into HLS code, RTL, microcode, firmware stubs, and documentation using stable prompt templates and predictable behavior.

Synthesis Mode does not involve conversation or exploration. It is a structured build pipeline designed for repeatability and automation. The full description of this mode is provided in the [synthesis mode document](./synthesis.md).
