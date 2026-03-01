# Hardware Architecture Planning Powered by Enhanced Copilot Chat

Planning Mode is the exploratory phase of PySilicon where developers describe their goals in natural language and receive structured architectural guidance. This mode exists to make early hardware design accessible, especially for developers who are not hardware specialists. It transforms the blank‑slate problem of accelerator architecture into a guided, conversational process grounded in PySilicon’s conventions and examples.

## Why Planning Mode Exists

Early hardware design is often the hardest part of accelerator development. Developers must decide how to partition functionality, define module boundaries, design memory interfaces, and reason about control flow. These decisions require experience that many software‑oriented engineers do not yet have.

Planning Mode addresses this gap by providing:

- a conversational environment for exploring architectural ideas  
- guidance grounded in PySilicon’s examples and idioms  
- explanations that help users understand tradeoffs and patterns  
- a way to converge on a clean, synthesizable architecture before generating code  

This makes PySilicon approachable for beginners while still supporting experienced engineers who want a fast, structured way to plan designs.

## Why Planning Mode Lives Inside VS Code

VS Code has become the standard environment for complex software and hardware‑adjacent development. It provides:

- a unified workspace model  
- rich language tooling  
- extension APIs for injecting domain knowledge  
- a natural place for conversational agents to interact with project files  

Because Planning Mode relies on workspace context and example retrieval, VS Code is the ideal host. Developers stay inside their editor, ask questions about their design, and receive answers grounded in the files already present in the project.

## Copilot Chat as the Planning Engine

Planning Mode uses Copilot Chat as its reasoning engine. Copilot Chat brings several capabilities that make it well‑suited for architectural planning:

- multi‑step reasoning over natural language queries  
- awareness of workspace files and examples  
- the ability to synthesize structured plans  
- conversational refinement across multiple turns  
- code search and pattern recognition  

This allows developers to ask questions such as:

> “How should I structure the memory interface for a streaming FFT block?”

and receive answers that reference PySilicon’s conventions, examples, and architectural patterns.

## How the PySilicon Extension Augments Copilot Chat

The PySilicon VS Code extension enriches Copilot Chat by placing curated domain knowledge directly into the workspace. When activated, the extension:

- copies instruction files into `.github/instructions/`  
- installs rich examples into `.pysilicon/`  
- provides templates, naming conventions, and interface patterns  
- ensures that Copilot Chat can retrieve these files naturally  

Because Copilot Chat uses workspace‑aware retrieval, it automatically incorporates these files into its reasoning. The instructions also explicitly point Copilot toward the examples, ensuring consistent grounding.

This creates a reproducible planning environment where the agent’s behavior is guided by visible, version‑controlled files rather than hidden heuristics.

## What Planning Mode Enables

With this setup, developers can:

- explore alternative architectures  
- ask about module boundaries and interface design  
- reason about memory systems and control flow  
- reference examples without manually searching for them  
- iterate conversationally until the design is ready for synthesis  

Planning Mode is intentionally open‑ended and exploratory. It is where creativity, tradeoff analysis, and architectural understanding happen.

## Example Interaction

A typical Planning Mode interaction might look like:

> **User:** “I need a memory interface for a vector unit with burst reads. What should the module boundary look like?”  
>  
> **Planning Agent:**  
> - identifies the relevant memory‑interface instructions  
> - references the example in `.pysilicon/examples/memif/`  
> - explains required signals and burst semantics  
> - proposes a clean module boundary  
> - points to the example HLS stub for implementation details  

This loop continues until the architecture is well‑defined and ready for Synthesis Mode.

## Example Instruction Excerpt

Instruction files guide the planning agent toward consistent, PySilicon‑aligned designs. For example:

> **pysilicon-memif-instructions.md (excerpt)**  
>  
> - A PySilicon memory interface (`MemIF`) must expose `read(addr, size)` and `write(addr, data)` semantics.  
> - For hardware synthesis, the generated HLS function should follow the pattern shown in `.pysilicon/examples/memif/memif_hls.cpp`.  
> - When generating a new memory interface, ensure that:  
>   - burst lengths are explicit  
>   - address widths match the module’s declared configuration  
>   - the interface is compatible with the `cmd_memif` and `prog_memif` conventions  
> - If the user asks for a memory interface, reference the example above and follow the naming conventions in `memif_spec.md`.

These files make the planning agent predictable, transparent, and grounded in PySilicon’s design philosophy.