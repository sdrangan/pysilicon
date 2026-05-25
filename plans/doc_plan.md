# Documentation Reorganization & Refresh Plan

> **Status: READY TO EXECUTE.** All blocking conditions from the previous
> draft are now met. PR #31 (build DAG reorg + cosim timing validator)
> rewrote `docs/examples/poly/` into a 5-page tutorial, which is the
> running example this plan documents against. The synthesis pipeline,
> HwTestbench codegen, `BoundRegMap`, `param_supports`, and the
> cycle-timing validator are all landed on `main`.

## Goal

Bring `docs/` in sync with the codebase as of PR #31 (commit `375434a`).
Three concerns:

1. **Document what's landed.** Significant new framework pieces have no
   docs yet: the synthesis pipeline, `HwTestbench`, `BoundRegMap`,
   `param_supports`, and cosim timing validation.
2. **Refresh what's stale.** `SchemaArray` references and other
   pre-dataschema-unification API names still appear in three files.
3. **Restructure for the audience.** Delete the obsolete
   `docs/architecture/` content. Slim `docs/overview/` to an elevator
   pitch. Add `docs/future/` as the canonical home for forward-looking
   work.

## Audience: this plan is written for a GitHub Copilot agent

This plan is intended to be executed by a Copilot agent over multiple
issues/PRs while the maintainer is travelling. Implications:

- **Phases are PR-sized.** Each phase corresponds to one branch, one
  PR, one commit. Eight phases total. Land them in order, but each is
  reviewable independently.
- **Verification commands are explicit.** Every phase ends with a
  command whose output you (the agent) check before declaring the phase
  done. If the command fails, do not commit — surface the failure.
- **Page templates are provided.** Where the plan asks you to create a
  new page, the frontmatter and skeleton headings are spelled out.
  Match the structure exactly; fill in prose under each heading.
- **Reference pages to mimic.** Where the plan says "match the style
  of X," X is a specific file path. Read it before writing the new one.
- **No subjective preservation calls.** Where this plan says "delete X,"
  delete it. Worth-preserving prose is named explicitly with file
  paths and line ranges where applicable.

## Already done in PR #31 (do NOT touch in this plan)

- `docs/examples/poly/` — fully restructured into a 5-page tutorial
  (`01_python_golden_model.md` through `05_cosim_timing.md` + `index.md`
  + the existing `poly_axi_stream.md`). This is the running example
  the synthesis guide refers to. **Do not modify these pages.**
- `docs/guide/build/codegen.md`, `corecomp.md`, `index.md`, `vitis.md`
  were refreshed in PR #31 to match the implemented pipeline.
  **Spot-check for `SchemaArray` references in Phase 6 sweep, but
  otherwise leave alone.**

## Settled design decisions (do NOT re-litigate)

1. **`docs/architecture/` is deleted entirely.** The current 10 files
   are 70%+ stale. Cross-cutting design principles move into the
   relevant `docs/guide/<area>/index.md` pages.
2. **`docs/overview/` shrinks to 3 pages:** `index.md` (1-page elevator
   pitch), `example.md` (poly pointer), `motivation.md` (why
   Python-as-source-of-truth). Delete `innovations.md` and `prior.md`.
3. **`docs/future/` is new.** One page per forward-looking area. Each
   page is forward-looking only — describes intent and cross-references
   the corresponding plan file, does not document existing behavior.
4. **`docs/guide/synthesis/` is new.** Largest writing effort. Seven
   pages: index, extractor, codegen, templating, param_supports,
   testbench, cosim_timing.
5. **`docs/guide/components/` is filled in.** Currently has only
   `index.md`. Add four sibling pages: `hwparam.md`, `hwconst.md`,
   `hwtestbench.md`, `lifecycle.md`.
6. **`BoundRegMap` is documented inside `docs/guide/interface/regmap.md`,
   NOT in a new `components/regmap.md`.** The existing regmap page in
   the interface section is the right home: regmap is a bus-side
   concept, the bound proxy is a bus-master concept, and the page is
   already there.
7. **Page style:** match [docs/guide/interface/stream.md](../docs/guide/interface/stream.md)
   for any single-concept page (frontmatter → concept → API → example →
   quick reference). Use [docs/guide/interface/index.md](../docs/guide/interface/index.md)
   as the template for any new `index.md`.
8. **`nav_order`:** assigned per phase. Final order is fixed in Phase 8:
   overview (1), examples (2), guide (3, with children), future (4).
   Inside guide: installation (1), schema (2), interface (3),
   components (4), synthesis (5), build (6), memory (7), timing (8),
   developer (9).
9. **All link updates are direct edits, no redirect pages.**
10. **No code changes in this plan.** Pure documentation.

## Stale references to sweep

`grep -rn "SchemaArray" docs/` currently returns hits in:

- `docs/guide/schema/dataarrays.md`
- `docs/guide/interface/array_transfer.md`
- `docs/guide/memory/vitis.md`

Phase 6 sweeps all of `docs/` for `SchemaArray`, `gen_array_utils`,
and `CodeGenConfig` references and updates each.

## Reference reading (read before starting)

Before writing any new content, read these as ground truth for the API
surfaces you're documenting:

- [pysilicon/hw/hw_component.py](../pysilicon/hw/hw_component.py) — `HwComponent`, `HwParam`, `HwConst`, `param_supports`.
- [pysilicon/hw/hw_testbench.py](../pysilicon/hw/hw_testbench.py) — `HwTestbench` class.
- [pysilicon/hw/regmap.py](../pysilicon/hw/regmap.py) — `RegMap`, `RegField`, `RegAccess`, `VitisRegMap`, `BoundRegMap` (new).
- [pysilicon/build/hwgen.py](../pysilicon/build/hwgen.py), [pysilicon/build/hwcodegen.py](../pysilicon/build/hwcodegen.py), [pysilicon/build/hwresolve.py](../pysilicon/build/hwresolve.py) — synthesis pipeline.
- [pysilicon/build/hwcodegen_steps.py](../pysilicon/build/hwcodegen_steps.py) — `HlsCodegenStep` (kernel mode + `is_testbench` mode).
- [pysilicon/build/cosim_steps.py](../pysilicon/build/cosim_steps.py) — `ExtractCosimTimingStep`, `ValidateTimingStep`.
- [pysilicon/utils/cosimparse.py](../pysilicon/utils/cosimparse.py) — cosim report parsing.
- [examples/poly/poly.py](../examples/poly/poly.py), [examples/poly/poly_build.py](../examples/poly/poly_build.py) — worked example.
- [docs/examples/poly/](../docs/examples/poly/) — already-written 5-page tutorial; learn the narrative and the example references to reuse.

## Working convention

- One commit per phase. One branch + PR per phase. PR title format:
  `docs: phase N — <short description>`.
- Commit messages: imperative, ~60-char subject line, body explaining
  what changed and why (1-3 paragraphs).
- After each phase: run the verification command in that phase's "Verify"
  block. Do not commit if it fails.
- Each PR should include a brief "what to look for in review" note
  in the PR description.

---

## Phase 1 — Add `docs/future/` with stub pages

**Goal:** Establish the canonical home for aspirational work. Cheap,
small, signals project direction without touching existing pages.

**Create exactly 6 files** under `docs/future/`:

1. `index.md`:
   ```markdown
   ---
   title: Future
   parent: PySilicon
   nav_order: 4
   has_children: true
   ---

   # Future Work

   This section describes planned features that are not yet
   implemented. Each page describes one area of forward-looking work;
   for details on what is currently being designed or built, see the
   corresponding plan file under [plans/](https://github.com/sdrangan/pysilicon/tree/main/plans).

   - [AI-Assisted Hook Completion](./ai_hook_completion.md)
   - [AI-Assisted Planning](./ai_planning.md)
   - [Cycle Model Training](./cycle_model_training.md)
   - [Design Analysis Automation](./design_analysis.md)
   - [Vivado IPI Backend](./vivado_backend.md)
   ```

2. `ai_hook_completion.md`, `ai_planning.md`, `cycle_model_training.md`,
   `design_analysis.md`, `vivado_backend.md` — each follows this template:
   ```markdown
   ---
   title: <Title>
   parent: Future
   ---

   # <Title>

   > **Status:** Not implemented. This page describes intended future work.

   ## Concept

   <1 paragraph describing what this feature is and why it's wanted.>

   ## Status

   <1 paragraph: what's designed, what's prototyped, what's not started.>

   ## See also

   <Links to relevant plan files under plans/, or "TBD".>
   ```

   Specific guidance for `cycle_model_training.md`:
   > Describes the workflow that fits `HwComponent` timing parameters
   > (e.g., `proc_latency`, `proc_ii`) from RTL cosim measurements.
   > Status: ground-truth measurement infrastructure (`ValidateTimingStep`)
   > landed in PR #31, with measured `delta=4` cycles on poly. Training
   > step itself is not yet started. See PR #31 for the validator.

**Verify:**
```bash
ls docs/future/ | sort
```
Expected output: exactly `ai_hook_completion.md`, `ai_planning.md`,
`cycle_model_training.md`, `design_analysis.md`, `index.md`,
`vivado_backend.md`.

**Commit:** `docs: add future/ section with stubs for aspirational features`

---

## Phase 2 — Write `docs/guide/synthesis/`

**Goal:** Reference documentation for the full Python-to-HLS synthesis
pipeline. Largest writing effort in the plan.

**Create exactly 7 files** under `docs/guide/synthesis/`:

| File | Topic | Source files to read |
|------|-------|----------------------|
| `index.md` | Concept overview: 5-stage pipeline (extractor → IR → resolver → emitter → BuildStep). How a Python `HwComponent` becomes a compilable HLS kernel set. ToC. | `pysilicon/build/hwcodegen.py`, `pysilicon/build/hwgen.py` |
| `extractor.md` | `HwStmtExtractor`, the synthesizable subset, no-implicit-capture rule, `extract_kernel(comp)` entry point, the `@synthesizable` decorator | `pysilicon/build/hwcodegen.py`, `pysilicon/hw/synth.py` |
| `codegen.md` | `kernel_files_to_str` flow, three generated file types (.hpp / .cpp / _impl.{cpp,tpp}), C++ type translation, namespacing via `cpp_namespace` | `pysilicon/build/hwgen.py`, `pysilicon/build/hwcodegen_steps.py` |
| `templating.md` | `HwParam` → C++ template params, `HwParamValue` auto-wrap, hook templating + `.tpp` pattern, sticky impl-file lifecycle | `pysilicon/hw/hw_component.py`, `pysilicon/build/hwcodegen_steps.py` |
| `param_supports.md` | Multi-variant kernel emission via `param_supports` dict, naming convention `<kernel>_<key>`, concrete top-level functions per variant, why Vitis can't template the top | `pysilicon/hw/hw_component.py`, the `param_supports` block in `examples/poly/poly.py` |
| `testbench.md` | `HwTestbench` class, `is_testbench=True` mode on `HlsCodegenStep`, `dut = MyComp()` binding, `dut.run()` lowering, push/pop/push_array/pop_array on streams, sequential-only constraint (and pointer to future SimPy-mode upgrade) | `pysilicon/hw/hw_testbench.py`, `pysilicon/build/hwcodegen_steps.py`, `examples/poly/poly.py` `PolyTBHls` class |
| `cosim_timing.md` | The cycle-timing validation triple (`ExtractPyTimingStep` → `ExtractCosimTimingStep` → `ValidateTimingStep`), `CosimReportParser` Vitis-version handling, structured `timing_verdict` JSON, the `delta=4` proof point from poly | `pysilicon/build/cosim_steps.py`, `pysilicon/utils/cosimparse.py` |

Each page uses this skeleton:
```markdown
---
title: <Title>
parent: Synthesis
nav_order: <N>
---

# <Title>

## Concept

<1-3 paragraphs: what this is, why it exists.>

## API

<The classes / functions / decorators. Show signatures verbatim from
source. Link to source with [name](../../../pysilicon/build/foo.py).>

## Example

<Code block showing real usage. Pull from examples/poly/ where possible.
Cite the source file with a line range.>

## Quick reference

<Bullet list of the most-used pieces. 3-8 lines.>
```

The `index.md` for the section uses this skeleton:
```markdown
---
title: Synthesis
parent: Guide
nav_order: 5
has_children: true
---

# Synthesis

<2-3 paragraphs explaining the 5-stage pipeline at the concept level.>

## In this section

- [Extractor](./extractor.md) — ...
- [Codegen](./codegen.md) — ...
- [Templating](./templating.md) — ...
- [Param supports](./param_supports.md) — ...
- [Testbench](./testbench.md) — ...
- [Cosim timing](./cosim_timing.md) — ...
```

**Style guide:** match the existing [docs/guide/interface/stream.md](../docs/guide/interface/stream.md)
for individual pages. Use it as the template even for the headings —
don't invent new section names.

**Verify:**
```bash
ls docs/guide/synthesis/ | sort
```
Expected: exactly `codegen.md`, `cosim_timing.md`, `extractor.md`,
`index.md`, `param_supports.md`, `templating.md`, `testbench.md`.

Each file must have valid frontmatter (`---` block with `title:` and
`parent:`).

**Commit:** `docs: add guide/synthesis/ section covering the full Python-to-HLS pipeline`

---

## Phase 3 — Fill in `docs/guide/components/`

**Goal:** Replace the stub `index.md` and add four sibling pages
covering `HwComponent` and `HwTestbench` in depth.

**Pages:**

1. **`index.md`** (rewrite from current stub):
   - Concept: `HwComponent` vs `Component` (latter is sim-only, former is also synthesizable).
   - The three variable categories: `HwConst[T]`, `HwParam[T]`, plain.
   - Endpoint declaration patterns (streams, AXI-Lite via regmap, AXI-MM).
   - ToC for sibling pages.

2. **`hwparam.md`** (NEW):
   - `HwParam[T]` declaration syntax.
   - `HwParamValue` wrapper and the `__post_init__` auto-wrap.
   - Immutability semantics — `HwParam` is read-only after construction.
   - How `HwParam` fields flow through to C++ template parameters
     (forward-reference to `docs/guide/synthesis/templating.md`).

3. **`hwconst.md`** (NEW):
   - `HwConst[T]` declaration syntax.
   - Class-level constants vs instance-level params: when to use each.
   - Note that C++ codegen of `HwConst` is currently deferred (no
     `static constexpr` emission yet); the value is used in Python sim
     and as an `int` literal where it appears in expressions.

4. **`hwtestbench.md`** (NEW):
   - `HwTestbench(HwComponent)` subclass.
   - `main(self)` method is the codegen entry point.
   - Sequential-only constraint: file I/O legal, blocking stream ops
     legal, `dut.run()` legal — but `env.process(dut.run_proc())` style
     concurrent stimulus/capture is not yet supported (future work in
     [docs/future/](../../future/) — link to a yet-unwritten page or
     "TBD").
   - Worked example: `PolyTBHls` from [examples/poly/poly.py](../../../examples/poly/poly.py).
   - Forward-reference to `docs/guide/synthesis/testbench.md` for
     codegen mechanics.

5. **`lifecycle.md`** (NEW):
   - `pre_sim`, `run_proc`, `on_start`, `post_sim` — when each fires.
   - When to use `on_start` (regmap-driven launch, kernel-mode style)
     vs `run_proc` (free-running SimPy process).
   - The `@sim_only` decorator for methods that should NOT be visited
     by the extractor.

**`nav_order` for the new pages:** `hwparam.md`=2, `hwconst.md`=3,
`hwtestbench.md`=4, `lifecycle.md`=5. The `index.md` has no
`nav_order` (it's the section landing page).

Reuse the same page skeleton from Phase 2 (concept / API / example /
quick reference).

**Verify:**
```bash
ls docs/guide/components/ | sort
```
Expected: exactly `hwconst.md`, `hwparam.md`, `hwtestbench.md`,
`index.md`, `lifecycle.md`.

**Commit:** `docs: fill in guide/components/ with HwComponent + HwTestbench reference`

---

## Phase 4 — Document `BoundRegMap` in `docs/guide/interface/regmap.md`

**Goal:** Add the host-side proxy section to the existing regmap page.

**Read first:** [docs/guide/interface/regmap.md](../docs/guide/interface/regmap.md)
in full. Identify the section that covers the in-process `RegMap.get` /
`RegMap.set` API (the kernel-side surface).

**Edit:** add a new sibling section titled `## Host-side: BoundRegMap`
immediately after the kernel-side `get` / `set` coverage. Contents:

- Why it exists: the kernel-side `RegMap.get/set` is in-process; the
  host side previously required spelling out
  `master.read_schema(SchemaCls, addr=base + offset_of(name))` plus
  `.val` casts at every call site.
- API:
  - `regmap.bind_master(master, base_addr=0) -> BoundRegMap`
  - `BoundRegMap.get(name)` (coroutine) — returns native Python value
    (int / IntEnum / float, or schema instance for DataArray / DataList).
  - `BoundRegMap.set(name, value)` (coroutine) — auto-wraps raw values
    via the field's schema, matching kernel-side ergonomics.
  - `BoundRegMap.start()` (coroutine) — convenience for `VitisRegMap`'s
    `ap_start`.
- Example: pull from [examples/poly/poly.py](../../../examples/poly/poly.py)
  `PolyTB.run_proc` (the SimPy-side testbench using `bind_master`).
  Show the call site with `rm = regmap.bind_master(...)` and the
  `yield from rm.get(...)` triplet at the end.
- Cite the source class with a link: `BoundRegMap` is in
  [pysilicon/hw/regmap.py](../../../pysilicon/hw/regmap.py).

**Also:** if `docs/guide/interface/regmap.md` references `SchemaArray`
or `gen_array_utils` anywhere, update those during this edit (Phase 6
will sweep but it's cheaper to handle here if you touch the file).

**Verify:**
```bash
grep -c "BoundRegMap" docs/guide/interface/regmap.md
```
Expected: ≥ 5 (concept, API, example, quick reference, plus inline
mentions).

**Commit:** `docs: document BoundRegMap host-side proxy in interface/regmap.md`

---

## Phase 5 — Refresh `docs/guide/schema/`

**Goal:** Update the schema docs to reflect the post-`SchemaArray`
unification and document `cpp_storage` modes + the `array()` factory.

**Page-by-page changes:**

**`docs/guide/schema/dataarrays.md`** — biggest rewrite:
- Drop the opening framing "This section does NOT discuss the `DataArray`
  schema class itself" — that framing is obsolete; `DataArray` IS the
  unified array concept now.
- Restructure as:
  1. `DataArray` declaration (`element_type`, `max_shape`, `static`).
  2. The `array(elem_type, data)` factory in [pysilicon/hw/arrayutils.py](../../../pysilicon/hw/arrayutils.py)
     for runtime construction.
  3. `cpp_storage="struct"` (default) vs `"raw"` lowering modes.
     Show generated C++ for each. Reference `CoeffArray` in
     `examples/poly/poly.py` as the `cpp_storage="raw"` example.
  4. Generated array utilities via `ArrayUtilsStep` (BuildStep,
     registered in a `BuildDag` — NOT a free function).
- Fix specific API references:
  - `gen_array_utils` → `ArrayUtilsStep`
  - `CodeGenConfig(root_dir=..., util_dir=...)` → `BuildConfig(root_dir=...)`
  - Typo `WORD_BWword_bw_supported` → `word_bw_supported`
- Add note: pipelined stream operations (`get_pipelined`,
  `write_pipelined`) are only legal inside `@synthesizable` hook
  bodies, not in `on_start` / `run_proc`. Cross-reference
  `docs/guide/synthesis/extractor.md` (written in Phase 2).

**`docs/guide/schema/datalists.md`** — verify the `CoeffArray(DataArray)`
example still reads correctly. The shape (a class inheriting
`DataArray` with `element_type` and `max_shape`) is still valid. Drop
any prose that references `SchemaArray`.

**`docs/guide/schema/codegen.md`** — verify the generated-codegen story
matches the current `DataSchemaStep` + `ArrayUtilsStep` flow. Update
any references to obsolete `CodeGenConfig` to `BuildConfig`.

**`docs/guide/schema/index.md`** — verify ToC and overview prose
reflects the current structure. Add a one-line pointer to
`docs/guide/synthesis/` for the kernel-codegen story.

**`docs/guide/schema/dataunion.md`** — read; if no `SchemaArray` or
`CodeGenConfig` references, no edit needed.

**Verify:**
```bash
grep -rn "SchemaArray\|gen_array_utils\|CodeGenConfig\|WORD_BWword_bw_supported" docs/guide/schema/
```
Expected: no output (zero matches).

**Commit:** `docs: refresh guide/schema/ for DataArray unification + cpp_storage`

---

## Phase 6 — Sweep stale references across all of `docs/`

**Goal:** Catch the remaining `SchemaArray` / `gen_array_utils` /
`CodeGenConfig` references that Phase 5 didn't reach.

**Sweep command:**
```bash
grep -rn "SchemaArray\|gen_array_utils\|CodeGenConfig" docs/
```

**Expected pre-edit hits** (after Phase 5):
- `docs/guide/interface/array_transfer.md`
- `docs/guide/memory/vitis.md`

For each match:
1. Read the surrounding paragraph.
2. If the prose described `SchemaArray`-only behavior (i.e., the
   reference was the topic), rewrite the paragraph for `DataArray`.
3. If the reference was incidental, replace `SchemaArray` with
   `DataArray` in place. Adjust the surrounding wording only as needed.
4. Replace `gen_array_utils` with `ArrayUtilsStep`. Replace
   `CodeGenConfig` with `BuildConfig`.

If unsure how to rewrite a passage, leave a clearly-marked TODO
comment in HTML form `<!-- TODO: this paragraph needs rewriting for
DataArray unification - copilot agent was unsure -->` and surface it
in the PR description rather than guessing.

**Verify:**
```bash
grep -rn "SchemaArray\|gen_array_utils\|CodeGenConfig" docs/
```
Expected: no output (zero matches). If you left TODOs, list them in
the PR description.

**Commit:** `docs: sweep remaining SchemaArray / gen_array_utils / CodeGenConfig refs`

---

## Phase 7 — Slim `docs/overview/`

**Goal:** Cut to 3 pages — elevator pitch, example pointer, motivation.

**Changes:**

1. **`docs/overview/index.md`** — rewrite as a 1-page elevator pitch.
   What PySilicon is, who it's for, the value proposition. Lead with
   the cycle-approximate-Python result from PR #31 (PySim and RTL
   cosim agree within ±4 cycles on the poly kernel) as concrete proof
   of the thesis. End with a pointer to the poly tutorial at
   `docs/examples/poly/`.

2. **`docs/overview/example.md`** — rewrite as a short poly walkthrough
   (≤ 1 page). Show the 5-stage pipeline at a glance with one code
   snippet per stage. Link to the full per-page tutorial at
   `docs/examples/poly/`.

3. **`docs/overview/motivation.md`** — keep the most-current prose
   from the existing version. Drop anything that's aspirational
   positioning. Anchor in: Python is the single source of truth for
   simulation, codegen, testbench, and timing verification, validated
   end-to-end. Cross-link to `docs/future/` for what's not yet built.

4. **DELETE** `docs/overview/innovations.md` and `docs/overview/prior.md`.

**Verify:**
```bash
ls docs/overview/ | sort
```
Expected: exactly `example.md`, `index.md`, `motivation.md`.

**Commit:** `docs: slim overview/ to elevator pitch + example + motivation`

---

## Phase 8 — Delete `docs/architecture/` and renumber nav

**Goal:** Final cut + sidebar order audit.

**Step 1: Pre-delete content review.**

Walk each of these files one last time, looking for prose that is
concrete (not aspirational, not stale) and not already covered by the
new `docs/guide/synthesis/`, `components/`, or `future/` pages:

- `docs/architecture/cpp.md`
- `docs/architecture/firmware.md`
- `docs/architecture/hwobj.md`
- `docs/architecture/index.md`
- `docs/architecture/modes.md`
- `docs/architecture/planning.md`
- `docs/architecture/rtlverification.md`
- `docs/architecture/simulation.md`
- `docs/architecture/synthesis.md`
- `docs/architecture/unittest.md`

For any preservation-worthy paragraph, copy it into the closest
`docs/guide/<area>/index.md` page (or `docs/future/<area>.md` if it's
aspirational). Cite the source file in a commit-time comment, not in
the doc body.

If unsure whether a passage is preservation-worthy, the default is
**drop it**. The codebase has moved past most of this content; it's
better to lose marginal prose than to leave stale claims.

**Step 2: Delete the directory.**

```bash
rm -rf docs/architecture/
```

**Step 3: Update internal links.**

```bash
grep -rn "architecture/" docs/
```

For each hit, update the link target to the new location (a page
under `docs/guide/<area>/` or `docs/future/`) or delete the reference
if the content was dropped.

**Step 4: Renumber `nav_order` across docs/.**

Final order:
- `docs/overview/` → `nav_order: 1`
- `docs/examples/` → `nav_order: 2`
- `docs/guide/` → `nav_order: 3` (has `has_children: true`)
- `docs/future/` → `nav_order: 4` (has `has_children: true`)

Inside `docs/guide/`:
- `installation/` → 1
- `schema/` → 2
- `interface/` → 3
- `components/` → 4
- `synthesis/` → 5
- `build/` → 6
- `memory/` → 7
- `timing/` → 8
- `developer/` → 9

Each `index.md` of these subsections gets the right `nav_order` in
its frontmatter.

**Step 5: Link audit.**

```bash
grep -rE '\]\([^)]*\.md[^)]*\)' docs/ | grep -v Binary
```

For each match, verify the target file exists relative to the source
file's directory. Fix any dead links.

**Verify:**
```bash
test ! -d docs/architecture/ && echo "architecture deleted: OK"
grep -rn "architecture/" docs/ && echo "FAIL: residual links" || echo "no residual links: OK"
```

Both lines should print "OK".

**Commit:** `docs: delete architecture/, renumber nav_order, fix internal links`

---

## Final acceptance

After all 8 phases land, the following must hold:

- `docs/architecture/` does not exist.
- `docs/overview/` has exactly 3 pages: `index.md`, `example.md`, `motivation.md`.
- `docs/guide/synthesis/` has exactly 7 pages.
- `docs/guide/components/` has exactly 5 pages (including `index.md`).
- `docs/guide/interface/regmap.md` contains a `## Host-side: BoundRegMap` section.
- `docs/future/` has exactly 6 pages (`index.md` + 5 stubs).
- `grep -rn "SchemaArray\|gen_array_utils\|CodeGenConfig" docs/` returns no output.
- `grep -rn "architecture/" docs/` returns no output.
- All 8 PRs merged to `main`.

## Out of scope (do NOT do)

- Documenting `HwConst` C++ codegen behavior beyond noting it's
  currently deferred. The code path doesn't exist yet.
- Adding the cycle-model-training workflow to `docs/guide/`. It's a
  future-work entry in `docs/future/cycle_model_training.md`; promote
  to a guide section when the workflow ships.
- Expanding `docs/examples/poly/` further. PR #31 wrote that; do not
  modify.
- Adding `docs/analysis/` as a top-level section. Aspirational; covered
  in `docs/future/design_analysis.md`.
- Renaming `docs/examples/` or `docs/guide/` themselves.
- Adding screenshots, diagrams (e.g., Mermaid), or auto-generated API
  reference. Optional polish for later.
- Touching the `_config.yml` Jekyll theme settings.
- Documenting features that haven't shipped (RTL co-sim with VCD
  extraction beyond cosim report, multi-component synthesis, AI
  features). Mention in `docs/future/`; do not write guides.

## Notes for the agent

- **Page count discipline.** Where the plan says "exactly N pages,"
  count them with `ls docs/<dir>/ | sort | wc -l` before committing.
- **No fabricated APIs.** Every method, class, or function named in
  a doc page must exist in the source. If unsure, grep the source
  before writing.
- **No fabricated examples.** Code examples in docs must come from
  `examples/` directly or be minimal illustrations using only
  documented APIs. Do not invent class names.
- **TODOs are acceptable, fabrications are not.** If you cannot
  determine the correct content for a passage, leave a clearly-marked
  HTML TODO comment and surface it in the PR description. Do not
  write speculative prose.
- **Cross-references via relative paths.** `[Foo](../components/hwparam.md)`,
  not absolute or repo-root paths.
- **Frontmatter is mandatory.** Every page must have a `---` block
  with at minimum `title:` and `parent:` keys. Pages that are section
  landing pages also need `has_children: true`.
- **Stop and ask** (in the PR description) if a design question arises
  that this plan does not answer. Do not invent a new convention.
