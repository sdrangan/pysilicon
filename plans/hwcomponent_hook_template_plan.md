# HwComponent Phase 10: Hook Templating + `.tpp` Impl Files

## Goal

When a `@synthesizable` user hook takes a stream-typed argument whose endpoint's bitwidth is a `HwParamValue`, the generated C++ for that hook becomes a template. Its definition lives in a sticky, user-owned `<component>_<hook>_impl.tpp` file that is `#include`d from `<component>.hpp`. Non-templated hooks (no stream args, or stream args from raw-int endpoints) continue to use the existing `.cpp` impl-file flow unchanged.

After this phase, a hook like:

```python
@synthesizable
def process(self, cmd: DemoCmdHdr, s_in: StreamIFSlave) -> ProcessGen[DemoError]: ...
```

— called from `on_start` with `self.s_in` (whose bitwidth is `HwParamValue(32, "in_bw")`) — generates:

```cpp
// demo.hpp (tail)
namespace demo {
    template <int in_bw>
    DemoError process(DemoCmdHdr cmd,
                      hls::stream<streamutils::axi4s_word<in_bw>>& s_in);
}
#include "demo_process_impl.tpp"
```

```cpp
// demo_process_impl.tpp   <-- sticky, user-owned
namespace demo {
template <int in_bw>
DemoError process(DemoCmdHdr cmd,
                  hls::stream<streamutils::axi4s_word<in_bw>>& s_in) {
    // TODO: implement process
    return DemoError(0);
}
}
```

Kernel body call site: `auto err = demo::process(cmd, s_in);` — C++ template deduction picks `in_bw` from the stream type.

## Already done (do NOT redo)

- Phases 3.5–9 — full pipeline: extractor, resolver, body codegen, file emission, namespacing, BuildStep wrapping, HwParam templating for the kernel.
- `HwParamValue` (int subclass with `.param_name`) is already on every endpoint whose bitwidth came from a HwParam field — see [pysilicon/hw/hw_component.py](../pysilicon/hw/hw_component.py).
- Kernel signature already emits `template <int in_bw, ...>` and uses template names in stream type expressions and stream-stmt bodies — see `kernel_signature`, `_stream_template_arg` in [pysilicon/build/hwgen.py](../pysilicon/build/hwgen.py).
- The currently-generated `.hpp` hardcodes `WORD_BW` in **hook** forward declarations (the only remaining place `WORD_BW` is emitted). This plan fixes it.

## Design decisions (already settled — do NOT re-litigate)

1. **`.tpp` file pattern.** Templated hook definitions live in `<component>_<hook>_impl.tpp`. The `.tpp` is `#include`d from `<component>.hpp` at the bottom, after all forward decls. This is the standard C++ idiom for template-implementation-in-separate-file.
2. **Per-hook routing.** A hook is "templated" iff at least one of its arg types depends on a `HwParamValue`. Per-hook decision rule lives in a helper (`hook_template_params`); the file-emission path branches on whether the returned list is empty.
3. **Template params discovered from the call site, not the hook annotation.** The hook's Python signature says `s_in: StreamIFSlave` (no bitwidth info). The codegen reads the actual endpoint instance passed at the call site and pulls `HwParamValue.param_name` from its bitwidth.
4. **Single-call-site requirement for templated hooks.** If a templated hook is called from more than one site with different param sources, raise `SynthesisError`. (DemoComponent and PolyAccelComponent both have one call site per hook, so this isn't a constraint in practice.)
5. **`.tpp` does NOT include `<component>.hpp`.** Would be circular. The `.tpp` relies on being included from the `.hpp` and has access to all the schema decls in that scope. Add a header-comment in the generated stub noting "this file is included from `<component>.hpp`; types defined there are in scope."
6. **Call sites use C++ template deduction.** `demo::process(cmd, s_in)` — no explicit `<in_bw>`. The stream argument type uniquely identifies the template param.
7. **Sticky-file lifecycle unchanged.** `.tpp` files are written only if absent, never overwritten. Same rule as `.cpp` impls.
8. **Stale impl file detection.** If a hook was previously non-templated and had a sticky `<hook>_impl.cpp`, and now becomes templated (codegen wants `<hook>_impl.tpp`), `HlsCodegenStep` raises a clear error pointing at the stale `.cpp` and asking the user to delete it. Do NOT auto-delete user-owned files.
9. **Non-templated hooks are unchanged.** No edits to existing `_emit_function_call`, `impl_stub_to_cpp`, etc. when the hook has no templated args. The branching happens at the discovery layer.
10. **HwComponent migration in this plan: DemoComponent only.** Refactor `DemoComponent.process` to take `s_in: StreamIFSlave` as an arg. This creates one templated-hook example for the demo to verify. PolyAccelComponent's evaluate hook is already in this shape — it'll work after this plan lands, but wiring it through `poly_build.py` is the next phase, not this one.

## Reference reading (read once before starting)

- [pysilicon/build/hwgen.py](../pysilicon/build/hwgen.py) — `hook_signature`, `hook_signature_str`, `header_to_cpp`, `impl_stub_to_cpp`, `kernel_files_to_str`, `_collect_hooks`, `_stream_template_arg`. Most edits land here.
- [pysilicon/build/hwcodegen_steps.py](../pysilicon/build/hwcodegen_steps.py) — `HlsCodegenStep.produces` and its `run()` method. Needs per-hook extension awareness.
- [pysilicon/hw/hwstmt.py](../pysilicon/hw/hwstmt.py) — `FunctionStmt`. `inputs` carries the resolved endpoint instances at the call site.
- [experiment/extract_demo.py](../experiment/extract_demo.py) — `DemoComponent`. You'll refactor `process` to take a stream arg in Phase 6.
- [experiment/buildstep_demo.py](../experiment/buildstep_demo.py) — visual deliverable target.

## Working convention

- One commit per phase, in order, push after each.
- Run `pytest tests/hw/ tests/build/` after every phase. All must stay green (the 12 pre-existing failures in `tests/build/test_build.py` are unrelated; ignore them, do not "fix").
- Final acceptance: `python experiment/buildstep_demo.py` shows a templated hook in `demo.hpp`, an `#include "demo_process_impl.tpp"` line at the bottom, and a `demo_process_impl.tpp` file in `experiment/gen/`.

---

## Phase 1: `hook_template_params` helper

**Goal:** Pure analysis function — given a `FunctionStmt`, return the ordered, deduplicated list of `HwParamValue.param_name`s used by its stream-typed call-site arguments.

**Changes:**

- In [pysilicon/build/hwgen.py](../pysilicon/build/hwgen.py), add:

  ```python
  from pysilicon.hw.interface import InterfaceEndpoint, StreamIFMaster, StreamIFSlave
  from pysilicon.hw.hw_component import HwParamValue


  def hook_template_params(stmt: FunctionStmt) -> list[str]:
      """Return the ordered, deduplicated list of HwParam names this hook is templated on.

      An empty list means the hook is NOT templated and is emitted in a .cpp file.
      A non-empty list means the hook is templated and goes in a .tpp file.
      """
      params: list[str] = []
      seen: set[str] = set()
      for inp in stmt.inputs:
          if isinstance(inp, (StreamIFSlave, StreamIFMaster)):
              bw = inp.bitwidth
              if isinstance(bw, HwParamValue) and bw.param_name not in seen:
                  params.append(bw.param_name)
                  seen.add(bw.param_name)
      return params


  def _validate_single_call_site(tree, hook_method, template_params: list[str]) -> None:
      """Templated hooks must be called from exactly one site with consistent template params.

      Walk the tree, find every FunctionStmt for this hook, ensure the template
      param list matches. Raise SynthesisError if not.
      """
      from pysilicon.build.hwcodegen import SynthesisError
      sites: list[list[str]] = []
      def visit(node):
          if isinstance(node, FunctionStmt) and node.method is hook_method:
              sites.append(hook_template_params(node))
          for child in _stmt_children(node):
              visit(child)
      visit(tree)
      if len(sites) > 1 and any(s != sites[0] for s in sites[1:]):
          raise SynthesisError(
              f"Templated hook '{hook_method.__name__}' called from {len(sites)} sites "
              f"with inconsistent template params: {sites}"
          )


  def _stmt_children(node) -> list:
      """Return child stmts of a node, for tree walks."""
      if isinstance(node, WhileStmt):
          return [node.body]
      if isinstance(node, SeqStmt):
          return list(node.stmts)
      if isinstance(node, CaseStmt):
          out = [node.if_true]
          if node.if_false is not None:
              out.append(node.if_false)
          return out
      return []
  ```

**Tests** (extend `tests/hw/test_hwgen.py`):

- `hook_template_params(FunctionStmt(...))` on a stmt with no inputs → `[]`.
- On a stmt with `HwVar`-only inputs (no endpoint args) → `[]`.
- On a stmt with one `_FakeParamEndpoint(param_name="in_bw")` input → `["in_bw"]`.
- On a stmt with two endpoints, both same param_name → deduplicated to one.
- On a stmt with two endpoints with different param_names → both, in input order.
- On a stmt with a raw-int endpoint (no HwParamValue) mixed with a param endpoint → only the param name returned.
- `_validate_single_call_site` raises when called twice with different params, passes when called once or with consistent params.

**Commit:** `hwgen: hook_template_params helper + single-call-site validation`

---

## Phase 2: `hook_signature_str` emits templated decls

**Goal:** Extend `hook_signature_str` to accept optional template params and emit `template <int in_bw, ...>` prefix when non-empty. Stream-typed arg types use the template names.

**Changes:**

- In [pysilicon/build/hwgen.py](../pysilicon/build/hwgen.py), modify `hook_signature_str` (and the underlying `hook_signature` if needed) to take a `template_params: list[str] | None = None` argument. When provided and non-empty:
  - Prefix the signature with `template <int p1, int p2, ...>\n`.
  - In the args block, stream-typed args use `axi4s_word<template_param>` instead of `axi4s_word<WORD_BW>`. The mapping from stream-arg position to template-param name comes from the order of stream args in the call site — first stream arg uses the first template param, etc. **Equivalent rule:** the i-th stream arg of the hook uses the i-th param name from `template_params` (which were collected in input order in Phase 1).

  Sketch:

  ```python
  def hook_signature_str(method, template_params: list[str] | None = None) -> str:
      ret_cpp, args = hook_signature(method, template_params=template_params)
      arg_str = ", ".join(arg_decl for _, arg_decl in args)
      prefix = ""
      if template_params:
          prefix = f"template <{', '.join(f'int {p}' for p in template_params)}>\n"
      return f"{prefix}{ret_cpp} {method.__name__}({arg_str})"
  ```

  And in `hook_signature`, when building the args list and encountering a stream-typed arg with `template_params` provided, pop the next param name from the list to use in the type expression.

**Tests:**

- `hook_signature_str(hook_method)` with no `template_params` argument → unchanged from current behavior (uses `WORD_BW`). Existing test `test_hook_signature_str_with_stream_endpoint` continues to pass.
- `hook_signature_str(hook_method, template_params=["in_bw"])` for a hook taking one stream arg → returns `"template <int in_bw>\n... process(... hls::stream<...axi4s_word<in_bw>>& s_in)"`.
- Two template params, two stream args → both substituted, prefix has both.

**Commit:** `hwgen: hook_signature_str accepts template_params for templated decl emission`

---

## Phase 3: Header emits templated forward decls + `.tpp` includes

**Goal:** `header_to_cpp` discovers per-hook template params from the tree, emits the appropriate forward decls (templated or not), and adds `#include "<component>_<hook>_impl.tpp"` lines at the bottom for each templated hook.

**Changes:**

- In [pysilicon/build/hwgen.py](../pysilicon/build/hwgen.py), modify `header_to_cpp`:

  ```python
  def header_to_cpp(comp) -> str:
      tree = extract_kernel(comp)
      schemas = _collect_schemas(tree, comp)
      lines = ["#pragma once", ""]
      lines.append('#include "include/streamutils_hls.h"')
      for s in schemas:
          lines.append(f'#include "include/{s.cpp_class_name().lower()}.h"')
      lines.append("")
      lines.append(_kernel_signature_decl(comp))

      # Hook forward declarations
      hooks = _collect_hooks_with_params(tree)   # NEW: returns [(method, [params])]
      if hooks:
          ns = resolved_namespace(type(comp))
          lines.append("")
          if ns is not None:
              lines.append(f"namespace {ns} {{")
              indent = "    "
          else:
              indent = ""
          for hook, tparams in hooks:
              for line in hook_signature_str(hook, template_params=tparams).split("\n"):
                  lines.append(f"{indent}{line}")
              lines[-1] += ";"
          if ns is not None:
              lines.append("}")

      # .tpp includes for templated hooks — at the very bottom, after all decls
      templated = [(h, p) for h, p in hooks if p]
      if templated:
          lines.append("")
          kn = cpp_kernel_name(type(comp))
          for hook, _ in templated:
              lines.append(f'#include "{kn}_{hook.__name__}_impl.tpp"')

      return "\n".join(lines) + "\n"


  def _collect_hooks_with_params(tree) -> list[tuple[object, list[str]]]:
      """Walk tree, return [(method, template_params)] for each unique FunctionStmt.

      Validates single-call-site consistency for each templated hook.
      """
      seen_methods: dict[int, tuple[object, list[str]]] = {}
      def visit(node):
          if isinstance(node, FunctionStmt):
              key = id(node.method)
              tparams = hook_template_params(node)
              if key not in seen_methods:
                  seen_methods[key] = (node.method, tparams)
              else:
                  _, existing = seen_methods[key]
                  if existing != tparams:
                      from pysilicon.build.hwcodegen import SynthesisError
                      raise SynthesisError(
                          f"Hook '{node.method.__name__}' called with inconsistent "
                          f"template params: {existing} vs {tparams}"
                      )
          for child in _stmt_children(node):
              visit(child)
      visit(tree)
      return list(seen_methods.values())
  ```

**Tests:**

- Non-templated hook (no stream args): forward decl emitted without `template` prefix. No `#include` line for `.tpp` at the bottom.
- Templated hook: forward decl includes `template <int in_bw>` line. `.tpp` include emitted at the bottom.
- Mixed: a component with one non-templated and one templated hook → forward decls for both, only the templated one gets a `.tpp` include.

**Commit:** `hwgen: header emits templated hook decls + .tpp includes`

---

## Phase 4: `.tpp` impl file generation + driver routing

**Goal:** New `impl_stub_to_tpp(comp, hook, template_params)` builds the `.tpp` content. `kernel_files_to_str` routes each hook to `.cpp` or `.tpp` based on whether `hook_template_params` is empty.

**Changes:**

- In [pysilicon/build/hwgen.py](../pysilicon/build/hwgen.py), add:

  ```python
  def impl_stub_to_tpp(comp, hook_method, template_params: list[str]) -> str:
      """Build the first-time stub content for one templated hook .tpp file."""
      ns = resolved_namespace(type(comp))
      ret_cpp, args = hook_signature(hook_method, template_params=template_params)
      arg_str = ", ".join(arg_decl for _, arg_decl in args)
      default = _stub_default_return(ret_cpp)
      body_lines = [f"    // TODO: implement {hook_method.__name__}"]
      if default:
          body_lines.append(f"    {default}")
      body = "\n".join(body_lines)
      tparam_str = ", ".join(f"int {p}" for p in template_params)
      func_def = (
          f"template <{tparam_str}>\n"
          f"{ret_cpp} {hook_method.__name__}({arg_str}) {{\n"
          f"{body}\n"
          f"}}"
      )
      if ns is not None:
          func_def = f"namespace {ns} {{\n{func_def}\n}}"
      header = (
          "// This file is included from "
          f"{cpp_kernel_name(type(comp))}.hpp at the bottom — types declared\n"
          "// there are in scope. Do not include this file directly except via the .hpp.\n\n"
      )
      return header + func_def + "\n"
  ```

- Modify `kernel_files_to_str` to route per-hook:

  ```python
  def kernel_files_to_str(comp) -> dict[str, str]:
      name = cpp_kernel_name(type(comp))
      files: dict[str, str] = {
          f"{name}.hpp": header_to_cpp(comp),
          f"{name}.cpp": kernel_to_cpp(comp),
      }
      tree = extract_kernel(comp)
      for hook, tparams in _collect_hooks_with_params(tree):
          if tparams:
              files[f"{name}_{hook.__name__}_impl.tpp"] = impl_stub_to_tpp(comp, hook, tparams)
          else:
              files[f"{name}_{hook.__name__}_impl.cpp"] = impl_stub_to_cpp(comp, hook)
      return files
  ```

**Tests:**

- `impl_stub_to_tpp` for a templated hook contains: the header comment, `template <int in_bw>`, the function signature, `// TODO: implement`, the default return, and is wrapped in `namespace demo { ... }` if applicable.
- `kernel_files_to_str` for a component with a templated hook returns a dict whose keys include `<component>_<hook>_impl.tpp` (not `.cpp`).
- For a component with a non-templated hook, the dict still includes `<component>_<hook>_impl.cpp`.
- Mixed: dict has both extensions, one per hook.

**Commit:** `hwgen: impl_stub_to_tpp + kernel_files_to_str per-hook extension routing`

---

## Phase 5: `HlsCodegenStep` per-hook extension awareness + stale-file detection

**Goal:** `HlsCodegenStep` correctly declares both `.cpp` and `.tpp` impl files in `produces`, writes them sticky, and detects the stale-file case (hook was non-templated, became templated — leaves a `.cpp` lying around).

**Changes:**

- In [pysilicon/build/hwcodegen_steps.py](../pysilicon/build/hwcodegen_steps.py), modify `_discover_hooks` and `produces`:

  ```python
  def _discover_hooks(self) -> list[tuple[str, str]]:
      """Return [(hook_name, extension)] for each hook on the component.

      Extension is 'cpp' or 'tpp' depending on whether the hook is templated.
      """
      from pysilicon.build.hwgen import _collect_hooks_with_params
      comp = self.comp_class(name="_codegen", sim=Simulation())
      tree = extract_kernel(comp)
      result: list[tuple[str, str]] = []
      for hook, tparams in _collect_hooks_with_params(tree):
          ext = "tpp" if tparams else "cpp"
          result.append((hook.__name__, ext))
      return result


  @property
  def produces(self) -> dict:
      out_dir = Path(self.output_dir)
      d = {
          f"{self._kernel_name}_hpp": out_dir / f"{self._kernel_name}.hpp",
          f"{self._kernel_name}_cpp": out_dir / f"{self._kernel_name}.cpp",
      }
      for hook_name, ext in self._hook_info:  # was self._hook_names
          key = f"{self._kernel_name}_{hook_name}_impl"
          d[key] = out_dir / f"{self._kernel_name}_{hook_name}_impl.{ext}"
      return d
  ```

  Rename `self._hook_names` → `self._hook_info` (list of `(name, ext)` tuples). Update `__post_init__` and `run()` accordingly.

- In `run()`, add the stale-file check:

  ```python
  # After writing .hpp and .cpp, before writing impl files:
  for hook_name, ext in self._hook_info:
      stem = f"{self._kernel_name}_{hook_name}_impl"
      other_ext = "cpp" if ext == "tpp" else "tpp"
      stale_path = out_root / f"{stem}.{other_ext}"
      if stale_path.exists():
          raise RuntimeError(
              f"Stale impl file detected: {stale_path}. Hook '{hook_name}' is "
              f"now {'templated' if ext == 'tpp' else 'non-templated'}; expected "
              f"file is {stem}.{ext}. Delete the stale file and re-run."
          )
  ```

**Tests** (extend `tests/build/test_hwcodegen_steps.py`):

- For a component with a templated hook: `produces` dict has key for `.tpp` (not `.cpp`); `run()` writes the `.tpp` to disk; sticky behavior works (second run preserves it).
- Stale-file scenario: write a `.cpp` impl manually, then re-run with a component whose hook is templated → `RuntimeError` raised mentioning the stale path.
- Mixed component with templated and non-templated hooks: both files written correctly with the right extensions.

**Commit:** `hwcodegen_steps: per-hook extension routing + stale-file detection`

---

## Phase 6: Refactor DemoComponent.process + verify

**Goal:** Give the demo a templated hook to exercise the new path end-to-end. Visual verification via `experiment/buildstep_demo.py`.

**Changes:**

- In [experiment/extract_demo.py](../experiment/extract_demo.py), refactor `DemoComponent.process` to take a stream arg:

  ```python
  @synthesizable
  def process(self, cmd: DemoCmdHdr, s_in: StreamIFSlave) -> ProcessGen[DemoError]:
      """User compute hook - now takes a stream arg, triggering templating."""
      yield self.env.timeout(0)
      if int(cmd.tx_id) == 0:
          return DemoError.BAD_INPUT
      return DemoError.OK
  ```

  And update the `on_start` call to pass `self.s_in`:

  ```python
  err = yield from self.process(cmd, self.s_in)
  ```

- Delete `experiment/gen/demo_process_impl.cpp` (stale; will be replaced by `.tpp`).

- Run `python experiment/buildstep_demo.py` and confirm:
  - Three files in `experiment/gen/`: `demo.hpp`, `demo.cpp`, `demo_process_impl.tpp` (note `.tpp`, not `.cpp`).
  - `demo.hpp` contains:
    - `template <int in_bw>` before the `process` forward decl in the `namespace demo` block.
    - `#include "demo_process_impl.tpp"` at the bottom.
  - `demo.cpp` body calls `demo::process(cmd, s_in);` (template deduction, no explicit `<in_bw>`).
  - `demo_process_impl.tpp` is the templated stub with `// TODO: implement process` and `return DemoError(0);`.

**Verification commands:**

```bash
pytest tests/hw/ tests/build/
mypy pysilicon/build/hwgen.py pysilicon/build/hwcodegen_steps.py
ruff check pysilicon/build/hwgen.py pysilicon/build/hwcodegen_steps.py tests/hw/test_hwgen.py tests/build/test_hwcodegen_steps.py
python experiment/buildstep_demo.py
```

**Commit:** `demo: process takes stream arg (triggers templating); verify .tpp pipeline`

---

## Final acceptance

- `pytest tests/hw/ tests/build/` passes (modulo the pre-existing 12 failures in `tests/build/test_build.py`).
- `mypy` and `ruff` clean on touched files.
- `python experiment/buildstep_demo.py` prints success for all steps and writes three files including `demo_process_impl.tpp`.
- `experiment/gen/demo.hpp` ends with `#include "demo_process_impl.tpp"`.
- `experiment/gen/demo.cpp` body calls `demo::process(cmd, s_in);` (no explicit template args).
- 6 commits on `main`, one per phase, pushed in order.
- New / modified files only:
  - [pysilicon/build/hwgen.py](../pysilicon/build/hwgen.py) — `hook_template_params`, `_validate_single_call_site`, `_collect_hooks_with_params`, `impl_stub_to_tpp`, updates to `hook_signature_str` / `header_to_cpp` / `kernel_files_to_str`.
  - [pysilicon/build/hwcodegen_steps.py](../pysilicon/build/hwcodegen_steps.py) — per-hook extension routing in `_discover_hooks`, `produces`, `run`.
  - [experiment/extract_demo.py](../experiment/extract_demo.py) — `process` takes a stream arg.
  - [tests/hw/test_hwgen.py](../tests/hw/test_hwgen.py) — extended.
  - [tests/build/test_hwcodegen_steps.py](../tests/build/test_hwcodegen_steps.py) — extended.

## Out of scope (do NOT do)

- Wiring `HlsCodegenStep` into `examples/poly/poly_build.py`. **Next plan.**
- Migrating PolyAccelComponent. (It already has the right `evaluate` shape; this plan just makes the generated code work. Actual integration is next.)
- Multi-call-site templated hooks. Validation raises if encountered; doesn't try to handle.
- Hook templating for non-stream arg types (e.g., `HwConst` values that vary across instances — not a real case since `HwConst` is class-level).
- Auto-deletion of stale `.cpp` impl files. Raise an error and ask the user to delete; do not delete user-owned files.
- Schema-header naming convention fix (`hwgen.py` emits `include/<lower>.h` vs `DataSchemaStep` produces `include/<snake_case>.h`). Will surface during the next plan's poly integration; address there.
- HwConst C++ codegen.

If a design question arises that this plan doesn't answer, stop and ask — do not invent a new convention.
