# HwComponent Phase 3.5: Extractor End-to-End on `on_start`

## Goal

Make `HwStmtExtractor` produce a valid `HwStmt` tree from the real `PolyAccelComponent.on_start` (in [examples/poly/poly.py](../examples/poly/poly.py)) by extending the synthesizable subset and adding the missing statement types. No C++ codegen — just the IR tree.

## Already done (do NOT redo)

- `@synthesizable`, `@sim_only`, `HwComponent`, `HwParam`, `SynthContext`, `ControlMode` — in [pysilicon/hw/synth.py](../pysilicon/hw/synth.py) and [pysilicon/hw/hw_component.py](../pysilicon/hw/hw_component.py).
- `HwStmt` IR (`SeqStmt`, `WhileStmt`, `CaseStmt`, `ContinueStmt`, `SynthCallStmt`, `HookStmt`, `HwVar`, `Ref`, `FieldRef`) — in [pysilicon/hw/hwstmt.py](../pysilicon/hw/hwstmt.py).
- `HwStmtExtractor` and `SynthesisError` — in [pysilicon/build/hwcodegen.py](../pysilicon/build/hwcodegen.py).
- Stream stmt classes (`StreamGetStmt`, `StreamWriteStmt`, `StreamDrainStmt`, `StreamGetPipelinedStmt`, `StreamWritePipelinedStmt`) — in [pysilicon/hw/interface.py](../pysilicon/hw/interface.py).
- `VitisRegMap` / `VitisRegMapMMIFSlave` — in [pysilicon/hw/regmap.py](../pysilicon/hw/regmap.py).

## Design decisions (already settled — do NOT re-litigate)

1. **Kernel body method:** for components with a `VitisRegMapMMIFSlave` endpoint, extract `on_start`. Otherwise, extract `run_proc`. Selection lives in a module-level `extract_kernel(comp)` helper.
2. **No implicit captures in synthesizable methods.** Inside any synthesizable method body, `self.foo` attribute *reads* are forbidden unless `foo` is `@sim_only`. Method *calls* on `self.X.method(...)` are allowed only if `method` is `@synthesizable`. Extractor enforces both at extraction time.
3. **User compute methods become `FunctionStmt`, NOT recursive extraction.** When the extractor sees `yield from self.method(...)` where `method` is `@synthesizable` with no `synth_fn`, produce a `FunctionStmt` referencing the method by name. Do not walk the method body. The C++ implementation is hand-written by the user.
4. **One impl file per hook:** `<component>_<function>_impl.cpp`. Override via `@synthesizable(impl_file="...")`.
5. **Prototypes live in `<component>.hpp`** for both the kernel function and every hook.
6. **`VitisRegMap.set` and `.get` are synthesizable.** They become AXI-Lite scalar writes/reads in C++. Each gets a dedicated `SynthCallStmt` subclass (`RegMapSetStmt`, `RegMapGetStmt`).
7. **`return` (with or without value)** is allowed inside `if` bodies and at the top level of `while True:`. Added as `ReturnStmt`.
8. **`!=` is allowed in `CaseStmt` test** in addition to `==`.

## Reference reading (read once before starting)

- [docs/guide/interface/regmap.md](../docs/guide/interface/regmap.md) — `VitisRegMap`, hook contract, `on_start` lifecycle.
- [pysilicon/hw/regmap.py](../pysilicon/hw/regmap.py) — actual `RegMap` / `VitisRegMap` API.
- [pysilicon/build/hwcodegen.py](../pysilicon/build/hwcodegen.py) — current extractor.
- [pysilicon/hw/hwstmt.py](../pysilicon/hw/hwstmt.py) — current IR.
- [examples/poly/poly.py](../examples/poly/poly.py) lines 149–235 — `PolyAccelComponent.__post_init__`, `on_start`, and `evaluate`. The end-to-end test target.

## Working convention

- Each phase is a separate commit; push after each commit.
- Run `pytest tests/hw/test_hw_component.py tests/hw/test_phase3.py` after every phase. All previously-passing tests must continue to pass.
- New tests live in `tests/hw/test_extract_poly.py` (one new file).

---

## Phase 1: `ReturnStmt`

**Goal:** Allow `return` (with or without a value) inside `if` bodies and at the top level of `while True:`.

**Changes:**

- In [pysilicon/hw/hwstmt.py](../pysilicon/hw/hwstmt.py), add:

  ```python
  @dataclass
  class ReturnStmt(HwStmt):
      """`return` from the kernel function. Optional return value."""
      value: HwExpr | None = None
  ```

- In [pysilicon/build/hwcodegen.py](../pysilicon/build/hwcodegen.py), extend `_visit_stmt` to handle `ast.Return`:
  - `return` with no value → `ReturnStmt(value=None)`.
  - `return <Name>` where the name is a bound `HwVar` → `ReturnStmt(value=Ref(var=...))`.
  - `return <Attribute>` where it's a known `FieldRef` pattern → `ReturnStmt(value=FieldRef(...))`.
  - Anything else → `SynthesisError`.

**Tests** (new file `tests/hw/test_extract_poly.py`):

- Extract a tiny synthetic component whose `run_proc` is `while True: yield from self.s_in.get(...); if x.f == V: return`. Assert the inner `if` body's first stmt is a `ReturnStmt`.
- Extract a `run_proc` that returns at top level after one operation. Assert the outer tree contains a `ReturnStmt`.

**Commit:** `extractor: add ReturnStmt support inside if-body and while-true`

---

## Phase 2: `!=` in `CaseStmt`

**Goal:** Allow `if x.f != V:` in addition to `if x.f == V:`.

**Changes:**

- In [pysilicon/hw/hwstmt.py](../pysilicon/hw/hwstmt.py), add an `op` field to `CaseStmt`:

  ```python
  @dataclass
  class CaseStmt(HwStmt):
      var:      HwVar
      field:    str
      value:    object
      if_true:  SeqStmt
      if_false: SeqStmt | None = None
      op:       str = '=='                # '==' or '!='
  ```

- In `_visit_if`, accept `ast.NotEq` in addition to `ast.Eq`. Store the operator on the resulting `CaseStmt`.

**Tests:**

- Extract a `run_proc` with `if err != PolyError.NO_ERROR:` and assert the resulting `CaseStmt.op == '!='`.
- Existing `==` cases continue to extract with `op == '=='` (regression check).

**Commit:** `extractor: support != in CaseStmt test`

---

## Phase 3: `extract_kernel(comp)` policy helper

**Goal:** The extractor must be able to target `on_start` for regmap-bearing components.

**Changes:**

- In [pysilicon/build/hwcodegen.py](../pysilicon/build/hwcodegen.py):
  - Add `method_name: str = 'run_proc'` parameter to `HwStmtExtractor.__init__`. Store on `self._method_name`. Use it in `extract()` where `comp.run_proc` is currently hard-coded.
  - Add a module-level helper:

    ```python
    def extract_kernel(comp: HwComponent) -> HwStmt:
        """Pick on_start if the component has a VitisRegMapMMIFSlave; else run_proc."""
        from pysilicon.hw.regmap import VitisRegMapMMIFSlave   # local import: avoid cycle
        for ep in getattr(comp, 'endpoints', {}).values():
            if isinstance(ep, VitisRegMapMMIFSlave):
                return HwStmtExtractor(comp, method_name='on_start').extract()
        return HwStmtExtractor(comp, method_name='run_proc').extract()
    ```

**Tests:**

- Extract from a small `HwComponent` with no regmap → confirm `extract_kernel` calls `run_proc`.
- Extract from a small `HwComponent` with a `VitisRegMapMMIFSlave` endpoint → confirm `extract_kernel` targets `on_start`.

**Commit:** `extractor: add method_name param and extract_kernel policy helper`

---

## Phase 4: No-implicit-capture rule

**Goal:** Extractor refuses any `self.foo` *attribute read* in a synthesizable method body unless `foo` is `@sim_only` or `foo.method(...)` is a `@synthesizable` method call. Method calls on `@synthesizable` paths (including endpoint methods and regmap methods) are always allowed.

**Changes:**

- In [pysilicon/build/hwcodegen.py](../pysilicon/build/hwcodegen.py):
  - Add a pre-pass `_validate_no_implicit_capture(func_def)` that walks the AST of the body before extraction. For every `ast.Attribute` node:
    - If the attribute root is `self` and the access is the *function* part of an `ast.Call` (i.e., `self.X.method(...)`), require `method` to be `@synthesizable`. Allow.
    - If the attribute root is `self` and the access is a *read* (not a call function), check if it resolves to a `@sim_only` callable. If yes (the read is `self.logger.log` etc.), allow. Otherwise raise `SynthesisError` naming the attribute and the line number.
    - `self.foo()` where `foo` is `@sim_only` → drop silently (already handled by `_visit_expr_stmt`).
  - The existing `_require_synthesizable` already handles method calls; the new check is specifically for *reads*.

- Note: `var.field` reads (where `var` is a bound `HwVar`) are still allowed — they become `FieldRef` expressions. The rule only applies to `self.X` reads, not to `<HwVar>.X` reads.

**Tests:**

- Component whose `run_proc` reads `self.proc_latency` (plain field, not `@sim_only`) → expect `SynthesisError` mentioning `proc_latency`.
- Component whose `run_proc` calls `self.logger.log(...)` where `log` is `@sim_only` → no error, statement is dropped.
- Component whose `run_proc` calls `self.s_in.get(...)` (endpoint, `@synthesizable`) → no error.

**Commit:** `extractor: enforce no-implicit-capture rule on self.X reads`

---

## Phase 5: `RegMapGetStmt`, `RegMapSetStmt`, decorate `VitisRegMap.get` and `.set`

**Goal:** Allow `self.regmap.get(name)` and `self.regmap.set(name, value)` calls in `on_start` and synthesizable user methods. Each becomes a typed `SynthCallStmt` subclass.

**Changes:**

- In [pysilicon/hw/regmap.py](../pysilicon/hw/regmap.py):
  - Add two stmt classes (place near the top of the file, after imports):

    ```python
    from pysilicon.hw.hwstmt import SynthCallStmt
    from dataclasses import dataclass

    @dataclass
    class RegMapGetStmt(SynthCallStmt):
        """Synthesizable read of a regmap field — emits an AXI-Lite scalar read."""

    @dataclass
    class RegMapSetStmt(SynthCallStmt):
        """Synthesizable write to a regmap field — emits an AXI-Lite scalar write."""
    ```

  - Decorate `RegMap.set` and `RegMap.get` (the base class — `VitisRegMap` inherits):
    - `set`: `@synthesizable(stmt_class=RegMapSetStmt)`
    - `get`: `@synthesizable(stmt_class=RegMapGetStmt)`
  - Import `synthesizable` from `pysilicon.hw.synth`. No `synth_fn` is provided (codegen lives in a later phase).
  - These decorators must not change runtime behavior — verify existing regmap tests still pass.

**Tests:**

- Extract a `run_proc` containing `self.regmap.set("error", v)` where `v` is a bound `HwVar`. Assert the produced stmt `isinstance(s, RegMapSetStmt)` and that its `inputs` list contains the right field name (as an `ast.Constant` or string) and the `HwVar`.
- Extract a `run_proc` containing `coeffs = self.regmap.get("coeffs")`. Assert the produced stmt is a `RegMapGetStmt`, `coeffs` is bound in scope.
- All tests in `tests/hw/test_regmap.py` continue to pass.

**Commit:** `regmap: make get/set synthesizable with RegMapGetStmt and RegMapSetStmt`

---

## Phase 6: `FunctionStmt` + `@synthesizable(impl_file=...)`

**Goal:** Calls to user-written `@synthesizable` methods (no `synth_fn`) produce a `FunctionStmt` carrying the hook name and impl-file location. The extractor does not recurse into the method body.

**Changes:**

- In [pysilicon/hw/synth.py](../pysilicon/hw/synth.py):
  - Add `impl_file: str | None = None` parameter to the `synthesizable` decorator. Store as `f._impl_file = impl_file`.

- In [pysilicon/hw/hwstmt.py](../pysilicon/hw/hwstmt.py):
  - Add:

    ```python
    @dataclass
    class FunctionStmt(SynthCallStmt):
        """Call to a user-written @synthesizable method (no synth_fn).

        Codegen emits a forward declaration in <component>.hpp and a call site
        in <component>.cpp. The implementation lives in the impl_file (default
        `<component>_<function>_impl.cpp`), hand-written by the user.
        """
        impl_file: str | None = None
    ```

- In [pysilicon/build/hwcodegen.py](../pysilicon/build/hwcodegen.py):
  - Modify `_make_call_stmt`: when the method has `_synth_fn is None` and `_is_synthesizable is True`, produce a `FunctionStmt`. Read `_impl_file` from the method and set it on the `FunctionStmt`.
  - Remove the `HookStmt` import.

- **Delete `HookStmt` entirely:**
  - Remove the class from [pysilicon/hw/hwstmt.py](../pysilicon/hw/hwstmt.py).
  - Remove any docstring references to `HookStmt` in [pysilicon/hw/synth.py](../pysilicon/hw/synth.py).
  - Update [tests/hw/test_hwstmt.py](../tests/hw/test_hwstmt.py) to remove any `HookStmt` tests; replace with equivalent `FunctionStmt` tests where relevant.
  - Run `grep -r HookStmt` to confirm zero remaining references in the tree.

**Tests:**

- Extract a `run_proc` that calls `yield from self.evaluate(cmd_hdr, ...)` where `evaluate` is `@synthesizable` (no `synth_fn`). Assert the produced stmt is a `FunctionStmt`, its `method.__name__ == 'evaluate'`, `impl_file is None`.
- Same with `@synthesizable(impl_file="custom.cpp")` → assert `impl_file == "custom.cpp"`.
- The `FunctionStmt` inputs list resolves call-site arguments (including endpoint references like `self.s_in`).

**Commit:** `extractor: produce FunctionStmt for user-written synthesizable methods`

---

## Phase 7: End-to-end test on `PolyAccelComponent.on_start`

**Goal:** Prove the extractor produces a structurally correct `HwStmt` tree for the real poly kernel body, with no errors.

**Changes:**

- In `tests/hw/test_extract_poly.py`, add:

  ```python
  def test_extract_poly_accel_on_start():
      from examples.poly.poly import PolyAccelComponent
      from pysilicon.build.hwcodegen import extract_kernel
      from pysilicon.hw.hwstmt import (
          WhileStmt, SeqStmt, CaseStmt, ReturnStmt, FunctionStmt,
      )
      from pysilicon.hw.interface import StreamGetStmt
      from pysilicon.hw.regmap import RegMapSetStmt
      from pysilicon.simulation.simulation import Simulation

      comp = PolyAccelComponent(name='p', sim=Simulation())
      tree = extract_kernel(comp)

      # Top level: WhileStmt with a SeqStmt body
      assert isinstance(tree, WhileStmt)
      body = tree.body.stmts

      # Expected (logging / _inc_job dropped via @sim_only):
      #   0: cmd_hdr = yield from self.s_in.get(PolyCmdHdr)   → StreamGetStmt
      #   1: if cmd_hdr.cmd_type == PolyCmdType.END: return    → CaseStmt(op='==')
      #   2: err = yield from self.evaluate(...)              → FunctionStmt
      #   3: if err != PolyError.NO_ERROR: ...                → CaseStmt(op='!=')
      assert isinstance(body[0], StreamGetStmt)

      assert isinstance(body[1], CaseStmt) and body[1].op == '=='
      end_branch = body[1].if_true.stmts
      assert any(isinstance(s, ReturnStmt) for s in end_branch)

      assert isinstance(body[2], FunctionStmt)
      assert body[2].method.__name__ == 'evaluate'

      assert isinstance(body[3], CaseStmt) and body[3].op == '!='
      halt_branch = body[3].if_true.stmts
      regmap_sets = [s for s in halt_branch if isinstance(s, RegMapSetStmt)]
      assert len(regmap_sets) == 3       # error, tx_id, halted
      assert any(isinstance(s, ReturnStmt) for s in halt_branch)
  ```

- Also add `test_extract_poly_accel_no_implicit_capture_violation`: a deliberately-malformed clone of `PolyAccelComponent` whose `on_start` reads `self.proc_latency`. Assert `extract_kernel(comp)` raises `SynthesisError` mentioning `proc_latency`.

**Verification commands:**

```bash
pytest tests/hw/test_extract_poly.py -v
pytest tests/hw/                            # nothing else regressed
pytest tests/                               # full suite still green
mypy pysilicon/hw/regmap.py pysilicon/hw/hwstmt.py pysilicon/hw/synth.py pysilicon/build/hwcodegen.py
ruff check pysilicon/hw/regmap.py pysilicon/hw/hwstmt.py pysilicon/hw/synth.py pysilicon/build/hwcodegen.py tests/hw/test_extract_poly.py
```

**Commit:** `tests: end-to-end extractor validation on PolyAccelComponent.on_start`

---

## Final acceptance

- `pytest tests/hw/test_extract_poly.py` passes.
- `pytest tests/` passes with no regressions.
- `mypy` and `ruff` clean on the touched files.
- 7 commits on the branch, one per phase, pushed in order.
- No changes outside the file list above (do **not** modify `examples/poly/poly.py`, build steps, or anything in `docs/`).

## Out of scope (do NOT do)

- `to_cpp()` on any `HwStmt` subclass.
- `HlsKernelStep` / `HlsImplStep` build steps.
- Stub file generation in `<component>_<function>_impl.cpp` (that's the next phase).
- `MMArrayReadStmt` / `MMArrayWriteStmt` for the MM endpoints (no poly path needs them).
- Modifying `PolyAccelComponent` or any other production code outside the extractor and IR.
- Type resolution for `HwVar` (leaving `typ=None` is fine for now).
- Any backend integration (Vitis, Vivado).

If a design question arises that this plan doesn't answer, stop and ask — do not invent a new convention.
