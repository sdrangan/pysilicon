"""Testbench-side ``HwComponent`` subclass.

A ``HwTestbench`` is a Python class whose ``main(self)`` method is both:

- runnable in simulation as a normal Python function (sequential reads/writes
  against the wrapped DUT's endpoints), and
- extractable to a C++ ``int main()`` testbench by ``HlsCodegenStep`` when
  configured with ``is_testbench=True``.

The two coexist with the existing simulation-side ``PolyTB(SimObj)`` style: the
SimPy concurrent testbench is preserved (it stays the timing-accurate model
used by ``PySimStep``).  ``HwTestbench`` is purely the codegen-source class
that produces a Vitis HLS ``main()`` C++ file.

The class itself is a thin marker; the extractor mode and the codegen
emitter handle the rest.  See ``plans/hwcomponent_testbench_codegen_plan.md``
Phase 1 design decisions for the full picture.
"""
from __future__ import annotations

from typing import ClassVar

from waveflow.hw.hw_component import HwComponent


class HwTestbench(HwComponent):
    """Base class for codegen-source testbenches.

    Subclasses override :meth:`main` with a **sequential** body that:

    - constructs a single DUT (e.g. ``dut = PolyAccelComponent(...)``),
    - reads input vectors from disk via the standard schema file-IO methods,
    - pushes stream data into the DUT's endpoints (``dut.s_in.push(...)``),
    - configures regmap fields (``dut.regmap.set(...)``),
    - calls ``dut.run()`` once,
    - pops the DUT's response streams (``dut.m_out.pop(...)``),
    - reads regmap status and writes the comparison artifacts to disk.

    Concurrent stimulus/capture coroutines (``env.process(...)``) are not
    supported in v1 — the body must be straight-line.  See
    ``plans/hwcomponent_testbench_codegen_plan.md`` Phase 14 scope.
    """

    #: Class-level marker. ``HlsCodegenStep.is_testbench`` auto-detects via
    #: ``issubclass(comp_class, HwTestbench)`` and falls back to this flag if
    #: someone declares a testbench-shaped class via mixin without inheriting
    #: directly from ``HwTestbench`` itself.
    _is_testbench: ClassVar[bool] = True

    #: Framework-provided handle on the testbench's data directory.  Reads
    #: of ``self.data_dir`` inside ``main()`` lower to the C++ ``data_dir``
    #: local that ``int main()`` populates from ``argv``.
    data_dir: ClassVar[str] = "data"

    def main(self) -> None:
        """Sequential testbench body. Subclasses override this."""
        raise NotImplementedError(
            f"{type(self).__name__} must override main()."
        )
