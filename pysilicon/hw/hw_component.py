from __future__ import annotations

import sys
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, ClassVar, Generic, TypeVar
import typing

from pysilicon.hw.component import Component

T = TypeVar('T')


class HwParam(Generic[T]):
    """Marks a dataclass field as a C++ template parameter.

    In simulation the field behaves as a normal Python attribute (dataclass
    does not enforce types). At build time the extractor collects all HwParam
    fields from ``get_type_hints()`` and maps them to C++ template names.

    C++ name convention: the Python field name verbatim — ``in_bw`` →
    ``in_bw``.
    """


class HwConst(Generic[T]):
    """Marks a class attribute as a class-level constant.

    Translates to ``static constexpr T name = value;`` in generated C++
    (codegen emission added in a follow-up phase). Immutable by convention —
    the framework does not prevent reassignment, but the marker signals
    "do not modify after class definition" to readers and to codegen.

    Usage::

        class CoeffArray(DataArray):
            ncoeff: HwConst[int] = 4
            max_shape = (ncoeff,)
    """


def discover_hw_const(cls) -> dict[str, Any]:
    """Walk the MRO and return ``{field_name: value}`` for every ``HwConst`` field.

    Order is class-MRO declaration order, deduplicated by name (subclass wins).
    Plain fields, ``HwParam`` fields, and ``ClassVar`` literals are excluded.
    """
    result: dict[str, Any] = {}
    for klass in reversed(cls.__mro__):
        hints = getattr(klass, '__annotations__', {})
        mod = sys.modules.get(klass.__module__)
        globs: dict = vars(mod) if mod is not None else {}
        for name, hint in hints.items():
            if isinstance(hint, str):
                try:
                    hint = eval(hint, globs)  # noqa: S307
                except Exception:
                    continue
            if typing.get_origin(hint) is HwConst:
                if hasattr(klass, name):
                    result[name] = getattr(klass, name)
    return result


class ControlMode(Enum):
    AUTO = auto()           # inferred from HwStmt root at build time
    FREE_RUNNING = auto()   # ap_ctrl_none  (WhileStmt at root)
    PER_INVOCATION = auto() # ap_ctrl_chain (SeqStmt at root)


@dataclass
class SynthContext:
    """Parameter context passed to every ``synth_fn`` during codegen."""

    component: HwComponent
    params: dict[str, str]  # Python name → C++ template param name

    def cpp_param(self, py_name: str) -> str:
        """Return the C++ expression for a parameter.

        Returns the template parameter name (e.g. ``'IN_BW'``) for
        ``HwParam`` fields, or ``repr(value)`` for ``ClassVar`` literals.
        """
        if py_name in self.params:
            return self.params[py_name]
        return repr(getattr(self.component, py_name))

    @classmethod
    def from_component(cls, comp: HwComponent) -> SynthContext:
        import sys
        params: dict[str, str] = {}
        comp_type = type(comp)
        # Walk only the HwComponent subclass layers — stop before HwComponent
        # itself to avoid evaluating SimObj/Component TYPE_CHECKING annotations.
        for klass in comp_type.__mro__:
            if klass is HwComponent:
                break
            if not issubclass(klass, HwComponent):
                break
            raw_ann = vars(klass).get('__annotations__', {})
            mod = sys.modules.get(klass.__module__)
            globs: dict = vars(mod) if mod is not None else {}
            for name, hint_val in raw_ann.items():
                if isinstance(hint_val, str):
                    try:
                        hint = eval(hint_val, globs)  # noqa: S307
                    except Exception:
                        continue
                else:
                    hint = hint_val
                if typing.get_origin(hint) is HwParam:
                    params[name] = name.upper()
        return cls(component=comp, params=params)


class HwParamValue(int):
    """Int subclass that remembers which ``HwParam`` field it was bound to.

    Created automatically by :meth:`HwComponent.__post_init__` when wrapping
    raw values for ``HwParam``-annotated fields. Behaves as a plain ``int``
    for arithmetic, comparison, and protocol checks. Codegen inspects the
    ``.param_name`` attribute to decide between emitting a template name vs
    a literal value.
    """

    param_name: str  # type-only; the runtime attribute is set in __new__

    def __new__(cls, value: int, param_name: str) -> "HwParamValue":
        obj = super().__new__(cls, int(value))
        obj.param_name = param_name
        return obj

    def __repr__(self) -> str:
        return f"HwParamValue({int(self)!r}, {self.param_name!r})"

    def __str__(self) -> str:
        # Format / print / f-string must show the integer value, not the
        # diagnostic repr — codegen f-strings that haven't yet been migrated
        # to ``_stream_template_arg`` rely on this to keep emitting literals.
        # Going through a plain int sidesteps int's __str__/__repr__ slot
        # collision on subclasses that override __repr__.
        return str(int(self))

    def __format__(self, spec: str) -> str:
        # Same reason as __str__: f-string formatting must yield the int.
        return format(int(self), spec)


class HwComponent(Component):
    """Base class for synthesizable hardware components.

    Subclasses annotate synthesis template parameters with ``HwParam[T]``
    and mark compute methods with ``@synthesizable``.
    """

    control_mode: ClassVar[ControlMode] = ControlMode.AUTO
    cpp_kernel_name: ClassVar[str | None] = None
    cpp_namespace: ClassVar[str | None] = None
    """Override for the C++ namespace wrapping hooks for this component.

        None (default): namespace is auto-derived from cpp_kernel_name(cls).
        "":             opt out; hooks emitted in global namespace.
        "<name>":       use this string as the namespace verbatim.

    The kernel function itself is always emitted in the global namespace
    (Vitis HLS requires this).
    """

    def __post_init__(self) -> None:
        # Wrap HwParam field values BEFORE super().__post_init__ so any
        # subclass setup that reads self.<param> after super() sees
        # HwParamValue instances.
        self._wrap_hw_params()
        super().__post_init__()
        # Sentinel that flips immutability on. HwParam fields cannot be
        # reassigned once construction has completed.
        object.__setattr__(self, '_hw_construction_complete', True)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, '_hw_construction_complete', False):
            for klass in type(self).__mro__:
                klass_hints = getattr(klass, '__annotations__', {})
                if name not in klass_hints:
                    continue
                hint = klass_hints[name]
                if isinstance(hint, str):
                    mod = sys.modules.get(klass.__module__)
                    globs: dict = vars(mod) if mod is not None else {}
                    try:
                        hint = eval(hint, globs)  # noqa: S307
                    except Exception:
                        break
                if typing.get_origin(hint) is HwParam:
                    current = getattr(self, name, None)
                    raise AttributeError(
                        f"Cannot reassign HwParam field '{name}' after "
                        f"construction (current value: {current!r})"
                    )
                break
        object.__setattr__(self, name, value)

    def _wrap_hw_params(self) -> None:
        """Replace each ``HwParam[T]`` field value with a ``HwParamValue`` wrapper."""
        for klass in type(self).__mro__:
            if klass is HwComponent:
                break
            if not issubclass(klass, HwComponent):
                break
            raw_ann = vars(klass).get('__annotations__', {})
            mod = sys.modules.get(klass.__module__)
            globs: dict = vars(mod) if mod is not None else {}
            for name, hint_val in raw_ann.items():
                if isinstance(hint_val, str):
                    try:
                        hint = eval(hint_val, globs)  # noqa: S307
                    except Exception:
                        continue
                else:
                    hint = hint_val
                if typing.get_origin(hint) is not HwParam:
                    continue
                value = getattr(self, name, None)
                if value is None or isinstance(value, HwParamValue):
                    continue
                # object.__setattr__ bypasses the Phase 3 immutability guard.
                object.__setattr__(
                    self, name, HwParamValue(int(value), name)
                )
