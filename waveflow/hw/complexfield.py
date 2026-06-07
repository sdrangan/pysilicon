"""``ComplexField`` -- a complex DataSchema type generic over a scalar inner field.

A ``ComplexField`` is two inner components (real, imag) of the **same** inner format,
generic over ``FloatField`` / ``FixedField`` / ``IntField``.  Like ``FixedField``, arrays
use ``DataArray[ComplexField]`` (no parallel class) and arithmetic is **free functions**
(:func:`cadd`/:func:`csub`/:func:`cmult`/:func:`conj`), wired into the merged operator
layer as ``+``/``-``/``*``.

Value representation (``.val``) depends on the inner (see :mod:`waveflow.utils.complexutils`):

- **float**     -> native numpy complex (``complex64``/``complex128``).
- **fixed/int** -> a numpy **structured** scalar ``[('re', D), ('im', D)]`` of stored ints.

Serialization is **interleaved I/Q** -- ``inner.serialize(re)`` then ``inner.serialize(im)``,
total width ``2 x inner`` -- by *composing the inner field's own (de)serialization* (so the
float IEEE bit-view and wide-int multi-word packing are reused verbatim).  A
``DataArray[ComplexField]`` therefore maps directly onto a ``std::complex<T>`` array.

Arithmetic **composes** the inner field's arithmetic on the re/im components
(:mod:`waveflow.utils.complexutils`, which lowers fixed/int to
:mod:`waveflow.utils.fixputils` and float to numpy); result formats follow the
``FixedField`` rules and inherit the >64-bit guard.  ``cmult``/``conj`` require a signed
inner (they produce signed results); ``cadd``/``csub`` follow the inner rule.
"""
from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from waveflow.hw.dataschema import DataArray, DataField, FloatField, IntField, _elem_kind
from waveflow.utils import complexutils as cx
from waveflow.utils.complexutils import complex_dtype, int_format
from waveflow.utils.fixputils import Format


def _inner_kind(inner: type[DataField]) -> str:
    """Classify a scalar inner field as ``float`` / ``fixed`` / ``int``."""
    kind = _elem_kind(inner)
    if kind == "other":
        raise TypeError(
            f"ComplexField inner must be a FloatField / FixedField / IntField, got "
            f"{inner.__name__}.")
    return kind


class ComplexField(DataField):
    """Complex element type generic over a scalar inner field (real, imag share a format)."""

    inner_type: ClassVar[type[DataField] | None] = None
    kind: ClassVar[str] = ""                      # "float" | "fixed" | "int"
    is_complex_field: ClassVar[bool] = True       # marker for the operator layer dispatch
    bitwidth: ClassVar[int | None] = None
    cpp_type: ClassVar[str | None] = None
    can_gen_include: ClassVar[bool] = False
    _specializations: ClassVar[dict[tuple[Any, ...], type["ComplexField"]]] = {}

    # --- specialization -------------------------------------------------------
    @classmethod
    def specialize(cls, inner: type[DataField], **kwargs: Any) -> type["ComplexField"]:
        """Return a cached ``ComplexField`` over a scalar ``inner`` field (one per inner)."""
        if not isinstance(inner, type) or not issubclass(inner, DataField):
            raise TypeError("ComplexField inner must be a scalar DataField subclass.")
        kind = _inner_kind(inner)

        overrides = cls.validate_specialize_kwargs(kwargs)
        override_items = tuple(sorted(overrides.items()))
        key = (cls, inner, override_items)
        cached = cls._specializations.get(key)
        if cached is not None:
            return cached

        inner_bw = inner.get_bitwidth()
        if kind == "int":
            cpp_type = f"wf_cint<{inner_bw}>"     # std::complex<ap_int> is non-standard
        else:
            cpp_type = f"std::complex<{inner.cpp_type}>"
        subclass_name = f"Complex_{inner.__name__}"
        specialized_attrs = cls.merge_specialize_attrs(
            {
                "inner_type": inner,
                "kind": kind,
                "bitwidth": 2 * inner_bw,
                "cpp_type": cpp_type,
                "__module__": cls.__module__,
                "__doc__": f"Specialized complex field: {cpp_type}.",
            },
            overrides,
        )
        specialized = type(subclass_name, (cls,), specialized_attrs)
        cls._specializations[key] = specialized
        return specialized

    # --- inner format / representation ---------------------------------------
    @classmethod
    def inner_format(cls) -> Format:
        """The fixed/int inner's :class:`Format` (raises for a float inner)."""
        inner = cls.inner_type
        if cls.kind == "fixed":
            return inner.get_format()
        if cls.kind == "int":
            return int_format(inner.get_bitwidth(), inner.signed)
        raise TypeError("a float inner has no integer Format.")

    @classmethod
    def _np_complex_type(cls) -> type:
        return np.complex128 if cls.inner_type.get_bitwidth() == 64 else np.complex64

    @classmethod
    def init_value(cls) -> Any:
        if cls.kind == "float":
            return cls._np_complex_type()(0)
        return np.zeros((), dtype=complex_dtype(cls.inner_format()))[()]

    def _convert(self, value: Any) -> Any:
        cls = type(self)
        if isinstance(value, ComplexField):
            value = value.val
        re, im = _extract_re_im(value)
        if cls.kind == "float":
            return cls._np_complex_type()(complex(float(re), float(im)))
        # fixed/int: route each component through the inner's wrapping, then pack stored ints
        inner = cls.inner_type
        rf, imf = inner(), inner()
        rf.val, imf.val = re, im
        out = np.zeros((), dtype=complex_dtype(cls.inner_format()))
        out["re"], out["im"] = rf.val, imf.val
        return out[()]

    # --- interleaved I/Q (de)serialization, composing the inner field ---------
    def _inner_fields(self) -> tuple[DataField, DataField]:
        cls = type(self)
        inner = cls.inner_type
        v = self.val
        rf, imf = inner(), inner()
        if cls.kind == "float":
            rf.val, imf.val = float(np.real(v)), float(np.imag(v))
        else:
            rf.val, imf.val = int(v["re"]), int(v["im"])
        return rf, imf

    def _serialize_recursive(
        self, word_bw: int, words: list[int], ipos0: int = 0, iword0: int = 0,
    ) -> tuple[int, int]:
        re_field, im_field = self._inner_fields()
        pos, word = re_field._serialize_recursive(word_bw, words, ipos0, iword0)
        return im_field._serialize_recursive(word_bw, words, pos, word)

    def _deserialize_recursive(
        self, word_bw: int, words: list[int], ipos0: int = 0, iword0: int = 0,
    ) -> tuple[int, int]:
        inner = type(self).inner_type
        re_field, im_field = inner(), inner()
        pos, word = re_field._deserialize_recursive(word_bw, words, ipos0, iword0)
        pos, word = im_field._deserialize_recursive(word_bw, words, pos, word)
        if type(self).kind == "float":
            self.val = complex(float(re_field.val), float(im_field.val))
        else:
            self.val = (int(re_field.val), int(im_field.val))
        return pos, word


def _extract_re_im(value: Any) -> tuple[Any, Any]:
    """Pull (re, im) out of a structured scalar / complex / (re, im) pair / real number."""
    if isinstance(value, ComplexField):
        value = value.val
    if isinstance(value, (np.void, np.ndarray)) and value.dtype.names:
        return value["re"], value["im"]
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError(f"complex (re, im) pair must have length 2, got {len(value)}.")
        return value[0], value[1]
    if isinstance(value, (complex, np.complexfloating)):
        return value.real, value.imag
    return value, 0


# --- DataArray[ComplexField] arithmetic (free functions; compose complexutils) ----
def _check_same_complex_kind(a: DataArray, b: DataArray, op: str) -> str:
    ea, eb = a.element_type, b.element_type
    if not (getattr(ea, "is_complex_field", False) and getattr(eb, "is_complex_field", False)):
        raise TypeError(f"complex {op} needs both operands to be DataArray[ComplexField].")
    if ea.kind != eb.kind:
        raise TypeError(
            f"cannot apply complex {op} to {ea.kind}- and {eb.kind}-inner complex arrays "
            "(operands must share an inner kind).")
    return ea.kind


def _result_inner(kind: str, r: Format) -> type[DataField]:
    if kind == "int":
        return IntField.specialize(r.W, r.signed)
    from waveflow.hw.fixpoint import FixedField
    return FixedField.specialize(r.W, r.int_bits, r.signed, r.q_mode, r.o_mode)


def _wrap_complex(val: np.ndarray, inner: type[DataField]) -> DataArray:
    cf = ComplexField.specialize(inner)
    arr = np.asarray(val)
    shape = arr.shape if arr.shape else (1,)
    return DataArray.specialize(cf, max_shape=tuple(shape))(arr.reshape(shape))


def _wrap_float(out: np.ndarray) -> DataArray:
    bw = 64 if np.asarray(out).dtype == np.complex128 else 32
    return _wrap_complex(out, FloatField.specialize(bw))


def cadd(a: DataArray, b: DataArray) -> DataArray:
    """``a + b`` componentwise (full precision; int bits +1)."""
    kind = _check_same_complex_kind(a, b, "add")
    if kind == "float":
        return _wrap_float(cx.cadd_float(np.asarray(a.val), np.asarray(b.val)))
    out, r = cx.cadd(np.asarray(a.val), a.element_type.inner_format(),
                     np.asarray(b.val), b.element_type.inner_format())
    return _wrap_complex(out, _result_inner(kind, r))


def csub(a: DataArray, b: DataArray) -> DataArray:
    """``a - b`` componentwise (full precision; always signed)."""
    kind = _check_same_complex_kind(a, b, "sub")
    if kind == "float":
        return _wrap_float(cx.csub_float(np.asarray(a.val), np.asarray(b.val)))
    out, r = cx.csub(np.asarray(a.val), a.element_type.inner_format(),
                     np.asarray(b.val), b.element_type.inner_format())
    return _wrap_complex(out, _result_inner(kind, r))


def cmult(a: DataArray, b: DataArray) -> DataArray:
    """``a * b`` -- complex multiply; result ``(2W+1, 2I+1, signed)`` (signed inner only)."""
    kind = _check_same_complex_kind(a, b, "mul")
    if kind == "float":
        return _wrap_float(cx.cmult_float(np.asarray(a.val), np.asarray(b.val)))
    out, r = cx.cmult(np.asarray(a.val), a.element_type.inner_format(),
                      np.asarray(b.val), b.element_type.inner_format())
    return _wrap_complex(out, _result_inner(kind, r))


def conj(a: DataArray) -> DataArray:
    """``conj(a)`` -- imag negated; result ``(W+1, I+1, signed)`` (signed inner only)."""
    ea = a.element_type
    if not getattr(ea, "is_complex_field", False):
        raise TypeError("conj needs a DataArray[ComplexField].")
    if ea.kind == "float":
        return _wrap_float(cx.conj_float(np.asarray(a.val)))
    out, r = cx.conj(np.asarray(a.val), ea.inner_format())
    return _wrap_complex(out, _result_inner(ea.kind, r))
