"""VMAC command/accelerator tests — encode/decode round-trips + the ``specialize`` cascade.

``VmacCmd`` is a plain ``DataList`` (nested ``Region`` / ``Scalar`` sub-lists, an
``EnumField`` mode, ``BooleanField`` flags), so it must serialize / deserialize back to an
identical value across word widths — the wire-format contract for the (Phase-3) HLS kernel.
The structural widths live on ``VmacAccel`` (an ``HwComponent`` with ``HwParam`` fields);
its computed ``Cmd`` specializes the command schema so a command's field widths track the
silicon (``addr`` = ``mem_awidth`` bits; immediate ``re`` / ``im`` = ``data_bw`` bits); same
params → same schema class object.
"""
import pytest

from examples.vmac.golden import VmacAccel
from examples.vmac.vmac_cmd import Region, Scalar, VmacCmd, VmacMode
from waveflow.utils.fixputils import OMode, QMode

# a concrete accelerator: 32-bit addresses, 16-bit operands/immediates
ACCEL = VmacAccel(mem_dwidth=512, mem_awidth=32, data_bw=16, acc_bw=48, out_bw=12)
Cmd = ACCEL.Cmd


def _full_cmd():
    cmd = Cmd()
    cmd.n_rows, cmd.n_cols = 6, 4
    cmd.a = {"addr": 0, "row_stride": 4, "col_stride": 1}
    cmd.b = {"addr": 24, "row_stride": 4, "col_stride": 1}
    cmd.c = {"addr": 48, "row_stride": 1, "col_stride": 6}        # transposed access
    cmd.d = {"addr": 72, "row_stride": 4, "col_stride": 1}
    cmd.alpha = {"direct": 1, "re": -16, "im": 8, "addr": 0, "stride": 0}
    cmd.beta = {"direct": 0, "re": 0, "im": 0, "addr": 96, "stride": 1}
    cmd.b_one, cmd.c_zero, cmd.b_conj, cmd.reduce_rows = 0, 1, 1, 1
    cmd.mode = VmacMode.COMPLEX
    cmd.int_bits, cmd.shift = 8, 13
    cmd.q_rnd, cmd.o_sat = 1, 1
    return cmd


# --- encode / decode round-trips ----------------------------------------------
@pytest.mark.parametrize("word_bw", [16, 32, 64])
def test_vmac_cmd_roundtrip(word_bw):
    cmd = _full_cmd()
    restored = Cmd().deserialize(cmd.serialize(word_bw), word_bw)
    assert restored.val == cmd.val


def test_nested_region_scalar_roundtrip():
    reg = Region(addr=12, row_stride=-3, col_stride=2)           # negative stride survives
    r2 = Region().deserialize(reg.serialize(32), 32)
    assert r2.val == reg.val == {"addr": 12, "row_stride": -3, "col_stride": 2}

    sc = Scalar(direct=1, re=-100, im=77, addr=5, stride=-1)
    s2 = Scalar().deserialize(sc.serialize(32), 32)
    assert s2.val == sc.val
    assert s2.direct is True                                     # BooleanField -> Python bool


def test_default_cmd_roundtrips():
    cmd = Cmd()                                                  # all defaults / zeros
    restored = Cmd().deserialize(cmd.serialize(32), 32)
    assert restored.val == cmd.val


def test_mode_enum_field_roundtrips_both_values():
    for mode in (VmacMode.REAL, VmacMode.COMPLEX):
        cmd = Cmd()
        cmd.mode = mode
        restored = Cmd().deserialize(cmd.serialize(32), 32)
        assert int(restored.mode) == int(mode)


def test_signed_fields_preserve_negatives():
    cmd = _full_cmd()
    cmd.a = {"addr": 10, "row_stride": -4, "col_stride": -1}
    cmd.alpha = {"direct": 1, "re": -32768, "im": -1, "addr": 0, "stride": 0}  # data_bw=16 range
    restored = Cmd().deserialize(cmd.serialize(32), 32)
    assert restored.a.row_stride == -4 and restored.a.col_stride == -1
    assert restored.alpha.re == -32768 and restored.alpha.im == -1
    assert restored.val == cmd.val


def test_flags_are_booleanfields():
    cmd = _full_cmd()
    for flag in (cmd.b_one, cmd.c_zero, cmd.b_conj, cmd.reduce_rows, cmd.q_rnd, cmd.o_sat):
        assert isinstance(flag, bool)
    assert cmd.c_zero is True and cmd.b_one is False


def test_q_o_mode_properties():
    cmd = Cmd()
    cmd.q_rnd, cmd.o_sat = 0, 0
    assert cmd.q_mode is QMode.AP_TRN and cmd.o_mode is OMode.AP_WRAP
    cmd.q_rnd, cmd.o_sat = 1, 1
    assert cmd.q_mode is QMode.AP_RND and cmd.o_mode is OMode.AP_SAT


# --- the instance -> type bridge + cascade ------------------------------------
def test_accel_cmd_shared_and_distinct():
    # instances are not cached, but the schema their HwParam widths specialize IS:
    # same widths -> the same cached Cmd class; different widths -> a different one.
    a = VmacAccel(mem_dwidth=512, mem_awidth=32, data_bw=16, acc_bw=48, out_bw=12)
    b = VmacAccel(mem_dwidth=999, mem_awidth=32, data_bw=16, acc_bw=99, out_bw=7)
    assert a.Cmd is ACCEL.Cmd                                    # Cmd depends only on (mem_awidth, data_bw)
    assert b.Cmd is ACCEL.Cmd
    c = VmacAccel(mem_awidth=24, data_bw=12)
    assert c.Cmd is not ACCEL.Cmd


def test_accel_is_hwcomponent_with_hwparams():
    from waveflow.hw.hw_component import HwComponent, _hw_param_names
    assert issubclass(VmacAccel, HwComponent)
    # the extractor sees all five structural widths as HwParam template params
    assert _hw_param_names(VmacAccel) >= {"mem_dwidth", "mem_awidth", "data_bw", "acc_bw", "out_bw"}


def test_accel_carries_structural_params():
    assert (int(ACCEL.mem_dwidth), int(ACCEL.mem_awidth), int(ACCEL.data_bw),
            int(ACCEL.acc_bw), int(ACCEL.out_bw)) == (512, 32, 16, 48, 12)


def test_cmd_specialize_cached_and_matches_accel():
    assert ACCEL.Cmd is VmacCmd.specialize(mem_awidth=32, data_bw=16)
    assert VmacCmd.specialize(mem_awidth=32, data_bw=16) is VmacCmd.specialize(mem_awidth=32, data_bw=16)
    assert issubclass(ACCEL.Cmd, VmacCmd)


def test_cmd_field_widths_track_mem_awidth_and_data_bw():
    # addr (and strides) follow mem_awidth; immediate re/im follow data_bw
    region = ACCEL.Cmd.get_element_schema("a")
    scalar = ACCEL.Cmd.get_element_schema("alpha")
    assert region.get_element_schema("addr").get_bitwidth() == 32
    assert region.get_element_schema("row_stride").get_bitwidth() == 32
    assert scalar.get_element_schema("re").get_bitwidth() == 16
    assert scalar.get_element_schema("im").get_bitwidth() == 16

    wide = VmacAccel(mem_dwidth=256, mem_awidth=40, data_bw=24, acc_bw=64, out_bw=24)
    assert wide.Cmd.get_element_schema("a").get_element_schema("addr").get_bitwidth() == 40
    assert wide.Cmd.get_element_schema("alpha").get_element_schema("re").get_bitwidth() == 24


def test_region_scalar_specialize_cached_and_sized():
    assert Region.specialize(mem_awidth=32) is Region.specialize(mem_awidth=32)
    assert Region.specialize(mem_awidth=40) is not Region.specialize(mem_awidth=32)
    assert Region.specialize(mem_awidth=40).get_element_schema("addr").get_bitwidth() == 40
    assert Scalar.specialize(mem_awidth=32, data_bw=16) is Scalar.specialize(mem_awidth=32, data_bw=16)
    assert Scalar.specialize(mem_awidth=32, data_bw=16).get_element_schema("re").get_bitwidth() == 16
    assert Scalar.specialize(mem_awidth=32, data_bw=24).get_element_schema("re").get_bitwidth() == 24


def test_in_bw_out_bw_acc_bw_removed_from_cmd():
    fields = set(VmacCmd.elements)
    assert {"in_bw", "out_bw", "acc_bw"}.isdisjoint(fields)     # moved to the accelerator
    assert {"int_bits", "shift", "q_rnd", "o_sat"} <= fields    # runtime params stay
