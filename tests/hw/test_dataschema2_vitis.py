import subprocess
from pathlib import Path

import numpy as np
import pytest

from pysilicon.codegen.build import CodeGenConfig
from pysilicon.codegen.streamutils import copy_streamutils
from pysilicon.hw.dataschema2 import DataList, FloatField, IntField
from pysilicon.xilinxutils import toolchain


class DemoPacket(DataList):
    elements = {
        "count": {
            "schema": IntField.specialize(bitwidth=16, signed=True),
            "description": "Signed sample count",
        },
        "gain": {
            "schema": FloatField.specialize(bitwidth=32),
            "description": "Linear gain value",
        },
        "mode": {
            "schema": IntField.specialize(bitwidth=8, signed=False),
            "description": "Unsigned mode selector",
        },
    }


RESOURCES_DIR = Path(__file__).with_name("resources")
ROUNDTRIP_CPP_PATH = RESOURCES_DIR / "dataschema2_roundtrip_test.cpp"
ROUNDTRIP_TCL_PATH = RESOURCES_DIR / "dataschema2_roundtrip_run.tcl"


@pytest.mark.vitis
def test_dataschema2_python_to_vitis_roundtrip(tmp_path: Path):
    vitis_path = toolchain.find_vitis_path()
    if not vitis_path:
        pytest.skip("Vitis installation not found; skipping dataschema2 integration test.")

    packet = DemoPacket(count=-21, gain=1.75, mode=5)
    input_words_path = tmp_path / "packet_words.txt"
    output_words_path = tmp_path / "packet_words_out.txt"

    packed = packet.serialize(word_bw=32)
    nwords = int(np.asarray(packed).size)
    np.savetxt(input_words_path, packed.astype(np.uint32), fmt="%u")

    cfg = CodeGenConfig(root_dir=tmp_path)
    include_path = DemoPacket.gen_include(cfg=cfg, word_bw_supported=[32])
    assert include_path.name == "demo_packet.h"

    copy_streamutils(cfg)

    cpp_src = ROUNDTRIP_CPP_PATH.read_text(encoding="utf-8").replace("__NWORDS__", str(nwords))
    (tmp_path / "roundtrip_test.cpp").write_text(cpp_src, encoding="utf-8")
    (tmp_path / "roundtrip_run.tcl").write_text(
        ROUNDTRIP_TCL_PATH.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    try:
        toolchain.run_vitis_hls(tmp_path / "roundtrip_run.tcl", work_dir=tmp_path)
    except RuntimeError as exc:
        pytest.skip(f"Vitis execution unavailable in current setup: {exc}")
    except subprocess.CalledProcessError as exc:
        pytest.fail(
            "Vitis execution failed for dataschema2 integration test.\n"
            f"Command: {exc.cmd}\n"
            f"Return code: {exc.returncode}\n"
            f"Stdout:\n{exc.stdout}\n"
            f"Stderr:\n{exc.stderr}"
        )

    got_words = np.loadtxt(output_words_path, dtype=np.uint32)
    got_words = np.asarray(got_words)
    if got_words.ndim == 0:
        got_words = got_words.reshape(1)

    restored = DemoPacket().deserialize(got_words, word_bw=32)
    assert restored.is_close(packet, rel_tol=1e-6, abs_tol=1e-6)