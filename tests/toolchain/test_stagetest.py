import pytest

from pysilicon.toolchain import StageTest, TestStage


def test_stage_flow_from_names_returns_ordered_range_names() -> None:
    flow = StageTest.from_names(("csim", "csynth", "cosim", "generate_vcd"))

    assert flow.stage_names == ("csim", "csynth", "cosim", "generate_vcd")
    assert flow.range_names("csynth", "cosim") == ("csynth", "cosim")


def test_stage_flow_rejects_duplicate_stage_names() -> None:
    with pytest.raises(ValueError, match="Duplicate stage name 'csim'"):
        StageTest.from_names(("csim", "csim"))


def test_stage_flow_rejects_unknown_stage() -> None:
    flow = StageTest.from_names(("csim", "csynth", "cosim"))

    with pytest.raises(ValueError, match="Unsupported stage 'generate_vcd'"):
        flow.stage_index("generate_vcd")


def test_stage_flow_rejects_invalid_stage_range() -> None:
    flow = StageTest.from_names(("csim", "csynth", "cosim"))

    with pytest.raises(ValueError, match="must not come after"):
        flow.validate_range("cosim", "csynth")


def test_stage_flow_run_executes_selected_subset_in_order() -> None:
    seen: list[str] = []
    flow = StageTest(
        (
            TestStage("csim", lambda: seen.append("csim") or "csim-result"),
            TestStage("csynth", lambda: seen.append("csynth") or "csynth-result"),
            TestStage("cosim", lambda: seen.append("cosim") or "cosim-result"),
        )
    )

    results = flow.run("csynth", "cosim")

    assert seen == ["csynth", "cosim"]
    assert results == [
        ("csynth", "csynth-result"),
        ("cosim", "cosim-result"),
    ]


def test_stage_definition_requires_nonempty_name() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        TestStage("")